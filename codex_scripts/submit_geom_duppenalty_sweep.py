#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional


REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_MAX_TIME_MARGIN_MINUTES = 10.0


@dataclass(frozen=True)
class SlurmResources:
    partition: str
    gres: str
    cpus_per_task: int
    mem: str
    time: str


@dataclass(frozen=True)
class RunSpec:
    run_name: str
    seed: int
    duplicate_penalty_scale: float


def _slurm_time_to_seconds(value: str) -> Optional[int]:
    raw = (value or "").strip()
    if not raw:
        return None

    days = 0
    rest = raw
    if "-" in raw:
        day_part, rest = raw.split("-", 1)
        try:
            days = int(day_part)
        except ValueError:
            return None

    parts = [p for p in rest.split(":") if p != ""]
    try:
        nums = [int(p) for p in parts]
    except ValueError:
        return None

    hours = 0
    minutes = 0
    seconds = 0
    if len(nums) == 3:
        hours, minutes, seconds = nums
    elif len(nums) == 2:
        minutes, seconds = nums
    elif len(nums) == 1:
        minutes = nums[0]
    else:
        return None

    total = days * 86400 + hours * 3600 + minutes * 60 + seconds
    if total <= 0:
        return None
    return total


def _has_override(overrides: Optional[List[str]], prefix: str) -> bool:
    if not overrides:
        return False
    for item in overrides:
        if str(item).strip().startswith(prefix):
            return True
    return False


def _compute_max_time_hours(
    resources: SlurmResources, *, margin_minutes: float = _DEFAULT_MAX_TIME_MARGIN_MINUTES
) -> Optional[float]:
    total_seconds = _slurm_time_to_seconds(resources.time)
    if total_seconds is None:
        return None
    margin_seconds = max(0.0, float(margin_minutes)) * 60.0
    usable_seconds = max(60.0, float(total_seconds) - margin_seconds)
    return usable_seconds / 3600.0


def _parse_csv_floats(value: str) -> List[float]:
    out: List[float] = []
    for part in (value or "").split(","):
        part = part.strip()
        if not part:
            continue
        out.append(float(part))
    if not out:
        raise ValueError("Float list is empty.")
    return out


def _parse_csv_ints(value: str) -> List[int]:
    out: List[int] = []
    for part in (value or "").split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    if not out:
        raise ValueError("Seed list is empty.")
    return out


def _sbatch_submit(sbatch_path: Path) -> Optional[int]:
    proc = subprocess.run(
        ["sbatch", "--parsable", str(sbatch_path)],
        cwd=str(REPO_ROOT),
        check=False,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        return None
    out = (proc.stdout or "").strip()
    if not out:
        return None
    try:
        return int(out.split(";", 1)[0])
    except Exception:
        return None


def _sbatch_submit_eval(
    *,
    eval_sbatch: Path,
    dependency_job_id: int,
    run_dir: Path,
    seed: int,
    num_molecules: int,
    checkpoint: str,
    no_share_initial_noise: bool,
    repair_invalid: bool,
    repair_unstable: bool,
    repair_add_h: bool,
    repair_steps: int,
    repair_alpha: float,
) -> Optional[int]:
    export_items = [
        "ALL",
        f"RUN_DIR={run_dir}",
        f"SEED={int(seed)}",
        f"NUM_MOLECULES={int(num_molecules)}",
    ]
    checkpoint = str(checkpoint or "").strip()
    if checkpoint:
        export_items.append(f"CHECKPOINT={checkpoint}")
    if no_share_initial_noise:
        export_items.append("NO_SHARE=1")
    if repair_invalid:
        export_items.append("REPAIR_INVALID=1")
    if repair_unstable:
        export_items.append("REPAIR_UNSTABLE=1")
    if repair_invalid or repair_unstable:
        export_items.append(f"REPAIR_STEPS={int(repair_steps)}")
        export_items.append(f"REPAIR_ALPHA={float(repair_alpha)}")
    if repair_add_h:
        export_items.append("REPAIR_ADD_H=1")

    proc = subprocess.run(
        [
            "sbatch",
            "--parsable",
            f"--dependency=afterok:{int(dependency_job_id)}",
            f"--export={','.join(export_items)}",
            str(eval_sbatch),
        ],
        cwd=str(REPO_ROOT),
        check=False,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        return None
    out = (proc.stdout or "").strip()
    if not out:
        return None
    try:
        return int(out.split(";", 1)[0])
    except Exception:
        return None


def _write_launch_sbatch(
    *,
    run_dir: Path,
    resume_checkpoint: Path,
    geom_data_file: Path,
    resources: SlurmResources,
    config_name: str,
    spec: RunSpec,
    wandb_project: str,
    hydra_overrides: Optional[List[str]] = None,
) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    sbatch_path = run_dir / "launch.sbatch"

    job_name = spec.run_name
    if len(job_name) > 128:
        job_name = job_name[:128]

    out_path = run_dir / "slurm_%j.out"
    err_path = run_dir / "slurm_%j.err"

    try:
        save_path = run_dir.relative_to(REPO_ROOT)
    except ValueError:
        save_path = run_dir

    cmd = [
        "python",
        "-u",
        "run_verl_diffusion.py",
        "--config-name",
        config_name,
        f"save_path={save_path}",
        "resume=true",
        f"checkpoint_path={resume_checkpoint}",
        f"seed={spec.seed}",
        f"dataloader.geom_data_file={geom_data_file}",
        "wandb.enabled=true",
        f"wandb.wandb_project={wandb_project}",
        f"wandb.wandb_name={spec.run_name}",
        f"filters.duplicate_penalty_scale={spec.duplicate_penalty_scale}",
    ]
    max_time_hours = _compute_max_time_hours(resources)
    if max_time_hours is not None and not _has_override(hydra_overrides, "train.max_time_hours="):
        cmd.append(f"train.max_time_hours={max_time_hours:.3f}")
    if hydra_overrides:
        cmd.extend([str(item) for item in hydra_overrides])

    lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name={job_name}",
        f"#SBATCH --partition={resources.partition}",
        f"#SBATCH --gres={resources.gres}",
        "#SBATCH --ntasks=1",
        f"#SBATCH --cpus-per-task={resources.cpus_per_task}",
        f"#SBATCH --mem={resources.mem}",
        f"#SBATCH --time={resources.time}",
        f"#SBATCH --output={out_path}",
        f"#SBATCH --error={err_path}",
        "",
        "set -euo pipefail",
        "",
        "module load miniconda >/dev/null 2>&1 || true",
        'if command -v conda >/dev/null 2>&1; then',
        '  eval \"$(conda shell.bash hook)\"',
        "  conda activate edm",
        "fi",
        "",
        f'cd \"{REPO_ROOT}\"',
        f'export PYTHONPATH=\"{REPO_ROOT}:{REPO_ROOT}/edm_source:${{PYTHONPATH:-}}\"',
        "export HF_HUB_OFFLINE=1",
        "export VERL_TQDM=0",
        "export HYDRA_FULL_ERROR=1",
        "export PYTHONUNBUFFERED=1",
        "export PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync",
        f'export WANDB_DIR="{run_dir}/wandb"',
        f'mkdir -p "{run_dir}/wandb"',
        "",
        'if [[ -n "${WANDB_API_KEY:-}" ]]; then',
        '  export WANDB_MODE="${WANDB_MODE:-online}"',
        'elif [[ -f "${HOME}/.netrc" ]] && grep -q "api.wandb.ai" "${HOME}/.netrc"; then',
        '  export WANDB_MODE="${WANDB_MODE:-online}"',
        "else",
        '  export WANDB_MODE="${WANDB_MODE:-offline}"',
        "fi",
        "",
        " ".join(cmd),
        "",
    ]
    sbatch_path.write_text("\n".join(lines))
    return sbatch_path


def parse_args() -> argparse.Namespace:
    default_run_root = REPO_ROOT / "outputs" / "verl" / "geom_stability_queue"
    user = os.environ.get("USER", "").strip()
    if user:
        default_run_root = Path("/home") / user / "logs" / "geom_stability_queue"

    parser = argparse.ArgumentParser(
        description=(
            "Submit GEOM duplicate-penalty sweeps starting from a frozen eval_checkpoint snapshot.\n"
            "Default config mirrors the current 1024-sample leaderboard leader settings."
        )
    )
    parser.add_argument(
        "--resume-checkpoint",
        type=str,
        required=True,
        help="Path to a DDPO checkpoint (.pth) to resume from (prefer eval_checkpoint snapshots).",
    )
    parser.add_argument(
        "--geom-data-file",
        type=str,
        default=str(REPO_ROOT / "geom_drugs_30.npy"),
        help="Path to geom_drugs_30.npy.",
    )
    parser.add_argument(
        "--config-name",
        type=str,
        default="ddpo_geom_best1174_sw7_lr1e7_kl015",
        help="Hydra config name to use as baseline.",
    )
    parser.add_argument(
        "--duplicate-penalty-scales",
        type=str,
        default="0.50,0.70",
        help="Comma-separated duplicate penalty scales to sweep.",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="90,91",
        help="Comma-separated training seeds. Runs all combinations with --duplicate-penalty-scales.",
    )
    parser.add_argument(
        "--run-root",
        type=str,
        default=str(default_run_root),
        help="Root directory for sweep outputs.",
    )
    parser.add_argument(
        "--group-name",
        type=str,
        default="",
        help="Optional group name (default: auto timestamp).",
    )
    parser.add_argument("--partition", type=str, default="gpu")
    parser.add_argument("--gres", type=str, default="gpu:h100:1")
    parser.add_argument("--cpus-per-task", type=int, default=8)
    parser.add_argument("--mem", type=str, default="64G")
    parser.add_argument("--time", type=str, default="12:00:00")
    parser.add_argument("--wandb-project", type=str, default="ddpo")
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Extra Hydra overrides applied to every run (repeatable).",
    )
    parser.add_argument("--dry-run", action="store_true", help="Write sbatch files but do not submit.")
    parser.add_argument(
        "--schedule-eval",
        action="store_true",
        help="Submit dependent 1024-sample eval jobs via codex_scripts/eval_geom_1024.sbatch.",
    )
    parser.add_argument(
        "--eval-sbatch",
        type=str,
        default=str(REPO_ROOT / "codex_scripts" / "eval_geom_1024.sbatch"),
        help="Path to eval sbatch script (expects RUN_DIR in --export).",
    )
    parser.add_argument("--eval-seed", type=int, default=42, help="Evaluation seed (default: 42).")
    parser.add_argument(
        "--eval-num-molecules",
        type=int,
        default=1024,
        help="Number of molecules to sample during evaluation (default: 1024).",
    )
    parser.add_argument(
        "--eval-checkpoint",
        type=str,
        default="",
        help=(
            "Optional checkpoint filename (relative to RUN_DIR) or absolute path passed to eval script.\n"
            "Defaults to letting eval_geom_1024.sbatch choose checkpoint_best/Latest."
        ),
    )
    parser.add_argument(
        "--eval-no-share",
        action="store_true",
        help="Set NO_SHARE=1 for evaluation (disables shared initial noise).",
    )
    parser.add_argument(
        "--eval-repair-invalid",
        action="store_true",
        help=(
            "Set REPAIR_INVALID=1 for evaluation (compute repaired metrics via UMA relaxation). "
            "When used with codex_scripts/eval_geom_1024.sbatch, this also writes the base (no-repair) metrics."
        ),
    )
    parser.add_argument(
        "--eval-repair-unstable",
        action="store_true",
        help="Set REPAIR_UNSTABLE=1 for evaluation (UMA relaxation on QM9-unstable samples).",
    )
    parser.add_argument(
        "--eval-repair-add-h",
        action="store_true",
        help="Set REPAIR_ADD_H=1 for evaluation (convert implicit H into explicit H before stability check).",
    )
    parser.add_argument("--eval-repair-steps", type=int, default=3, help="REPAIR_STEPS for eval (default: 3).")
    parser.add_argument("--eval-repair-alpha", type=float, default=0.02, help="REPAIR_ALPHA for eval (default: 0.02).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    resume_checkpoint = Path(args.resume_checkpoint).expanduser()
    if not resume_checkpoint.is_absolute():
        resume_checkpoint = (REPO_ROOT / resume_checkpoint).resolve()
    if not resume_checkpoint.exists():
        raise FileNotFoundError(f"Resume checkpoint not found: {resume_checkpoint}")

    geom_data_file = Path(args.geom_data_file).expanduser()
    if not geom_data_file.is_absolute():
        geom_data_file = (REPO_ROOT / geom_data_file).resolve()
    if not geom_data_file.exists():
        raise FileNotFoundError(f"GEOM data file not found: {geom_data_file}")

    run_root = Path(args.run_root).expanduser()
    if not run_root.is_absolute():
        run_root = (REPO_ROOT / run_root).resolve()
    run_root.mkdir(parents=True, exist_ok=True)

    eval_sbatch = Path(args.eval_sbatch).expanduser()
    if not eval_sbatch.is_absolute():
        eval_sbatch = (REPO_ROOT / eval_sbatch).resolve()
    if args.schedule_eval and not eval_sbatch.exists():
        raise FileNotFoundError(f"Eval sbatch not found: {eval_sbatch}")

    group_name = (args.group_name or "").strip()
    if not group_name:
        group_name = datetime.now().strftime("%Y%m%d_%H%M%S")

    dup_scales = _parse_csv_floats(args.duplicate_penalty_scales)
    seeds = _parse_csv_ints(args.seeds)

    resources = SlurmResources(
        partition=str(args.partition),
        gres=str(args.gres),
        cpus_per_task=int(args.cpus_per_task),
        mem=str(args.mem),
        time=str(args.time),
    )

    group_dir = run_root / f"branch_best1174_dupsweep_{group_name}"
    group_dir.mkdir(parents=True, exist_ok=True)

    submitted = []
    for dup_scale in dup_scales:
        dup_tag = f"dup{int(round(dup_scale * 100)):02d}"
        for seed in seeds:
            run_name = f"gb1174_{dup_tag}_seed{seed}_{group_name}"
            run_dir = group_dir / run_name
            spec = RunSpec(run_name=run_name, seed=int(seed), duplicate_penalty_scale=float(dup_scale))
            sbatch_path = _write_launch_sbatch(
                run_dir=run_dir,
                resume_checkpoint=resume_checkpoint,
                geom_data_file=geom_data_file,
                resources=resources,
                config_name=str(args.config_name),
                spec=spec,
                wandb_project=str(args.wandb_project),
                hydra_overrides=[str(item) for item in (args.override or [])],
            )
            if args.dry_run:
                submitted.append(
                    {
                        "run_dir": str(run_dir),
                        "job_id": None,
                        "sbatch": str(sbatch_path),
                        "eval_job_id": None,
                        "eval_sbatch": str(eval_sbatch) if args.schedule_eval else None,
                    }
                )
                continue

            job_id = _sbatch_submit(sbatch_path)
            eval_job_id = None
            if args.schedule_eval and job_id is not None:
                eval_job_id = _sbatch_submit_eval(
                    eval_sbatch=eval_sbatch,
                    dependency_job_id=int(job_id),
                    run_dir=run_dir,
                    seed=int(args.eval_seed),
                    num_molecules=int(args.eval_num_molecules),
                    checkpoint=str(args.eval_checkpoint),
                    no_share_initial_noise=bool(args.eval_no_share),
                    repair_invalid=bool(args.eval_repair_invalid),
                    repair_unstable=bool(args.eval_repair_unstable),
                    repair_add_h=bool(args.eval_repair_add_h),
                    repair_steps=int(args.eval_repair_steps),
                    repair_alpha=float(args.eval_repair_alpha),
                )
            submitted.append(
                {
                    "run_dir": str(run_dir),
                    "job_id": job_id,
                    "sbatch": str(sbatch_path),
                    "eval_job_id": eval_job_id,
                    "eval_sbatch": str(eval_sbatch) if args.schedule_eval else None,
                }
            )

    print(json.dumps({"group_dir": str(group_dir), "runs": submitted}, indent=2))


if __name__ == "__main__":
    main()
