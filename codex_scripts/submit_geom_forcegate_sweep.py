#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional


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
    config_name: str
    run_name: str
    seed: int


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


def _parse_seeds(value: str) -> List[int]:
    seeds: List[int] = []
    for part in (value or "").split(","):
        part = part.strip()
        if not part:
            continue
        seeds.append(int(part))
    if not seeds:
        raise ValueError("Seed list is empty.")
    return seeds


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


def _write_launch_sbatch(
    *,
    run_dir: Path,
    resume_checkpoint: Path,
    geom_data_file: Path,
    resources: SlurmResources,
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
        spec.config_name,
        f"save_path={save_path}",
        "resume=true",
        f"checkpoint_path={resume_checkpoint}",
        f"seed={spec.seed}",
        f"dataloader.geom_data_file={geom_data_file}",
        "wandb.enabled=true",
        f"wandb.wandb_project={wandb_project}",
        f"wandb.wandb_name={spec.run_name}",
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
        '  eval "$(conda shell.bash hook)"',
        "  conda activate edm",
        "fi",
        "",
        f'cd "{REPO_ROOT}"',
        f'export PYTHONPATH="{REPO_ROOT}:{REPO_ROOT}/edm_source:${{PYTHONPATH:-}}"',
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


def _iter_specs(
    *,
    group_tag: str,
    inv060_seeds: Iterable[int],
    inv080_seeds: Iterable[int],
) -> List[RunSpec]:
    configs = [
        (
            "ddpo_geom_energy_force_vxu_suffix_tuned_stability_guard_shaping_forcealign_sp700_inv060_dup090_forcegate",
            "i60",
            inv060_seeds,
        ),
        (
            "ddpo_geom_energy_force_vxu_suffix_tuned_stability_guard_shaping_forcealign_sp700_inv080_dup090_forcegate",
            "i80",
            inv080_seeds,
        ),
    ]
    specs: List[RunSpec] = []
    for config_name, inv_tag, seeds in configs:
        for seed in seeds:
            run_name = f"gb1175_sp700_{inv_tag}_d90_forcegate_seed{seed}_{group_tag}"
            specs.append(RunSpec(config_name=config_name, run_name=run_name, seed=int(seed)))
    return specs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Submit GEOM forcegate sweep jobs to Slurm.")
    default_run_root = REPO_ROOT / "outputs" / "verl" / "geom_stability_queue"
    user = os.environ.get("USER", "").strip()
    if user:
        default_run_root = Path("/home") / user / "logs" / "geom_stability_queue"
    parser.add_argument(
        "--resume-checkpoint",
        type=str,
        required=True,
        help="Path to the DDPO checkpoint (.pth) to resume from (use an eval_checkpoint snapshot).",
    )
    parser.add_argument(
        "--geom-data-file",
        type=str,
        default=str(REPO_ROOT / "geom_drugs_30.npy"),
        help="Path to geom_drugs_30.npy.",
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
    parser.add_argument(
        "--inv060-seeds",
        type=str,
        default="79,80",
        help="Comma-separated seeds for inv060 jobs.",
    )
    parser.add_argument(
        "--inv080-seeds",
        type=str,
        default="81,82",
        help="Comma-separated seeds for inv080 jobs.",
    )
    parser.add_argument(
        "--force-alignment-weight",
        type=float,
        default=None,
        help=(
            "Optional override for train.force_alignment_weight. "
            "When set, the sweep appends a `faXX` tag to the run name (XX = round(100*w))."
        ),
    )
    parser.add_argument("--partition", type=str, default="gpu")
    parser.add_argument("--gres", type=str, default="gpu:h100:1")
    parser.add_argument("--cpus-per-task", type=int, default=8)
    parser.add_argument("--mem", type=str, default="64G")
    parser.add_argument("--time", type=str, default="12:00:00")
    parser.add_argument("--wandb-project", type=str, default="ddpo")
    parser.add_argument("--dry-run", action="store_true", help="Write sbatch files but do not submit.")
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

    group_name = (args.group_name or "").strip()
    if not group_name:
        group_name = datetime.now().strftime("%Y%m%d_%H%M%S")

    inv060_seeds = _parse_seeds(args.inv060_seeds)
    inv080_seeds = _parse_seeds(args.inv080_seeds)

    hydra_overrides: List[str] = []
    group_tag_suffix = ""
    if args.force_alignment_weight is not None:
        weight = float(args.force_alignment_weight)
        if weight < 0.0:
            raise ValueError("--force-alignment-weight must be >= 0")
        hydra_overrides.append(f"train.force_alignment_weight={weight}")
        fa_tag = f"fa{int(round(weight * 100)):02d}"
        group_tag_suffix = f"_{fa_tag}"

    resources = SlurmResources(
        partition=str(args.partition),
        gres=str(args.gres),
        cpus_per_task=int(args.cpus_per_task),
        mem=str(args.mem),
        time=str(args.time),
    )

    group_dir = run_root / f"branch_best1175_forcegate_{group_name}"
    group_dir.mkdir(parents=True, exist_ok=True)

    specs = _iter_specs(
        group_tag=f"{group_name}{group_tag_suffix}",
        inv060_seeds=inv060_seeds,
        inv080_seeds=inv080_seeds,
    )

    submitted = []
    for spec in specs:
        run_dir = group_dir / spec.run_name
        sbatch_path = _write_launch_sbatch(
            run_dir=run_dir,
            resume_checkpoint=resume_checkpoint,
            geom_data_file=geom_data_file,
            resources=resources,
            spec=spec,
            wandb_project=str(args.wandb_project),
            hydra_overrides=hydra_overrides,
        )
        if args.dry_run:
            submitted.append({"run_dir": str(run_dir), "job_id": None, "sbatch": str(sbatch_path)})
            continue

        job_id = _sbatch_submit(sbatch_path)
        submitted.append({"run_dir": str(run_dir), "job_id": job_id, "sbatch": str(sbatch_path)})

    print(json.dumps({"group_dir": str(group_dir), "runs": submitted}, indent=2))


if __name__ == "__main__":
    main()
