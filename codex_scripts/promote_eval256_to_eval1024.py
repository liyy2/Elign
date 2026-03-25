#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class PromotionCandidate:
    run_dir: Path
    metrics_path: Path
    validity: float
    uniqueness: float
    stability_rate: Optional[float]
    atom_stability_mean: Optional[float]
    checkpoint: Optional[Path]
    checkpoint_epoch: Optional[int]


def _default_run_root() -> Path:
    user = os.environ.get("USER", "").strip()
    if user:
        return Path("/home") / user / "logs" / "geom_stability_queue"
    return REPO_ROOT / "outputs" / "verl" / "geom_stability_queue"


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _extract_rdkit_metrics(payload: Dict[str, Any]) -> Tuple[Optional[float], Optional[float]]:
    rdkit_metrics = payload.get("rdkit_metrics")
    if isinstance(rdkit_metrics, dict):
        validity = rdkit_metrics.get("validity")
        uniqueness = rdkit_metrics.get("uniqueness")
        try:
            validity = float(validity) if validity is not None else None
        except Exception:
            validity = None
        try:
            uniqueness = float(uniqueness) if uniqueness is not None else None
        except Exception:
            uniqueness = None
        return validity, uniqueness

    validity_raw = payload.get("rdkit_validity")
    uniqueness_raw = payload.get("rdkit_uniqueness")
    validity = None
    uniqueness = None
    try:
        validity = float(validity_raw) if validity_raw is not None else None
    except Exception:
        validity = None
    try:
        uniqueness = float(uniqueness_raw) if uniqueness_raw is not None else None
    except Exception:
        uniqueness = None
    return validity, uniqueness


def _extract_optional_float(payload: Dict[str, Any], key: str) -> Optional[float]:
    value = payload.get(key)
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _extract_optional_int(payload: Dict[str, Any], key: str) -> Optional[int]:
    value = payload.get(key)
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return None


def _iter_metrics_files(run_root: Path, pattern: str) -> Iterable[Path]:
    yield from sorted(run_root.rglob(pattern))


def _has_1024_eval(run_dir: Path, eval_seed: int) -> bool:
    needle = f"eval_metrics_1024_seed{eval_seed}"
    for path in run_dir.glob(f"{needle}*skipmlff.json"):
        if path.is_file():
            return True
    return False


def _build_candidate(metrics_path: Path) -> Optional[PromotionCandidate]:
    payload = _load_json(metrics_path)
    if not isinstance(payload, dict):
        return None

    validity, uniqueness = _extract_rdkit_metrics(payload)
    if validity is None or uniqueness is None:
        return None

    run_dir = metrics_path.parent
    stability_rate = _extract_optional_float(payload, "stability_rate")
    atom_stability_mean = _extract_optional_float(payload, "atom_stability_mean")

    checkpoint = payload.get("eval_checkpoint")
    checkpoint_path: Optional[Path] = None
    if isinstance(checkpoint, str) and checkpoint.strip():
        checkpoint_path = Path(checkpoint).expanduser()
        if not checkpoint_path.is_absolute():
            checkpoint_path = (run_dir / checkpoint_path).resolve()
        if not checkpoint_path.exists():
            checkpoint_path = None

    checkpoint_epoch = _extract_optional_int(payload, "eval_checkpoint_epoch")

    return PromotionCandidate(
        run_dir=run_dir,
        metrics_path=metrics_path,
        validity=float(validity),
        uniqueness=float(uniqueness),
        stability_rate=stability_rate,
        atom_stability_mean=atom_stability_mean,
        checkpoint=checkpoint_path,
        checkpoint_epoch=checkpoint_epoch,
    )


def _submit_eval(
    *,
    eval_sbatch: Path,
    candidate: PromotionCandidate,
    eval_seed: int,
    num_molecules: int,
    no_share: bool,
    dry_run: bool,
) -> Optional[int]:
    export_items = [
        "ALL",
        f"RUN_DIR={candidate.run_dir}",
        f"SEED={int(eval_seed)}",
        f"NUM_MOLECULES={int(num_molecules)}",
    ]

    tag = None
    if candidate.checkpoint_epoch is not None:
        tag = f"ckptbest_e{candidate.checkpoint_epoch}_from256"
        export_items.append(f"TAG={tag}")

    if candidate.checkpoint is not None:
        export_items.append(f"CHECKPOINT={candidate.checkpoint}")

    if no_share:
        export_items.append("NO_SHARE=1")

    cmd = [
        "sbatch",
        "--parsable",
        f"--export={','.join(export_items)}",
        str(eval_sbatch),
    ]

    if dry_run:
        print("[DRY RUN]", " ".join(cmd))
        return None

    proc = subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        check=False,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        stderr = (proc.stderr or "").strip()
        stdout = (proc.stdout or "").strip()
        msg = stderr or stdout or f"sbatch failed with code {proc.returncode}"
        raise RuntimeError(msg)

    raw = (proc.stdout or "").strip()
    if not raw:
        return None
    try:
        return int(raw.split(";", 1)[0])
    except Exception:
        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Scan eval_metrics_256 files and submit 1024-sample evaluations for runs that pass a validity threshold."
        )
    )
    parser.add_argument(
        "--run-root",
        type=str,
        default=str(_default_run_root()),
        help="Root directory containing run subfolders (default: /home/$USER/logs/geom_stability_queue).",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="eval_metrics_256_seed*_skipmlff.json",
        help="Glob pattern (relative to each run dir) used to find eval256 metric files.",
    )
    parser.add_argument(
        "--eval-sbatch",
        type=str,
        default=str(REPO_ROOT / "codex_scripts" / "eval_geom_1024.sbatch"),
        help="Path to eval sbatch script (accepts RUN_DIR/NUM_MOLECULES/SEED/CHECKPOINT/TAG via --export).",
    )
    parser.add_argument("--eval-seed", type=int, default=42, help="Evaluation seed for promoted 1024 eval.")
    parser.add_argument("--num-molecules", type=int, default=1024, help="Number of molecules for promoted eval.")
    parser.add_argument(
        "--validity-threshold",
        type=float,
        default=0.99,
        help="Promote runs with RDKit validity >= this threshold.",
    )
    parser.add_argument(
        "--uniqueness-threshold",
        type=float,
        default=0.0,
        help="Optional: require RDKit uniqueness >= this threshold (default: disabled).",
    )
    parser.add_argument(
        "--mol-stability-threshold",
        type=float,
        default=0.0,
        help="Optional: require stability_rate >= this threshold (default: disabled).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=8,
        help="Maximum number of 1024 eval jobs to submit in one invocation.",
    )
    parser.add_argument(
        "--no-share",
        action="store_true",
        help="Disable shared initial noise during evaluation (sets NO_SHARE=1).",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print sbatch commands without submitting.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_root = Path(args.run_root).expanduser()
    if not run_root.is_absolute():
        run_root = (REPO_ROOT / run_root).resolve()
    if not run_root.exists():
        raise FileNotFoundError(f"Run root not found: {run_root}")

    eval_sbatch = Path(args.eval_sbatch).expanduser()
    if not eval_sbatch.is_absolute():
        eval_sbatch = (REPO_ROOT / eval_sbatch).resolve()
    if not eval_sbatch.exists():
        raise FileNotFoundError(f"Eval sbatch not found: {eval_sbatch}")

    candidates: List[PromotionCandidate] = []
    for metrics_path in _iter_metrics_files(run_root, str(args.pattern)):
        candidate = _build_candidate(metrics_path)
        if candidate is None:
            continue
        if _has_1024_eval(candidate.run_dir, int(args.eval_seed)):
            continue
        if candidate.validity < float(args.validity_threshold):
            continue
        if candidate.uniqueness < float(args.uniqueness_threshold or 0.0):
            continue
        if float(args.mol_stability_threshold or 0.0) > 0.0:
            if (candidate.stability_rate or 0.0) < float(args.mol_stability_threshold):
                continue
        candidates.append(candidate)

    candidates.sort(key=lambda c: (c.validity, c.uniqueness, c.stability_rate or 0.0), reverse=True)
    to_submit = candidates[: max(0, int(args.limit or 0))]

    submitted: List[Dict[str, Any]] = []
    for candidate in to_submit:
        job_id = _submit_eval(
            eval_sbatch=eval_sbatch,
            candidate=candidate,
            eval_seed=int(args.eval_seed),
            num_molecules=int(args.num_molecules),
            no_share=bool(args.no_share),
            dry_run=bool(args.dry_run),
        )
        submitted.append(
            {
                "run_dir": str(candidate.run_dir),
                "metrics": str(candidate.metrics_path),
                "validity": candidate.validity,
                "uniqueness": candidate.uniqueness,
                "stability_rate": candidate.stability_rate,
                "job_id": job_id,
            }
        )

    print(json.dumps({"submitted": submitted}, indent=2))


if __name__ == "__main__":
    main()

