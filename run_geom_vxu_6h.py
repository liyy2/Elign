#!/usr/bin/env python3
import argparse
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run GEOM VERL/DDPO experiments focused on RDKit validity×uniqueness.\n"
            "Phase 1 (experiment A): force-only (suffix-only shared-prefix rollouts).\n"
            "Phase 2 (experiment B): energy+force (resume from phase 1 best checkpoint).\n"
            "Each phase has its own max-hour budget (defaults to 6h each)."
        )
    )
    parser.add_argument("--phase1-hours", type=float, default=6.0)
    parser.add_argument("--phase2-hours", type=float, default=6.0)
    parser.add_argument("--check-minutes", type=float, default=30.0)
    parser.add_argument("--phase1-config", type=str, default="ddpo_geom_force_vxu_suffix")
    parser.add_argument("--phase2-config", type=str, default="ddpo_geom_energy_force_vxu_suffix")
    parser.add_argument("--out-root", type=str, default="outputs/verl/geom_vxu")
    parser.add_argument("--run-name", type=str, default=None)
    return parser.parse_args()


def _safe_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def read_checkpoint_metrics(run_dir: Path) -> Optional[Tuple[int, Dict[str, Any]]]:
    ckpt_path = run_dir / "checkpoint_latest.pth"
    if not ckpt_path.exists():
        return None
    try:
        payload = torch.load(ckpt_path, map_location="cpu")
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    epoch = payload.get("epoch")
    metrics = payload.get("metrics") or {}
    if epoch is None or not isinstance(metrics, dict):
        return None
    try:
        epoch = int(epoch)
    except Exception:
        return None
    return epoch, metrics


def format_metrics(metrics: Dict[str, Any]) -> str:
    fields = [
        ("reward", metrics.get("reward")),
        ("rdkit_validity", metrics.get("rdkit_validity")),
        ("rdkit_uniqueness", metrics.get("rdkit_uniqueness")),
        ("validity_x_uniqueness", metrics.get("validity_x_uniqueness")),
    ]
    parts = []
    for key, value in fields:
        value_f = _safe_float(value)
        if value_f is None:
            continue
        parts.append(f"{key}={value_f:.4f}")
    return " ".join(parts) if parts else "(no metrics yet)"


def run_phase(
    phase_name: str,
    config_name: str,
    run_dir: Path,
    max_time_hours: float,
    check_minutes: float,
    resume: bool = False,
    checkpoint_path: Optional[Path] = None,
) -> int:
    run_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "python",
        "run_verl_diffusion.py",
        "--config-name",
        config_name,
        f"save_path={str(run_dir)}",
        f"train.max_time_hours={max_time_hours}",
    ]
    if resume:
        cmd.append("resume=true")
    if checkpoint_path is not None:
        cmd.append(f"checkpoint_path={str(checkpoint_path)}")

    print(f"[{phase_name}] launch: {' '.join(cmd)}")
    proc = subprocess.Popen(cmd, cwd=str(Path(__file__).resolve().parent))

    check_seconds = max(10.0, float(check_minutes) * 60.0)
    last_epoch = None
    while proc.poll() is None:
        time.sleep(check_seconds)
        latest = read_checkpoint_metrics(run_dir)
        if latest is None:
            print(f"[{phase_name}] checkpoint not written yet")
            continue
        epoch, metrics = latest
        if last_epoch is None or epoch != last_epoch:
            print(f"[{phase_name}] epoch={epoch} {format_metrics(metrics)}")
            last_epoch = epoch

    rc = int(proc.returncode or 0)
    print(f"[{phase_name}] done: exit_code={rc}")
    return rc


def main() -> None:
    args = parse_args()

    phase1_hours = max(0.0, float(args.phase1_hours))
    phase2_hours = max(0.0, float(args.phase2_hours))

    run_name = args.run_name
    if not run_name:
        run_name = datetime.now().strftime("geom_vxu_%Y%m%d_%H%M%S")

    out_root = Path(args.out_root).resolve()
    phase1_dir = out_root / f"{run_name}_phase1_force"
    phase2_dir = out_root / f"{run_name}_phase2_energy"

    print(f"Run root: {out_root}")
    if phase1_hours > 0.0:
        print(f"Phase 1: {phase1_dir} ({phase1_hours:.2f}h)")
    else:
        print("Phase 1: (skipped)")
    if phase2_hours > 0.0:
        print(f"Phase 2: {phase2_dir} ({phase2_hours:.2f}h)")
    else:
        print("Phase 2: (skipped)")

    resume_ckpt = None
    if phase1_hours > 0.0:
        rc1 = run_phase(
            phase_name="phase1_force",
            config_name=args.phase1_config,
            run_dir=phase1_dir,
            max_time_hours=phase1_hours,
            check_minutes=float(args.check_minutes),
            resume=False,
            checkpoint_path=None,
        )
        if rc1 != 0:
            raise SystemExit(rc1)

        best_ckpt = phase1_dir / "checkpoint_best.pth"
        latest_ckpt = phase1_dir / "checkpoint_latest.pth"
        resume_ckpt = best_ckpt if best_ckpt.exists() else latest_ckpt if latest_ckpt.exists() else None
        if resume_ckpt is None:
            raise FileNotFoundError("Phase 1 produced no checkpoint to resume from.")
    if phase2_hours <= 0.0:
        return
    if resume_ckpt is None:
        raise FileNotFoundError("Need a phase1 checkpoint to run phase2 (set --phase1-hours > 0).")

    rc2 = run_phase(
        phase_name="phase2_energy",
        config_name=args.phase2_config,
        run_dir=phase2_dir,
        max_time_hours=phase2_hours,
        check_minutes=float(args.check_minutes),
        resume=True,
        checkpoint_path=resume_ckpt,
    )
    if rc2 != 0:
        raise SystemExit(rc2)


if __name__ == "__main__":
    os.environ.setdefault("WANDB_MODE", "offline")
    main()
