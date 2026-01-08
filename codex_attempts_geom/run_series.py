#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
ATTEMPTS_ROOT = Path(__file__).resolve().parent


@dataclass
class RunResult:
    run_dir: Path
    log_path: Path
    status: str
    start_time: str
    duration_hours: float
    best_epoch: Optional[int]
    best_metrics: Dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a multi-experiment GEOM RL series with periodic monitoring and "
            "auto-logging into codex_attempts_geom/."
        )
    )
    parser.add_argument(
        "--plan",
        type=str,
        default=str(ATTEMPTS_ROOT / "series_plan.yaml"),
        help="YAML plan file (see codex_attempts_geom/series_plan.yaml).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned commands but do not launch training.",
    )
    return parser.parse_args()


def _safe_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_int(value: Any) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _next_attempt_id() -> int:
    max_id = 0
    for path in ATTEMPTS_ROOT.glob("attempt_*"):
        if not path.is_dir():
            continue
        try:
            attempt_id = int(path.name.split("_", 1)[1])
        except Exception:
            continue
        max_id = max(max_id, attempt_id)
    return max_id + 1


def read_checkpoint_metrics(run_dir: Path, which: str = "latest") -> Optional[Dict[str, Any]]:
    ckpt_name = "checkpoint_latest.pth" if which == "latest" else "checkpoint_best.pth"
    ckpt_path = run_dir / ckpt_name
    if not ckpt_path.exists():
        return None
    try:
        payload = torch.load(ckpt_path, map_location="cpu")
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    metrics = payload.get("metrics")
    if not isinstance(metrics, dict):
        return None
    return metrics


def format_metrics(metrics: Dict[str, Any]) -> str:
    keys = ["reward", "rdkit_validity", "rdkit_uniqueness", "validity_x_uniqueness"]
    parts = []
    for k in keys:
        v = _safe_float(metrics.get(k))
        if v is None:
            continue
        parts.append(f"{k}={v:.4f}")
    epoch = _safe_int(metrics.get("epoch"))
    if epoch is not None:
        parts.insert(0, f"epoch={epoch}")
    return " ".join(parts) if parts else "(no metrics yet)"


def _merge_env(base_env: Dict[str, Any], override_env: Optional[Dict[str, Any]]) -> Dict[str, str]:
    env = {k: str(v) for k, v in base_env.items()}
    if override_env:
        for k, v in override_env.items():
            if v is None:
                env.pop(k, None)
            else:
                env[k] = str(v)
    merged = os.environ.copy()
    merged.update(env)
    return merged


def _write_attempt_files(
    attempt_id: int,
    result: RunResult,
    description: str,
    config_name: str,
    overrides: list[str],
    resume_from: Optional[str],
) -> None:
    attempt_dir = ATTEMPTS_ROOT / f"attempt_{attempt_id:03d}"
    attempt_dir.mkdir(parents=True, exist_ok=True)

    best_metrics = dict(result.best_metrics or {})
    best_epoch = result.best_epoch
    if best_epoch is None:
        best_epoch = _safe_int(best_metrics.get("epoch"))

    payload: Dict[str, Any] = {
        "description": description,
        "start_time": result.start_time,
        "duration_hours": result.duration_hours,
        "status": result.status,
        "log_path": str(result.log_path),
        "run_dir": str(result.run_dir),
        "config_name": config_name,
        "overrides": overrides,
    }
    if resume_from:
        payload["resume_from"] = resume_from
    if best_epoch is not None:
        payload["best_epoch"] = best_epoch

    for k in [
        "reward",
        "atom_stability",
        "molecule_stability",
        "rdkit_validity",
        "rdkit_uniqueness",
        "validity_x_uniqueness",
    ]:
        if k in best_metrics:
            payload[k] = best_metrics[k]

    (attempt_dir / "metrics.json").write_text(json.dumps(payload, indent=2) + "\n")

    run_config = result.run_dir / "config.yaml"
    if run_config.exists():
        shutil.copyfile(run_config, attempt_dir / "config.yaml")
    else:
        (attempt_dir / "config.yaml").write_text(
            yaml.safe_dump(
                {
                    "config_name": config_name,
                    "save_path": str(result.run_dir),
                    "checkpoint_path": resume_from,
                    "overrides": overrides,
                },
                sort_keys=False,
            )
        )

    best_line = (
        f"- Best epoch: {best_epoch}\n"
        f"- Atom stability: {payload.get('atom_stability', '--')}\n"
        f"- Mol stability: {payload.get('molecule_stability', '--')}\n"
        f"- RDKit validity: {payload.get('rdkit_validity', '--')}\n"
        f"- RDKit uniqueness: {payload.get('rdkit_uniqueness', '--')}\n"
        f"- Validity × Uniqueness: {payload.get('validity_x_uniqueness', '--')}\n"
    )

    notes = (
        f"# Attempt {attempt_id:03d}: {description}\n\n"
        f"**Date**: {payload['start_time'].split(' ')[0]}  \n"
        f"**Status**: {payload['status']}  \n"
        f"**Duration**: {payload['duration_hours']:.3f}h  \n\n"
        "## Run\n\n"
        f"- Log: `{payload['log_path']}`\n"
        f"- Run dir: `{payload['run_dir']}`\n\n"
        "## Best Metrics (checkpoint_best)\n\n"
        f"{best_line}\n"
        "## Config\n\n"
        f"- Config: `{config_name}`\n"
        + (f"- Resume from: `{resume_from}`\n" if resume_from else "")
        + (f"- Overrides: `{overrides}`\n" if overrides else "")
    )
    (attempt_dir / "NOTES.md").write_text(notes)

    subprocess.run(["python", str(ATTEMPTS_ROOT / "update_leaderboard.py")], cwd=str(REPO_ROOT), check=False)


def _run_one_experiment(
    attempt_id: int,
    series_name: str,
    out_root: Path,
    check_minutes: float,
    base_env: Dict[str, Any],
    exp: Dict[str, Any],
    prev_best_ckpt: Optional[Path],
    dry_run: bool,
) -> tuple[RunResult, Optional[Path]]:
    exp_name = str(exp["name"])
    config_name = str(exp["config"])
    description = str(exp.get("description") or exp_name)
    max_hours = float(exp.get("max_hours", 6.0))
    plateau_patience_minutes = _safe_float(exp.get("plateau_patience_minutes"))
    min_delta = _safe_float(exp.get("min_delta")) or 0.0

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = out_root / f"{series_name}_attempt{attempt_id:03d}_{exp_name}_{timestamp}"
    log_path = out_root / f"{run_dir.name}.log"

    overrides = [str(x) for x in (exp.get("overrides") or [])]

    resume_from = exp.get("resume_from")
    resume_ckpt: Optional[Path] = None
    if isinstance(resume_from, str) and resume_from.strip():
        if resume_from == "prev_best":
            resume_ckpt = prev_best_ckpt
        else:
            resume_ckpt = Path(resume_from)
    elif resume_from is None:
        resume_ckpt = None

    cmd = [
        "python",
        "run_verl_diffusion.py",
        "--config-name",
        config_name,
        f"save_path={str(run_dir)}",
        f"train.max_time_hours={max_hours}",
    ]
    if resume_ckpt is not None:
        cmd += ["resume=true", f"checkpoint_path={str(resume_ckpt)}"]
    cmd += overrides

    env = _merge_env(base_env, exp.get("env"))

    print(f"[attempt {attempt_id:03d}] {description}")
    print(f"[attempt {attempt_id:03d}] cmd: {' '.join(cmd)}")
    print(f"[attempt {attempt_id:03d}] log: {log_path}")

    if dry_run:
        return (
            RunResult(
                run_dir=run_dir,
                log_path=log_path,
                status="running",
                start_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                duration_hours=0.0,
                best_epoch=None,
                best_metrics={},
            ),
            resume_ckpt,
        )

    run_dir.mkdir(parents=True, exist_ok=True)
    start_wall = time.time()
    start_time_str = datetime.fromtimestamp(start_wall).strftime("%Y-%m-%d %H:%M:%S")

    with open(log_path, "a", encoding="utf-8") as log_f:
        log_f.write(f"[runner] start_time={start_time_str}\n")
        log_f.write(f"[runner] description={description}\n")
        log_f.write(f"[runner] cmd={' '.join(cmd)}\n")
        log_f.flush()

        proc = subprocess.Popen(
            cmd,
            cwd=str(REPO_ROOT),
            stdout=log_f,
            stderr=log_f,
            env=env,
            start_new_session=True,
        )

        best_seen = float("-inf")
        last_improve_ts = time.time()
        interrupted = False

        try:
            while proc.poll() is None:
                time.sleep(max(10.0, check_minutes * 60.0))
                metrics = read_checkpoint_metrics(run_dir, which="latest") or {}
                metric_value = _safe_float(metrics.get("validity_x_uniqueness"))
                if metric_value is not None and metric_value > best_seen + min_delta:
                    best_seen = metric_value
                    last_improve_ts = time.time()
                print(f"[attempt {attempt_id:03d}] {format_metrics(metrics)}")

                if plateau_patience_minutes is not None:
                    elapsed_min = (time.time() - last_improve_ts) / 60.0
                    if elapsed_min >= plateau_patience_minutes:
                        print(f"[attempt {attempt_id:03d}] plateau for {elapsed_min:.1f}m; sending SIGINT to stop")
                        os.killpg(proc.pid, signal.SIGINT)
                        interrupted = True
        except KeyboardInterrupt:
            print(f"[attempt {attempt_id:03d}] interrupted by user; sending SIGINT to stop")
            try:
                os.killpg(proc.pid, signal.SIGINT)
            except Exception:
                pass
            interrupted = True

        rc = int(proc.wait() or 0)

    duration_hours = (time.time() - start_wall) / 3600.0
    status = "completed" if rc == 0 else "failed"
    if interrupted:
        status = "early_stop"

    best_metrics = read_checkpoint_metrics(run_dir, which="best") or {}
    best_epoch = _safe_int(best_metrics.get("epoch"))

    result = RunResult(
        run_dir=run_dir,
        log_path=log_path,
        status=status,
        start_time=start_time_str,
        duration_hours=duration_hours,
        best_epoch=best_epoch,
        best_metrics=best_metrics,
    )

    _write_attempt_files(
        attempt_id=attempt_id,
        result=result,
        description=description,
        config_name=config_name,
        overrides=overrides,
        resume_from=str(resume_ckpt) if resume_ckpt is not None else None,
    )

    next_best_ckpt = run_dir / "checkpoint_best.pth"
    if next_best_ckpt.exists():
        return result, next_best_ckpt
    next_latest_ckpt = run_dir / "checkpoint_latest.pth"
    if next_latest_ckpt.exists():
        return result, next_latest_ckpt
    return result, resume_ckpt


def main() -> None:
    args = parse_args()
    plan_path = Path(args.plan).resolve()
    plan = yaml.safe_load(plan_path.read_text())
    if not isinstance(plan, dict):
        raise SystemExit(f"Invalid plan file: {plan_path}")

    series_name = str(plan.get("series_name") or "codex_geom_vxu")
    out_root = Path(plan.get("out_root") or "outputs/verl/geom_vxu")
    if not out_root.is_absolute():
        out_root = (REPO_ROOT / out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    base_env = dict(plan.get("env") or {})
    check_minutes = float(plan.get("check_minutes", 30))
    experiments = plan.get("experiments") or []
    if not isinstance(experiments, list) or not experiments:
        raise SystemExit("Plan must define a non-empty `experiments:` list.")

    prev_best_ckpt: Optional[Path] = None
    for exp in experiments:
        if not isinstance(exp, dict):
            raise SystemExit("Each experiment must be a mapping/dict.")
        attempt_id = _next_attempt_id()
        result, prev_best_ckpt = _run_one_experiment(
            attempt_id=attempt_id,
            series_name=series_name,
            out_root=out_root,
            check_minutes=check_minutes,
            base_env=base_env,
            exp=exp,
            prev_best_ckpt=prev_best_ckpt,
            dry_run=bool(args.dry_run),
        )
        if result.status == "failed":
            print(f"[attempt {attempt_id:03d}] failed; stopping series")
            break


if __name__ == "__main__":
    main()
