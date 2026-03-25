#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import signal
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _format_seconds(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    if seconds < 60:
        return f"{seconds:.0f}s"
    minutes = seconds / 60.0
    if minutes < 60:
        return f"{minutes:.1f}m"
    hours = minutes / 60.0
    return f"{hours:.2f}h"


def _now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _is_finite(value: float) -> bool:
    return math.isfinite(value)


def read_checkpoint_metrics(run_dir: Path) -> Optional[Dict[str, Any]]:
    ckpt_path = run_dir / "checkpoint_latest.pth"
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


def read_checkpoint_metrics_named(run_dir: Path, which: str) -> Optional[Dict[str, Any]]:
    name = "checkpoint_best.pth" if which == "best" else "checkpoint_latest.pth"
    ckpt_path = run_dir / name
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


def pick_metric(metrics: Dict[str, Any], preferred: Optional[str]) -> Tuple[Optional[str], Optional[float]]:
    if preferred:
        value = _safe_float(metrics.get(preferred))
        if value is not None:
            return preferred, value
    for key in ("validity_x_uniqueness_x_stability", "validity_x_uniqueness", "molecule_stability", "reward"):
        value = _safe_float(metrics.get(key))
        if value is not None:
            return key, value
    return None, None


def _pid_is_running(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _run_cmd(cmd: List[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=str(REPO_ROOT), check=False, capture_output=True, text=True)


def _slurm_job_state(job_id: int) -> Optional[str]:
    if job_id <= 0:
        return None
    proc = _run_cmd(["squeue", "-h", "-j", str(job_id), "-o", "%T"])
    if proc.returncode != 0:
        return None
    state = (proc.stdout or "").strip()
    return state or None


def _slurm_cancel(job_id: int, sig: str = "INT") -> None:
    if job_id <= 0:
        return
    _run_cmd(["scancel", f"--signal={sig}", str(job_id)])


def _send_sigint(pid: int) -> None:
    if pid <= 0:
        return
    try:
        os.kill(pid, signal.SIGINT)
    except Exception:
        return


def _has_override(overrides: List[str], prefix: str) -> bool:
    return any(str(item).startswith(prefix) for item in overrides)


@dataclass
class TargetConfig:
    name: str
    kind: str  # local | slurm
    run_dir: Path
    pid: Optional[int] = None  # local supervisor PID
    job_id: Optional[int] = None  # slurm job id
    monitor_metric: Optional[str] = None
    mode: str = "max"  # max | min
    min_delta: float = 0.0
    smoothing_window: int = 1
    warmup_minutes: float = 0.0
    patience_minutes: float = 180.0
    stall_minutes: float = 30.0
    stop_signal: str = "INT"


@dataclass
class TargetState:
    start_ts: float = field(default_factory=time.time)
    last_epoch: Optional[int] = None
    last_epoch_ts: Optional[float] = None
    metric_name: Optional[str] = None
    last_metric: Optional[float] = None
    history: List[Tuple[float, float]] = field(default_factory=list)  # (ts, metric_value)
    best_smoothed: Optional[float] = None
    last_improve_ts: Optional[float] = None
    stopped: bool = False
    stop_reason: Optional[str] = None


@dataclass
class QueueExperiment:
    name: str
    config: str
    overrides: List[str] = field(default_factory=list)


@dataclass
class SlurmConfig:
    partition: str = "gpu"
    gres: str = "gpu:h100:1"
    cpus_per_task: int = 8
    mem: str = "64G"
    time: str = "08:00:00"


@dataclass
class QueueConfig:
    enabled: bool = False
    max_active: int = 0
    run_root: Path = field(default_factory=lambda: REPO_ROOT / "outputs" / "verl" / "geom_autopilot")
    resume_checkpoint: Optional[Path] = None
    update_resume_from_best: bool = True
    max_time_hours: float = 6.0
    env: Dict[str, str] = field(default_factory=dict)
    slurm: SlurmConfig = field(default_factory=SlurmConfig)
    common_overrides: List[str] = field(default_factory=list)
    experiments: List[QueueExperiment] = field(default_factory=list)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Monitor one or more GEOM RL runs (local or Slurm) by polling checkpoints and "
            "auto-stopping on stalls/plateaus with noise-tolerant smoothing."
        )
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="YAML config file specifying targets + monitoring thresholds.",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run a single monitoring cycle and exit (debug).",
    )
    return parser.parse_args()


def _load_config(path: Path) -> Tuple[float, List[TargetConfig], Optional[Path]]:
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict):
        raise SystemExit(f"Invalid config file: {path}")

    poll_minutes = float(raw.get("poll_minutes", 10))
    state_path = raw.get("state_path")
    state_path_p = Path(state_path).expanduser() if state_path else None

    targets_raw = raw.get("targets")
    if not isinstance(targets_raw, list) or not targets_raw:
        raise SystemExit("Config must define a non-empty `targets:` list.")

    targets: List[TargetConfig] = []
    for entry in targets_raw:
        if not isinstance(entry, dict):
            raise SystemExit("Each target must be a mapping/dict.")
        name = str(entry.get("name") or "")
        kind = str(entry.get("kind") or "")
        run_dir = str(entry.get("run_dir") or "")
        if not name or not kind or not run_dir:
            raise SystemExit("Each target needs `name`, `kind`, and `run_dir`.")
        pid = entry.get("pid")
        job_id = entry.get("job_id")
        targets.append(
            TargetConfig(
                name=name,
                kind=kind,
                run_dir=(REPO_ROOT / run_dir).resolve() if not Path(run_dir).is_absolute() else Path(run_dir),
                pid=None if pid is None else int(pid),
                job_id=None if job_id is None else int(job_id),
                monitor_metric=entry.get("monitor_metric"),
                mode=str(entry.get("mode") or "max"),
                min_delta=float(entry.get("min_delta") or 0.0),
                smoothing_window=int(entry.get("smoothing_window") or 1),
                warmup_minutes=float(entry.get("warmup_minutes") or 0.0),
                patience_minutes=float(entry.get("patience_minutes") or 180.0),
                stall_minutes=float(entry.get("stall_minutes") or 30.0),
                stop_signal=str(entry.get("stop_signal") or "INT"),
            )
        )

    poll_minutes = max(0.2, poll_minutes)
    for t in targets:
        t.smoothing_window = max(1, int(t.smoothing_window))
        t.mode = "min" if t.mode.lower() == "min" else "max"
        t.kind = t.kind.lower()
        if t.kind not in {"local", "slurm"}:
            raise SystemExit(f"Unsupported target kind '{t.kind}' for {t.name}. Use 'local' or 'slurm'.")

    queue_cfg = raw.get("queue")
    queue = QueueConfig(enabled=False)
    if isinstance(queue_cfg, dict) and queue_cfg.get("enabled"):
        queue.enabled = True
        queue.max_active = int(queue_cfg.get("max_active") or 0)
        run_root = queue_cfg.get("run_root")
        if not run_root:
            raise SystemExit("queue.enabled=true requires queue.run_root")
        queue.run_root = (REPO_ROOT / run_root).resolve() if not Path(str(run_root)).is_absolute() else Path(run_root)
        resume_checkpoint = queue_cfg.get("resume_checkpoint")
        if resume_checkpoint:
            queue.resume_checkpoint = (
                (REPO_ROOT / resume_checkpoint).resolve()
                if not Path(str(resume_checkpoint)).is_absolute()
                else Path(resume_checkpoint)
            )
        queue.update_resume_from_best = bool(queue_cfg.get("update_resume_from_best", True))
        queue.max_time_hours = float(queue_cfg.get("max_time_hours", 6.0))
        env_raw = queue_cfg.get("env") or {}
        if isinstance(env_raw, dict):
            queue.env = {str(k): str(v) for k, v in env_raw.items() if v is not None}
        slurm_raw = queue_cfg.get("slurm") or {}
        if isinstance(slurm_raw, dict):
            queue.slurm = SlurmConfig(
                partition=str(slurm_raw.get("partition") or "gpu"),
                gres=str(slurm_raw.get("gres") or "gpu:h100:1"),
                cpus_per_task=int(slurm_raw.get("cpus_per_task") or 8),
                mem=str(slurm_raw.get("mem") or "64G"),
                time=str(slurm_raw.get("time") or "08:00:00"),
            )
        common_overrides = queue_cfg.get("common_overrides") or []
        if not isinstance(common_overrides, list):
            raise SystemExit("queue.common_overrides must be a list of strings")
        queue.common_overrides = [str(x) for x in common_overrides]
        experiments_raw = queue_cfg.get("experiments") or []
        if not isinstance(experiments_raw, list) or not experiments_raw:
            raise SystemExit("queue.enabled=true requires a non-empty queue.experiments list")
        for exp in experiments_raw:
            if not isinstance(exp, dict):
                raise SystemExit("queue.experiments entries must be dicts")
            exp_name = str(exp.get("name") or "")
            exp_config = str(exp.get("config") or "")
            if not exp_name or not exp_config:
                raise SystemExit("queue.experiments entries need name and config")
            exp_overrides = exp.get("overrides") or []
            if not isinstance(exp_overrides, list):
                raise SystemExit("queue.experiments.overrides must be a list")
            queue.experiments.append(
                QueueExperiment(name=exp_name, config=exp_config, overrides=[str(x) for x in exp_overrides])
            )

    return poll_minutes, targets, state_path_p, queue


def _compute_smoothed(history: List[Tuple[float, float]], window: int) -> Optional[float]:
    if not history:
        return None
    window = max(1, int(window))
    values = [v for _, v in history[-window:]]
    return float(sum(values) / len(values))


def _should_improve(mode: str, value: float, best: Optional[float], min_delta: float) -> bool:
    if best is None:
        return True
    if mode == "min":
        return value < best - min_delta
    return value > best + min_delta


def _stop_target(cfg: TargetConfig, reason: str) -> None:
    print(f"[{_now_str()}] STOP {cfg.name}: {reason}")
    if cfg.kind == "local":
        if cfg.pid:
            _send_sigint(int(cfg.pid))
    else:
        if cfg.job_id:
            _slurm_cancel(int(cfg.job_id), sig=str(cfg.stop_signal or "INT"))


def _sbatch_submit(sbatch_path: Path) -> Optional[int]:
    proc = _run_cmd(["sbatch", "--parsable", str(sbatch_path)])
    if proc.returncode != 0:
        return None
    out = (proc.stdout or "").strip()
    if not out:
        return None
    try:
        return int(out.split(";", 1)[0])
    except Exception:
        return None


def _write_sbatch(
    *,
    sbatch_path: Path,
    job_name: str,
    run_dir: Path,
    config_name: str,
    resume_checkpoint: Path,
    max_time_hours: float,
    env: Dict[str, str],
    slurm: SlurmConfig,
    overrides: List[str],
) -> None:
    sbatch_path.parent.mkdir(parents=True, exist_ok=True)
    out_path = run_dir / "slurm_%j.out"
    err_path = run_dir / "slurm_%j.err"
    lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name={job_name}",
        f"#SBATCH --partition={slurm.partition}",
        f"#SBATCH --gres={slurm.gres}",
        f"#SBATCH --cpus-per-task={slurm.cpus_per_task}",
        f"#SBATCH --mem={slurm.mem}",
        f"#SBATCH --time={slurm.time}",
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
        f"cd \"{REPO_ROOT}\"",
        f"export PYTHONPATH=\"{REPO_ROOT}:{REPO_ROOT}/edm_source:${{PYTHONPATH:-}}\"",
    ]
    for k, v in env.items():
        lines.append(f"export {k}={json.dumps(str(v))}")
    lines += [
        "",
        'if [[ -n "${WANDB_API_KEY:-}" ]]; then',
        '  export WANDB_MODE="${WANDB_MODE:-online}"',
        "else",
        '  export WANDB_MODE="${WANDB_MODE:-offline}"',
        "fi",
    ]
    cmd = [
        "python",
        "-u",
        "run_verl_diffusion.py",
        "--config-name",
        config_name,
        f"save_path={run_dir}",
        f"train.max_time_hours={max_time_hours}",
        "resume=true",
        f"checkpoint_path={resume_checkpoint}",
    ] + overrides
    lines += ["", " ".join(cmd), ""]
    sbatch_path.write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    cfg_path = Path(args.config).resolve()
    poll_minutes, targets, state_path, queue = _load_config(cfg_path)

    states: Dict[str, TargetState] = {t.name: TargetState() for t in targets}
    pending_queue: List[QueueExperiment] = list(queue.experiments)
    global_best_metric: Optional[float] = None
    global_best_ckpt: Optional[Path] = queue.resume_checkpoint

    print(f"[{_now_str()}] monitor: {cfg_path}")
    print(f"[{_now_str()}] poll_every={poll_minutes:.1f}m targets={len(targets)}")
    if queue.enabled:
        print(
            f"[{_now_str()}] queue: enabled max_active={queue.max_active} "
            f"pending={len(pending_queue)} run_root={queue.run_root}"
        )

    poll_seconds = poll_minutes * 60.0
    while True:
        any_active = False
        submitted_any = False
        snapshot: Dict[str, Any] = {"time": _now_str(), "targets": {}}

        for t in list(targets):
            st = states[t.name]
            if st.stopped:
                snapshot["targets"][t.name] = {"stopped": True, "reason": st.stop_reason}
                continue

            if t.kind == "local":
                running = _pid_is_running(int(t.pid or -1))
                state = "RUNNING" if running else "DONE"
            else:
                state = _slurm_job_state(int(t.job_id or -1)) or "DONE"
                running = state not in {"DONE", "COMPLETED", "CANCELLED", "FAILED", "TIMEOUT", "OUT_OF_MEMORY"}

            snapshot["targets"][t.name] = {"state": state, "run_dir": str(t.run_dir)}

            if not running:
                st.stopped = True
                st.stop_reason = f"finished ({state})"

                # Update global best checkpoint (optional) once the job finishes.
                if queue.enabled and queue.update_resume_from_best:
                    best_metrics = read_checkpoint_metrics_named(t.run_dir, "best") or read_checkpoint_metrics_named(
                        t.run_dir, "latest"
                    )
                    if best_metrics:
                        _, best_value = pick_metric(best_metrics, t.monitor_metric)
                        if best_value is not None and _is_finite(best_value):
                            if global_best_metric is None or _should_improve(
                                t.mode, best_value, global_best_metric, t.min_delta
                            ):
                                global_best_metric = best_value
                                candidate = t.run_dir / "checkpoint_best.pth"
                                global_best_ckpt = candidate if candidate.exists() else t.run_dir / "checkpoint_latest.pth"
                                print(
                                    f"[{_now_str()}] global_best updated by {t.name}: "
                                    f"{t.monitor_metric or 'metric'}={best_value:.6f} ckpt={global_best_ckpt}"
                                )
                continue

            any_active = True

            metrics = read_checkpoint_metrics(t.run_dir)
            if not metrics:
                print(f"[{_now_str()}] {t.name}: no checkpoint yet ({state})")
                continue

            epoch = metrics.get("epoch")
            epoch_i = None
            try:
                epoch_i = int(epoch) if epoch is not None else None
            except Exception:
                epoch_i = None

            metric_name, metric_value = pick_metric(metrics, t.monitor_metric)
            if metric_value is None or metric_name is None:
                print(f"[{_now_str()}] {t.name}: checkpoint has no usable metric keys")
                continue

            if not _is_finite(metric_value):
                st.stopped = True
                st.stop_reason = f"non-finite {metric_name}={metric_value}"
                _stop_target(t, st.stop_reason)
                continue

            now = time.time()
            if epoch_i is not None and (st.last_epoch is None or epoch_i != st.last_epoch):
                st.last_epoch = epoch_i
                st.last_epoch_ts = now

            st.metric_name = metric_name
            st.last_metric = metric_value
            if not st.history or abs(st.history[-1][1] - metric_value) > 1e-12:
                st.history.append((now, metric_value))

            smoothed = _compute_smoothed(st.history, t.smoothing_window)
            if smoothed is not None:
                if st.best_smoothed is None:
                    st.best_smoothed = smoothed
                    st.last_improve_ts = now
                elif _should_improve(t.mode, smoothed, st.best_smoothed, t.min_delta):
                    st.best_smoothed = smoothed
                    st.last_improve_ts = now

            stall_ok = True
            if st.last_epoch_ts is not None and t.stall_minutes > 0:
                stall_age = (now - st.last_epoch_ts) / 60.0
                if stall_age >= t.stall_minutes:
                    stall_ok = False

            warmup_ok = (now - st.start_ts) >= (t.warmup_minutes * 60.0)
            plateau_ok = True
            if warmup_ok and st.last_improve_ts is not None and t.patience_minutes > 0:
                plateau_age = (now - st.last_improve_ts) / 60.0
                if plateau_age >= t.patience_minutes:
                    plateau_ok = False

            line = (
                f"[{_now_str()}] {t.name} ({state}) "
                f"epoch={st.last_epoch or '--'} "
                f"{metric_name}={metric_value:.6f} "
                f"smoothed={'' if smoothed is None else f'{smoothed:.6f}'} "
                f"best_smoothed={'' if st.best_smoothed is None else f'{st.best_smoothed:.6f}'} "
                f"since_best={'' if st.last_improve_ts is None else _format_seconds(now - st.last_improve_ts)}"
            )
            print(line)

            snapshot["targets"][t.name].update(
                {
                    "epoch": st.last_epoch,
                    "metric": metric_name,
                    "value": metric_value,
                    "smoothed": smoothed,
                    "best_smoothed": st.best_smoothed,
                    "since_best_s": None if st.last_improve_ts is None else (now - st.last_improve_ts),
                }
            )

            if not stall_ok:
                st.stopped = True
                st.stop_reason = f"stalled: no new epoch for {t.stall_minutes:.1f}m"
                _stop_target(t, st.stop_reason)
                continue

            if not plateau_ok:
                st.stopped = True
                st.stop_reason = (
                    f"plateau: no improvement in {t.patience_minutes:.1f}m "
                    f"(best_smoothed={st.best_smoothed})"
                )
                _stop_target(t, st.stop_reason)
                continue

        if queue.enabled:
            active_count = 0
            for t in targets:
                st = states.get(t.name)
                if st is None or st.stopped:
                    continue
                if t.kind == "local":
                    running = _pid_is_running(int(t.pid or -1))
                else:
                    state = _slurm_job_state(int(t.job_id or -1)) or "DONE"
                    running = state not in {"DONE", "COMPLETED", "CANCELLED", "FAILED", "TIMEOUT", "OUT_OF_MEMORY"}
                if running:
                    active_count += 1

            while active_count < int(queue.max_active or 0) and pending_queue:
                exp = pending_queue.pop(0)
                resume_ckpt = global_best_ckpt or queue.resume_checkpoint
                if resume_ckpt is None:
                    print(f"[{_now_str()}] queue: missing resume checkpoint; cannot submit {exp.name}")
                    break

                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                run_dir = queue.run_root / f"{exp.name}_{ts}"
                run_dir.mkdir(parents=True, exist_ok=True)
                sbatch_path = run_dir / "launch.sbatch"
                overrides = list(queue.common_overrides) + list(exp.overrides)
                # Ensure queue-launched runs don't clobber each other in W&B.
                # (`ddpo_config.yaml` defaults wandb_name to a constant.)
                if not _has_override(overrides, "wandb.enabled="):
                    overrides.append("wandb.enabled=true")
                if not _has_override(overrides, "wandb.wandb_project="):
                    overrides.append("wandb.wandb_project=ddpo")
                if not _has_override(overrides, "wandb.wandb_name="):
                    overrides.append(f"wandb.wandb_name={run_dir.name}")
                job_name = f"geom_{exp.name}"
                _write_sbatch(
                    sbatch_path=sbatch_path,
                    job_name=job_name,
                    run_dir=run_dir,
                    config_name=exp.config,
                    resume_checkpoint=resume_ckpt,
                    max_time_hours=queue.max_time_hours,
                    env=queue.env,
                    slurm=queue.slurm,
                    overrides=overrides,
                )
                job_id = _sbatch_submit(sbatch_path)
                if job_id is None:
                    print(f"[{_now_str()}] queue: failed to submit {exp.name}")
                    break

                target_name = f"queue_{exp.name}_{job_id}"
                new_target = TargetConfig(
                    name=target_name,
                    kind="slurm",
                    run_dir=run_dir,
                    job_id=job_id,
                    pid=None,
                    monitor_metric=None,
                    mode="max",
                    min_delta=0.0005,
                    smoothing_window=5,
                    warmup_minutes=60,
                    patience_minutes=300,
                    stall_minutes=90,
                    stop_signal="INT",
                )
                targets.append(new_target)
                states[target_name] = TargetState()
                print(f"[{_now_str()}] queue: submitted {exp.name} as job {job_id} -> {run_dir}")
                active_count += 1
                submitted_any = True

        if state_path is not None:
            try:
                state_path.parent.mkdir(parents=True, exist_ok=True)
                state_path.write_text(json.dumps(snapshot, indent=2) + "\n")
            except Exception:
                pass

        if args.once:
            break
        if submitted_any:
            any_active = True
        if not any_active:
            print(f"[{_now_str()}] all targets finished/stopped; exiting monitor")
            break
        time.sleep(poll_seconds)


if __name__ == "__main__":
    main()
