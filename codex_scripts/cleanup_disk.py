#!/usr/bin/env python3
"""
Lightweight disk-cleanup helper.

Primary use-case: prune RL run checkpoints under outputs/verl_geom so we don't hit
home quota while training/evaluating.

This script is intentionally conservative:
  - only deletes files that match checkpoint_epoch_*.pth
  - keeps checkpoint_best.pth / checkpoint_latest.pth untouched
  - can keep a time-based safety window to avoid deleting a file being written

Also supports pruning local W&B run folders (small, but helps keep things tidy).
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple


EPOCH_CKPT_RE = re.compile(r"^checkpoint_epoch_(\d+)\.pth$")
WANDB_RUN_DIR_RE = re.compile(r"^(offline-run|run)-")


@dataclass(frozen=True)
class DeleteAction:
    path: Path
    size_bytes: int


def _human_bytes(n: int) -> str:
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if n < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}PB"


def _iter_run_dirs(runs_root: Path) -> Iterable[Path]:
    if not runs_root.exists():
        return []
    for p in runs_root.iterdir():
        if p.is_dir():
            yield p


def _collect_epoch_checkpoints(run_dir: Path) -> List[Tuple[int, Path]]:
    out: List[Tuple[int, Path]] = []
    for p in run_dir.iterdir():
        if not p.is_file():
            continue
        m = EPOCH_CKPT_RE.match(p.name)
        if m is None:
            continue
        out.append((int(m.group(1)), p))
    out.sort(key=lambda x: x[0])
    return out


def prune_epoch_checkpoints(
    runs_root: Path,
    keep_last: int,
    recent_hours: float,
    dry_run: bool,
) -> List[DeleteAction]:
    now = time.time()
    recent_cutoff = now - recent_hours * 3600.0

    deletions: List[DeleteAction] = []
    for run_dir in _iter_run_dirs(runs_root):
        ckpts = _collect_epoch_checkpoints(run_dir)
        if len(ckpts) <= keep_last:
            continue

        # Keep by epoch index.
        keep_by_epoch = {p for _, p in ckpts[-keep_last:]}

        # Safety: also keep anything modified very recently.
        keep_by_time = set()
        for _, p in ckpts:
            try:
                if p.stat().st_mtime >= recent_cutoff:
                    keep_by_time.add(p)
            except FileNotFoundError:
                # File may have disappeared mid-run; ignore.
                continue

        keep = keep_by_epoch | keep_by_time

        for _, p in ckpts:
            if p in keep:
                continue
            try:
                size = p.stat().st_size
            except FileNotFoundError:
                continue
            deletions.append(DeleteAction(path=p, size_bytes=size))

    # Execute deletions last to reduce the chance of partial cleanup.
    if not dry_run:
        for d in deletions:
            try:
                d.path.unlink()
            except FileNotFoundError:
                pass

    return deletions


def prune_wandb_runs(wandb_root: Path, keep_last: int, dry_run: bool) -> List[DeleteAction]:
    if not wandb_root.exists():
        return []

    candidates: List[Path] = []
    for p in wandb_root.iterdir():
        if not p.is_dir():
            continue
        if p.name == "latest-run":
            continue
        if WANDB_RUN_DIR_RE.match(p.name):
            candidates.append(p)

    # Newest last
    candidates.sort(key=lambda p: p.stat().st_mtime)
    to_delete = candidates[:-keep_last] if keep_last > 0 else candidates

    deletions: List[DeleteAction] = []
    for p in to_delete:
        try:
            size = sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
        except FileNotFoundError:
            size = 0
        deletions.append(DeleteAction(path=p, size_bytes=size))

    if not dry_run:
        for d in deletions:
            shutil.rmtree(d.path, ignore_errors=True)

    return deletions


def main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", type=Path, default=Path("outputs/verl_geom"))
    ap.add_argument("--keep-epoch", type=int, default=20, help="Keep last K checkpoint_epoch_*.pth per run dir.")
    ap.add_argument(
        "--recent-hours",
        type=float,
        default=2.0,
        help="Also keep any checkpoint_epoch_*.pth modified within this many hours (safety for running jobs).",
    )
    ap.add_argument("--wandb-root", type=Path, default=Path("wandb"))
    ap.add_argument("--keep-wandb", type=int, default=5, help="Keep last K wandb run dirs in wandb/ (run-* + offline-run-*).")
    ap.add_argument("--dry-run", action="store_true", help="Print what would be deleted, but do not delete.")
    args = ap.parse_args(argv)

    if args.keep_epoch < 0 or args.keep_wandb < 0:
        ap.error("--keep-epoch/--keep-wandb must be >= 0")

    epoch_deletions = prune_epoch_checkpoints(
        runs_root=args.runs_root,
        keep_last=args.keep_epoch,
        recent_hours=args.recent_hours,
        dry_run=args.dry_run,
    )
    wandb_deletions = prune_wandb_runs(
        wandb_root=args.wandb_root,
        keep_last=args.keep_wandb,
        dry_run=args.dry_run,
    )

    total_bytes = sum(d.size_bytes for d in epoch_deletions) + sum(d.size_bytes for d in wandb_deletions)
    print(f"prune_epoch_checkpoints: {len(epoch_deletions)} files -> {_human_bytes(sum(d.size_bytes for d in epoch_deletions))}")
    print(f"prune_wandb_runs:        {len(wandb_deletions)} dirs  -> {_human_bytes(sum(d.size_bytes for d in wandb_deletions))}")
    print(f"TOTAL: {_human_bytes(total_bytes)} {'(dry-run)' if args.dry_run else ''}")

    if args.dry_run and epoch_deletions:
        # Print a short preview of deletions to help sanity-check.
        print("\nExample deletions (first 10):")
        for d in epoch_deletions[:10]:
            print(f"  {d.path} ({_human_bytes(d.size_bytes)})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
