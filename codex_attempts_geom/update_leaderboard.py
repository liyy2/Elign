#!/usr/bin/env python3
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class AttemptRow:
    attempt: int
    date: str
    description: str
    atom_stability: Optional[float]
    molecule_stability: Optional[float]
    rdkit_validity: Optional[float]
    rdkit_uniqueness: Optional[float]
    validity_x_uniqueness: Optional[float]
    duration_hours: Optional[float]
    status: str


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text())


def _safe_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _fmt_float(value: Optional[float], digits: int = 3) -> str:
    if value is None:
        return "--"
    return f"{value:.{digits}f}"


def _fmt_duration(hours: Optional[float]) -> str:
    if hours is None:
        return "--"
    return f"{hours:.3f}h"


def _parse_date(start_time: Any) -> str:
    if isinstance(start_time, str) and start_time.strip():
        try:
            dt = datetime.fromisoformat(start_time.replace("Z", ""))
            return dt.date().isoformat()
        except ValueError:
            return start_time.split(" ")[0]
    return "--"


def _iter_attempt_rows() -> list[AttemptRow]:
    rows: list[AttemptRow] = []
    for attempt_dir in sorted(ROOT.glob("attempt_*")):
        if not attempt_dir.is_dir():
            continue
        try:
            attempt_id = int(attempt_dir.name.split("_", 1)[1])
        except Exception:
            continue

        metrics_path = attempt_dir / "metrics.json"
        if not metrics_path.exists():
            continue
        metrics = _load_json(metrics_path)

        rows.append(
            AttemptRow(
                attempt=attempt_id,
                date=_parse_date(metrics.get("start_time")),
                description=str(metrics.get("description") or "--"),
                atom_stability=_safe_float(metrics.get("atom_stability")),
                molecule_stability=_safe_float(metrics.get("molecule_stability")),
                rdkit_validity=_safe_float(metrics.get("rdkit_validity")),
                rdkit_uniqueness=_safe_float(metrics.get("rdkit_uniqueness")),
                validity_x_uniqueness=_safe_float(metrics.get("validity_x_uniqueness")),
                duration_hours=_safe_float(metrics.get("duration_hours")),
                status=str(metrics.get("status") or "--"),
            )
        )
    rows.sort(key=lambda r: r.attempt)
    return rows


def _best_attempt(rows: list[AttemptRow]) -> Optional[AttemptRow]:
    best: Optional[AttemptRow] = None
    for row in rows:
        if row.validity_x_uniqueness is None:
            continue
        if best is None or row.validity_x_uniqueness > (best.validity_x_uniqueness or float("-inf")):
            best = row
    return best


def build_leaderboard_md(rows: list[AttemptRow]) -> str:
    best = _best_attempt(rows)
    best_line = (
        f"Current best is **attempt {best.attempt:03d}** with V×U="
        f"{_fmt_float(best.validity_x_uniqueness, 3)}."
        if best is not None
        else "No completed attempts with V×U metrics yet."
    )

    lines: list[str] = []
    lines.append("# GEOM Experiment Leaderboard")
    lines.append("")
    lines.append("This file tracks iterative GEOM RL experiments focused on **RDKit validity × RDKit uniqueness**.")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(
        "| Attempt | Date | Description | AtomStab | MolStab | Valid | Uniq | V×U | Duration | Status |"
    )
    lines.append(
        "|---------|------|-------------|----------|---------|-------|------|-----|----------|--------|"
    )
    for r in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    f"{r.attempt:03d}",
                    r.date,
                    r.description,
                    _fmt_float(r.atom_stability),
                    _fmt_float(r.molecule_stability),
                    _fmt_float(r.rdkit_validity),
                    _fmt_float(r.rdkit_uniqueness),
                    _fmt_float(r.validity_x_uniqueness),
                    _fmt_duration(r.duration_hours),
                    r.status,
                ]
            )
            + " |"
        )

    lines.append("")
    lines.append("## Best Run So Far")
    lines.append("")
    lines.append(best_line)
    lines.append("")
    lines.append("## Metrics Definitions")
    lines.append("")
    lines.append("- **AtomStab**: atom stability rate (training rollouts).")
    lines.append("- **MolStab**: molecule stability rate (training rollouts).")
    lines.append("- **Valid**: RDKit validity fraction (training rollouts).")
    lines.append("- **Uniq**: RDKit uniqueness among valid molecules (training rollouts).")
    lines.append("- **V×U**: product of validity and uniqueness (primary objective).")
    lines.append("- **Duration**: wall-clock time inferred from log timestamps + file mtime.")
    lines.append("- **Status**: `completed`, `early_stop`, `failed`, `running`.")
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- Compute budget per attempt: **≤6 hours** (unless explicitly shorter).")
    lines.append("- These metrics come from the trainer's rollout batch (often small), so expect noise.")
    return "\n".join(lines) + "\n"


def main() -> None:
    rows = _iter_attempt_rows()
    out_path = ROOT / "LEADERBOARD.md"
    out_path.write_text(build_leaderboard_md(rows))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()

