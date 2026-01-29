import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run eval_elign_rollout.py + compute_elign_metrics.py for multiple seeds and aggregate "
            "RDKit validity/uniqueness/novelty + stability metrics."
        )
    )
    parser.add_argument("--run-dir", type=str, required=True, help="ELIGN run directory.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoint_latest.pth",
        help="Checkpoint path (absolute or relative to --run-dir).",
    )
    parser.add_argument(
        "--args-pickle",
        type=str,
        default="pretrained/edm/edm_qm9/args.pickle",
        help="Path to EDM args.pickle.",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="123,456,789",
        help="Comma-separated list of integer seeds.",
    )
    parser.add_argument(
        "--num-molecules",
        type=int,
        default=256,
        help="Number of molecules to sample per seed.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device string passed to eval + metrics scripts.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size for MLFF metrics evaluation (defaults to --num-molecules).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to write aggregated JSON. Defaults to <run-dir>/eval_metrics_multi_seed.json.",
    )
    parser.add_argument(
        "--repair-invalid",
        action="store_true",
        help=(
            "If set, compute_elign_metrics.py will run an UMA-based repair step on RDKit-invalid samples "
            "before computing metrics. This typically improves RDKit validity without touching already-valid molecules."
        ),
    )
    parser.add_argument(
        "--repair-steps",
        type=int,
        default=3,
        help="Number of UMA force update steps for invalid-sample repair (default: 3).",
    )
    parser.add_argument(
        "--repair-alpha",
        type=float,
        default=0.01,
        help="Step size for invalid-sample repair: pos <- pos + alpha * force (default: 0.01).",
    )
    return parser.parse_args()


def _abs_path(value: str, base_dir: Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return path


def _load_metrics_json(path: Path) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def main() -> None:
    args = parse_args()
    run_dir = _abs_path(args.run_dir, Path.cwd())
    checkpoint_path = _abs_path(args.checkpoint, run_dir)
    args_pickle = _abs_path(args.args_pickle, Path.cwd())
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    if not seeds:
        raise ValueError("--seeds must contain at least one integer seed")

    num_molecules = int(args.num_molecules)
    if num_molecules <= 0:
        raise ValueError("--num-molecules must be > 0")

    batch_size = int(args.batch_size) if args.batch_size is not None else num_molecules
    if batch_size <= 0:
        raise ValueError("--batch-size must be > 0")

    per_seed: Dict[str, Dict[str, Any]] = {}
    for seed in seeds:
        samples_path = run_dir / f"eval_{num_molecules}_seed{seed}.pt"
        metrics_path = run_dir / f"eval_metrics_{num_molecules}_seed{seed}.json"

        subprocess.check_call(
            [
                sys.executable,
                "eval_elign_rollout.py",
                "--run-dir",
                str(run_dir),
                "--args-pickle",
                str(args_pickle),
                "--checkpoint",
                str(checkpoint_path),
                "--output",
                str(samples_path),
                "--num-molecules",
                str(num_molecules),
                "--seed",
                str(seed),
                "--device",
                args.device,
            ]
        )

        metrics_cmd = [
            sys.executable,
            "compute_elign_metrics.py",
            "--run-dir",
            str(run_dir),
            "--samples",
            str(samples_path),
            "--args-pickle",
            str(args_pickle),
            "--batch-size",
            str(batch_size),
            "--device",
            args.device,
            "--output",
            str(metrics_path),
        ]
        if args.repair_invalid:
            metrics_cmd.extend(
                [
                    "--repair-invalid",
                    "--repair-steps",
                    str(int(args.repair_steps)),
                    "--repair-alpha",
                    str(float(args.repair_alpha)),
                ]
            )
        subprocess.check_call(metrics_cmd)

        payload = _load_metrics_json(metrics_path)
        rdkit = payload.get("rdkit_metrics", {}) if isinstance(payload, dict) else {}
        rdkit_raw = payload.get("rdkit_metrics_raw") if isinstance(payload, dict) else None
        per_seed[str(seed)] = {
            "num_total": int(rdkit.get("num_total", num_molecules)),
            "num_valid": int(rdkit.get("num_valid", 0)),
            "num_unique": int(rdkit.get("num_unique", 0)),
            "rdkit_validity": float(rdkit.get("validity", 0.0)),
            "rdkit_uniqueness": float(rdkit.get("uniqueness", 0.0)),
            "novelty_frac": float(rdkit.get("novelty_frac", 0.0)),
            "stability_rate": float(payload.get("stability_rate", 0.0)),
            "atom_stability_rate": float(payload.get("atom_stability", {}).get("mean", 0.0)),
        }
        per_seed[str(seed)]["validity_x_uniqueness"] = (
            per_seed[str(seed)]["rdkit_validity"] * per_seed[str(seed)]["rdkit_uniqueness"]
        )
        if isinstance(rdkit_raw, dict):
            per_seed[str(seed)]["rdkit_validity_raw"] = float(rdkit_raw.get("validity", 0.0))
            per_seed[str(seed)]["rdkit_uniqueness_raw"] = float(rdkit_raw.get("uniqueness", 0.0))
            per_seed[str(seed)]["validity_x_uniqueness_raw"] = (
                per_seed[str(seed)]["rdkit_validity_raw"] * per_seed[str(seed)]["rdkit_uniqueness_raw"]
            )

    def _avg(key: str) -> float:
        return sum(per_seed[str(seed)][key] for seed in seeds) / len(seeds)

    avg = {
        "rdkit_validity": _avg("rdkit_validity"),
        "rdkit_uniqueness": _avg("rdkit_uniqueness"),
        "validity_x_uniqueness": _avg("validity_x_uniqueness"),
        "stability_rate": _avg("stability_rate"),
        "atom_stability_rate": _avg("atom_stability_rate"),
        "novelty_frac": _avg("novelty_frac"),
    }
    if all("validity_x_uniqueness_raw" in per_seed[str(seed)] for seed in seeds):
        avg["rdkit_validity_raw"] = _avg("rdkit_validity_raw")
        avg["rdkit_uniqueness_raw"] = _avg("rdkit_uniqueness_raw")
        avg["validity_x_uniqueness_raw"] = _avg("validity_x_uniqueness_raw")

    result = {
        "run_dir": str(run_dir),
        "checkpoint": str(checkpoint_path.name),
        "eval": {
            "num_seeds": len(seeds),
            "num_molecules_per_seed": num_molecules,
            "avg": avg,
            "per_seed": per_seed,
        },
    }

    output_path = _abs_path(
        args.output if args.output else "eval_metrics_multi_seed.json",
        run_dir,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Wrote aggregated metrics to {output_path}")


if __name__ == "__main__":
    main()
