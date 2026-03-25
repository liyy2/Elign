#!/usr/bin/env python
import argparse
import csv
import os
import pickle
import statistics
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch
import logging

REPO_ROOT = Path(__file__).resolve().parents[1]
EDM_SOURCE_ROOT = REPO_ROOT / "edm_source"
import sys

for path in (REPO_ROOT, EDM_SOURCE_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from edm_source.configs.datasets_config import get_dataset_info
from edm_source.qm9.dataset import retrieve_dataloaders
from edm_source.qm9.models import get_model
from edm_source.mlff_modules.mlff_guided_diffusion_core import MLFFGuidedDiffusion
from verl_diffusion.model.edm_model import EDMModel


def _parse_int_list(value: str) -> List[int]:
    items = []
    for token in str(value).split(","):
        token = token.strip()
        if not token:
            continue
        items.append(int(token))
    if not items:
        raise ValueError("Empty n_nodes list")
    return items


def _select_device(device_arg: Optional[str]) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    # Prefer an explicit index for reproducibility and to avoid accidental CPU runs.
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _make_masks(batch_size: int, n_nodes: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    node_mask = torch.ones((batch_size, n_nodes, 1), device=device)
    eye = torch.eye(n_nodes, device=device)
    edge_mask = (1.0 - eye).unsqueeze(0).repeat(batch_size, 1, 1)
    edge_mask = edge_mask.view(batch_size * n_nodes * n_nodes, 1)
    return node_mask, edge_mask


def _time_call(fn, device: torch.device) -> float:
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    fn()
    if device.type == "cuda":
        torch.cuda.synchronize()
    return time.perf_counter() - t0


def _seed_all(seed: int, device: torch.device) -> None:
    torch.manual_seed(int(seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(seed))


def _load_edm_config(path: Path):
    with open(path, "rb") as f:
        cfg = pickle.load(f)
    return cfg


def _load_checkpoint_into_edm_model(edm_model: EDMModel, checkpoint_path: Path, device: torch.device) -> None:
    # Reuse the robust loader from eval_verl_rollout.py
    from eval_verl_rollout import load_model_weights  # noqa: WPS433

    load_model_weights(edm_model, checkpoint_path, device)


def main() -> None:
    # Keep guidance benchmarks from being dominated by per-step INFO logs.
    logging.getLogger("edm_source.mlff_modules").setLevel(logging.WARNING)
    logging.getLogger("edm_source.mlff_modules.mlff_logger").setLevel(logging.WARNING)
    logging.getLogger("edm_source.mlff_modules.mlff_force_computer").setLevel(logging.ERROR)

    parser = argparse.ArgumentParser(description="Benchmark inference scaling vs molecule size.")
    parser.add_argument("--args-pickle", type=str, default="pretrained/edm/edm_qm9/args.pickle")
    parser.add_argument("--base-weights", type=str, default="pretrained/edm/edm_qm9/generative_model_ema.npy")
    parser.add_argument(
        "--posttrained-weights",
        type=str,
        default="outputs/smoke_qm9_uma/generative_model_ema.npy",
        help="RL/post-trained checkpoint (state_dict). Set to empty to benchmark base sampler instead.",
    )
    parser.add_argument("--n-nodes", type=str, default="10,15,20,25,29")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--time-step", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--time-stat",
        type=str,
        default="median",
        choices=("median", "mean", "min"),
        help="How to aggregate repeated timings for a single n_nodes.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Base RNG seed for sampling reproducibility.")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--mlff-model", type=str, default="uma-s-1p1")
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument("--guidance-iterations", type=int, default=1)
    parser.add_argument(
        "--noise-threshold",
        type=float,
        default=1.0,
        help="Set to 1.0 to force MLFF queries at every diffusion step (worst-case guidance cost).",
    )
    parser.add_argument("--compute-energy", action="store_true", help="Also query MLFF energies per step.")
    parser.add_argument("--out-csv", type=str, default="benchmarks/results/inference_scaling.csv")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output CSV instead of appending.")
    args = parser.parse_args()

    device = _select_device(args.device)
    n_nodes_list = _parse_int_list(args.n_nodes)
    batch_size = int(args.batch_size)
    time_step = int(args.time_step)

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    edm_config = _load_edm_config(Path(args.args_pickle))
    dataset_name = getattr(edm_config, "dataset", "")
    if isinstance(dataset_name, str) and "qm9" in dataset_name:
        edm_config.datadir = "qm9/temp"
    # Some dataset loaders expect these attributes on the args Namespace.
    edm_config.cuda = device.type == "cuda"
    edm_config.device = device
    if hasattr(edm_config, "no_cuda"):
        edm_config.no_cuda = not edm_config.cuda

    dataset_info = get_dataset_info(edm_config.dataset, edm_config.remove_h)

    dataloaders, _ = retrieve_dataloaders(edm_config)
    flow, _, _ = get_model(edm_config, device, dataset_info, dataloaders["train"])
    flow.to(device)
    flow.eval()

    # Base weights
    base_edm = EDMModel(flow, edm_config).to(device)
    _load_checkpoint_into_edm_model(base_edm, Path(args.base_weights), device)

    # Post-trained weights (optional)
    post_edm = None
    post_path = str(args.posttrained_weights).strip()
    if post_path:
        # Clone a fresh model instance so weights don't overwrite base.
        flow_post, _, _ = get_model(edm_config, device, dataset_info, dataloaders["train"])
        flow_post.to(device)
        flow_post.eval()
        post_edm = EDMModel(flow_post, edm_config).to(device)
        _load_checkpoint_into_edm_model(post_edm, Path(post_path), device)

    # Guided sampler wraps the base diffusion model.
    guided = MLFFGuidedDiffusion(
        base_diffusion=base_edm.model,
        mlff_model=args.mlff_model,
        mlff_predictor=None,
        guidance_scale=float(args.guidance_scale),
        guidance_iterations=int(args.guidance_iterations),
        noise_threshold=float(args.noise_threshold),
        force_clip_threshold=None,
        displacement_clip=None,
        compute_energy=bool(args.compute_energy),
        position_scale=float(getattr(edm_config, "normalize_factors", [1.0])[0]),
        use_wandb=False,
        device=str(device),
    )

    rows: List[Dict[str, object]] = []

    def _bench_one(mode: str, sampler, n_nodes: int) -> Dict[str, object]:
        # EnVariationalDiffusion uses self.T; override for benchmarking.
        sampler_base = sampler.base_diffusion if hasattr(sampler, "base_diffusion") else sampler
        setattr(sampler_base, "T", int(time_step))

        node_mask, edge_mask = _make_masks(batch_size, n_nodes, device)

        def _run():
            if mode == "guided":
                sampler.sample(
                    batch_size,
                    n_nodes,
                    node_mask=node_mask,
                    edge_mask=edge_mask,
                    context=None,
                    fix_noise=False,
                    dataset_info=dataset_info,
                )
            else:
                sampler.sample(batch_size, n_nodes, node_mask, edge_mask, context=None, fix_noise=False)

        for _ in range(int(args.warmup)):
            _seed_all(int(args.seed) + 10_000 + int(n_nodes), device)
            _run()

        mlff_pred_device = ""
        mlff_param_device = ""
        if mode == "guided" and getattr(sampler, "mlff_predictor", None) is not None:
            mlff_pred_device = str(getattr(sampler.mlff_predictor, "device", ""))
            try:
                if hasattr(sampler.mlff_predictor, "model"):
                    mlff_param_device = str(next(sampler.mlff_predictor.model.parameters()).device)
            except Exception:
                mlff_param_device = ""

        # Optional: count how often MLFF evaluation falls back (e.g., "no edges found").
        mlff_calls_total = 0
        mlff_calls_failed = 0
        mlff_samples_total = 0
        mlff_samples_failed = 0

        orig_compute = None
        if mode == "guided" and getattr(sampler, "force_computer", None) is not None:
            orig_compute = sampler.force_computer.compute_mlff_forces
            fallback_val = float(getattr(sampler.force_computer, "fallback_force_magnitude", 5.0))

            def _wrapped_compute(z, node_mask_in, dataset_info_in):  # noqa: WPS430
                nonlocal mlff_calls_total, mlff_calls_failed, mlff_samples_total, mlff_samples_failed
                mlff_calls_total += 1
                out = orig_compute(z, node_mask_in, dataset_info_in)
                forces = out[0] if isinstance(out, (tuple, list)) else out
                # Fallback forces are a constant vector [fallback, 0, 0] for every valid atom.
                mask = node_mask_in[..., 0].bool()
                call_failed = False
                for i in range(int(forces.shape[0])):
                    if not bool(mask[i].any()):
                        continue
                    fi = forces[i][mask[i]]
                    mlff_samples_total += 1
                    is_fallback = bool(
                        torch.all(fi[:, 0] == fallback_val)
                        and torch.all(fi[:, 1] == 0.0)
                        and torch.all(fi[:, 2] == 0.0)
                    )
                    if is_fallback:
                        mlff_samples_failed += 1
                        call_failed = True
                if call_failed:
                    mlff_calls_failed += 1
                return out

            sampler.force_computer.compute_mlff_forces = _wrapped_compute

        times = []
        try:
            for rep in range(int(args.repeats)):
                _seed_all(int(args.seed) + rep + 1_000 * int(n_nodes), device)
                times.append(_time_call(_run, device))
        finally:
            if orig_compute is not None:
                sampler.force_computer.compute_mlff_forces = orig_compute

        if str(args.time_stat) == "mean":
            dt = float(sum(times) / max(1, len(times)))
        elif str(args.time_stat) == "min":
            dt = float(min(times))
        else:
            dt = float(statistics.median(times))

        total_steps = float(batch_size * time_step)
        return {
            "mode": mode,
            "n_nodes": int(n_nodes),
            "batch_size": int(batch_size),
            "time_step": int(time_step),
            "device": str(device),
            "elapsed_sec": dt,
            "steps_per_sec": total_steps / dt if dt > 0 else 0.0,
            "molecules_per_sec": float(batch_size) / dt if dt > 0 else 0.0,
            "sec_per_molecule": dt / float(batch_size) if batch_size > 0 else 0.0,
            "sec_per_step_per_molecule": dt / total_steps if total_steps > 0 else 0.0,
            "mlff_calls_total": int(mlff_calls_total),
            "mlff_calls_failed": int(mlff_calls_failed),
            "mlff_samples_total": int(mlff_samples_total),
            "mlff_samples_failed": int(mlff_samples_failed),
            "mlff_call_failure_rate": (mlff_calls_failed / mlff_calls_total) if mlff_calls_total > 0 else 0.0,
            "mlff_sample_failure_rate": (mlff_samples_failed / mlff_samples_total) if mlff_samples_total > 0 else 0.0,
            "mlff_predictor_device": mlff_pred_device,
            "mlff_model_param_device": mlff_param_device,
        }

    for n_nodes in n_nodes_list:
        rows.append(_bench_one("guided", guided, n_nodes))
        if post_edm is not None:
            rows.append(_bench_one("posttrained", post_edm.model, n_nodes))
        else:
            rows.append(_bench_one("baseline", base_edm.model, n_nodes))

    fieldnames = list(rows[0].keys()) if rows else []
    mode = "w" if args.overwrite else "a"
    write_header = args.overwrite or not out_csv.exists()
    with open(out_csv, mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {out_csv}")


if __name__ == "__main__":
    main()
