import argparse
import math
import os
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional
import sys
REPO_ROOT = Path(__file__).resolve().parent
EDM_SOURCE_ROOT = REPO_ROOT / "edm_source"
for path in (str(REPO_ROOT), str(EDM_SOURCE_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)
import torch
from tqdm import tqdm
from omegaconf import OmegaConf

from edm_source.configs.datasets_config import get_dataset_info
from edm_source.qm9.dataset import retrieve_dataloaders
from edm_source.qm9.models import get_model
from edm_source.qm9.rdkit_functions import retrieve_qm9_smiles
from elign.dataloader.dataloader import EDMDataLoader
from elign.model.edm_model import EDMModel
from elign.worker.rollout.edm_rollout import EDMRollout
from elign.utils.rdkit_metrics import compute_rdkit_metrics



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate rollouts with a post-trained ELIGN diffusion checkpoint (FED-GRPO). "
            "The script loads the original EDM args.pickle, restores the ELIGN fine-tuned weights, "
            "and samples a user-specified number of molecules."
        )
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        required=True,
        help=(
            "Directory that stores the ELIGN post-training artifacts (expects config.yaml / args.pickle / checkpoints). "
            "Used as the default base path for relative inputs and outputs."
        ),
    )
    parser.add_argument(
        "--args-pickle",
        type=str,
        default=None,
        help=(
            "Path to the EDM args.pickle file. If omitted, the script tries (1) <run-dir>/args.pickle, "
            "then (2) <run-dir>/config.yaml -> model.config."
        ),
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help=(
            "Path to the ELIGN checkpoint (.pth or .npy saved with torch.save). "
            "Defaults to <run-dir>/checkpoint_latest.pth if present, otherwise checkpoint_best.pth."
        ),
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help=(
            "Output file to store sampled molecules (torch.save). "
            "Defaults to <run-dir>/eval_rollouts.pt."
        ),
    )
    parser.add_argument(
        "--num-molecules",
        type=int,
        default=1024,
        help="Total number of molecules to sample from the policy.",
    )
    parser.add_argument(
        "--sample-group-size",
        type=int,
        default=None,
        help="Override for dataloader.sample_group_size (defaults to the training config value or 1).",
    )
    parser.add_argument(
        "--each-prompt-sample",
        type=int,
        default=None,
        help="Override for dataloader.each_prompt_sample (defaults to the training config value or 24).",
    )
    parser.add_argument(
        "--micro-batch-size",
        type=int,
        default=None,
        help="Override for dataloader.micro_batch_size (defaults to dataset batch size).",
    )
    parser.add_argument(
        "--time-step",
        type=int,
        default=None,
        help="Override for model.time_step (defaults to config or 1000).",
    )
    parser.add_argument(
        "--share-initial-noise",
        action="store_true",
        help="Force shared initial noise across grouped samples (overrides config).",
    )
    parser.add_argument(
        "--skip-prefix",
        type=int,
        default=0,
        help="Number of diffusion steps to treat as fixed prefix (default disables prefix sharing).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to run evaluation on (cpu, cuda, cuda:0, …). Defaults to cuda if available.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base seed for dataloader sampling.",
    )
    return parser.parse_args()


def make_absolute(path_value: Optional[str], base_dir: Path) -> Optional[Path]:
    if not path_value:
        return None
    path = Path(path_value)
    if path.is_absolute():
        return path
    # Prefer paths relative to the current working directory (friendlier when passing
    # `pretrained/...` or `outputs/...`), and fall back to interpreting them relative
    # to the run directory for backwards compatibility.
    cwd_candidate = (Path.cwd() / path).resolve()
    if cwd_candidate.exists():
        return cwd_candidate
    return (base_dir / path).resolve()


def make_output_path(path_value: Optional[str], run_dir: Path) -> Optional[Path]:
    """Resolve output paths without relying on existence checks.

    For outputs we often want to create new files, so the existence-based fallback in
    `make_absolute()` can accidentally nest paths under `run_dir` when callers pass a
    repo-relative path like `outputs/elign/.../eval.pt`.

    Rule:
    - Absolute paths are kept as-is.
    - Bare filenames (no parent dirs) are interpreted as relative to `run_dir` (convenient).
    - Paths with parent dirs are interpreted as relative to CWD.
    """
    if not path_value:
        return None
    path = Path(path_value)
    if path.is_absolute():
        return path
    if path.parent != Path("."):
        return (Path.cwd() / path).resolve()
    return (run_dir / path).resolve()


def load_edm_config(args_pickle: Path) -> Any:
    with open(args_pickle, "rb") as f:
        edm_config = pickle.load(f)
    if isinstance(edm_config, dict):
        edm_config = OmegaConf.create(edm_config)
    if hasattr(edm_config, "datadir") and edm_config.datadir is None:
        edm_config.datadir = "qm9/temp"
    if not hasattr(edm_config, "normalization_factor"):
        edm_config.normalization_factor = 1
    if not hasattr(edm_config, "aggregation_method"):
        edm_config.aggregation_method = "sum"
    return edm_config


def infer_checkpoint_path(run_dir: Path, explicit_path: Optional[str]) -> Path:
    if explicit_path:
        return make_absolute(explicit_path, run_dir)
    candidates = [
        run_dir / "checkpoint_latest.pth",
        run_dir / "checkpoint_best.pth",
        run_dir / "generative_model_ema.npy",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Could not locate a checkpoint. Pass --checkpoint explicitly or ensure checkpoint_latest.pth exists."
    )


def load_run_config(run_dir: Path) -> Dict[str, Any]:
    config_path = run_dir / "config.yaml"
    if not config_path.exists():
        return {}
    import yaml

    with open(config_path, "r") as f:
        data = yaml.safe_load(f) or {}
    return data


def select_device(device_arg: Optional[str]) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def sanitize_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if not state_dict:
        return state_dict
    sample_key = next(iter(state_dict))
    if sample_key.startswith("module."):
        return {k[len("module.") :]: v for k, v in state_dict.items()}
    return state_dict


def load_model_weights(model: EDMModel, checkpoint_path: Path, device: torch.device) -> None:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state = checkpoint["model_state_dict"]
    else:
        state = checkpoint
    if not isinstance(state, dict):
        raise ValueError(f"Unsupported checkpoint format at {checkpoint_path} (expected a state_dict dict).")

    state = sanitize_state_dict(state)

    # Pretrained EDM checkpoints are stored as the underlying flow state_dict
    # (keys like 'dynamics.*'), whereas ELIGN/FED-GRPO checkpoints save the full
    # EDMModel state_dict (keys like 'model.dynamics.*'). Detect and load both.
    keys = list(state.keys())
    is_flow_state_dict = not any(k.startswith("model.") for k in keys) and any(
        k.startswith("dynamics.") or k.startswith("gamma.") or k.startswith("buffer") for k in keys
    )
    if is_flow_state_dict:
        flow = getattr(model, "model", None)
        if flow is None:
            raise ValueError("EDMModel is missing the underlying `model` (flow) attribute.")
        missing, unexpected = flow.load_state_dict(state, strict=False)
    else:
        missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[WARN] Missing keys while loading checkpoint: {missing}")
    if unexpected:
        print(f"[WARN] Unexpected keys while loading checkpoint: {unexpected}")


def prepare_sampling_config(
    run_config: Dict[str, Any],
    overrides: argparse.Namespace,
) -> Dict[str, Any]:
    cfg = {
        "distributed": {
            "rank": 0,
            "world_size": 1,
            "local_rank": 0,
            "is_main_process": True,
        },
        "queue_size": 8,
        "model": {},
        "dataloader": {},
        "train": {},
        "reward": {},
    }

    model_cfg = run_config.get("model", {}) if isinstance(run_config, dict) else {}
    dataloader_cfg = run_config.get("dataloader", {}) if isinstance(run_config, dict) else {}

    cfg["model"]["time_step"] = (
        overrides.time_step
        if overrides.time_step is not None
        else int(model_cfg.get("time_step", 1000))
    )
    cfg["model"]["share_initial_noise"] = bool(overrides.share_initial_noise)
    cfg["model"]["config"] = model_cfg.get("config")
    cfg["model"]["model_path"] = model_cfg.get("model_path")

    cfg["dataloader"]["sample_group_size"] = (
        overrides.sample_group_size
        if overrides.sample_group_size is not None
        else int(dataloader_cfg.get("sample_group_size", 1))
    )
    cfg["dataloader"]["each_prompt_sample"] = (
        overrides.each_prompt_sample
        if overrides.each_prompt_sample is not None
        else int(dataloader_cfg.get("each_prompt_sample", 24))
    )
    cfg["dataloader"]["micro_batch_size"] = (
        overrides.micro_batch_size
        if overrides.micro_batch_size is not None
        else int(dataloader_cfg.get("micro_batch_size", cfg["dataloader"]["each_prompt_sample"]))
    )
    cfg["dataloader"]["smiles_path"] = dataloader_cfg.get("smiles_path", "qm9/temp/qm9_smiles.pickle")
    cfg["dataloader"]["epoches"] = dataloader_cfg.get("epoches", 1)

    cfg["train"]["force_alignment_enabled"] = False
    cfg["train"]["force_alignment_weight"] = 0.0

    skip_prefix = max(int(overrides.skip_prefix), 0)
    cfg["reward"]["shaping"] = {
        "enabled": False,
        "skip_prefix": skip_prefix,
        "scheduler": {"skip_prefix": skip_prefix},
    }

    return cfg


def serialize_samples(
    samples: List[Dict[str, Any]],
    output_path: Path,
    rdkit_metrics: Optional[Dict[str, Any]] = None,
) -> None:
    payload = {"samples": samples}
    if rdkit_metrics is not None:
        payload["rdkit_metrics"] = rdkit_metrics
    torch.save(payload, output_path)


def main() -> None:
    args = parse_args()
    cwd = Path(os.getcwd())
    run_dir = make_absolute(args.run_dir, cwd) if not Path(args.run_dir).is_absolute() else Path(args.run_dir)
    run_dir = run_dir.resolve()

    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory '{run_dir}' does not exist.")

    run_config = load_run_config(run_dir)

    args_pickle_path: Optional[Path] = None
    if args.args_pickle:
        args_pickle_path = make_absolute(args.args_pickle, run_dir)
    else:
        candidate = make_absolute("args.pickle", run_dir)
        if candidate is not None and candidate.exists():
            args_pickle_path = candidate
        elif isinstance(run_config, dict):
            model_cfg = run_config.get("model")
            if isinstance(model_cfg, dict) and model_cfg.get("config"):
                inferred = make_absolute(str(model_cfg["config"]), run_dir)
                if inferred is not None and inferred.exists():
                    args_pickle_path = inferred

    if args_pickle_path is None or not args_pickle_path.exists():
        raise FileNotFoundError(
            "Could not locate the EDM args.pickle.\n"
            f"- Tried: {run_dir / 'args.pickle'}\n"
            "- Tried: <run-dir>/config.yaml -> model.config\n"
            "Fix: pass --args-pickle explicitly, or copy args.pickle into the run directory."
        )

    checkpoint_path = infer_checkpoint_path(run_dir, args.checkpoint)
    output_path = (
        make_output_path(args.output, run_dir)
        if args.output
        else run_dir / "eval_rollouts.pt"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    sampling_config = prepare_sampling_config(run_config, args)

    edm_config = load_edm_config(args_pickle_path)
    edm_config.datadir = getattr(edm_config, "datadir", "qm9/temp") or "qm9/temp"

    device = select_device(args.device)
    edm_config.cuda = device.type == "cuda"
    if hasattr(edm_config, "no_cuda"):
        edm_config.no_cuda = not edm_config.cuda
    edm_config.device = device

    dataset_info = get_dataset_info(edm_config.dataset, edm_config.remove_h)
    retrieve_qm9_smiles(dataset_info)
    dataloaders, _ = retrieve_dataloaders(edm_config)
    flow, nodes_dist, prop_dist = get_model(edm_config, device, dataset_info, dataloaders["train"])
    flow.to(device)

    model = EDMModel(flow, edm_config)
    model.to(device)

    base_model_path = sampling_config["model"].get("model_path")
    if base_model_path:
        base_model_path = make_absolute(base_model_path, cwd)
        if base_model_path and base_model_path.exists():
            try:
                model.load(model_path=str(base_model_path))
            except Exception as exc:
                print(f"[WARN] Failed to load base EDM weights from {base_model_path}: {exc}")

    load_model_weights(model, checkpoint_path, device)
    model.eval()

    batch_size = sampling_config["dataloader"]["sample_group_size"] * sampling_config["dataloader"]["each_prompt_sample"]
    total_batches = math.ceil(args.num_molecules / max(batch_size, 1))

    dataloader = EDMDataLoader(
        config=sampling_config,
        dataset_info=dataset_info,
        nodes_dist=nodes_dist,
        prop_dist=prop_dist,
        device=device,
        condition=False,
        num_batches=total_batches,
        rank=0,
        world_size=1,
        base_seed=args.seed,
    )

    rollout = EDMRollout(model, sampling_config)

    generated: List[Dict[str, Any]] = []
    target = args.num_molecules
    sample_group_size = sampling_config["dataloader"]["sample_group_size"]
    each_prompt_sample = sampling_config["dataloader"]["each_prompt_sample"]
    progress = tqdm(
        total=target,
        desc="Sampling molecules",
        unit="mol",
        postfix={
            "group_size": sample_group_size,
            "per_prompt": each_prompt_sample,
        },
    )

    for prompts in dataloader:
        with torch.no_grad():
            samples = rollout.generate_samples(prompts)

        x = samples.batch["x"].to("cpu")
        one_hot = samples.batch["categorical"].to("cpu")
        nodesxsample = samples.batch["nodesxsample"].to("cpu")
        group_index = samples.batch.get("group_index")
        if group_index is not None:
            group_index = group_index.to("cpu")
        timesteps = samples.batch.get("timesteps")
        if timesteps is not None:
            timesteps = timesteps.to("cpu")

        batch_count = x.shape[0]
        for idx in range(batch_count):
            if len(generated) >= target:
                break
            num_nodes = int(nodesxsample[idx].item())
            positions = x[idx, :num_nodes].clone()
            atom_one_hot = one_hot[idx, :num_nodes].clone()
            atom_types = torch.argmax(atom_one_hot, dim=-1)

            sample_entry: Dict[str, Any] = {
                "positions": positions,
                "atom_types": atom_types,
                "num_atoms": num_nodes,
            }
            if group_index is not None:
                sample_entry["group_index"] = int(group_index[idx].item())
            if timesteps is not None:
                sample_entry["timesteps"] = timesteps[idx].clone()

            generated.append(sample_entry)

        completed = min(len(generated), target)
        if completed > progress.n:
            progress.update(completed - progress.n)

        if len(generated) >= target:
            break

    progress.close()

    final_samples = generated[:target]
    rdkit_metrics = compute_rdkit_metrics(final_samples, dataset_info)
    serialize_samples(final_samples, output_path, rdkit_metrics)
    print(
        f"Saved {len(final_samples)} molecules to {output_path}. "
        f"(batch_size={batch_size}, batches={total_batches}, device={device})"
    )
    rdkit_error = rdkit_metrics.get("error") if isinstance(rdkit_metrics, dict) else None
    if rdkit_error:
        print(f"[WARN] Skipped RDKit metrics: {rdkit_error}")
    else:
        validity_pct = rdkit_metrics["validity"] * 100.0
        uniqueness_pct = rdkit_metrics["uniqueness"] * 100.0
        num_total = rdkit_metrics["num_total"]
        num_valid = rdkit_metrics["num_valid"]
        num_unique = rdkit_metrics["num_unique"]
        print(
            f"RDKit validity: {validity_pct:.2f}% ({num_valid}/{num_total})"
        )
        if num_valid > 0:
            print(
                f"RDKit uniqueness: {uniqueness_pct:.2f}% ({num_unique}/{num_valid} valid)"
            )
        else:
            print("RDKit uniqueness: n/a (no valid molecules)")


if __name__ == "__main__":
    main()
