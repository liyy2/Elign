import argparse
import math
import os
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional
import sys

REPO_ROOT = Path(__file__).resolve().parent
EDM_SOURCE_ROOT = REPO_ROOT / "edm_source"
for path in (REPO_ROOT, EDM_SOURCE_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)
import torch
from tqdm import tqdm
from omegaconf import OmegaConf
from torch.distributions import Categorical

from edm_source.configs.datasets_config import get_dataset_info
from edm_source.qm9.dataset import retrieve_dataloaders
from edm_source.qm9.models import get_model, get_latent_diffusion
from edm_source.qm9.rdkit_functions import retrieve_qm9_smiles
from verl_diffusion.dataloader.dataloader import EDMDataLoader
from verl_diffusion.model.edm_model import EDMModel
from verl_diffusion.worker.rollout.edm_rollout import EDMRollout
from verl_diffusion.utils.rdkit_metrics import compute_rdkit_metrics



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate rollouts with a post-trained VERL diffusion checkpoint. "
            "The script loads the original EDM args.pickle, restores the VERL fine-tuned weights, "
            "and samples a user-specified number of molecules."
        )
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        required=True,
        help=(
            "Directory that stores the VERL training artifacts (expects config.yaml / args.pickle / checkpoints). "
            "Used as the default base path for relative inputs and outputs."
        ),
    )
    parser.add_argument(
        "--args-pickle",
        type=str,
        default=None,
        help="Path to the EDM args.pickle file. Defaults to <run-dir>/args.pickle.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help=(
            "Path to the VERL checkpoint (.pth or .npy saved with torch.save). "
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
        "--fixed-num-atoms",
        type=int,
        default=None,
        help=(
            "Force the node-count prior to always sample this many atoms (total nodes). "
            "Useful for generating baseline/RL samples at roughly the same size."
        ),
    )
    parser.add_argument(
        "--nodes-focus-min",
        type=int,
        default=None,
        help="Override dataloader.nodes_dist_focus_min for eval sampling.",
    )
    parser.add_argument(
        "--nodes-focus-max",
        type=int,
        default=None,
        help="Override dataloader.nodes_dist_focus_max for eval sampling.",
    )
    parser.add_argument(
        "--nodes-focus-multiplier",
        type=float,
        default=None,
        help="Override dataloader.nodes_dist_focus_multiplier for eval sampling.",
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
        dest="share_initial_noise",
        action="store_true",
        default=None,
        help="Force shared initial noise across grouped samples (overrides config).",
    )
    parser.add_argument(
        "--no-share-initial-noise",
        dest="share_initial_noise",
        action="store_false",
        default=None,
        help="Disable shared initial noise across grouped samples (overrides config).",
    )
    parser.add_argument(
        "--skip-prefix",
        type=int,
        default=None,
        help=(
            "Number of diffusion steps to treat as fixed prefix. "
            "Defaults to the training config value (or 0 if unset)."
        ),
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
    parser.add_argument(
        "--geom-data-file",
        type=str,
        default=None,
        help="Optional override for the GEOM `geom_drugs_30.npy` path.",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="auto",
        choices=["auto", "edm", "geoldm"],
        help="Model backend to use. Defaults to auto-detect from args.pickle.",
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
    repo-relative path like `outputs/verl/.../eval.pt`.

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
    # (keys like 'dynamics.*'), whereas VERL/DDPO checkpoints save the full
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


def resolve_args_pickle_path(
    args: argparse.Namespace,
    run_dir: Path,
    run_config: Dict[str, Any],
) -> Path:
    if args.args_pickle:
        candidate = make_absolute(args.args_pickle, run_dir)
        if candidate is not None and candidate.exists():
            return candidate
        raise FileNotFoundError(f"Could not locate args.pickle at '{candidate}'.")

    candidate = run_dir / "args.pickle"
    if candidate.exists():
        return candidate

    model_cfg = run_config.get("model", {}) if isinstance(run_config, dict) else {}
    config_value = model_cfg.get("config")
    if config_value:
        candidate = make_absolute(str(config_value), run_dir)
        if candidate is not None and candidate.exists():
            return candidate

    raise FileNotFoundError(
        "Could not locate args.pickle. Pass --args-pickle explicitly, copy args.pickle into --run-dir, "
        "or ensure config.yaml contains model.config pointing to the EDM args.pickle."
    )


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
    cfg["model"]["share_initial_noise"] = (
        bool(overrides.share_initial_noise)
        if overrides.share_initial_noise is not None
        else bool(model_cfg.get("share_initial_noise", False))
    )
    cfg["model"]["config"] = model_cfg.get("config")
    cfg["model"]["model_path"] = model_cfg.get("model_path")
    cfg["model"]["return_suffix_only"] = bool(model_cfg.get("return_suffix_only", False))
    cfg["model"]["backend"] = str(model_cfg.get("backend", "auto") or "auto")

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
    # GEOM requires an explicit `.npy` file path for loading conformations.
    cfg["dataloader"]["geom_data_file"] = dataloader_cfg.get("geom_data_file")
    if overrides.geom_data_file is not None:
        cfg["dataloader"]["geom_data_file"] = overrides.geom_data_file
    cfg["dataloader"]["smiles_path"] = dataloader_cfg.get("smiles_path", "qm9/temp/qm9_smiles.pickle")
    cfg["dataloader"]["epoches"] = dataloader_cfg.get("epoches", 1)
    cfg["dataloader"]["nodes_dist_focus_min"] = (
        int(overrides.nodes_focus_min)
        if overrides.nodes_focus_min is not None
        else dataloader_cfg.get("nodes_dist_focus_min")
    )
    cfg["dataloader"]["nodes_dist_focus_max"] = (
        int(overrides.nodes_focus_max)
        if overrides.nodes_focus_max is not None
        else dataloader_cfg.get("nodes_dist_focus_max")
    )
    cfg["dataloader"]["nodes_dist_focus_multiplier"] = (
        float(overrides.nodes_focus_multiplier)
        if overrides.nodes_focus_multiplier is not None
        else dataloader_cfg.get("nodes_dist_focus_multiplier", 1.0)
    )
    cfg["dataloader"]["nodes_dist_fixed"] = int(overrides.fixed_num_atoms) if overrides.fixed_num_atoms else None

    cfg["train"]["force_alignment_enabled"] = False
    cfg["train"]["force_alignment_weight"] = 0.0

    if overrides.skip_prefix is not None:
        skip_prefix = max(int(overrides.skip_prefix), 0)
    else:
        skip_prefix = max(int(model_cfg.get("skip_prefix", 0) or 0), 0)
    cfg["model"]["skip_prefix"] = skip_prefix
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

    checkpoint_path = infer_checkpoint_path(run_dir, args.checkpoint)
    output_path = (
        make_output_path(args.output, run_dir)
        if args.output
        else run_dir / "eval_rollouts.pt"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    run_config = load_run_config(run_dir)
    sampling_config = prepare_sampling_config(run_config, args)

    args_pickle_path = resolve_args_pickle_path(args, run_dir, run_config)
    edm_config = load_edm_config(args_pickle_path)
    backend = str(args.backend or sampling_config["model"].get("backend", "auto") or "auto").lower()
    if backend in {"", "auto"}:
        backend = "geoldm" if bool(getattr(edm_config, "train_diffusion", False)) else "edm"
    if backend not in {"edm", "geoldm"}:
        raise ValueError(f"Unsupported backend '{backend}'.")
    print(f"Using diffusion backend: {backend}")

    # Match the training entrypoint (`run_verl_diffusion.py`):
    # - QM9 uses a processed cache directory.
    # - GEOM requires an explicit `.npy` path (geom_data_file / geom_data_path).
    dataset_name = getattr(edm_config, "dataset", "")
    if isinstance(dataset_name, str) and "qm9" in dataset_name:
        edm_config.datadir = "qm9/temp"

    dataloader_cfg = sampling_config.get("dataloader") or {}
    if isinstance(dataloader_cfg, dict):
        geom_data_file = dataloader_cfg.get("geom_data_file")
        if geom_data_file:
            setattr(edm_config, "geom_data_file", geom_data_file)
            setattr(edm_config, "geom_data_path", geom_data_file)

    device = select_device(args.device)
    edm_config.cuda = device.type == "cuda"
    if hasattr(edm_config, "no_cuda"):
        edm_config.no_cuda = not edm_config.cuda
    edm_config.device = device

    dataset_info = get_dataset_info(edm_config.dataset, edm_config.remove_h)
    # Only QM9 needs a dataset SMILES list; GEOM evaluation doesn't.
    if "qm9" in str(dataset_info.get("name", "")):
        retrieve_qm9_smiles(dataset_info)
    dataloaders, _ = retrieve_dataloaders(edm_config)
    if backend == "geoldm":
        flow, nodes_dist, prop_dist = get_latent_diffusion(edm_config, device, dataset_info, dataloaders["train"])
    else:
        flow, nodes_dist, prop_dist = get_model(edm_config, device, dataset_info, dataloaders["train"])
    flow.to(device)

    # Mirror the RL rollout node-count reweighting so offline eval matches training prompts.
    dataloader_cfg = sampling_config.get("dataloader") or {}
    if isinstance(dataloader_cfg, dict) and nodes_dist is not None:
        focus_min = dataloader_cfg.get("nodes_dist_focus_min")
        focus_max = dataloader_cfg.get("nodes_dist_focus_max")
        focus_multiplier = dataloader_cfg.get("nodes_dist_focus_multiplier", 1.0)
        try:
            focus_multiplier = float(focus_multiplier) if focus_multiplier is not None else 1.0
        except (TypeError, ValueError):
            focus_multiplier = 1.0

        if focus_min is not None and focus_max is not None and focus_multiplier != 1.0:
            try:
                focus_min = int(focus_min)
                focus_max = int(focus_max)
            except (TypeError, ValueError):
                focus_min = None
                focus_max = None

        if focus_min is not None and focus_max is not None and focus_multiplier != 1.0:
            if hasattr(nodes_dist, "prob") and hasattr(nodes_dist, "n_nodes") and hasattr(nodes_dist, "m"):
                prob = nodes_dist.prob.detach().clone().to(dtype=torch.float64)
                n_nodes = nodes_dist.n_nodes.detach().to(dtype=torch.long)
                mask = (n_nodes >= focus_min) & (n_nodes <= focus_max)
                if mask.any():
                    prob[mask] = prob[mask] * focus_multiplier
                    prob = prob / prob.sum().clamp(min=1e-12)
                    nodes_dist.prob = prob.to(dtype=torch.float32)
                    nodes_dist.m = Categorical(nodes_dist.prob)
                    print(
                        f"Reweighted n_nodes prior for eval rollouts: "
                        f"[{focus_min}, {focus_max}] x {focus_multiplier}"
                    )

        fixed_nodes = dataloader_cfg.get("nodes_dist_fixed")
        if fixed_nodes is not None:
            try:
                fixed_nodes = int(fixed_nodes)
            except (TypeError, ValueError):
                fixed_nodes = None

        if fixed_nodes is not None:
            if hasattr(nodes_dist, "prob") and hasattr(nodes_dist, "n_nodes") and hasattr(nodes_dist, "m"):
                n_nodes = nodes_dist.n_nodes.detach().to(dtype=torch.long)
                mask = n_nodes == int(fixed_nodes)
                if not mask.any():
                    raise ValueError(
                        f"--fixed-num-atoms={fixed_nodes} is not supported by this dataset prior "
                        f"(min={int(n_nodes.min())}, max={int(n_nodes.max())})."
                    )
                prob = torch.zeros_like(nodes_dist.prob, dtype=torch.float64)
                prob[mask] = 1.0
                prob = prob / prob.sum().clamp(min=1e-12)
                nodes_dist.prob = prob.to(dtype=torch.float32)
                nodes_dist.m = Categorical(nodes_dist.prob)
                print(f"Fixed n_nodes prior for eval rollouts: n_nodes={fixed_nodes}")

    model = EDMModel(flow, edm_config, backend=backend)
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
