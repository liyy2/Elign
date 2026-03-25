import argparse
import math
import os
import pickle
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import sys

REPO_ROOT = Path(__file__).resolve().parent
EDM_SOURCE_ROOT = REPO_ROOT / "edm_source"
for path in (REPO_ROOT, EDM_SOURCE_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.distributions import Categorical
from tqdm import tqdm

from edm_source.configs.datasets_config import get_dataset_info
from edm_source.qm9.dataset import retrieve_dataloaders
from edm_source.qm9.models import get_model
from edm_source.qm9.rdkit_functions import retrieve_qm9_smiles
from edm_source.mlff_modules.mlff_force_computer import MLFFForceComputer
from edm_source.mlff_modules.mlff_utils import get_mlff_predictor

from verl_diffusion.dataloader.dataloader import EDMDataLoader
from verl_diffusion.model.edm_model import EDMModel
from verl_diffusion.worker.reward.scheduler import RewardScheduler


def _set_global_seed(seed: int) -> None:
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_absolute(path_value: Optional[str], base_dir: Path) -> Optional[Path]:
    if not path_value:
        return None
    path = Path(path_value)
    if path.is_absolute():
        return path
    candidate = (base_dir / path).resolve()
    if candidate.exists():
        return candidate
    candidate = (Path.cwd() / path).resolve()
    if candidate.exists():
        return candidate
    return candidate


def load_run_config(run_dir: Path) -> Dict[str, Any]:
    config_path = run_dir / "config.yaml"
    if not config_path.exists():
        return {}
    import yaml

    with open(config_path, "r") as f:
        data = yaml.safe_load(f) or {}
    return data


def infer_checkpoint_path(run_dir: Path, explicit_path: Optional[str]) -> Path:
    if explicit_path:
        resolved = make_absolute(explicit_path, run_dir)
        if resolved is None:
            raise FileNotFoundError(f"Could not resolve checkpoint path: {explicit_path}")
        return resolved
    candidates = [
        run_dir / "checkpoint_best.pth",
        run_dir / "checkpoint_latest.pth",
        run_dir / "checkpoint.pth",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Could not locate a post-trained checkpoint. Pass --posttrained explicitly or ensure checkpoint_best.pth exists."
    )


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


def select_device(device_arg: Optional[str]) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _resolve_str_device(device_like: Optional[str], fallback: torch.device) -> str:
    if device_like is None:
        candidate = fallback
    else:
        candidate = device_like
    if isinstance(candidate, torch.device):
        if candidate.type == "cuda":
            return f"cuda:{candidate.index}" if candidate.index is not None else "cuda"
        return "cpu"
    return str(candidate)


def _reweight_nodes_dist(nodes_dist, dataloader_cfg: Dict[str, Any]) -> None:
    if nodes_dist is None or not isinstance(dataloader_cfg, dict):
        return

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
                print(f"Reweighted n_nodes prior: [{focus_min}, {focus_max}] x {focus_multiplier}")


def _maybe_fix_nodes_dist(nodes_dist, dataloader_cfg: Dict[str, Any]) -> None:
    if nodes_dist is None or not isinstance(dataloader_cfg, dict):
        return
    fixed_nodes = dataloader_cfg.get("nodes_dist_fixed")
    if fixed_nodes is None:
        return
    try:
        fixed_nodes = int(fixed_nodes)
    except (TypeError, ValueError):
        return

    if hasattr(nodes_dist, "prob") and hasattr(nodes_dist, "n_nodes") and hasattr(nodes_dist, "m"):
        n_nodes = nodes_dist.n_nodes.detach().to(dtype=torch.long)
        mask = n_nodes == fixed_nodes
        if not mask.any():
            raise ValueError(
                f"nodes_dist_fixed={fixed_nodes} unsupported (min={int(n_nodes.min())}, max={int(n_nodes.max())})."
            )
        prob = torch.zeros_like(nodes_dist.prob, dtype=torch.float64)
        prob[mask] = 1.0
        prob = prob / prob.sum().clamp(min=1e-12)
        nodes_dist.prob = prob.to(dtype=torch.float32)
        nodes_dist.m = Categorical(nodes_dist.prob)
        print(f"Fixed n_nodes prior: n_nodes={fixed_nodes}")


@dataclass
class RunningMoments:
    count: int = 0
    sum: float = 0.0
    sum_sq: float = 0.0
    min: float = float("inf")
    max: float = float("-inf")

    def update(self, values: torch.Tensor) -> None:
        if values.numel() == 0:
            return
        values_f = values.detach().to(dtype=torch.float64)
        count = int(values_f.numel())
        self.count += count
        self.sum += float(values_f.sum().item())
        self.sum_sq += float((values_f * values_f).sum().item())
        self.min = min(self.min, float(values_f.min().item()))
        self.max = max(self.max, float(values_f.max().item()))

    def mean(self) -> Optional[float]:
        if self.count <= 0:
            return None
        return self.sum / float(self.count)

    def std(self) -> Optional[float]:
        if self.count <= 1:
            return None
        mean = self.sum / float(self.count)
        var = self.sum_sq / float(self.count) - mean * mean
        var = max(var, 0.0)
        return math.sqrt(var)


@dataclass
class RunningCorrelation:
    count: int = 0
    sum_x: float = 0.0
    sum_y: float = 0.0
    sum_x2: float = 0.0
    sum_y2: float = 0.0
    sum_xy: float = 0.0

    def update(self, x: torch.Tensor, y: torch.Tensor) -> None:
        if x.numel() == 0 or y.numel() == 0:
            return
        if x.shape != y.shape:
            raise ValueError(f"Correlation update expects matching shapes, got {x.shape} vs {y.shape}")
        x_f = x.detach().to(dtype=torch.float64)
        y_f = y.detach().to(dtype=torch.float64)
        n = int(x_f.numel())
        self.count += n
        self.sum_x += float(x_f.sum().item())
        self.sum_y += float(y_f.sum().item())
        self.sum_x2 += float((x_f * x_f).sum().item())
        self.sum_y2 += float((y_f * y_f).sum().item())
        self.sum_xy += float((x_f * y_f).sum().item())

    def pearson(self) -> Optional[float]:
        n = float(self.count)
        if n <= 1.0:
            return None
        num = self.sum_xy - (self.sum_x * self.sum_y) / n
        den_x = self.sum_x2 - (self.sum_x * self.sum_x) / n
        den_y = self.sum_y2 - (self.sum_y * self.sum_y) / n
        den = math.sqrt(max(den_x, 0.0) * max(den_y, 0.0))
        if den <= 0.0:
            return None
        return float(num / den)


class ReservoirSampler:
    """Uniform reservoir sampling over a stream of equal-length 1D arrays."""

    def __init__(self, max_size: int, seed: int, dtypes: Dict[str, Any]):
        if max_size <= 0:
            raise ValueError("max_size must be > 0")
        self.max_size = int(max_size)
        self.rng = np.random.default_rng(int(seed))
        self.dtypes = dict(dtypes)
        self.data = {k: np.empty(self.max_size, dtype=self.dtypes[k]) for k in self.dtypes}
        self.filled = 0
        self.seen = 0  # total items processed in the stream

    def update(self, batch: Dict[str, np.ndarray]) -> None:
        if not batch:
            return
        keys = list(self.dtypes.keys())
        for k in keys:
            if k not in batch:
                raise KeyError(f"Reservoir update missing key '{k}'")
        lengths = {int(len(batch[k])) for k in keys}
        if len(lengths) != 1:
            raise ValueError(f"Reservoir update expects equal lengths, got {sorted(lengths)}")
        m = int(next(iter(lengths)))
        if m <= 0:
            return

        # Fill remaining capacity first.
        if self.filled < self.max_size:
            take = min(self.max_size - self.filled, m)
            end = self.filled + take
            for k in keys:
                self.data[k][self.filled:end] = batch[k][:take].astype(self.dtypes[k], copy=False)
            self.filled = end
            self.seen += take
            if take == m:
                return
            # Continue with the remainder.
            batch = {k: batch[k][take:] for k in keys}
            m = m - take

        # Reservoir sampling for the remainder (vectorized).
        # Items have stream indices [seen, seen+m-1] (0-indexed). For each item i with
        # global index g, draw j ~ Uniform{0..g}. If j < K, replace reservoir[j].
        K = self.max_size
        g = np.arange(self.seen, self.seen + m, dtype=np.int64)
        j = (self.rng.random(m) * (g + 1)).astype(np.int64)
        mask = j < K
        if np.any(mask):
            targets = j[mask]
            for k in keys:
                self.data[k][targets] = batch[k][mask].astype(self.dtypes[k], copy=False)
        self.seen += m

    def to_dict(self) -> Dict[str, np.ndarray]:
        size = int(self.filled)
        return {k: self.data[k][:size].copy() for k in self.data}


def _compute_forces(
    force_computer: MLFFForceComputer,
    z: torch.Tensor,
    node_mask: torch.Tensor,
    dataset_info: Dict[str, Any],
    chunk_size: int,
) -> torch.Tensor:
    total = int(z.shape[0])
    chunk_size = max(1, int(chunk_size))
    forces = []
    for start in range(0, total, chunk_size):
        end = min(total, start + chunk_size)
        forces.append(force_computer.compute_mlff_forces(z[start:end], node_mask[start:end], dataset_info))
    return torch.cat(forces, dim=0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Post-hoc analysis of force-alignment: compare the learned drift deviation "
            "(mu_post - mu_pre) against MLFF forces, using cosine similarity and correlations."
        )
    )
    parser.add_argument("--run-dir", type=str, default=None, help="VERL run directory containing config.yaml + checkpoints.")
    parser.add_argument("--args-pickle", type=str, default=None, help="Path to pretrained EDM args.pickle.")
    parser.add_argument("--pretrained", type=str, default=None, help="Path to pretrained EDM checkpoint (flow state_dict).")
    parser.add_argument("--posttrained", type=str, default=None, help="Path to post-trained VERL checkpoint.")
    parser.add_argument("--num-molecules", type=int, default=256, help="How many molecules to sample for analysis.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--device", type=str, default=None, help="Torch device for diffusion model eval (e.g. cuda:0).")
    parser.add_argument("--mlff-device", type=str, default=None, help="Device for MLFF predictor/force computation.")
    parser.add_argument("--mlff-model", type=str, default=None, help="MLFF model name (e.g. uma-s-1p1).")
    parser.add_argument("--disable-mlff", action="store_true", help="Skip MLFF calls; only report drift delta stats.")
    parser.add_argument(
        "--stage",
        type=str,
        default="fine",
        choices=("fine", "coarse", "all"),
        help="Which scheduler stage(s) to include in the alignment stats.",
    )
    parser.add_argument("--min-force", type=float, default=None, help="Minimum force norm to include an atom in stats.")
    parser.add_argument("--min-delta", type=float, default=None, help="Minimum |mu_post - mu_pre| norm to include an atom.")
    parser.add_argument("--mlff-batch-size", type=int, default=None, help="Chunk size for MLFF inference.")

    parser.add_argument("--time-step", type=int, default=None, help="Diffusion steps (T). Defaults from run config / args.")
    parser.add_argument("--sample-group-size", type=int, default=None, help="Prompt group size for sampling.")
    parser.add_argument("--each-prompt-sample", type=int, default=None, help="Samples per prompt.")
    parser.add_argument("--share-initial-noise", type=int, default=None, help="1 to share initial noise (DanceGRPO), 0 to disable.")
    parser.add_argument("--skip-prefix", type=int, default=None, help="How many diffusion steps to share/skip.")
    parser.add_argument("--return-suffix-only", type=int, default=None, help="1 to return suffix-only rollouts, 0 for full.")

    parser.add_argument("--scheduler-mode", type=str, default=None, choices=("adaptive", "uniform"))
    parser.add_argument("--scheduler-skip-prefix", type=int, default=None)
    parser.add_argument("--scheduler-uniform-stride", type=int, default=None)
    parser.add_argument("--scheduler-include-terminal", type=int, default=None)
    parser.add_argument("--scheduler-coarse-stride", type=int, default=None)
    parser.add_argument("--scheduler-fine-stride", type=int, default=None)
    parser.add_argument("--scheduler-threshold-fraction", type=float, default=None)
    parser.add_argument("--out", type=str, default=None, help="Optional JSON output path.")
    parser.add_argument(
        "--save-npz",
        type=str,
        default=None,
        help="Optional path to save a downsampled set of per-atom/per-step pairs for plotting (npz).",
    )
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=200000,
        help="Max number of atom-step pairs to store when --save-npz is used.",
    )
    return parser.parse_args()


def _resolve_paths(args: argparse.Namespace, run_config: Dict[str, Any], run_dir: Optional[Path]) -> Tuple[Path, Path, Path]:
    if args.args_pickle:
        args_pickle = make_absolute(args.args_pickle, run_dir or Path.cwd())
        if args_pickle is None or not args_pickle.exists():
            raise FileNotFoundError(f"args.pickle not found at {args.args_pickle}")
    else:
        model_cfg = run_config.get("model", {}) if isinstance(run_config, dict) else {}
        cfg_value = model_cfg.get("config")
        if not cfg_value:
            raise FileNotFoundError("args.pickle not specified and not found in run config (model.config).")
        args_pickle = make_absolute(str(cfg_value), run_dir or Path.cwd())
        if args_pickle is None or not args_pickle.exists():
            raise FileNotFoundError(f"args.pickle not found at {cfg_value}")

    if args.pretrained:
        pretrained = make_absolute(args.pretrained, run_dir or Path.cwd())
        if pretrained is None or not pretrained.exists():
            raise FileNotFoundError(f"Pretrained checkpoint not found at {args.pretrained}")
    else:
        model_cfg = run_config.get("model", {}) if isinstance(run_config, dict) else {}
        cfg_value = model_cfg.get("model_path")
        if not cfg_value:
            raise FileNotFoundError("pretrained not specified and not found in run config (model.model_path).")
        pretrained = make_absolute(str(cfg_value), run_dir or Path.cwd())
        if pretrained is None or not pretrained.exists():
            raise FileNotFoundError(f"Pretrained checkpoint not found at {cfg_value}")

    if run_dir is not None:
        posttrained = infer_checkpoint_path(run_dir, args.posttrained)
    else:
        if not args.posttrained:
            raise FileNotFoundError("posttrained checkpoint is required when --run-dir is not provided.")
        posttrained = make_absolute(args.posttrained, Path.cwd())
        if posttrained is None or not posttrained.exists():
            raise FileNotFoundError(f"Posttrained checkpoint not found at {args.posttrained}")

    return args_pickle, pretrained, posttrained


def _build_scheduler(args: argparse.Namespace, run_config: Dict[str, Any]) -> RewardScheduler:
    shaping_cfg = {}
    if isinstance(run_config, dict):
        reward_cfg = run_config.get("reward", {}) or {}
        if isinstance(reward_cfg, dict):
            shaping_cfg = reward_cfg.get("shaping", {}) or {}
    scheduler_cfg = shaping_cfg.get("scheduler", {}) if isinstance(shaping_cfg, dict) else {}

    mode = args.scheduler_mode or (scheduler_cfg.get("mode", "adaptive") if isinstance(scheduler_cfg, dict) else "adaptive")
    skip_prefix = (
        args.scheduler_skip_prefix
        if args.scheduler_skip_prefix is not None
        else int(scheduler_cfg.get("skip_prefix", 0) or 0)
    )
    uniform_stride = (
        args.scheduler_uniform_stride
        if args.scheduler_uniform_stride is not None
        else int(scheduler_cfg.get("uniform_stride", 1) or 1)
    )
    include_terminal_cfg = scheduler_cfg.get("include_terminal", True) if isinstance(scheduler_cfg, dict) else True
    include_terminal = include_terminal_cfg
    if args.scheduler_include_terminal is not None:
        include_terminal = bool(int(args.scheduler_include_terminal))

    adaptive_cfg_in = scheduler_cfg.get("adaptive", {}) if isinstance(scheduler_cfg, dict) else {}
    adaptive_cfg = {
        "coarse_stride": args.scheduler_coarse_stride if args.scheduler_coarse_stride is not None else adaptive_cfg_in.get("coarse_stride", 10),
        "fine_stride": args.scheduler_fine_stride if args.scheduler_fine_stride is not None else adaptive_cfg_in.get("fine_stride", 2),
        "threshold_fraction": args.scheduler_threshold_fraction if args.scheduler_threshold_fraction is not None else adaptive_cfg_in.get("threshold_fraction", 0.25),
    }
    return RewardScheduler(
        mode=mode,
        skip_prefix=skip_prefix,
        uniform_stride=uniform_stride,
        adaptive_config=adaptive_cfg,
        include_terminal=include_terminal,
    )


def main() -> None:
    args = parse_args()
    _set_global_seed(args.seed)

    run_dir = None
    run_config: Dict[str, Any] = {}
    if args.run_dir:
        run_dir = make_absolute(args.run_dir, Path.cwd())
        if run_dir is None or not run_dir.exists():
            raise FileNotFoundError(f"Run directory not found: {args.run_dir}")
        run_config = load_run_config(run_dir)

    args_pickle_path, pretrained_path, posttrained_path = _resolve_paths(args, run_config, run_dir)
    edm_config = load_edm_config(args_pickle_path)

    # Match the training entrypoint (`run_verl_diffusion.py`):
    dataset_name = getattr(edm_config, "dataset", "")
    if isinstance(dataset_name, str) and "qm9" in dataset_name:
        edm_config.datadir = "qm9/temp"

    dataloader_cfg = (run_config.get("dataloader") if isinstance(run_config, dict) else {}) or {}
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
    if "qm9" in str(dataset_info.get("name", "")):
        retrieve_qm9_smiles(dataset_info)
    dataloaders, _ = retrieve_dataloaders(edm_config)

    flow_pre, nodes_dist, prop_dist = get_model(edm_config, device, dataset_info, dataloaders["train"])
    flow_post, _, _ = get_model(edm_config, device, dataset_info, dataloaders["train"])
    flow_pre.to(device)
    flow_post.to(device)

    _reweight_nodes_dist(nodes_dist, dataloader_cfg)
    _maybe_fix_nodes_dist(nodes_dist, dataloader_cfg)

    model_pre = EDMModel(flow_pre, edm_config).to(device)
    model_post = EDMModel(flow_post, edm_config).to(device)

    # Load pretrained weights into both models.
    model_pre.load(model_path=str(pretrained_path))
    model_post.load(model_path=str(pretrained_path))
    load_model_weights(model_post, posttrained_path, device)
    model_pre.eval()
    model_post.eval()

    # Sampling config
    model_cfg = run_config.get("model", {}) if isinstance(run_config, dict) else {}
    time_step = int(args.time_step if args.time_step is not None else model_cfg.get("time_step", getattr(edm_config, "diffusion_steps", 1000)))
    sample_group_size = int(args.sample_group_size if args.sample_group_size is not None else dataloader_cfg.get("sample_group_size", 1))
    each_prompt_sample = int(args.each_prompt_sample if args.each_prompt_sample is not None else dataloader_cfg.get("each_prompt_sample", 24))
    share_initial_noise = bool(int(args.share_initial_noise)) if args.share_initial_noise is not None else bool(model_cfg.get("share_initial_noise", False))
    skip_prefix = int(args.skip_prefix if args.skip_prefix is not None else model_cfg.get("skip_prefix", 0) or 0)
    return_suffix_only = bool(int(args.return_suffix_only)) if args.return_suffix_only is not None else bool(model_cfg.get("return_suffix_only", False))

    batch_size = sample_group_size * each_prompt_sample
    total_batches = math.ceil(int(args.num_molecules) / max(batch_size, 1))

    scheduler = _build_scheduler(args, run_config)

    train_cfg = run_config.get("train", {}) if isinstance(run_config, dict) else {}
    min_force = float(args.min_force) if args.min_force is not None else float(train_cfg.get("force_alignment_min_force", 1e-4) or 1e-4)
    min_delta = float(args.min_delta) if args.min_delta is not None else float(train_cfg.get("force_alignment_min_delta", 1e-4) or 1e-4)

    mlff_batch_size = args.mlff_batch_size
    if mlff_batch_size is None:
        reward_cfg = run_config.get("reward", {}) if isinstance(run_config, dict) else {}
        shaping_cfg = reward_cfg.get("shaping", {}) if isinstance(reward_cfg, dict) else {}
        mlff_batch_size = shaping_cfg.get("mlff_batch_size") if isinstance(shaping_cfg, dict) else None
    mlff_batch_size = int(mlff_batch_size) if mlff_batch_size is not None else 64

    # MLFF setup
    mlff_model = args.mlff_model
    if mlff_model is None and isinstance(run_config, dict):
        reward_cfg = run_config.get("reward", {}) or {}
        if isinstance(reward_cfg, dict):
            mlff_model = reward_cfg.get("mlff_model")
    mlff_device_str = _resolve_str_device(args.mlff_device, device)
    force_computer = None
    if not args.disable_mlff:
        predictor = get_mlff_predictor(mlff_model=mlff_model or "uma-s-1p1", device=mlff_device_str)
        if predictor is None:
            print("[WARN] MLFF predictor failed to load; running without MLFF forces.")
        else:
            position_scale = 1.0
            norm_values = dataset_info.get("normalize_factors")
            if isinstance(norm_values, (list, tuple)) and len(norm_values) > 0:
                position_scale = float(norm_values[0])
            force_computer = MLFFForceComputer(
                mlff_predictor=predictor,
                position_scale=position_scale,
                device=mlff_device_str,
                compute_energy=False,
            )

    dataloader_config = {
        "distributed": {"rank": 0, "world_size": 1, "local_rank": 0, "is_main_process": True},
        "model": {
            "time_step": time_step,
            "share_initial_noise": share_initial_noise,
            "skip_prefix": skip_prefix,
            "return_suffix_only": return_suffix_only,
        },
        "dataloader": {
            "sample_group_size": sample_group_size,
            "each_prompt_sample": each_prompt_sample,
            "micro_batch_size": dataloader_cfg.get("micro_batch_size", each_prompt_sample),
            "epoches": dataloader_cfg.get("epoches", total_batches),
            "nodes_dist_focus_min": dataloader_cfg.get("nodes_dist_focus_min"),
            "nodes_dist_focus_max": dataloader_cfg.get("nodes_dist_focus_max"),
            "nodes_dist_focus_multiplier": dataloader_cfg.get("nodes_dist_focus_multiplier", 1.0),
            "nodes_dist_fixed": dataloader_cfg.get("nodes_dist_fixed"),
        },
    }

    dataloader = EDMDataLoader(
        config=dataloader_config,
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

    cosine_stats = RunningMoments()
    step_cosine_pre_stats = RunningMoments()
    step_cosine_post_stats = RunningMoments()
    step_proj_pre_stats = RunningMoments()
    step_proj_post_stats = RunningMoments()
    delta_proj_stats = RunningMoments()
    delta_pos_norm_stats = RunningMoments()
    delta_type_norm_stats = RunningMoments()
    force_norm_stats = RunningMoments()
    corr_components = RunningCorrelation()
    corr_magnitudes = RunningCorrelation()
    per_timestep: Dict[int, RunningMoments] = {}

    eps = 1e-12
    processed = 0

    pair_sampler = None
    if args.save_npz:
        pair_sampler = ReservoirSampler(
            max_size=int(args.max_pairs),
            seed=int(args.seed),
            dtypes={
                "timestep": np.int32,
                "force_norm": np.float32,
                "delta_norm": np.float32,
                "cos_delta": np.float32,
                "proj_delta": np.float32,
                "cos_step_pre": np.float32,
                "cos_step_post": np.float32,
                "proj_step_pre": np.float32,
                "proj_step_post": np.float32,
            },
        )

    progress = tqdm(total=int(args.num_molecules), desc="Analyzing", unit="mol")
    for prompts in dataloader:
        if processed >= int(args.num_molecules):
            break

        nodesxsample = prompts.batch["nodesxsample"]
        group_index = prompts.batch.get("group_index")
        max_n_nodes = int(prompts.meta_info.get("max_n_nodes", dataset_info["max_n_nodes"]))
        batch_size_actual = int(nodesxsample.shape[0])

        model_ref = model_post.module if hasattr(model_post, "module") else model_post
        node_mask, edge_mask = model_ref.get_mask(nodesxsample, batch_size_actual, max_n_nodes)
        node_mask = node_mask.to(device)
        edge_mask = edge_mask.to(device)

        with torch.no_grad():
            _, _, latents_list, _, timesteps_list, mus_list, _, z0_preds_list = model_post.sample(
                n_samples=batch_size_actual,
                n_nodes=max_n_nodes,
                node_mask=node_mask,
                edge_mask=edge_mask,
                timestep=time_step,
                group_index=group_index,
                share_initial_noise=share_initial_noise,
                skip_prefix=skip_prefix,
                return_suffix_only=return_suffix_only,
            )

        latents = torch.stack(latents_list, dim=1)  # [B, S+1, N, D]
        mus = torch.stack(mus_list, dim=1)  # [B, S, N, D]
        z0_preds = torch.stack(z0_preds_list, dim=1)  # [B, S, N, D]
        timesteps = torch.tensor(timesteps_list, device=device)  # [S]

        diffusion_steps = int(timesteps.numel() - 1)  # exclude decode transition
        if diffusion_steps <= 0:
            continue

        timesteps_diff = timesteps[:diffusion_steps]
        mus_diff = mus[:, :diffusion_steps]
        z0_preds_diff = z0_preds[:, :diffusion_steps]
        latents_diff = latents[:, :diffusion_steps]

        schedule = scheduler.build(timesteps=timesteps_diff.detach().cpu(), end_idx=diffusion_steps - 1)
        schedule_indices = schedule.indices.to(device)
        fine_mask = schedule.fine_mask.to(device)

        selected_count = int(schedule_indices.numel())
        if selected_count <= 0:
            continue

        # Prepare per-step s/t arrays and masks for the expanded batch.
        s_vals = timesteps_diff.index_select(0, schedule_indices).to(dtype=torch.float32)
        s_array = (s_vals / float(time_step)).view(1, selected_count).expand(batch_size_actual, -1).reshape(-1, 1)
        t_array = ((s_vals + 1.0) / float(time_step)).view(1, selected_count).expand(batch_size_actual, -1).reshape(-1, 1)

        zt_selected = latents_diff.index_select(1, schedule_indices).contiguous()  # [B, K, N, D]
        mu_post_selected = mus_diff.index_select(1, schedule_indices).contiguous()
        z0_selected = z0_preds_diff.index_select(1, schedule_indices).contiguous()

        zt_flat = zt_selected.view(batch_size_actual * selected_count, max_n_nodes, -1)
        mu_post_flat = mu_post_selected.view(batch_size_actual * selected_count, max_n_nodes, -1)
        z0_flat = z0_selected.view(batch_size_actual * selected_count, max_n_nodes, -1)

        node_mask_flat = node_mask.repeat_interleave(selected_count, dim=0)
        edge_mask_view = edge_mask.view(batch_size_actual, max_n_nodes * max_n_nodes, -1)
        edge_mask_flat = edge_mask_view.repeat_interleave(selected_count, dim=0).reshape(-1, edge_mask_view.shape[-1])

        with torch.no_grad():
            _, _, mu_pre_flat, _, _ = model_pre.sample_p_zs_given_zt(
                s_array,
                t_array,
                zt_flat,
                node_mask_flat,
                edge_mask_flat,
                context=None,
                fix_noise=True,
            )

        delta = mu_post_flat - mu_pre_flat
        delta_pos = delta[:, :, :3]
        delta_type = delta[:, :, 3:]

        delta_pos_norm = torch.norm(delta_pos, dim=-1)
        delta_type_norm = torch.norm(delta_type, dim=-1) if delta_type.numel() > 0 else torch.zeros_like(delta_pos_norm)

        delta_pos_norm_stats.update(delta_pos_norm)
        delta_type_norm_stats.update(delta_type_norm)

        forces_flat = None
        if force_computer is not None:
            z0_forces = z0_flat
            node_forces = node_mask_flat
            if mlff_device_str != str(device):
                z0_forces = z0_forces.to(mlff_device_str)
                node_forces = node_forces.to(mlff_device_str)
            forces_flat = _compute_forces(force_computer, z0_forces, node_forces, dataset_info, mlff_batch_size)
            if forces_flat.device != device:
                forces_flat = forces_flat.to(device)

        if forces_flat is not None:
            force_norm = torch.norm(forces_flat, dim=-1)
            force_norm_stats.update(force_norm)

            # Stage gating (fine/coarse/all) per selected diffusion step.
            if args.stage == "all":
                stage_mask = torch.ones_like(fine_mask, dtype=torch.bool)
            elif args.stage == "fine":
                stage_mask = fine_mask.to(dtype=torch.bool)
            else:
                stage_mask = (~fine_mask).to(dtype=torch.bool)
            stage_mask_flat = stage_mask.view(1, selected_count).expand(batch_size_actual, -1).reshape(-1)

            node_valid = node_mask_flat[:, :, 0] > 0.5
            valid = (
                node_valid
                & (force_norm > min_force)
                & (delta_pos_norm > min_delta)
                & stage_mask_flat.unsqueeze(-1)
            )

            if valid.any():
                delta_vec = delta_pos
                force_vec = forces_flat
                dot = (delta_vec * force_vec).sum(dim=-1)
                cos = dot / (delta_pos_norm * force_norm + eps)
                cos = cos.clamp(-1.0, 1.0)

                cosine_stats.update(cos[valid])
                force_unit = force_vec / (force_norm.unsqueeze(-1) + eps)
                delta_proj_stats.update(dot[valid] / (force_norm[valid] + eps))
                corr_magnitudes.update(delta_pos_norm[valid], force_norm[valid])

                delta_comp = delta_vec[valid].reshape(-1)
                force_comp = force_vec[valid].reshape(-1)
                corr_components.update(delta_comp, force_comp)

                zt_pos = zt_flat[:, :, :3]
                mu_pre_pos = mu_pre_flat[:, :, :3]
                mu_post_pos = mu_post_flat[:, :, :3]
                step_pre = mu_pre_pos - zt_pos
                step_post = mu_post_pos - zt_pos
                step_pre_norm = torch.norm(step_pre, dim=-1)
                step_post_norm = torch.norm(step_post, dim=-1)

                valid_step = node_valid & (force_norm > min_force) & stage_mask_flat.unsqueeze(-1)
                valid_step = valid_step & (step_pre_norm > eps) & (step_post_norm > eps)
                if valid_step.any():
                    dot_pre = (step_pre * force_vec).sum(dim=-1)
                    dot_post = (step_post * force_vec).sum(dim=-1)
                    cos_pre = (dot_pre / (step_pre_norm * force_norm + eps)).clamp(-1.0, 1.0)
                    cos_post = (dot_post / (step_post_norm * force_norm + eps)).clamp(-1.0, 1.0)
                    step_cosine_pre_stats.update(cos_pre[valid_step])
                    step_cosine_post_stats.update(cos_post[valid_step])

                    proj_pre = (step_pre * force_unit).sum(dim=-1)
                    proj_post = (step_post * force_unit).sum(dim=-1)
                    step_proj_pre_stats.update(proj_pre[valid_step])
                    step_proj_post_stats.update(proj_post[valid_step])

                if pair_sampler is not None:
                    store_mask = valid & (step_pre_norm > eps) & (step_post_norm > eps)
                    if store_mask.any():
                        # Build timestep per (flat_sample, node) pair without materializing huge tensors.
                        timestep_rows = s_vals.to(dtype=torch.float32)
                        timestep_flat = (
                            timestep_rows.view(1, selected_count)
                            .expand(batch_size_actual, -1)
                            .reshape(-1)
                        )
                        timestep_pair = timestep_flat.view(-1, 1).expand(-1, max_n_nodes)
                        store = {
                            "timestep": timestep_pair[store_mask].detach().to("cpu").to(torch.int32).numpy(),
                            "force_norm": force_norm[store_mask].detach().to("cpu").to(torch.float32).numpy(),
                            "delta_norm": delta_pos_norm[store_mask].detach().to("cpu").to(torch.float32).numpy(),
                            "cos_delta": cos[store_mask].detach().to("cpu").to(torch.float32).numpy(),
                            "proj_delta": (dot / (force_norm + eps))[store_mask]
                            .detach()
                            .to("cpu")
                            .to(torch.float32)
                            .numpy(),
                            "cos_step_pre": cos_pre[store_mask].detach().to("cpu").to(torch.float32).numpy(),
                            "cos_step_post": cos_post[store_mask].detach().to("cpu").to(torch.float32).numpy(),
                            "proj_step_pre": proj_pre[store_mask].detach().to("cpu").to(torch.float32).numpy(),
                            "proj_step_post": proj_post[store_mask].detach().to("cpu").to(torch.float32).numpy(),
                        }
                        pair_sampler.update(store)

                cos_reshaped = cos.view(batch_size_actual, selected_count, max_n_nodes)
                valid_reshaped = valid.view(batch_size_actual, selected_count, max_n_nodes)
                s_vals_int = s_vals.detach().to("cpu").to(dtype=torch.long).tolist()
                for step_idx, timestep_val in enumerate(s_vals_int):
                    mask_step = valid_reshaped[:, step_idx]
                    if not mask_step.any():
                        continue
                    stats = per_timestep.get(int(timestep_val))
                    if stats is None:
                        stats = RunningMoments()
                        per_timestep[int(timestep_val)] = stats
                    stats.update(cos_reshaped[:, step_idx][mask_step])

        processed += batch_size_actual
        progress.update(min(batch_size_actual, int(args.num_molecules) - progress.n))

    progress.close()

    result = {
        "num_molecules": int(args.num_molecules),
        "processed_molecules": int(processed),
        "time_step": int(time_step),
        "skip_prefix": int(skip_prefix),
        "return_suffix_only": bool(return_suffix_only),
        "stage": args.stage,
        "min_force": float(min_force),
        "min_delta": float(min_delta),
        "paths": {
            "args_pickle": str(args_pickle_path),
            "pretrained": str(pretrained_path),
            "posttrained": str(posttrained_path),
        },
        "delta_pos_norm": {
            "mean": delta_pos_norm_stats.mean(),
            "std": delta_pos_norm_stats.std(),
            "min": None if delta_pos_norm_stats.count == 0 else delta_pos_norm_stats.min,
            "max": None if delta_pos_norm_stats.count == 0 else delta_pos_norm_stats.max,
            "count": int(delta_pos_norm_stats.count),
        },
        "delta_type_norm": {
            "mean": delta_type_norm_stats.mean(),
            "std": delta_type_norm_stats.std(),
            "min": None if delta_type_norm_stats.count == 0 else delta_type_norm_stats.min,
            "max": None if delta_type_norm_stats.count == 0 else delta_type_norm_stats.max,
            "count": int(delta_type_norm_stats.count),
        },
    }

    if force_computer is None:
        result["alignment"] = {"error": "MLFF disabled/unavailable; no force alignment stats computed."}
    else:
        result["alignment"] = {
            "cosine": {
                "mean": cosine_stats.mean(),
                "std": cosine_stats.std(),
                "min": None if cosine_stats.count == 0 else cosine_stats.min,
                "max": None if cosine_stats.count == 0 else cosine_stats.max,
                "count": int(cosine_stats.count),
            },
            "delta_projection_onto_force_unit": {
                "mean": delta_proj_stats.mean(),
                "std": delta_proj_stats.std(),
                "min": None if delta_proj_stats.count == 0 else delta_proj_stats.min,
                "max": None if delta_proj_stats.count == 0 else delta_proj_stats.max,
                "count": int(delta_proj_stats.count),
            },
            "step_cosine_pre": {
                "mean": step_cosine_pre_stats.mean(),
                "std": step_cosine_pre_stats.std(),
                "min": None if step_cosine_pre_stats.count == 0 else step_cosine_pre_stats.min,
                "max": None if step_cosine_pre_stats.count == 0 else step_cosine_pre_stats.max,
                "count": int(step_cosine_pre_stats.count),
            },
            "step_cosine_post": {
                "mean": step_cosine_post_stats.mean(),
                "std": step_cosine_post_stats.std(),
                "min": None if step_cosine_post_stats.count == 0 else step_cosine_post_stats.min,
                "max": None if step_cosine_post_stats.count == 0 else step_cosine_post_stats.max,
                "count": int(step_cosine_post_stats.count),
            },
            "step_projection_onto_force_unit_pre": {
                "mean": step_proj_pre_stats.mean(),
                "std": step_proj_pre_stats.std(),
                "min": None if step_proj_pre_stats.count == 0 else step_proj_pre_stats.min,
                "max": None if step_proj_pre_stats.count == 0 else step_proj_pre_stats.max,
                "count": int(step_proj_pre_stats.count),
            },
            "step_projection_onto_force_unit_post": {
                "mean": step_proj_post_stats.mean(),
                "std": step_proj_post_stats.std(),
                "min": None if step_proj_post_stats.count == 0 else step_proj_post_stats.min,
                "max": None if step_proj_post_stats.count == 0 else step_proj_post_stats.max,
                "count": int(step_proj_post_stats.count),
            },
            "force_norm": {
                "mean": force_norm_stats.mean(),
                "std": force_norm_stats.std(),
                "min": None if force_norm_stats.count == 0 else force_norm_stats.min,
                "max": None if force_norm_stats.count == 0 else force_norm_stats.max,
                "count": int(force_norm_stats.count),
            },
            "pearson_corr": {
                "components": corr_components.pearson(),
                "magnitudes": corr_magnitudes.pearson(),
            },
            "per_timestep": [
                {
                    "timestep": int(k),
                    "mean_cosine": v.mean(),
                    "std_cosine": v.std(),
                    "count": int(v.count),
                }
                for k, v in sorted(per_timestep.items(), key=lambda item: item[0], reverse=True)
            ],
        }

    print("=== Post-hoc force alignment (pretrained vs posttrained) ===")
    print(f"Processed molecules: {processed} (target={int(args.num_molecules)})")
    print(f"Delta mu_pos norm mean: {result['delta_pos_norm']['mean']}")
    if force_computer is not None:
        cosine_mean = result["alignment"]["cosine"]["mean"]
        corr_comp = result["alignment"]["pearson_corr"]["components"]
        corr_mag = result["alignment"]["pearson_corr"]["magnitudes"]
        step_cos_pre = result["alignment"]["step_cosine_pre"]["mean"]
        step_cos_post = result["alignment"]["step_cosine_post"]["mean"]
        print(f"Mean cosine(delta_mu_pos, force): {cosine_mean}")
        print(f"Mean cosine(step_pre=mu_pre-zt, force): {step_cos_pre}")
        print(f"Mean cosine(step_post=mu_post-zt, force): {step_cos_post}")
        print(f"Pearson corr (components): {corr_comp}")
        print(f"Pearson corr (magnitudes): {corr_mag}")

    if args.out:
        out_path = make_absolute(args.out, run_dir or Path.cwd())
        if out_path is None:
            raise ValueError(f"Could not resolve output path: {args.out}")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        import json

        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Saved: {out_path}")

    if args.save_npz and pair_sampler is not None:
        npz_path = make_absolute(args.save_npz, run_dir or Path.cwd())
        if npz_path is None:
            raise ValueError(f"Could not resolve npz path: {args.save_npz}")
        npz_path.parent.mkdir(parents=True, exist_ok=True)
        payload = pair_sampler.to_dict()
        # Attach lightweight metadata for convenience.
        payload["_meta_num_molecules"] = np.array([int(args.num_molecules)], dtype=np.int32)
        payload["_meta_stage"] = np.array([args.stage], dtype=object)
        np.savez_compressed(npz_path, **payload)
        print(f"Saved pairs (npz): {npz_path} (stored={len(payload['timestep'])}, seen={pair_sampler.seen})")


if __name__ == "__main__":
    main()
