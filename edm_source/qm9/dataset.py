from torch.utils.data import DataLoader
import os

from .data.args import init_argparse
from .data.collate import PreprocessQM9
from .data.utils import initialize_datasets


def _resolve_geom_data_file(cfg) -> str:
    explicit = getattr(cfg, "geom_data_file", None) or getattr(cfg, "geom_data_path", None)
    if explicit:
        return str(explicit)

    datadir = getattr(cfg, "datadir", None)
    if isinstance(datadir, str) and datadir:
        if datadir.endswith(".npy") and os.path.exists(datadir):
            return datadir
        candidate = os.path.join(datadir, "geom_drugs_30.npy")
        if os.path.exists(candidate):
            return candidate

    cwd_candidate = os.path.join(os.getcwd(), "data", "geom", "geom_drugs_30.npy")
    if os.path.exists(cwd_candidate):
        return cwd_candidate

    cwd_datasets_candidate = os.path.join(os.getcwd(), "datasets", "geom", "geom_drugs_30.npy")
    if os.path.exists(cwd_datasets_candidate):
        return cwd_datasets_candidate

    repo_candidate = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data",
        "geom",
        "geom_drugs_30.npy",
    )
    if os.path.exists(repo_candidate):
        return repo_candidate

    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    repo_datasets_candidate = os.path.join(repo_root, "datasets", "geom", "geom_drugs_30.npy")
    if os.path.exists(repo_datasets_candidate):
        return repo_datasets_candidate

    raise FileNotFoundError(
        "GEOM conformation file not found. Provide `geom_data_file` (or `datadir`) "
        "pointing to `geom_drugs_30.npy`, or place it at `datasets/geom/geom_drugs_30.npy` "
        "or `data/geom/geom_drugs_30.npy`."
    )


def retrieve_dataloaders(cfg):
    if 'qm9' in cfg.dataset:
        batch_size = cfg.batch_size
        num_workers = cfg.num_workers
        filter_n_atoms = cfg.filter_n_atoms
        # Initialize dataloader
        args = init_argparse('qm9')
        # data_dir = cfg.data_root_dir
        args, datasets, num_species, charge_scale = initialize_datasets(args, cfg.datadir, cfg.dataset,
                                                                        subtract_thermo=args.subtract_thermo,
                                                                        force_download=args.force_download,
                                                                        remove_h=cfg.remove_h)
        qm9_to_eV = {'U0': 27.2114, 'U': 27.2114, 'G': 27.2114, 'H': 27.2114, 'zpve': 27211.4, 'gap': 27.2114, 'homo': 27.2114,
                     'lumo': 27.2114}

        for dataset in datasets.values():
            dataset.convert_units(qm9_to_eV)

        if filter_n_atoms is not None:
            print("Retrieving molecules with only %d atoms" % filter_n_atoms)
            datasets = filter_atoms(datasets, filter_n_atoms)

        # Construct PyTorch dataloaders from datasets
        preprocess = PreprocessQM9(load_charges=cfg.include_charges)
        dataloaders = {split: DataLoader(dataset,
                                         batch_size=batch_size,
                                         shuffle=args.shuffle if (split == 'train') else False,
                                         num_workers=num_workers,
                                         collate_fn=preprocess.collate_fn)
                             for split, dataset in datasets.items()}
    elif 'geom' in cfg.dataset:
        try:
            import build_geom_dataset  # type: ignore[import-not-found]
        except ModuleNotFoundError:
            from edm_source import build_geom_dataset  # type: ignore[import-not-found]

        try:
            from configs.datasets_config import get_dataset_info  # type: ignore[import-not-found]
        except ModuleNotFoundError:
            from edm_source.configs.datasets_config import get_dataset_info  # type: ignore[import-not-found]

        data_file = _resolve_geom_data_file(cfg)
        dataset_info = get_dataset_info(cfg.dataset, cfg.remove_h)

        # Retrieve QM9 dataloaders
        split_data = build_geom_dataset.load_split_data(
            data_file,
            val_proportion=0.1,
            test_proportion=0.1,
            filter_size=cfg.filter_molecule_size,
        )
        transform = build_geom_dataset.GeomDrugsTransform(dataset_info,
                                                          cfg.include_charges,
                                                          cfg.device,
                                                          cfg.sequential)
        dataloaders = {}
        for key, indices in zip(['train', 'val', 'test'], split_data["splits"]):
            dataset = build_geom_dataset.GeomDrugsDataset(
                conformation_file=split_data["conformation_file"],
                indices=indices,
                starts=split_data["starts"],
                lengths=split_data["lengths"],
                transform=transform,
                sequential=cfg.sequential,
            )
            shuffle = (key == 'train') and not cfg.sequential

            # Sequential dataloading disabled for now.
            dataloaders[key] = build_geom_dataset.GeomDrugsDataLoader(
                sequential=cfg.sequential, dataset=dataset,
                batch_size=cfg.batch_size,
                shuffle=shuffle)
        charge_scale = None
    else:
        raise ValueError(f'Unknown dataset {cfg.dataset}')

    return dataloaders, charge_scale


def filter_atoms(datasets, n_nodes):
    for key in datasets:
        dataset = datasets[key]
        idxs = dataset.data['num_atoms'] == n_nodes
        for key2 in dataset.data:
            dataset.data[key2] = dataset.data[key2][idxs]

        datasets[key].num_pts = dataset.data['one_hot'].size(0)
        datasets[key].perm = None
    return datasets
