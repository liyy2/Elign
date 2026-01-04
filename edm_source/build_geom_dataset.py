import msgpack
import os
import numpy as np
import torch
from torch.utils.data import BatchSampler, DataLoader, Dataset, SequentialSampler
import argparse
from typing import Optional

try:
    from qm9.data import collate as qm9_collate
except ModuleNotFoundError:
    from edm_source.qm9.data import collate as qm9_collate


def _select_lowest_energy_conformers(conformers, num_conformations: int):
    """Return indices of the lowest-energy conformers (up to ``num_conformations``)."""
    if not conformers:
        return []
    energies_list = []
    for conf in conformers:
        if not isinstance(conf, dict):
            energies_list.append(np.inf)
            continue
        value = conf.get("totalenergy", np.inf)
        try:
            energy = float(value) if value is not None else np.inf
        except (TypeError, ValueError):
            energy = np.inf
        energies_list.append(energy)

    energies = np.asarray(energies_list, dtype=np.float64)
    if energies.size == 0:
        return []
    k = int(num_conformations)
    if k <= 0:
        return []
    if energies.size <= k:
        return list(np.argsort(energies))
    # Partial selection avoids sorting the full list for large conformer pools.
    idx = np.argpartition(energies, k - 1)[:k]
    idx = idx[np.argsort(energies[idx])]
    return idx.tolist()


def _normalize_conformers(conformers) -> list:
    """Normalize GEOM conformer containers into a list of conformer dicts."""
    if conformers is None:
        return []
    if isinstance(conformers, dict):
        values = conformers.values()
    elif isinstance(conformers, (list, tuple)):
        values = conformers
    else:
        return []

    return [conf for conf in values if isinstance(conf, dict)]


def extract_conformers(args):
    drugs_file = os.path.join(args.data_dir, args.data_file)
    save_file = f"geom_drugs_{'no_h_' if args.remove_h else ''}{args.conformations}"
    smiles_list_file = 'geom_drugs_smiles.txt'
    number_atoms_file = f"geom_drugs_n_{'no_h_' if args.remove_h else ''}{args.conformations}"

    if getattr(args, "output_dtype", None):
        output_dtype = str(args.output_dtype).lower()
    else:
        output_dtype = "float32"
    dtype = np.float32 if output_dtype in {"float32", "f4"} else np.float64

    output_base = os.path.join(args.data_dir, save_file)
    output_path = output_base if output_base.endswith(".npy") else f"{output_base}.npy"

    smiles_path = os.path.join(args.data_dir, smiles_list_file)
    n_atoms_base = os.path.join(args.data_dir, number_atoms_file)
    n_atoms_path = n_atoms_base if n_atoms_base.endswith(".npy") else f"{n_atoms_base}.npy"

    use_streaming = bool(getattr(args, "streaming", True))
    if not use_streaming:
        unpacker = msgpack.Unpacker(open(drugs_file, "rb"), raw=False, strict_map_key=False)

        all_smiles = []
        all_number_atoms = []
        dataset_conformers = []
        mol_id = 0
        for i, drugs_1k in enumerate(unpacker):
            if not isinstance(drugs_1k, dict):
                continue
            print(f"Unpacking file {i}...")
            for smiles, all_info in drugs_1k.items():
                if not isinstance(smiles, str) or not isinstance(all_info, dict):
                    continue
                all_smiles.append(smiles)
                conformers = _normalize_conformers(all_info.get('conformers', []))
                lowest_energies = _select_lowest_energy_conformers(conformers, args.conformations)
                for conf_idx in lowest_energies:
                    conformer = conformers[conf_idx]
                    if "xyz" not in conformer:
                        continue
                    try:
                        coords = np.asarray(conformer["xyz"], dtype=dtype)
                    except Exception:
                        continue
                    if coords.ndim != 2 or coords.shape[1] < 4:
                        continue
                    coords = coords[:, :4]  # [atomic_number, x, y, z]
                    if args.remove_h:
                        coords = coords[coords[:, 0] != 1.0]
                    n = coords.shape[0]
                    if n == 0:
                        continue
                    all_number_atoms.append(n)
                    mol_id_arr = float(mol_id) * np.ones((n, 1), dtype=dtype)
                    id_coords = np.hstack((mol_id_arr, coords))

                    dataset_conformers.append(id_coords)
                    mol_id += 1

        print("Total number of conformers saved", mol_id)
        all_number_atoms = np.asarray(all_number_atoms, dtype=np.int32)
        dataset = np.vstack(dataset_conformers)

        print("Total number of atoms in the dataset", dataset.shape[0])
        print("Average number of atoms per molecule", dataset.shape[0] / mol_id)

        np.save(output_base, dataset)
        with open(smiles_path, 'w', encoding="utf-8") as f:
            for s in all_smiles:
                f.write(s)
                f.write('\n')

        np.save(n_atoms_base, all_number_atoms)
        print("Dataset processed.")
        return

    # Streaming build (two-pass): suitable for the full GEOM-Drugs dataset without large RAM.
    total_atoms = 0
    total_conformers = 0
    with open(drugs_file, "rb") as handle:
        unpacker = msgpack.Unpacker(handle, raw=False, strict_map_key=False)
        for i, drugs_1k in enumerate(unpacker):
            if not isinstance(drugs_1k, dict):
                continue
            if i % 10 == 0:
                print(f"[pass1] Unpacking chunk {i}...")
            for smiles, all_info in drugs_1k.items():
                if not isinstance(smiles, str) or not isinstance(all_info, dict):
                    continue
                conformers = _normalize_conformers(all_info.get("conformers", []))
                lowest = _select_lowest_energy_conformers(conformers, args.conformations)
                for conf_idx in lowest:
                    conformer = conformers[conf_idx]
                    if "xyz" not in conformer:
                        continue
                    try:
                        coords = np.asarray(conformer["xyz"], dtype=dtype)
                    except Exception:
                        continue
                    if coords.ndim != 2 or coords.shape[1] < 4:
                        continue
                    coords = coords[:, :4]
                    if args.remove_h:
                        coords = coords[coords[:, 0] != 1.0]
                    n = int(coords.shape[0])
                    if n == 0:
                        continue
                    total_atoms += n
                    total_conformers += 1

    print("Total number of conformers saved", total_conformers)
    print("Total number of atoms in the dataset", total_atoms)
    if total_conformers > 0:
        print("Average number of atoms per molecule", total_atoms / float(total_conformers))

    os.makedirs(args.data_dir, exist_ok=True)
    dataset_memmap = np.lib.format.open_memmap(
        output_path, mode="w+", dtype=dtype, shape=(total_atoms, 5)
    )
    n_atoms_per_conf = np.empty((total_conformers,), dtype=np.int32)

    cursor = 0
    mol_id = 0
    with open(drugs_file, "rb") as handle, open(smiles_path, "w", encoding="utf-8") as smiles_handle:
        unpacker = msgpack.Unpacker(handle, raw=False, strict_map_key=False)
        for i, drugs_1k in enumerate(unpacker):
            if not isinstance(drugs_1k, dict):
                continue
            if i % 10 == 0:
                print(f"[pass2] Unpacking chunk {i} (written atoms={cursor}, conformers={mol_id})...")
            for smiles, all_info in drugs_1k.items():
                if not isinstance(smiles, str) or not isinstance(all_info, dict):
                    continue
                smiles_handle.write(smiles)
                smiles_handle.write("\n")
                conformers = _normalize_conformers(all_info.get("conformers", []))
                lowest = _select_lowest_energy_conformers(conformers, args.conformations)
                for conf_idx in lowest:
                    conformer = conformers[conf_idx]
                    if "xyz" not in conformer:
                        continue
                    try:
                        coords = np.asarray(conformer["xyz"], dtype=dtype)
                    except Exception:
                        continue
                    if coords.ndim != 2 or coords.shape[1] < 4:
                        continue
                    coords = coords[:, :4]
                    if args.remove_h:
                        coords = coords[coords[:, 0] != 1.0]
                    n = int(coords.shape[0])
                    if n == 0:
                        continue
                    end = cursor + n
                    dataset_memmap[cursor:end, 0] = float(mol_id)
                    dataset_memmap[cursor:end, 1:] = coords
                    n_atoms_per_conf[mol_id] = n
                    cursor = end
                    mol_id += 1

    if cursor != total_atoms or mol_id != total_conformers:
        raise RuntimeError(
            f"Streaming build mismatch: wrote atoms={cursor} (expected {total_atoms}), "
            f"conformers={mol_id} (expected {total_conformers})."
        )

    dataset_memmap.flush()
    np.save(n_atoms_base, n_atoms_per_conf)
    print(f"Saved conformations to {output_path}")
    print(f"Saved smiles list to {smiles_path}")
    print(f"Saved atoms-per-conformer to {n_atoms_path}")
    print("Dataset processed.")


def load_split_data(conformation_file, val_proportion=0.1, test_proportion=0.1,
                    filter_size=None):
    from pathlib import Path
    path = Path(conformation_file)
    base_path = path.parent.absolute()

    all_data = np.load(conformation_file, mmap_mode="r")
    if all_data.ndim != 2 or all_data.shape[1] != 5:
        raise ValueError(
            f"Expected GEOM conformations array with shape [num_atoms, 5], got {all_data.shape}"
        )

    n_atoms_file = None
    stem = path.stem  # e.g., geom_drugs_30 / geom_drugs_no_h_30
    if stem.startswith("geom_drugs_"):
        suffix = stem[len("geom_drugs_") :]
        candidate = base_path / f"geom_drugs_n_{suffix}.npy"
        if candidate.exists():
            n_atoms_file = candidate

    if n_atoms_file is not None:
        lengths = np.load(n_atoms_file).astype(np.int32, copy=False)
        if lengths.ndim != 1:
            raise ValueError(f"Expected atoms-per-conformer vector at {n_atoms_file}, got {lengths.shape}")
        starts = np.empty((lengths.shape[0],), dtype=np.int64)
        starts[0] = 0
        if lengths.shape[0] > 1:
            starts[1:] = np.cumsum(lengths[:-1], dtype=np.int64)
    else:
        # Fallback: derive starts/lengths by scanning the mol_id column. This is
        # memory-safe but slower than using the `geom_drugs_n_*.npy` companion.
        mol_id = all_data[:, 0]
        total_atoms = int(mol_id.shape[0])
        if total_atoms == 0:
            raise ValueError("Empty GEOM dataset.")

        chunk_size = 5_000_000
        boundary_chunks = []
        prev_last = None
        for offset in range(0, total_atoms, chunk_size):
            end = min(offset + chunk_size, total_atoms)
            chunk = mol_id[offset:end]
            if chunk.size == 0:
                continue
            boundaries = np.nonzero(chunk[1:] != chunk[:-1])[0] + offset + 1
            if prev_last is not None and chunk[0] != prev_last:
                boundaries = np.concatenate(([offset], boundaries))
            boundary_chunks.append(boundaries.astype(np.int64, copy=False))
            prev_last = chunk[-1]

        split_indices = (
            np.concatenate(boundary_chunks) if boundary_chunks else np.empty((0,), dtype=np.int64)
        )
        starts = np.concatenate(([0], split_indices)).astype(np.int64, copy=False)
        ends = np.concatenate((split_indices, [total_atoms])).astype(np.int64, copy=False)
        lengths = (ends - starts).astype(np.int32, copy=False)

    total_atoms_expected = int(starts[-1] + lengths[-1]) if lengths.size else 0
    if total_atoms_expected != int(all_data.shape[0]):
        raise RuntimeError(
            f"GEOM index mismatch: starts/lengths imply {total_atoms_expected} atoms "
            f"but file has {all_data.shape[0]} rows. Regenerate the dataset or its `geom_drugs_n_*.npy`."
        )

    num_conformers = int(lengths.shape[0])
    if filter_size is not None:
        keep = np.nonzero(lengths <= int(filter_size))[0].astype(np.int32, copy=False)
        if keep.size == 0:
            raise ValueError("No molecules left after filter.")
        valid_indices = keep
    else:
        valid_indices = np.arange(num_conformers, dtype=np.int32)

    perm_path = os.path.join(base_path, 'geom_permutation.npy')
    use_saved_perm = filter_size is None and os.path.exists(perm_path)

    perm = None
    if use_saved_perm:
        try:
            perm = np.load(perm_path)
        except Exception:
            perm = None

    if perm is None or len(perm) != len(valid_indices):
        rng = np.random.RandomState(0)
        perm = rng.permutation(len(valid_indices)).astype('int32')
        if filter_size is None:
            try:
                np.save(perm_path, perm)
            except Exception:
                pass

    ordered_indices = valid_indices[perm]

    num_mol = len(ordered_indices)
    val_index = int(num_mol * val_proportion)
    test_index = val_index + int(num_mol * test_proportion)
    val_idx, test_idx, train_idx = np.split(ordered_indices, [val_index, test_index])

    return {
        "conformation_file": str(conformation_file),
        "starts": starts,
        "lengths": lengths,
        "splits": (train_idx, val_idx, test_idx),
    }


class GeomDrugsDataset(Dataset):
    def __init__(
        self,
        data_list=None,
        transform=None,
        *,
        conformation_file: Optional[str] = None,
        indices: Optional[np.ndarray] = None,
        starts: Optional[np.ndarray] = None,
        lengths: Optional[np.ndarray] = None,
        sequential: bool = False,
    ):
        """
        Args:
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.transform = transform

        self._file_backed = conformation_file is not None
        self.sequential = bool(sequential)
        if self._file_backed:
            if indices is None or starts is None or lengths is None:
                raise ValueError("File-backed GEOM dataset requires indices/starts/lengths.")
            self._all_data = np.load(conformation_file, mmap_mode="r")
            self._starts = np.asarray(starts, dtype=np.int64)
            self._lengths = np.asarray(lengths, dtype=np.int32)
            self.indices = np.asarray(indices, dtype=np.int64)

            if self.sequential:
                subset_lengths = self._lengths[self.indices]
                order = np.argsort(subset_lengths)
                self.indices = self.indices[order]
                subset_lengths = subset_lengths[order]
                self.split_indices = np.unique(subset_lengths, return_index=True)[1][1:]
            else:
                self.split_indices = np.asarray([], dtype=np.int64)
        else:
            if data_list is None:
                raise ValueError("Provide either data_list or conformation_file.")
            # Sort the data list by size
            list_lengths = [s.shape[0] for s in data_list]
            argsort = np.argsort(list_lengths)               # Sort by decreasing size
            self.data_list = [data_list[i] for i in argsort]
            # Store indices where the size changes
            self.split_indices = np.unique(np.sort(list_lengths), return_index=True)[1][1:]

    def __len__(self):
        if self._file_backed:
            return int(self.indices.shape[0])
        return len(self.data_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self._file_backed:
            conf_idx = int(self.indices[idx])
            start = int(self._starts[conf_idx])
            n = int(self._lengths[conf_idx])
            sample = np.array(self._all_data[start : start + n, 1:], copy=True)
        else:
            sample = self.data_list[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample


class CustomBatchSampler(BatchSampler):
    """ Creates batches where all sets have the same size. """
    def __init__(self, sampler, batch_size, drop_last, split_indices):
        super().__init__(sampler, batch_size, drop_last)
        self.split_indices = split_indices

    def __iter__(self):
        batch = []
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size or idx + 1 in self.split_indices:
                yield batch
                batch = []
        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self):
        count = 0
        batch = 0
        for idx in self.sampler:
            batch += 1
            if batch == self.batch_size or idx + 1 in self.split_indices:
                count += 1
                batch = 0
        if batch > 0 and not self.drop_last:
            count += 1
        return count


def collate_fn(batch):
    batch = {prop: qm9_collate.batch_stack([mol[prop] for mol in batch])
             for prop in batch[0].keys()}

    atom_mask = batch['atom_mask']

    # Obtain edges
    batch_size, n_nodes = atom_mask.size()
    edge_mask = atom_mask.unsqueeze(1) * atom_mask.unsqueeze(2)

    # mask diagonal
    diag_mask = ~torch.eye(edge_mask.size(1), dtype=torch.bool,
                           device=edge_mask.device).unsqueeze(0)
    edge_mask *= diag_mask

    # edge_mask = atom_mask.unsqueeze(1) * atom_mask.unsqueeze(2)
    batch['edge_mask'] = edge_mask.view(batch_size * n_nodes * n_nodes, 1)

    return batch


class GeomDrugsDataLoader(DataLoader):
    def __init__(self, sequential, dataset, batch_size, shuffle, drop_last=False):

        if sequential:
            # This goes over the data sequentially, advantage is that it takes
            # less memory for smaller molecules, but disadvantage is that the
            # model sees very specific orders of data.
            assert not shuffle
            sampler = SequentialSampler(dataset)
            batch_sampler = CustomBatchSampler(sampler, batch_size, drop_last,
                                               dataset.split_indices)
            super().__init__(dataset, batch_sampler=batch_sampler)

        else:
            # Dataloader goes through data randomly and pads the molecules to
            # the largest molecule size.
            super().__init__(dataset, batch_size, shuffle=shuffle,
                             collate_fn=collate_fn, drop_last=drop_last)


class GeomDrugsTransform(object):
    def __init__(self, dataset_info, include_charges, device, sequential):
        self.atomic_number_list = torch.Tensor(dataset_info['atomic_nb'])[None, :]
        self.device = device
        self.include_charges = include_charges
        self.sequential = sequential

    def __call__(self, data):
        n = data.shape[0]
        new_data = {}
        new_data['positions'] = torch.from_numpy(data[:, -3:])
        atom_types = torch.from_numpy(data[:, 0].astype(int)[:, None])
        one_hot = atom_types == self.atomic_number_list
        new_data['one_hot'] = one_hot
        if self.include_charges:
            new_data['charges'] = torch.zeros(n, 1, device=self.device)
        else:
            new_data['charges'] = torch.zeros(0, device=self.device)
        new_data['atom_mask'] = torch.ones(n, device=self.device)

        if self.sequential:
            edge_mask = torch.ones((n, n), device=self.device)
            edge_mask[~torch.eye(edge_mask.shape[0], dtype=torch.bool)] = 0
            new_data['edge_mask'] = edge_mask.flatten()
        return new_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--conformations", type=int, default=30,
                        help="Max number of conformations kept for each molecule.")
    parser.add_argument("--remove_h", action='store_true', help="Remove hydrogens from the dataset.")
    parser.add_argument("--data_dir", type=str, default='~/diffusion/data/geom/')
    parser.add_argument("--data_file", type=str, default="drugs_crude.msgpack")
    parser.add_argument(
        "--output_dtype",
        type=str,
        default="float32",
        choices=["float32", "float64"],
        help="Numeric dtype for the saved .npy conformations array.",
    )
    parser.add_argument(
        "--streaming",
        dest="streaming",
        action="store_true",
        help="Build the output array via a two-pass streaming pipeline (low RAM, slower).",
    )
    parser.add_argument(
        "--no-streaming",
        dest="streaming",
        action="store_false",
        help="Build the output array in-memory (fast, requires large RAM).",
    )
    parser.set_defaults(streaming=True)
    args = parser.parse_args()
    extract_conformers(args)
