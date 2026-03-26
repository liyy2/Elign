from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


_ELEMENT_TO_Z = {
    "H": 1,
    "C": 6,
    "N": 7,
    "O": 8,
    "F": 9,
    "P": 15,
    "S": 16,
    "CL": 17,
}


@dataclass(frozen=True)
class AtomRecord:
    index: int
    serial: int
    name: str
    residue_name: str
    residue_id: int
    chain_id: str
    element: str
    atomic_number: int


@dataclass(frozen=True)
class TopologyInfo:
    atoms: List[AtomRecord]
    atom_names: List[str]
    residue_names: List[str]
    residue_ids: List[int]
    chain_ids: List[str]
    elements: List[str]
    atomic_numbers: List[int]
    phi_indices: Tuple[int, int, int, int]
    psi_indices: Tuple[int, int, int, int]

    def to_dict(self) -> Dict[str, object]:
        payload = asdict(self)
        payload["atoms"] = [asdict(atom) for atom in self.atoms]
        return payload

    @classmethod
    def from_dict(cls, payload: Dict[str, object]) -> "TopologyInfo":
        atoms = [AtomRecord(**atom_payload) for atom_payload in payload["atoms"]]  # type: ignore[arg-type]
        return cls(
            atoms=atoms,
            atom_names=list(payload["atom_names"]),  # type: ignore[arg-type]
            residue_names=list(payload["residue_names"]),  # type: ignore[arg-type]
            residue_ids=[int(v) for v in payload["residue_ids"]],  # type: ignore[arg-type]
            chain_ids=list(payload["chain_ids"]),  # type: ignore[arg-type]
            elements=list(payload["elements"]),  # type: ignore[arg-type]
            atomic_numbers=[int(v) for v in payload["atomic_numbers"]],  # type: ignore[arg-type]
            phi_indices=tuple(int(v) for v in payload["phi_indices"]),  # type: ignore[arg-type]
            psi_indices=tuple(int(v) for v in payload["psi_indices"]),  # type: ignore[arg-type]
        )


def infer_element(atom_name: str, element_field: str) -> str:
    """Infer an element symbol from a PDB atom name/field."""

    element = str(element_field or "").strip().upper()
    if element:
        if len(element) >= 2 and element[:2] in _ELEMENT_TO_Z:
            return element[:2].title()
        return element[0].upper()

    stripped = "".join(ch for ch in atom_name.strip() if ch.isalpha()).upper()
    if len(stripped) >= 2 and stripped[:2] in _ELEMENT_TO_Z:
        return stripped[:2].title()
    if stripped:
        return stripped[0].upper()
    raise ValueError(f"Unable to infer element for atom name '{atom_name}'.")


def atomic_number_from_element(element: str) -> int:
    """Map a chemical element symbol to an atomic number."""

    key = str(element).strip().upper()
    if key not in _ELEMENT_TO_Z:
        raise KeyError(f"Unsupported element '{element}'.")
    return int(_ELEMENT_TO_Z[key])


def _parse_residue_id(res_seq: str, insertion_code: str) -> int:
    residue_id = int(res_seq.strip())
    insertion_code = str(insertion_code or "").strip()
    if not insertion_code:
        return residue_id
    return residue_id * 100 + ord(insertion_code[0])


def parse_pdb(pdb_path: str | Path) -> Tuple[TopologyInfo, np.ndarray]:
    """Parse atom metadata and coordinates from a PDB file."""

    atoms: List[AtomRecord] = []
    coordinates: List[List[float]] = []

    path = Path(pdb_path)
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            record = line[:6].strip().upper()
            if record not in {"ATOM", "HETATM"}:
                continue
            atom_name = line[12:16].strip()
            residue_name = line[17:20].strip()
            chain_id = line[21].strip() or "_"
            residue_id = _parse_residue_id(line[22:26], line[26:27])
            element = infer_element(atom_name, line[76:78])
            atomic_number = atomic_number_from_element(element)
            coord = [
                float(line[30:38]),
                float(line[38:46]),
                float(line[46:54]),
            ]
            atom = AtomRecord(
                index=len(atoms),
                serial=int(line[6:11]),
                name=atom_name,
                residue_name=residue_name,
                residue_id=residue_id,
                chain_id=chain_id,
                element=element,
                atomic_number=atomic_number,
            )
            atoms.append(atom)
            coordinates.append(coord)

    if not atoms:
        raise ValueError(f"No ATOM/HETATM records found in '{path}'.")

    phi_indices, psi_indices = infer_phi_psi_indices(atoms)
    topology = TopologyInfo(
        atoms=atoms,
        atom_names=[atom.name for atom in atoms],
        residue_names=[atom.residue_name for atom in atoms],
        residue_ids=[atom.residue_id for atom in atoms],
        chain_ids=[atom.chain_id for atom in atoms],
        elements=[atom.element for atom in atoms],
        atomic_numbers=[atom.atomic_number for atom in atoms],
        phi_indices=phi_indices,
        psi_indices=psi_indices,
    )
    return topology, np.asarray(coordinates, dtype=np.float32)


def _group_residues(atoms: Sequence[AtomRecord]) -> List[Tuple[Tuple[str, int, str], Dict[str, int]]]:
    groups: List[Tuple[Tuple[str, int, str], Dict[str, int]]] = []
    by_key: Dict[Tuple[str, int, str], Dict[str, int]] = {}
    ordered_keys: List[Tuple[str, int, str]] = []
    for atom in atoms:
        key = (atom.chain_id, atom.residue_id, atom.residue_name)
        if key not in by_key:
            by_key[key] = {}
            ordered_keys.append(key)
        by_key[key][atom.name.upper()] = atom.index
    for key in ordered_keys:
        groups.append((key, by_key[key]))
    return groups


def infer_phi_psi_indices(atoms: Sequence[AtomRecord]) -> Tuple[Tuple[int, int, int, int], Tuple[int, int, int, int]]:
    """Infer central-residue phi/psi torsion atom indices from fixed Ala2 topology."""

    residues = _group_residues(atoms)
    if len(residues) < 3:
        raise ValueError("Expected at least three residues to infer central Ala2 phi/psi torsions.")

    for i in range(1, len(residues) - 1):
        _prev_key, prev_atoms = residues[i - 1]
        _curr_key, curr_atoms = residues[i]
        _next_key, next_atoms = residues[i + 1]

        if {"N", "CA", "C"}.issubset(curr_atoms) and "C" in prev_atoms and "N" in next_atoms:
            phi = (prev_atoms["C"], curr_atoms["N"], curr_atoms["CA"], curr_atoms["C"])
            psi = (curr_atoms["N"], curr_atoms["CA"], curr_atoms["C"], next_atoms["N"])
            return phi, psi

    raise ValueError("Could not infer central phi/psi torsions from PDB topology.")


def topology_to_json(topology: TopologyInfo, path: str | Path) -> None:
    """Write topology metadata to a JSON file."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(topology.to_dict(), handle, indent=2, sort_keys=True)


def topology_from_json(path: str | Path) -> TopologyInfo:
    """Read topology metadata from a JSON file."""

    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return TopologyInfo.from_dict(payload)


def build_one_hot_features(atomic_numbers: Sequence[int]) -> Tuple[np.ndarray, List[int]]:
    """Build deterministic one-hot node features from atomic numbers."""

    unique_atomic_numbers = sorted({int(z) for z in atomic_numbers})
    feature_map = {z: idx for idx, z in enumerate(unique_atomic_numbers)}
    features = np.zeros((len(atomic_numbers), len(unique_atomic_numbers)), dtype=np.float32)
    for atom_idx, atomic_number in enumerate(atomic_numbers):
        features[atom_idx, feature_map[int(atomic_number)]] = 1.0
    return features, unique_atomic_numbers


def residue_summary(topology: TopologyInfo) -> List[Dict[str, object]]:
    """Return ordered residue-level metadata for logging/debugging."""

    residues = []
    for (chain_id, residue_id, residue_name), atom_map in _group_residues(topology.atoms):
        residues.append(
            {
                "chain_id": chain_id,
                "residue_id": residue_id,
                "residue_name": residue_name,
                "atom_names": sorted(atom_map.keys()),
            }
        )
    return residues
