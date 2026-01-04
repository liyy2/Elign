#!/bin/bash
set -euo pipefail

# Build the full GEOM-Drugs conformations array used by the EDM/VERL pipelines.
#
# Outputs (under $DATA_DIR):
#   - geom_drugs_30.npy         # [num_atoms_total, 5] rows: [mol_id, atomic_number, x, y, z]
#   - geom_drugs_n_30.npy       # [num_conformers] atom counts per conformer id
#   - geom_drugs_smiles.txt     # SMILES list (one per unique molecule in msgpack)
#
# Requirements:
#   - network access (downloads from Harvard Dataverse / GEOM repo)
#   - enough disk (archive ~42.7GB; extracted msgpack + generated .npy can be much larger)
#   - python env with msgpack + numpy (and the repo deps)
#
# Usage:
#   bash prepare_geom_drugs_30.sh
#
# Optional overrides:
#   DATA_DIR=/path/to/datasets/geom CONFORMERS=30 OUTPUT_DTYPE=float32 bash prepare_geom_drugs_30.sh
#   DATAVERSE_FILE_ID=4360331 bash prepare_geom_drugs_30.sh

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${DATA_DIR:-${REPO_ROOT}/datasets/geom}"
CONFORMERS="${CONFORMERS:-30}"
OUTPUT_DTYPE="${OUTPUT_DTYPE:-float32}" # float32|float64
PYTHON_BIN="${PYTHON_BIN:-python}"

# Harvard Dataverse (GEOM) dataset: doi:10.7910/DVN/JNGTDF
# File we need: drugs_crude.msgpack.tar.gz
DATAVERSE_FILE_ID="${DATAVERSE_FILE_ID:-4360331}"

ARCHIVE_NAME="drugs_crude.msgpack.tar.gz"
ARCHIVE_PATH="${DATA_DIR}/${ARCHIVE_NAME}"
MSGPACK_PATH="${DATA_DIR}/drugs_crude.msgpack"

mkdir -p "${DATA_DIR}"

if [[ ! -f "${ARCHIVE_PATH}" ]]; then
  echo "Downloading ${ARCHIVE_NAME} to ${ARCHIVE_PATH}"
  echo "Source: https://dataverse.harvard.edu/ (doi:10.7910/DVN/JNGTDF)"
  curl -L \
    -H "User-Agent: Mozilla/5.0" \
    --fail \
    --retry 10 \
    --retry-delay 10 \
    -C - \
    -o "${ARCHIVE_PATH}" \
    "https://dataverse.harvard.edu/api/access/datafile/${DATAVERSE_FILE_ID}"
else
  echo "Found existing archive: ${ARCHIVE_PATH}"
fi

if [[ ! -f "${MSGPACK_PATH}" ]]; then
  echo "Extracting ${ARCHIVE_NAME} into ${DATA_DIR}"
  tar -xzf "${ARCHIVE_PATH}" -C "${DATA_DIR}"
else
  echo "Found existing msgpack: ${MSGPACK_PATH}"
fi

if [[ ! -f "${MSGPACK_PATH}" ]]; then
  echo "ERROR: expected ${MSGPACK_PATH} after extraction"
  exit 1
fi

if ! "${PYTHON_BIN}" -c "import msgpack, numpy" >/dev/null 2>&1; then
  echo "ERROR: ${PYTHON_BIN} is missing required packages (msgpack, numpy)."
  echo "Activate your environment (e.g., conda activate edm) or set PYTHON_BIN=/path/to/python."
  exit 1
fi

echo "Building geom_drugs_${CONFORMERS}.npy (streaming, ${OUTPUT_DTYPE})"
"${PYTHON_BIN}" -u "${REPO_ROOT}/edm_source/build_geom_dataset.py" \
  --data_dir "${DATA_DIR}" \
  --data_file "drugs_crude.msgpack" \
  --conformations "${CONFORMERS}" \
  --output_dtype "${OUTPUT_DTYPE}" \
  --streaming

OUT_PATH="${DATA_DIR}/geom_drugs_${CONFORMERS}.npy"
N_ATOMS_PATH="${DATA_DIR}/geom_drugs_n_${CONFORMERS}.npy"

echo "Done."
echo "Conformations: ${OUT_PATH}"
echo "Atom counts:   ${N_ATOMS_PATH}"
echo "SMILES list:   ${DATA_DIR}/geom_drugs_smiles.txt"
