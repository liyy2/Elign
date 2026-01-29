# Contributing

Thanks for your interest in contributing to ELIGN.

## Development setup

```bash
conda create -n elign-dev python=3.10 -c conda-forge rdkit
conda activate elign-dev
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

## Running tests

```bash
pytest -q
```

## Style

- Follow PEP 8 and keep changes minimal and focused.
- Prefer small, testable functions and explicit configuration.

## Pull requests

- Clearly describe what changed and why.
- Include the command(s) used to reproduce results when relevant.
- Avoid committing large artifacts (checkpoints, datasets, `outputs/`, `wandb/`).

