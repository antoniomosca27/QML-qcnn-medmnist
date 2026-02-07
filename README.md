# QML-qcnn-medmnist

[![CI](https://github.com/antoniomosca27/QML-qcnn-medmnist/actions/workflows/ci.yml/badge.svg)](https://github.com/antoniomosca27/QML-qcnn-medmnist/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

QML-qcnn-medmnist implements a hybrid quantum-classical convolution pipeline for medMNIST classification using Qiskit and PyTorch. The workflow preprocesses medMNIST images, extracts patch-level quantum features, and trains a compact classifier. The project includes command-line interfaces for preprocessing, training, report generation, heatmap visualization, and learning-curve plotting. A notebook is provided for end-to-end reproducible execution.

## Installation
```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Quickstart
```bash
qcnn-preprocess --dataset bloodmnist
qcnn-train --dataset bloodmnist --epochs 5 --stride 3
qcnn-report --dataset bloodmnist --logdir logs/bloodmnist_run_001 --stride 3
qcnn-heatmap --dataset bloodmnist --logdir logs/bloodmnist_run_001 --idx 0 --stride 3
qcnn-plot-curves --logdir logs/bloodmnist_run_001
```

For `--logdir`, all analysis CLIs accept either:
- a run directory path (absolute or relative), e.g. `logs/bloodmnist_run_001`;
- a run folder name inside `--logs-dir`, e.g. `bloodmnist_run_001`.

## Repository Layout
- `src/`: Python package source (`src.datasets`, `src.models`, `src.quantum`, `src.scripts`, `src.training`, `src.utils`).
- `tests/`: automated test suite.
- `notebooks/QML-qcnn-medmnist_pipeline.ipynb`: end-to-end workflow notebook.
- `pyproject.toml`: package metadata, dependencies, and CLI entry points.

## Runtime Outputs
- `data/raw/`: downloaded medMNIST archives.
- `data/processed/<dataset>/`: preprocessed tensors (`train.pt`, `val.pt`, `test.pt`).
- `logs/<dataset>_run_XXX/`: run metrics and model checkpoints.
- `reports/<dataset>_run_XXX/`: confusion matrix, heatmaps, learning curves, predictions, and report metadata.

`data/`, `logs/`, and `reports/` are runtime artifact directories and are ignored by Git, except for `.gitkeep` placeholders.

## Reproducibility
- Use `--seed` in CLI commands to initialize Python, NumPy, PyTorch, and Qiskit randomness.
- `PYTHONHASHSEED` is set by the seed utility for deterministic hashing behavior.
- Set `QCNN_CPUS` to control process-level parallelism in quantum convolution workers.

## Developer
- Remove local test/build artifacts with `bash scripts/clean_artifacts.sh`.

## Dataset Acknowledgment
This project uses [medMNIST](https://medmnist.com/). Datasets are downloaded at runtime and are not tracked in version control.

## License
This project is licensed under the MIT License.
