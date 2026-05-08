# adversarial-attack-on-image-classifier

Codebase for a MATH6503 group project on adversarial attacks against a CIFAR-10 image classifier.

This repository contains:

- a pretrained ResNet-20 model for CIFAR-10
- several adversarial attack implementations
- a small script for running one experiment configuration
- helper utilities for evaluation and visualization
- saved CSV results and figure assets
- a notebook for manual analysis

## Overview

The project is organized around a pretrained CIFAR-10 classifier and a set of attack methods implemented in PyTorch. The current workflow is script-based rather than package-based:

1. `cifar10_resnet20.py` loads the dataset and pretrained model.
2. `attack.py` provides attack implementations such as FGSM, IFGSM, MI-FGSM, PGD, Frank-Wolfe, and LBFGS.
3. `run.py` selects correctly classified test images, runs one chosen attack, and writes a CSV file to `results/`.
4. `utils.py` provides visualization, evaluation, and result-saving helpers.
5. `data-analysis.ipynb` is used for exploratory analysis of saved CSV outputs.

## Repository Tree

```text
.
├── .gitignore
├── README.md
├── attack.py
├── cifar10_resnet20.py
├── cifar10_resnet20_pretrained.pth
├── data/
│   ├── cifar-10-python.tar.gz
│   └── cifar-10-batches-py/
│       ├── batches.meta
│       ├── data_batch_1
│       ├── data_batch_2
│       ├── data_batch_3
│       ├── data_batch_4
│       ├── data_batch_5
│       ├── readme.html
│       └── test_batch
├── data-analysis.ipynb
├── essay-images/
│   ├── fig-1-1pixel.png
│   ├── fig-1-fw-eg1.png
│   ├── fig-1-fw-eg2.png
│   ├── fig-2-fw-eg1.png
│   ├── fig-2-fw-eg2.png
│   ├── fig-inf-fw-eg.png
│   ├── fig-inf-fw-eg2.png
│   ├── fig-inf-lbfgs-1.png
│   ├── fig-inf-pgd-1.png
│   └── fig-inf-pgd-2.png
├── results/
│   ├── l1/
│   │   └── *.csv
│   ├── l2/
│   │   └── *.csv
│   └── linf/
│       └── *.csv
├── run.py
├── utils.py
└── requirements.txt
```

## File Guide

- `attack.py`
  Implements the adversarial attacks used in this project. The file currently includes `lbfgs`, `projected_gradient_descent`, `frank_wolfe`, `fgsm`, `ifgsm`, `mifgsm`, and an `L1` projection helper.

- `cifar10_resnet20.py`
  Defines the ResNet-20 architecture, normalization transform, CIFAR-10 train/test datasets, data loaders, class names, and loads the pretrained weights from `cifar10_resnet20_pretrained.pth`.

- `run.py`
  Main experiment script. It picks a subset of correctly classified test images, sets the attack configuration through hardcoded variables near the bottom of the file, runs the attack, and saves results to the corresponding `results/` folder.

- `utils.py`
  Utility functions for:
  - converting normalized tensors back to displayable images
  - plotting dataset examples
  - showing adversarial examples
  - measuring class-wise and overall accuracy
  - selecting correctly classified samples
  - computing attack success rate and exporting CSV files

- `data-analysis.ipynb`
  Notebook for exploratory analysis of the saved results. It loads CSV files from `results/`, compares attack success rates, and inspects selected adversarial examples.

- `cifar10_resnet20_pretrained.pth`
  Pretrained weights for the offline ResNet-20 model loaded by `cifar10_resnet20.py`.

- `data/cifar-10-python.tar.gz`
  Original compressed CIFAR-10 dataset archive.

- `data/cifar-10-batches-py/`
  Extracted CIFAR-10 Python-format dataset used by `torchvision.datasets.CIFAR10` with `download=False`.

- `essay-images/`
  Figure assets, likely used in a report, presentation, or written project submission.

- `results/l1/`, `results/l2/`, `results/linf/`
  Saved experiment outputs as CSV files grouped by the attack constraint norm. Filenames follow the pattern:
  `attack_name-e{epsilon}-p{p}-N{steps}.csv`

- `.gitignore`
  Ignore rules for local data, caches, the virtual environment, and the pretrained weight file.

- `requirements.txt`
  Minimal Python dependency list for running the current scripts and notebook support.

## Requirements

Install dependencies with:

```bash
pip install -r requirements.txt
```

Main libraries used in this repository:

- PyTorch
- torchvision
- NumPy
- pandas
- matplotlib
- tqdm
- IPython kernel support for notebook usage

## Setup

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

This repository expects the following files to exist locally:

- `cifar10_resnet20_pretrained.pth`
- `data/cifar-10-batches-py/`

Important note: `cifar10_resnet20.py` uses `download=False`, so the CIFAR-10 dataset must already be present in `data/`.

## How to Run an Experiment

The current project does not expose a command-line interface. Instead, experiment settings are configured directly in `run.py`.

At the bottom of `run.py`, the following variables define the experiment:

```python
e = 0.01
p = 1
N = 10
attack = mifgsm
```

To run the experiment:

```bash
python run.py
```

or, if you are using the local virtual environment:

```bash
.venv/bin/python run.py
```

The script will:

1. load the pretrained model and CIFAR-10 test set
2. select correctly classified test images
3. run the chosen attack
4. compute success statistics
5. save a CSV file under `results/l1/`, `results/l2/`, or `results/linf/`

## Supported Attacks

According to the code comments in `attack.py`, the supported attacks by norm are:

- L-BFGS, PGD, Frank-Wolfe, FGSM, I-FGSM, MI-FGSM


## Result Files

Each CSV written by `utils.attack_success_rate(...)` contains:

- `l1`: L1 norm of the perturbation
- `l2`: L2 norm of the perturbation
- `linf`: Linf norm of the perturbation
- `times`: average runtime per sample in the batch
- `prob_org`: model confidence for the true label before the attack
- `prob_adv`: model confidence for the true label after the attack
- `success`: whether the predicted label changed
- `global_index`: row index within the evaluated subset

## Notebook Usage

To work with `data-analysis.ipynb`, start Jupyter from the project root after installing the dependencies you need for notebooks.


## Notes

- The current code loads the model, dataset, and data loaders at import time.
- `run.py` uses hardcoded experiment parameters rather than CLI arguments.
- The repository already contains many saved result files, so you can inspect outputs without rerunning all experiments.
- This README documents the repository as it currently exists and does not change or reinterpret the implementation details in the code.
