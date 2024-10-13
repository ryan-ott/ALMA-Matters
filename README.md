# ALMA-Matters

Compressing ALMA for the DL4NLP course

## Overview

This repository focuses on compressing the ALMA model using techniques like pruning, quantisation and distillation while retaining translation performance. It includes code for running experiments, evaluating performance, and more.

> The repository also includes the [TransformerCompression](https://github.com/ryan-ott/TransformerCompression) submodule, a fork of the SliceGPT paper's code which provides the framework used for model pruning.

## Installation

### 1. Clone the Repository (with Submodules)
To get started, clone the repository and ensure that you initialize and fetch the submodule at the same time:

```bash
git clone --recurse-submodules https://github.com/ryan-ott/ALMA-Matters.git
```
If you've already cloned the repository without the `--recurse-submodules` flag, you can initialize the submodules afterward by running:

```bash
git submodule update --init --recursive
```

### 2. Install Dependencies
After cloning the repository, you need to set up the required Python environment. The environment configuration is provided in the `DL4_env.yml` file.

To create and activate the environment:

Simply run the installation script:

```bash
bash utils/scripts/install_env.job
```

Alternatively, you can manually create the environment using the provided `DL4_env.yml` file:

```bash
conda env create -f DL4_env.yml
conda activate DL4_env
```

#### 2.1. SliceGPT Installation
The `TransformerCompression` submodule is arranged as a Python package, so to run specific "experiemnts" such as slicing, run the following commands:

```bash
cd compression/pruning/TransformerCompression
conda run -n DL4_env pip install -e .[experiment,finetune]
```

#### 2.2. QLoRA Installation
The `qlora` submodule is independently usable (of the SliceGPT env), so to run (one of the) qlora training configurations, use the provided job file `run_qlora.job`; like this:

```bash
sbatch run_qlora.job
```

### 3. Updating the Submodule (When Necessary)
If a submodule is updated in the remote repository and you need to pull the latest changes, run:

```bash
git submodule update --remote
```
This will pull the latest version of the submodule from its repository.

### 4. Running the Code
Once the environment is set up and submodules are initialized, you can start running the experiments and scripts in the repository. For example, to evaluate a translation model, refer to the `evaluation/translation_eval2.py` script (filename will hopefully change ;).