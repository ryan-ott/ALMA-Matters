# ALMA-Matters

Compressing ALMA for the DL4NLP course

## Overview

This repository focuses on compressing the ALMA model using techniques like slicing, pruning, and distillation. It includes code for running experiments, evaluating translation performance, and more.

> The repository also includes the [TransformerCompression](https://github.com/ryan-ott/TransformerCompression) submodule, a fork of the SliceGPT paper's code which provides the framework used for model slicing.

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

```bash
conda env create -f DL4_env.yml
conda activate DL4_env
```

### 3. Updating the Submodule (When Necessary)
If the submodule is updated in the remote repository and you need to pull the latest changes, run:

```bash
git submodule update --remote
```
This will pull the latest version of the submodule from its repository.

### 4. Running the Code
Once the environment is set up and submodules are initialized, you can start running the experiments and scripts in the repository. For example, to evaluate a translation model, refer to the `evaluation/translation_eval2.py` script (filename will hopefully change ;).