#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --job-name=InstallEnvironment
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --time=04:00:00
#SBATCH --output=/home/scur1769/ALMA-Matters/utils/script_outputs/InstallEnvironment_%A.out

module purge
module load 2022
module load Anaconda3/2022.05

cd $HOME
pwd
conda env create -f "/home/scur1769/ALMA-Matters/DL4_env.yml"

# Activate the conda environment
source activate DL4_env

# Print packages in the environment
conda list

# for any additional packages, edit the DL4_env.yml file and rerun this script with conda env update -f DL4_env.yml