#!/bin/bash

#SBATCH -N 1
#SBATCH -n 1
#SBATCH --partition=xeon16
#SBATCH --time=10:00:00
#SBATCH --job-name=mkte100
#SBATCH --output=%x-%j.out
#SBATCH --mem=40G

module purge

export PYTHONPATH=$HOME/development/structural-template-matching/build/lib.linux-x86_64-3.6:$HOME/development/PyQSTEM/build/lib.linux-x86_64-3.6:$HOME/development/ase
module load ASE
module load scikit-image scikit-learn tqdm

python make_cluster_testing_data_100.py
