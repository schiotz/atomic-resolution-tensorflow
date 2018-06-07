#!/bin/bash

#SBATCH -N 1
#SBATCH -n 16
#SBATCH --partition=xeon16
#SBATCH --time=10:00:00
#SBATCH --output=learningcurve-%j.out
#SBATCH --mem=0
#SBATCH --gres=gpu:K20Xm:4

# zapmodules

module purge
export PYTHONPATH=$HOME/development/structural-template-matching/build/lib.linux-x86_64-3.6:$HOME/development/PyQSTEM/build/lib.linux-x86_64-3.6:$HOME/development/ase
module load matplotlib
if [[ "`hostname`" == "thul.fysik.dtu.dk" ]]; then
    echo "Loading Keras for CPU only."
    module load Keras/2.1.3-foss-2017b-Tensorflow-bin-Python-3.6.3
else
    echo "Loading GPU-enabled version of Keras."
    module load Keras/2.1.3-foss-2017b-Tensorflow-GPU-Python-3.6.3
fi
module load scikit-image scikit-learn tqdm

time python learningcurve.py graphs-initial

