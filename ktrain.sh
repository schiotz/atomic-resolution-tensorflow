#!/bin/bash

#SBATCH -N 1
#SBATCH -n 16
#SBATCH --partition=xeon16_256
#SBATCH --time=20:00:00
#SBATCH --output=ktrain-%j.out
#SBATCH --mem=0
#SBATCH --gres=gpu:K20Xm:4

# zapmodules

module purge
export PYTHONPATH=$HOME/development/structural-template-matching/build/lib.linux-x86_64-3.6:$HOME/development/PyQSTEM/build/lib.linux-x86_64-3.6:$HOME/development/ase

module load matplotlib/2.1.2-foss-2018a-Python-3.6.4
if [[ "`hostname`" == "thul.fysik.dtu.dk" ]]; then
    echo "Loading Keras for CPU only."
    module load Keras/2.2.0-foss-2018a-Python-3.6.4
    module unload TensorFlow
    module load TensorFlow/1.7.0-foss-2018a-Python-3.6.4
else
    echo "Loading GPU-enabled version of Keras."
    module load Keras/2.2.0-foss-2018a-Python-3.6.4
    module unload TensorFlow
    module load TensorFlow/1.7.0-foss-2018a-Python-3.6.4-CUDA-9.1.85
fi
module load scikit-image/0.13.1-foss-2018a-Python-3.6.4
module load scikit-learn/0.19.1-foss-2018a-Python-3.6.4
module load tqdm/4.23.4-foss-2018a-Python-3.6.4

time python ktrain.py correct110_v2
time python learningcurve.py graphs-correct110_v2
time python validatescale.py graphs-correct110_v2
time python validatedose.py graphs-correct110_v2



