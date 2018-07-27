#!/bin/bash
### General options
### -- specify queue --   NOTE: TitanX is significantly faster than K80
##BSUB -q gputitanxpascal
#BSUB -q gpuv100
#BSUB -gpu "num=1:mode=exclusive_process"
### -- set the job Name --
#BSUB -J analyze
### -- ask for number of cores (default: 1) --
#BSUB -n 5
#BSUB -R "span[hosts=1]"
### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 20:00
# request 5GB of memory
#BSUB -R "rusage[mem=3GB]"
### -- send notification at start --
##BSUB -B
### -- send notification at completion--
##BSUB -N
### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#BSUB -o analyze-%J.out
###BSUB -e ktrain-%J.err
# -- end of LSF options --


# zapmodules

module purge
source ~/asap-pyqstem.bashrc
export PYTHONPATH=$HOME/development/structural-template-matching/build/lib.linux-x86_64-3.6:$PYTHONPATH

module load tensorflow/1.5-gpu-python-3.6.2


time python3 analyze_expt_movie.py



