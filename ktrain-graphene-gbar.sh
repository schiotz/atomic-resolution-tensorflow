#!/bin/bash
### General options
### -- specify queue --   NOTE: TitanX is significantly faster than K80
##BSUB -q gpuk80
#BSUB -q gpuv100
##BSUB -gpu "num=8:mode=exclusive_process"
##BSUB -q gputitanxpascal
#BSUB -gpu "num=2:mode=exclusive_process"
### -- set the job Name --
#BSUB -J trainC2
### -- ask for number of cores (default: 1) --
#BSUB -n 10
#BSUB -R "span[hosts=1]"
### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 12:00
# request 5GB of memory
#BSUB -R "rusage[mem=3GB]"
### -- send notification at start --
##BSUB -B
### -- send notification at completion--
##BSUB -N
### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#BSUB -o trainC-%J.out
###BSUB -e ktrain-%J.err
# -- end of LSF options --


# zapmodules

module purge
source ~/asap-pyqstem.bashrc
export PYTHONPATH=$HOME/development/structural-template-matching/build/lib.linux-x86_64-3.6:$PYTHONPATH

module load tensorflow/1.5-gpu-python-3.6.2


time python3 ktrain-graphene.py graphene-hidose-paper
time python3 learningcurve-graphene.py graphs-graphene-hidose-paper
#time python3 validatescale.py graphs-110-peismovie-final
time python3 validatedose-graphene.py graphs-graphene-hidose-paper



