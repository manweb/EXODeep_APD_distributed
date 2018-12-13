#!/bin/bash -l

#SBATCH -p regular
#SBATCH -C haswell
#SBATCH -t 2:00:00
module load tensorflow/intel-1.12.0-py36
srun python -u TrainCNN.py $@
