#!/bin/bash -l

#SBATCH -p regular
#SBATCH -C haswell
#SBATCH -t 0:10:00
module load tensorflow/intel-1.12.0-py36
srun python -u test_batch.py
