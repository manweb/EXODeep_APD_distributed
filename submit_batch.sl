#!/bin/bash -l

#SBATCH -q regular
#SBATCH -C haswell
#SBATCH -t 1:00:00

export OMP_NUM_THREADS=28
export OMP_PROC_BIND=true
export OMP_PLACES=threads

module load tensorflow/intel-1.12.0-py36
srun -c 64 python -u TrainCNN.py $@
