# EXODeep position and energy reconstruction from scintillation light (distributed version)
This repository contains the code for training and evaluating the position and energy reconstruction from
the raw data of the scintillation signal. It is optimized to run distributed on multiple nodes at NERSC.
## Single machine training
To run single machine training, use the following command
> python TrainCNN.py <options>
or use the batch script and specify the number of nodes to be 1
> submit_batch.sl -N 1 <options>

