## EXODeep position and energy reconstruction from scintillation light (distributed version)
This repository contains the code for training and evaluating the position and energy reconstruction from
the raw data of the scintillation signal. It is optimized to run distributed on multiple nodes at NERSC.
### Single machine training
To run single machine training, use the following command
```
> python TrainCNN.py <options>
```
or use the batch script and specify the number of nodes to be 1
```
> sbatch -N 1 submit_batch.sl <options>
```
Use command --help to display all available options
### Distributed training
For distributed training specify the number of nodes N > 1 when submitting the job. You can specify the number of parameter serves
to be used with option --numPS
```
> sbatch -N 5 submit_batch.sl --numPS 2 <options>
```
Above command will run the training with 5 nodes of which 2 are parameter servers the remaining 3 the workers

