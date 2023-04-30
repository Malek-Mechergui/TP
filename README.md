# CS535 - Group 6 - A study in Distributed Optimizer performance

## Malek Mechergui, Saira Sabeen, Brendan Robert

The goal of this software project is to compare the performance of PyTorch optimizers in a single CPU node against 
a distributed Torch environment while changing as little as possible so that it is hopefully a fair comparison.

## Code Design

### Overview of program code files

The initial single-core code is in the notebook `LeNet_IMAGENET.ipynb` and can be executed independently on either
CPU or GPU.  This allowed us to explore the base model and data set to ensure our network was training appropriately
before scaling up in a headless torchrun environment.  `LeNet.ipynb` was a warm-up for us to explore LeNet using the
MNIST data set.  This wasn't part of our experiment directly, but rather a way for us to get up and running quickly
on the model before switching over to the much more complicated IMAGENET data set.

The same code was modified into a TorchRun-compatible standalone program `train_script.py`.  This can be run on one
host (single-cpu) as well as multiple via torchrun.  The execution across multiple servers is performed using the
shell script `run_distributed.sh`.  This script has a set of four hosts and opens a connection via SSH to each and
executes the appropriate torchrun command so that each node can call back to the primary host (rank 0).  Execution
is run within a GNU Screen window for safety against disconnection, and in addition the results are saved in pickle
files for further evaluation as necessary.

A second `run_distributed2.sh` bash script allows parallel execution against 4 additional hosts which allowed us to
run multiple trials at the same time without affecting the performance of one another.

Both run scripts have paired `stop_distributed.sh` scripts as well to terminate training as needed.  This allowed us
to better free-up resources quickly while re-adjusting program code and/or hyper parameters.  In particular this was
extremely necessary when trying to get L-BFGS to work in a parallel computing environment appropriately.

### Training data

Our training data was obtained from the downsampled IMAGENET data set: https://patrykchrabaszcz.github.io/Imagenet32/
This requires labels in `imagenet1000_clsidx_to_labels.txt`, validation data in a pickle file `val_data` and finally
the batched training in pickle files named sequentially `train_data_batch_N` where N ranges from 1 to 10, stored
in a folder `Imagenet32_train`.  The code also supports the 64x64 data set to allower further exploration.

### Details about train_script 

Train_script must be executed via `torchrun`, even if running on a single node.  This ensures certain environment
variables are set correctly at the start of execution.  The simplest invocation is:

``` bash
torchrun --nnodes=1 --node_rank=0 train_script.py [OPTIMIZER]
```

where [OPTIMIZER] is one of the following: adam, sgd, lbfgs.  As stated earlier, execution on multiple hosts requires
the run_distributed shell script to ensure that the appropriate command is executed on each host and also assign the
correct sequential rank to each.

Once the program begins, a communication backend is chosen based on the computation style available.  If GPU is used
then NCCL is used, otherwise GLOO is used as a communication backend.  The model and optimizer are wrapped with the 
apporiate Torch Distributed classes.  These handle all the necessary synchronization at the beginning of each epoch and 
forward pass.  However, some additional logic was necessary (as noted in the code) to keep L-BFGS stable, which is 
described further in the next section.

Cross-entropy loss is used as our evaluation function since our accuracy is based on a softmax of one-hot outputs.

A data loader provides the pre-processed/normalized data batches for the training function to consume.  During batch 
training, a new set is picked out of each batch after it was shuffled.  This is different in each distributed node so
that ideally when all weights are combined, a better convergence is obtainable since the weight adjustment considers a
wider range of samples.  This is especially useful in GPU training where memory is usually much more limited and the
more common solution is to train with smaller batches at the risk of over-fitting or otherwise poorer convergence (or,
on the other hand, many more epochs needed because learning rate had to be reduced, etc.)

At the end of each epoch, the running statistics and model file are saved in pickle format.  This allows us the ability
to resume training from a checkpoint later if needed, or to use the final results for further comparison if desired.
Fortunately during training, we were spared the disruption which would have warranted writing the "restore state" portion
of the code.

### More notes on L-BFGS optimizer

Early on, we had a convergence problem with L-BFGS because its state was not syncing correctly across processes.  
It was not trivial to wrap it with the appropriate distributed optimizer package as the optimizer step requires a lambda
function that enables it to re-execute the forward/backward pass if it chooses; One defining feature of L-BFGS is this
self-guided repeat pass to get gradients closer to a desired tolerance.  However, if we abandon this entirely (prevent any
additional forward-backward by loosening tolerance) then the model simply converges poorly. A hopefully better solution we 
discovered (attribution to the online discussion is provided in-code) is to use a reduction of errors across nodes during
the forward/backward pass so that each node reaches the same decision and therefore applies the same number of iterations 
during this process.  This stabalizes training, but the added communication across multiple hosts is expensive, and as a 
result we still needed to find a balance of fewer passes to balance this out somewhat.

The code in its current form represents this final compromise of parameters.

## Data results

Results from each trial were stored here for further evaluation.  The first two trials (trial1 and trial2) were early tests
of the program which were aborted quickly due to bugs causing poor convergence (see above section about L-BFGS for details.)

For our project we opted for CPU-based comparison as it was easier to scale without requiring specialized
hardware.

The four other trial directories correspond to the trails for each of the distributed executions.  We also ran the same
code on a single node in `solo trials` which consist of results from adam, sgd and lbfgs (using the final settings).  The
original LBGS (defaults) were not consiered for solo trials due to concerns of execution time, and it was sufficient data
in-hand that we didn't feel it was worth further exploration at this time.  If our goal were to focus on the effect of
network communication as a bottleneck, this would be the first area to dig into further though.

