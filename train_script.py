####
# Start this program with torchrun, i.e.
# torchrun 
# 
# Source inspired by:
# - https://github.com/pytorch/examples/blob/main/distributed/ddp-tutorial-series/multinode.py
# - https://lambdalabs.com/blog/multi-node-pytorch-distributed-training-guide
# - https://huggingface.co/blog/pytorch-ddp-accelerate-transformers
# 
# Required libraries:
# torch torchvision numpy matplotlib pickle
####

import os
import pickle
import numpy as np
import torch
import torch.distributed as dist
#import torch.distributed.autograd as dist_autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
from datetime import datetime
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.optim import DistributedOptimizer
from torch.distributed import init_process_group, destroy_process_group
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

# Helps identify size of cluster and if this process is the root (rank 0) process.
world_size = int(os.environ["WORLD_SIZE"]) 
RANK = int(os.environ["RANK"])
LOCAL_RANK = int(os.environ["LOCAL_RANK"] or 0)
# parameters (TODO: hyperparameter search with self-guided genetic algorithm)
RANDOM_SEED = 42
LEARNING_RATE = 0.001
BATCH_SIZE = 64
N_EPOCHS = 15

IMG_SIZE = 32
N_CLASSES = 1000
DATA_PATH = "/s/chopin/b/grad/bdvision/Downloads"
TRAIN_PATH = DATA_PATH + "/Imagenet" + str(IMG_SIZE) + "_train"
VAL_PATH = os.path.join(DATA_PATH, 'val_data')

# This is set when the program initalizes
device = None

## ------------------------ Dataset functions

class Dataset(torch.utils.data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, images, labels):
        'Initialization'
        self.labels = labels
        self.images = images

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.labels)

  def __getitem__(self, index):
        'Generates one sample of data'

        # Load data and get label
        X = self.images[index]
        y = self.labels[index]

        return X, y

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo)
    return dict

def load_databatch(data_folder, idx, img_size=32):
    data_file = os.path.join(data_folder, 'train_data_batch_' + str(idx))

    d = unpickle(data_file)
    x = d['data']
    y = d['labels']
    mean_image = d['mean']

    x = x/np.float32(255)
    mean_image = mean_image/np.float32(255)

    # Labels are indexed from 1, shift it so that indexes start at 0
    y = [i-1 for i in y if i < 501]
    data_size = x.shape[0]

    x -= mean_image

    img_size2 = img_size * img_size

    x = np.dstack((x[:, :img_size2], x[:, img_size2:2*img_size2], x[:, 2*img_size2:]))
    x = x.reshape((x.shape[0], img_size, img_size, 3)).transpose(0, 3, 1, 2)

    # create mirrored images
    X_train = np.array(x[0:data_size, :, :, :])
    Y_train = np.array(y[0:data_size])

    return dict(
        X_train=torch.from_numpy(X_train),
        Y_train=torch.from_numpy(Y_train),
        mean=mean_image)
def load_valdata(data_file , mean_image, img_size=IMG_SIZE):
    d = unpickle(data_file)
    x = d['data']
    y = d['labels']

    x = x/np.float32(255)

    # Labels are indexed from 1, shift it so that indexes start at 0
    y = [i-1 for i in y if i < 501]
    data_size = x.shape[0]

    x -= mean_image

    img_size2 = img_size * img_size

    x = np.dstack((x[:, :img_size2], x[:, img_size2:2*img_size2], x[:, 2*img_size2:]))
    x = x.reshape((x.shape[0], img_size, img_size, 3)).transpose(0, 3, 1, 2)

    # create mirrored images
    X_val = np.array(x[0:data_size, :, :, :])
    Y_val = np.array(y[0:data_size])

    return dict(
        X_val=torch.from_numpy(X_val),
        Y_val=torch.from_numpy(Y_val))

## ------------------------ Network definition
class LeNet5(nn.Module):

    def __init__(self, n_classes):
        super(LeNet5, self).__init__()
        
        self.feature_extractor = nn.Sequential(            
            nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, stride=1),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5, stride=1),
            nn.Tanh()
        )

        self.classifier = nn.Sequential(
            nn.Linear(in_features=120, out_features=84),
            nn.Tanh(),
            nn.Linear(in_features=84, out_features=n_classes),
        )


    def forward(self, x):
        x = self.feature_extractor(x)
        x = torch.flatten(x, 1)
        logits = self.classifier(x)
        probs = F.softmax(logits, dim=1)
        return logits, probs

## ------------------------ Training
def get_accuracy(model, data_loader, device):
    '''
    Function for computing the accuracy of the predictions over the entire data_loader
    '''
    
    correct_pred = 0 
    n = 0
    
    with torch.no_grad():
        model.eval()
        for X, y_true in data_loader:

            X = X.to(device)
            y_true = y_true.to(device)

            _, y_prob = model(X)
            _, predicted_labels = torch.max(y_prob, 1)

            n += y_true.size(0)
            correct_pred += (predicted_labels == y_true).sum()

    return correct_pred.float() / n
    
def train(train_loader, model, criterion, optimizer, device):
    '''
    Function for the training step of the training loop
    '''

    model.train()
    running_loss = 0
    
    for X, y_true in train_loader:

        # optimizer.zero_grad()
        
        X = X.to(device)
        y_true = y_true.to(device)
    
        def closure():
            y_hat, _ = model(X) 
            optimizer.zero_grad()
            loss = criterion(y_hat, y_true)            
            loss.backward()
            return loss

        loss = optimizer.step(closure)
        running_loss += loss * X.size(0)

        # The following works for ADAM optimizer, but not L-BFGS
        '''
            # Forward pass
            y_hat, _ = model(X) 
            loss = criterion(y_hat, y_true) 
            running_loss += loss.item() * X.size(0)

            # Backward pass
            loss.backward()
            optimizer.step()
        '''

        # Use distributed autograd (we're not using this at the moment)
        '''
        with dist_autograd.context() as context_id:
            y_hat, _ = model(X) 
            loss = criterion(y_hat, y_true) 
            running_loss += loss.item() * X.size(0)

            # Backward pass
            #loss.backward()
            dist_autograd.backward(context_id, [loss])

            optimizer.step(context_id)
        '''
    epoch_loss = running_loss / len(train_loader.dataset)
    return model, optimizer, epoch_loss

def validate(valid_loader, model, criterion, device):
    '''
    Function for the validation step of the training loop
    '''
   
    model.eval()
    running_loss = 0
    
    for X, y_true in valid_loader:
    
        X = X.to(device)
        y_true = y_true.to(device)

        # Forward pass and record loss
        y_hat, _ = model(X) 
        loss = criterion(y_hat, y_true) 
        running_loss += loss.item() * X.size(0)

    epoch_loss = running_loss / len(valid_loader.dataset)
        
    return model, epoch_loss

def training_loop(model, criterion, optimizer, epochs, device, print_every=1):
    '''
    Function defining the entire training loop
    '''
    
    # set objects for storing metrics
 #   best_loss = 1e10
 #   best_epoch = 0
    train_losses = []
    valid_losses = []
    train_accs = []
    valid_accs = []
 
    # Train model
    for epoch in range(epochs):
        for batch_no in range(10):
#            print(f'{datetime.now().time().replace(microsecond=0)} --- PID {RANK}; epoch {epoch}, batch {batch_no}');
            batch = load_databatch(TRAIN_PATH, batch_no+1)
            val_data = load_valdata(VAL_PATH, batch['mean'])

            train_dataset = Dataset(batch['X_train'], batch['Y_train'])
            val_dataset = Dataset(val_data['X_val'], val_data['Y_val'])

            train_loader = DataLoader(
                dataset=train_dataset,
                batch_size=BATCH_SIZE,
                shuffle=True,
                #sampler=DistributedSampler(train_dataset)
            )
            valid_loader = DataLoader(
                dataset=val_dataset,
                batch_size=BATCH_SIZE,
                shuffle=True,
                #sampler=DistributedSampler(val_dataset)
            )

            # training
            model, optimizer, train_loss = train(train_loader, model, criterion, optimizer, device)
#            if (train_loss < best_loss) :
#                best_epoch = epoch
#                best_loss = train_loss
            train_losses.append(train_loss)

        # validation
        with torch.no_grad():
            model, valid_loss = validate(valid_loader, model, criterion, device)
            valid_losses.append(valid_loss)

# TODO: Add model save, etc from https://github.com/pytorch/examples/blob/main/distributed/ddp-tutorial-series/multinode.py
#            if LOCAL_RANK == 0 and epoch % self.save_every == 0:
#                self._save_snapshot(epoch)
        if RANK == 0 :
            torch.save(model, "model.pt")
            train_acc = get_accuracy(model, train_loader, device=device)
            valid_acc = get_accuracy(model, valid_loader, device=device)
            train_accs.append(train_acc)
            valid_accs.append(valid_acc)
            torch.save(train_losses, "train_loss.pt")
            torch.save(valid_losses, "valid_loss.pt")
            torch.save(train_accs, "train_accs.pt")
            torch.save(valid_accs, "valid_accs.pt")
            if (epoch+1) % print_every == 0:            
                print(f'{datetime.now().time().replace(microsecond=0)} --- '
                    f'Epoch: {epoch}\t'
                    f'Train loss: {train_loss:.4f}\t'
                    f'Valid loss: {valid_loss:.4f}\t'
                    f'Train accuracy: {100 * train_acc:.2f}\t'
                    f'Valid accuracy: {100 * valid_acc:.2f}')            

    #plot_losses(train_losses, valid_losses)
    
    return model, optimizer, (train_losses, valid_losses)

## ------------------------ Main program (init/run)
def run():
    model = LeNet5(N_CLASSES).to(device)
    # optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    optimizer = torch.optim.LBFGS(model.parameters(), lr=LEARNING_RATE)
    # 1) That wasn't right, because we're not dealing with rrefs, 2) Makes more sense for individual nodes to have their own optimizers since we're pooling weights
    #optimizer = DistributedOptimizer(torch.optim.Adam, model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    ### Partial starting point for DDP example
    #ddp_model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[LOCAL_RANK], output_device=LOCAL_RANK)
    ddp_model = DDP(model)
    # Note: the training loop sets up the distributed sampler with the data set, in some examples you'd see that here
    # It works this way becasue the data set is not loaded all at once and instead is loaded in the training loop in batches,
    model, optimizer, results = training_loop(ddp_model, criterion, optimizer, N_EPOCHS, device)

    # TODO: save results so they can be graphed, etc


## Initialize and run the program
def init():
    global device
    torch.manual_seed(RANDOM_SEED)
    ## By setting device tensor.to(device) should always work correctly for CPU and GPU
    if (torch.cuda.device_count() > 0):
        device = torch.device("cuda:{}".format(LOCAL_RANK))
        dist.init_process_group(backend="nccl")
    else:
        device = torch.device("cpu")
        dist.init_process_group(backend="gloo")
    print("Started process " + str(RANK))
    
if __name__ == "__main__":
    init()
    run()