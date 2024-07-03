
import torch
import config as c
from utils import load_datasets, make_dataloaders
from tqdm import tqdm
import numpy as np

# placeholders
psum = torch.tensor([0.0, 0.0, 0.0])
psum_sq = torch.tensor([0.0, 0.0, 0.0])

train_set, test_set, validation_set = load_datasets(c.dataset_path, c.class_name)
train_loader, test_loader, validation_loader = make_dataloaders(train_set, test_set, validation_set)

# loop through images
#for inputs in tqdm(train_loader):
 #   psum    += inputs.sum(axis        = [0, 2, 3])
  #  psum_sq += (inputs ** 2).sum(axis = [0, 2, 3])

nimages = 0
mean = 0.0
var = 0.0
# REMEMBER TO ELIMINATE NORMALIZATION IN MAKE DATASET BEFORE RUNNING THIS
for i_batch, batch_target in enumerate(tqdm(validation_loader, disable=c.hide_tqdm_bar)):
    batch = batch_target[0]
    # Rearrange batch to be the shape of [B, C, W * H]
    batch = batch.view(batch.size(0), batch.size(1), -1)
    # Update total number of images
    nimages += batch.size(0)
    # Compute mean and std here
    mean += batch.mean(2).sum(0) 
    var += batch.var(2).sum(0)

mean /= nimages
var /= nimages
std = torch.sqrt(var)

print(mean)
print(std)
