'''This is the repo which contains the original code to the WACV 2022 paper
"Fully Convolutional Cross-Scale-Flows for Image-based Defect Detection"
by Marco Rudolph, Tom Wehrbein, Bodo Rosenhahn and Bastian Wandt.
For further information contact Marco Rudolph (rudolph@tnt.uni-hannover.de)'''

import torch
import numpy as np
import random
import config as c
from train import train, train_multi_model
from utils import load_datasets, make_dataloaders
from tiredatasetscript import *
import os
from utils import *


#set seed for all the project
torch.manual_seed(c.seed)
torch.cuda.manual_seed(c.seed)
torch.cuda.manual_seed_all(c.seed)
random.seed(c.seed)
np.random.seed(c.seed)

torch.use_deterministic_algorithms(True) 

#I change the location where pytorch saves pretrained models
os.environ['TORCH_HOME'] = 'models\\net' #setting the environment variable


train_set, test_set, validation_set = load_datasets(c.dataset_path, c.class_name)
train_loader, test_loader, validation_loader = make_dataloaders(train_set, test_set, validation_set)
if c.multi_model:
    train_multi_model(train_loader, test_loader, test_set, validation_loader, validation_set)
else:
    train(train_loader, test_loader, test_set, validation_loader, validation_set)
