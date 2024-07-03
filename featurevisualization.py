import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import config as c
from model import get_cs_flow_model, save_model, FeatureExtractor, nf_forward
from utils import *
import config as c
import neptuneparams as nep_params # comment this statement if you don't use neptune
from neptune.new.types import File # comment this statement if you don't use neptune
import matplotlib.pyplot as plt
from model import *
import shutil
import pandas as pd
import itertools

from tiredatasetscript import get_mask_patch

c.batch_size = 1



model_weights_path = "weights/tire_model9999.pth"
os.environ['TORCH_HOME'] = 'models\\net'
train_set, test_set, validation_set = load_datasets(c.dataset_path, c.class_name)
if c.use_sidelight:
    if c.multi_model:
        img_paths = np.empty((c.num_horizontal_patches,c.num_vertical_patches),dtype=object)
        for hor,ver in itertools.product(range(c.num_horizontal_patches), range(c.num_vertical_patches)):
            img_paths[hor,ver] = [p for p, l in validation_set[hor,ver].diffuse.samples]
        
    else:
        img_paths = validation_set.diffuse.paths if c.pre_extracted else [p for p, l in validation_set.diffuse.samples]
else:
    if c.multi_model:
        img_paths = np.empty((c.num_horizontal_patches,c.num_vertical_patches),dtype=object)
        for hor,ver in itertools.product(range(c.num_horizontal_patches), range(c.num_vertical_patches)):
            img_paths[hor,ver] = [p for p, l in validation_set[hor,ver].samples]
    else:
        img_paths = validation_set.paths if c.pre_extracted else [p for p, l in validation_set.samples]
_, test_loader, validation_loader = make_dataloaders(train_set, test_set, validation_set)

if c.multi_model:
    multi_models, _ = load_weights_multimodel(model_weights_path)
    multi_models.set_device(c.device)
    for hor,ver in itertools.product(range(c.num_horizontal_patches), range(c.num_vertical_patches)):
        multi_models.models[hor,ver].eval()
else:
    model, _ , _ = load_weights(model_weights_path)
    model.to(c.device)
    model.eval()

if not c.pre_extracted and not c.multi_model:
    fe = FeatureExtractor()
    fe.eval()
    fe.to(c.device)
    for param in fe.parameters():
        param.requires_grad = False

if c.multi_model == False:
    with torch.no_grad():
        data = next(iter(validation_loader))
        inputs, labels = preprocess_batch(data)
        if not c.pre_extracted:
            inputs = fe(inputs)
        z = model(inputs)
        #print(z)
        #print(model._modules)

        # this network has a gigantic children called ModuleList, that's why we can't use only children() method to split the network
        print(list(list(model.children())[0].children()))
        print('CUT LAYER')
        print(list(list(model.children())[0].children())[0:2])

        cut_model = torch.nn.Sequential(*list(list(model.children())[0].children())[0:2])
        middle_tensor = cut_model(inputs)[0]
        print("before: ",middle_tensor.shape)
        middle_tensor = torch.squeeze(middle_tensor,0)
        print("squeeze: ",middle_tensor.shape)
        middle_tensor = torch.mean(middle_tensor ** 2, dim=(0,)) / c.n_feat
        #middle_tensor = torch.mean(middle_tensor, dim=(0,)) / c.n_feat
        middle_tensor = t2np(middle_tensor)
        print("after: ",middle_tensor.shape)
        plt.imshow(middle_tensor, cmap='viridis')
        plt.savefig('study_elements/middle_tensor.jpg')
        print(cut_model(inputs))