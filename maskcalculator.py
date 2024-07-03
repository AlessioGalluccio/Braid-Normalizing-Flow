from pathlib import Path
from distutils.dir_util import copy_tree
import config as c
import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split
from pycocotools.coco import COCO
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
from pycocotools.coco import COCO
import itertools
import numpy as np


path_masks = "data_splitted/copertoni/tire_split/masks/anomaly"
sum_masks = np.zeros((2000,800),dtype=int)
for filename_full_mask in os.listdir(path_masks):
    im = Image.open(os.path.join(path_masks, filename_full_mask))
    im = np.array(im)
    #plt.imshow(im, cmap='plasma')
    #im.resize(sum_masks.shape)
    #sum_masks = sum_masks + im
    sum_masks[:im.shape[0],:im.shape[1]] += im
print(np.amax(sum_masks))
plt.imshow(sum_masks, cmap='plasma')
plt.savefig('study_elements/masks_sum.jpg', dpi=400)