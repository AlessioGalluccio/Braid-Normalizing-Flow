'''This file configures the training procedure because handling arguments in every single function is so exhaustive for
research purposes. Don't try this code if you are a software engineer.'''

# device settings
device = 'cuda'  # or 'cpu'

#seed
seed = 42

# neptune
neptune_activate = False
# data settings
from_object_dataset = False
dataset_path = "data_splitted/copertoni/"
class_name = "object_split"
modelname = "object_model"
pre_extracted = False  # were feature preextracted with extract_features?

img_size = (256, 256)  # dimage size of highest scale, others are //2, //4
img_dims = [3] + list(img_size)

# transformation settings
norm_mean, norm_std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

# network hyperparameters
n_scales = 3  # number of scales at which features are extracted, img_size is the highest - others are //2, //4,...
clamp = 3  # clamping parameter
max_grad_norm = 1e0  # clamp gradients to this norm
n_coupling_blocks = 2  # higher = more flexible = more unstable
fc_internal = 1024  # * 4 # number of neurons in hidden layers of s-t-networks
lr_init = 2e-4  # inital learning rate
use_gamma = True


# flow types
# MODES: 
# 'parallel_glow_coupling_layer'   standard cs-flow   
# 'triple_cross_coupling_layer'     braid cs-flow
flow_type = "triple_cross_coupling_layer"
extractor = 'efficientnet-b0'  # feature dataset name (which was used in 'extract_features.py' as 'export_name')
#n_feat = {"efficientnet-b5": 304, "efficientnet-b7": 384}[extractor]  # dependend from feature extractor
#n_feat = 912 #b5 sidelight
#n_feat = 1152 #b7 sidelight
n_feat = 336 #b0 sidelight
#n_feat = 112 #b0 no sidelight
n_layer_extractor = {"efficientnet-b0": 8,"efficientnet-b5": 36, "efficientnet-b7": 51}[extractor]
map_size = (img_size[0] // 16, img_size[1] // 16)

# dataloader parameters
batch_size = 64  # actual batch size is this value multiplied by n_transforms(_test)
kernel_sizes = [3] * (n_coupling_blocks - 1) + [1]

# total epochs = meta_epochs * sub_epochs
# evaluation after <sub_epochs> epochs
meta_epochs = 100  # total epochs = meta_epochs * sub_epochs
sub_epochs = 1  # evaluate after this number of epochs

load_previous_weight = False
weight_location = "weights/object_model9999.pth"

#sidelight options
use_sidelight = True

#patches options
use_patches = True
patches_mod = 'itself' # None,'itself', 'group'
num_vertical_patches = 6
num_horizontal_patches = 2
overlap_hor = 0
overlap_ver = 0

#multi model
multi_model = False
if not use_patches:
    multi_model = False

# permutation modes of normalizing flows channels
# MODES: 
# 'normal'      permutation is done as in paper
# 'no_perm'     channels are not permutated
permutation_mode = 'no_perm'


# output settings
verbose = True
hide_tqdm_bar = False
save_model = True
save_only_bests = False
