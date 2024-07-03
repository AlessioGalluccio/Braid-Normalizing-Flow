import numpy as np
import os
import torch
import torch.nn.functional as F
from torch import nn
from efficientnet_pytorch import EfficientNet
import config as c
from freia_funcs import *
import itertools

WEIGHT_DIR = './weights'
MODEL_DIR = './models/tmp'


def get_cs_flow_model(input_dim=c.n_feat):
    nodes = list()
    nodes.append(InputNode(input_dim, c.map_size[0], c.map_size[1], name='input'))
    nodes.append(InputNode(input_dim, c.map_size[0] // 2, c.map_size[1] // 2, name='input2'))
    nodes.append(InputNode(input_dim, c.map_size[0] // 4, c.map_size[1] // 4, name='input3'))

    for k in range(c.n_coupling_blocks):
        if k == 0:
            node_to_permute = [nodes[-3].out0, nodes[-2].out0, nodes[-1].out0]
        else:
            node_to_permute = [nodes[-1].out0, nodes[-1].out1, nodes[-1].out2]

        if c.permutation_mode == 'normal':
            nodes.append(Node(node_to_permute, ParallelPermute, {'seed': k}, name=F'permute_{k}'))
        elif c.permutation_mode == 'no_perm':
            nodes.append(Node(node_to_permute, NoParallelPermute, {'seed': k}, name=F'permute_{k}'))

        module_freia = __import__('freia_funcs')
        flow_type = getattr(module_freia, c.flow_type)    
        nodes.append(Node([nodes[-1].out0, nodes[-1].out1, nodes[-1].out2], flow_type,
                          {'clamp': c.clamp, 'F_class': CrossConvolutions,
                           'F_args': {'channels_hidden': c.fc_internal,
                                      'kernel_size': c.kernel_sizes[k], 'block_no': k}},
                          name=F'fc1_{k}'))

    nodes.append(OutputNode([nodes[-1].out0], name='output_end0'))
    nodes.append(OutputNode([nodes[-2].out1], name='output_end1'))
    nodes.append(OutputNode([nodes[-3].out2], name='output_end2'))
    nf = ReversibleGraphNet(nodes, n_jac=3)
    return nf

def nf_forward(model, inputs):
    # UNCOMMENT THIS ONE
    #print("Input to NF: ", inputs[0].shape)

    # nn.DataParallel wraps model and we can reach the attributes only with model.module
    if torch.cuda.device_count() > 1:
        return model.module(inputs), model.module.jacobian(run_forward=False)
    else:
        return model(inputs), model.jacobian(run_forward=False)

class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.feature_extractor = EfficientNet.from_pretrained(c.extractor)
        self.feature_extractor.eval()

    def eff_ext(self, x, use_layer=c.n_layer_extractor):
        #TL;DR this command pass x through a part of EfficientNet. From conv_steam to swish
        #_bn0 is batch normalization
        #_conv_stem is the convolution of EfficientNet
        #swish is swish function (an activation function)
        x = self.feature_extractor._swish(self.feature_extractor._bn0(self.feature_extractor._conv_stem(x)))
        #print(self.feature_extractor._blocks)
        # Blocks
        for idx, block in enumerate(self.feature_extractor._blocks):
            drop_connect_rate = self.feature_extractor._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self.feature_extractor._blocks)  # scale drop connect_rate
            x = block(x, drop_connect_rate=drop_connect_rate)
            if idx == use_layer:
                return x

    def forward(self, x):
        y = list()
        #print("in feat ",len(x))
        for s in range(c.n_scales):
            #reshape for different scales
            sizes_image = [c.img_size[0] // (2**s), c.img_size[1] // (2**s)] if s > 0 else c.img_size
            #print("Size before: ", sizes_image)
            if c.use_sidelight:
                feat_diffuse = F.interpolate(x[0], size=sizes_image)
                feat_diffuse = self.eff_ext(feat_diffuse)
                feat_side1 = F.interpolate(x[1], size=sizes_image)
                feat_side1 = self.eff_ext(feat_side1)
                feat_side2 = F.interpolate(x[2], size=sizes_image)
                feat_side2 = self.eff_ext(feat_side2)
                #create a batch of images of 3*3 channels
                #feat_s = torch.cat([feat_diffuse, feat_side1,feat_side2],1)
                feat_s = torch.cat([feat_side1,feat_side2,feat_diffuse],1)
                #feat_s = torch.cat([feat_side1,feat_side2],1)
                #print("feat_s ",feat_s.shape)
            else:
                feat_s = F.interpolate(x, size=sizes_image)
                feat_s = self.eff_ext(feat_s)
                #print("feat_s ",feat_s.shape)
                #print(sizes_image)

            y.append(feat_s)
            #print("Y: ",len(y))
        return y

class MultiModel():
    def __init__(self):
        if(not c.multi_model):
            raise ValueError('Multi model mode is not selected!')
        self.models = self._init_models()
        self.optimizers = self._init_optimizers()
        self.fes = self._init_fes()

    def _init_models(self):
        multi_models = np.empty((c.num_horizontal_patches,c.num_vertical_patches),dtype=object)
        for i in range(c.num_horizontal_patches):
            for j in range(c.num_vertical_patches):
                multi_models[i][j] = get_cs_flow_model()
        return multi_models
    
    def _init_optimizers(self):
        multi_optimizers = np.empty((c.num_horizontal_patches,c.num_vertical_patches),dtype=object)
        for i in range(c.num_horizontal_patches):
            for j in range(c.num_vertical_patches):
                multi_optimizers[i][j] = torch.optim.Adam(self.models[i][j].parameters(), lr=c.lr_init, eps=1e-04, weight_decay=1e-5)
        return multi_optimizers

    def set_multi_device(self):
        for i,j in itertools.product(range(c.num_horizontal_patches), range(c.num_vertical_patches)):
            self.models[i][j] = nn.DataParallel(self.models[i][j])
            self.fes[i][j] = nn.DataParallel(self.fes[i][j])
            for param in fe[i][j].parameters():
                param.requires_grad = False
    
    def set_device(self, device, hor=None, ver=None):
        #hor and ver permit to set a single model
        if hor is not None and ver is not None:
            #print(F"Single model {hor} {ver} set to {device}")
            self.models[hor][ver].to(device)
            self.fes[hor][ver].to(device)
        else:
            for i,j in itertools.product(range(c.num_horizontal_patches), range(c.num_vertical_patches)):
                self.models[i][j].to(device)
                self.fes[i][j].to(device)
        
    
    def _init_fes(self):
        fes = np.empty((c.num_horizontal_patches,c.num_vertical_patches),dtype=object)
        for i,j in itertools.product(range(c.num_horizontal_patches), range(c.num_vertical_patches)):
            fe = FeatureExtractor()
            fe.eval()
            fe.to(c.device)
            for param in fe.parameters():
                param.requires_grad = False
            fes[i][j] = fe
        return fes       


def save_model(model, filename):
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    torch.save(model, os.path.join(MODEL_DIR, filename))

def save_weights_multimodel(multimodel, filename, epoch):
    if not os.path.exists(WEIGHT_DIR):
        os.makedirs(WEIGHT_DIR)
    models_dict = np.empty((c.num_horizontal_patches,c.num_vertical_patches),dtype=object)
    optimizers_dict = np.empty((c.num_horizontal_patches,c.num_vertical_patches),dtype=object)
    for hor,ver in itertools.product(range(c.num_horizontal_patches), range(c.num_vertical_patches)):
        models_dict[hor,ver] = multimodel.models[hor,ver].state_dict()
        optimizers_dict[hor,ver] = multimodel.optimizers[hor,ver].state_dict()
    torch.save({
        'epoch': epoch,
        'optimizer_state_dict': optimizers_dict,
        'model_state_dict': models_dict
    }, os.path.join(WEIGHT_DIR, filename + str(epoch) + ".pth"))

def save_weights(model, optimizer, filename, epoch):
    if not os.path.exists(WEIGHT_DIR):
        os.makedirs(WEIGHT_DIR)
    torch.save({
        'epoch': epoch,
        'optimizer_state_dict': optimizer.state_dict(),
        'model_state_dict': model.state_dict()
    }, os.path.join(WEIGHT_DIR, filename + str(epoch) + ".pth"))

def load_weights(filename):
    model = get_cs_flow_model()
    model.to(c.device)
    checkpoint = torch.load(filename, map_location=c.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer = torch.optim.Adam(model.parameters(), lr=c.lr_init, eps=1e-04, weight_decay=1e-5)
    try:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    except:
        print("Can't load optimizer state")
    epoch = checkpoint['epoch']
    #model.to(c.device)
    return model, optimizer, epoch

def load_weights_multimodel(filename):
    multimodel = MultiModel()
    multimodel.set_device('cpu')
    models_dict = []
    optimizers_dict = []
    checkpoint = torch.load(filename, map_location='cpu')
    models_dict = checkpoint['model_state_dict']
    #print(multimodel.models[1,2].state_dict().keys())
    for hor,ver in itertools.product(range(c.num_horizontal_patches), range(c.num_vertical_patches)):
        multimodel.models[hor,ver].load_state_dict(models_dict[hor,ver])   
        try:
            multimodel.optimizers[hor,ver].load_state_dict(checkpoint['optimizer_state_dict'][hor,ver])
        except:
            print(F"Can't load optimizer state for {hor} {ver}")
    epoch = checkpoint['epoch']
    return multimodel, epoch


def load_model(filename):
    path = os.path.join(MODEL_DIR, filename)
    model = torch.load(path)
    return model
