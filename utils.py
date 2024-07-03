import os
import torch
from torchvision import datasets, transforms
import config as c
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt
import PIL
import torch.nn.functional as F
import itertools


def t2np(tensor):
    '''pytorch tensor -> numpy array'''
    return tensor.cpu().data.numpy() if tensor is not None else None


def flat(tensor):
    return tensor.reshape(tensor.shape[0], -1)


def concat_maps(maps):
    flat_maps = list()
    for m in maps:
        flat_maps.append(flat(m))
    return torch.cat(flat_maps, dim=1)[..., None]


def get_loss(z, jac):
    z = torch.cat([z[i].reshape(z[i].shape[0], -1) for i in range(len(z))], dim=1)
    jac = sum(jac)
    return torch.mean(0.5 * torch.sum(z ** 2, dim=(1,)) - jac) / z.shape[1]


def cat_maps(z):
    return torch.cat([z[i].reshape(z[i].shape[0], -1) for i in range(len(z))], dim=1)

def is_valid_multi_model(x: str, hor, ver) -> bool:
    tag = 'P_' + str(hor) + '_' + str(ver) + '_'
    if tag in x:
        return True
    else:
        return False



def load_datasets(dataset_path, class_name):
    '''
    Expected folder/file format to find anomalies of class <class_name> from dataset location <dataset_path>:

    train data:

            dataset_path/class_name/train/good/any_filename.png
            dataset_path/class_name/train/good/another_filename.tif
            dataset_path/class_name/train/good/xyz.png
            [...]

    test data:

        'normal data' = non-anomalies

            dataset_path/class_name/test/good/name_the_file_as_you_like_as_long_as_there_is_an_image_extension.webp
            dataset_path/class_name/test/good/did_you_know_the_image_extension_webp?.png
            dataset_path/class_name/test/good/did_you_know_that_filenames_may_contain_question_marks????.png
            dataset_path/class_name/test/good/dont_know_how_it_is_with_windows.png
            dataset_path/class_name/test/good/just_dont_use_windows_for_this.png
            [...]

        anomalies - assume there are anomaly classes 'crack' and 'curved'

            dataset_path/class_name/test/crack/dat_crack_damn.png
            dataset_path/class_name/test/crack/let_it_crack.png
            dataset_path/class_name/test/crack/writing_docs_is_fun.png
            [...]

            dataset_path/class_name/test/curved/wont_make_a_difference_if_you_put_all_anomalies_in_one_class.png
            dataset_path/class_name/test/curved/but_this_code_is_practicable_for_the_mvtec_dataset.png
            [...]
    '''

    def target_transform(target):
        return class_perm[target]

    if c.pre_extracted:
        trainset = FeatureDataset(train=True)
        testset = FeatureDataset(train=False)
        validationset = FeatureDataset(train=False)
    else:
        data_dir_train = os.path.join(dataset_path, class_name, 'train')
        data_dir_test = os.path.join(dataset_path, class_name, 'test')

        classes = os.listdir(data_dir_test)
        if 'good' not in classes:
            print('There should exist a subdirectory "good". Read the doc of this function for further information.')
            exit()
        classes.sort()
        class_perm = list()
        class_idx = 1
        for cl in classes:
            if cl == 'good':
                class_perm.append(0)
            else:
                class_perm.append(class_idx)
                class_idx += 1

        tfs = [transforms.Resize(c.img_size), transforms.ToTensor(), transforms.Normalize(c.norm_mean, c.norm_std) ]
        transform_train = transforms.Compose(tfs)

        if(c.use_sidelight):
            trainset = DiffuseAndSideLightDataset(data_dir_train, transform=transform_train)
            testset = DiffuseAndSideLightDataset(data_dir_test, transform=transform_train, target_transform=target_transform)
            if c.multi_model:
                    trainset = np.empty((c.num_horizontal_patches,c.num_vertical_patches),dtype=object)
                    testset = np.empty((c.num_horizontal_patches,c.num_vertical_patches),dtype=object)
                    for hor, ver in itertools.product(range(c.num_horizontal_patches), range(c.num_vertical_patches)):
                        trainset[hor][ver] = DiffuseAndSideLightDataset(data_dir_train, transform=transform_train, is_valid_file=lambda x : is_valid_multi_model(x,hor,ver))
                        testset[hor][ver] = DiffuseAndSideLightDataset(data_dir_test, transform=transform_train, target_transform=target_transform, is_valid_file=lambda x : is_valid_multi_model(x,hor,ver))

        else:
            trainset = ImageFolder(data_dir_train, transform=transform_train)
            testset = ImageFolder(data_dir_test, transform=transform_train, target_transform=target_transform)
            if c.multi_model:
                    trainset = np.empty((c.num_horizontal_patches,c.num_vertical_patches),dtype=object)
                    testset = np.empty((c.num_horizontal_patches,c.num_vertical_patches),dtype=object)
                    for hor, ver in itertools.product(range(c.num_horizontal_patches), range(c.num_vertical_patches)):
                        trainset[hor][ver] = ImageFolder(data_dir_train, transform=transform_train, is_valid_file=lambda x : is_valid_multi_model(x,hor,ver))
                        testset[hor][ver] = ImageFolder(data_dir_test, transform=transform_train, target_transform=target_transform, is_valid_file=lambda x : is_valid_multi_model(x,hor,ver))



        try:
            data_dir_validation = os.path.join(dataset_path, class_name, 'validation')
            classes = os.listdir(data_dir_validation)
            if 'good' not in classes:
                print('There should exist a subdirectory "good" in validation. Read the doc of this function for further information.')
                exit()
            classes.sort()
            class_perm = list()
            class_idx = 1
            for cl in classes:
                if cl == 'good':
                    class_perm.append(0)
                else:
                    class_perm.append(class_idx)
                    class_idx += 1
            if(c.use_sidelight):
                validationset = DiffuseAndSideLightDataset(data_dir_validation, transform=transform_train, target_transform=target_transform)
                if c.multi_model:
                    validationset = np.empty((c.num_horizontal_patches,c.num_vertical_patches),dtype=object)
                    for hor, ver in itertools.product(range(c.num_horizontal_patches), range(c.num_vertical_patches)):
                        validationset[hor][ver] = DiffuseAndSideLightDataset(data_dir_validation, transform=transform_train, target_transform=target_transform,
                         is_valid_file=lambda x : is_valid_multi_model(x,hor,ver))
                        
            else:
                validationset = ImageFolder(data_dir_validation, transform=transform_train, target_transform=target_transform)
                if c.multi_model:
                    validationset = np.empty((c.num_horizontal_patches,c.num_vertical_patches),dtype=object)
                    for hor, ver in itertools.product(range(c.num_horizontal_patches), range(c.num_vertical_patches)):
                        validationset[hor][ver] = ImageFolder(data_dir_validation, transform=transform_train, target_transform=target_transform,
                         is_valid_file=lambda x : is_valid_multi_model(x,hor,ver))
                        

        except FileNotFoundError:
            print("No validation dataset avaliable")
            validationset = None
    #print(trainset, testset, validationset)
    return trainset, testset, validationset


class FeatureDataset(Dataset):
    def __init__(self, root="data/features/" + c.class_name + '/', n_scales=c.n_scales, train=False):

        super(FeatureDataset, self).__init__()
        self.data = list()
        self.n_scales = n_scales
        self.train = train
        suffix = 'train' if train else 'test'

        for s in range(c.n_scales):
            self.data.append(np.load(root + c.class_name + '_scale_' + str(s) + '_' + suffix + '.npy'))

        self.labels = np.load(os.path.join(root, c.class_name + '_labels.npy')) if not train else np.zeros(
            [len(self.data[0])])
        self.paths = np.load(os.path.join(root, c.class_name + '_image_paths.npy'))
        self.class_names = [img_path.split('/')[-2] for img_path in self.paths]

    def __len__(self):
        return len(self.data[0])

    def __getitem__(self, index):
        out = list()
        for d in self.data:
            sample = d[index]
            sample = torch.FloatTensor(sample)
            out.append(sample)
        return out, self.labels[index]

class DiffuseAndSideLightDataset(Dataset):
    def __init__(self,data_dir, transform, target_transform=None, is_valid_file=None):
        #print(data_dir)
        side_directory = os.path.join(data_dir.replace(c.dataset_path, "sidelight/"))
        side1_directory = side_directory.replace(c.class_name, c.class_name + "/1")
        side2_directory = side_directory.replace(c.class_name, c.class_name + "/2")
        if target_transform == None:
            self.diffuse=ImageFolder(data_dir, transform=transform, is_valid_file=is_valid_file)
            self.side1=ImageFolder(side1_directory, transform=transform, is_valid_file=is_valid_file)
            self.side2=ImageFolder(side2_directory, transform=transform, is_valid_file=is_valid_file)
        else:
            self.diffuse=ImageFolder(data_dir, transform=transform, target_transform=target_transform, is_valid_file=is_valid_file)
            self.side1=ImageFolder(side1_directory, transform=transform, target_transform=target_transform, is_valid_file=is_valid_file)
            self.side2=ImageFolder(side2_directory, transform=transform, target_transform=target_transform, is_valid_file=is_valid_file)
            print("Diffuse init len: ",self.diffuse)
  
    def __len__(self):
        return(len(self.diffuse))
    
    def __getitem__(self,idx):
        #I return the array of diffuse, side1, side2 and the label value
        # in second dimension 0 is image, 1 is label
        diffuse_value_label = self.diffuse[idx]
        #print("Side general: ", self.side1)
        '''
        try:
            side1 = self.side1[idx][0]
        except IndexError:
            print("Index error in: ", idx)
            print("Diffuse general: ", self.diffuse)
            print("Diffuse content: ", self.diffuse[idx])
            print("Side general: ", self.side1)
            print("Side content previous: ", self.side1[idx-1])
            print("Side content: ", self.side1[idx])
        side2 = self.side2[idx][0]
        label = diffuse_value_label[1]
        '''
        return [diffuse_value_label[0], self.side1[idx][0], self.side2[idx][0]], diffuse_value_label[1]



def make_dataloaders(trainset, testset, validationset=None):
    if c.multi_model:
        trainloader = np.empty((c.num_horizontal_patches,c.num_vertical_patches),dtype=object)
        validationloader = np.empty((c.num_horizontal_patches,c.num_vertical_patches),dtype=object)
        testloader = np.empty((c.num_horizontal_patches,c.num_vertical_patches),dtype=object)
        #print(trainloader)
        for hor, ver in itertools.product(range(c.num_horizontal_patches), range(c.num_vertical_patches)):
            trainloader[hor][ver] = torch.utils.data.DataLoader(trainset[hor][ver], pin_memory=True, batch_size=c.batch_size, shuffle=True,
                                                drop_last=False)
            testloader[hor][ver] = torch.utils.data.DataLoader(testset[hor][ver], pin_memory=True, batch_size=c.batch_size, shuffle=False,
                                                drop_last=False)
            validationloader[hor][ver] = torch.utils.data.DataLoader(validationset[hor][ver], pin_memory=True, batch_size=c.batch_size,
                                                        shuffle=False, drop_last=False)
            print(F"validation set {hor} {ver} len: ",len(validationset[hor][ver]))

    else:
        trainloader = torch.utils.data.DataLoader(trainset, pin_memory=True, batch_size=c.batch_size, shuffle=True,
                                                drop_last=False)
        testloader = torch.utils.data.DataLoader(testset, pin_memory=True, batch_size=c.batch_size, shuffle=False,
                                                drop_last=False)
        if validationset is not None:
            validationloader = torch.utils.data.DataLoader(validationset, pin_memory=True, batch_size=c.batch_size,
                                                        shuffle=False, drop_last=False)
        else:
            validationloader = None
    return trainloader, testloader, validationloader


def preprocess_batch(data):
    if c.use_sidelight:
        '''move data to device and reshape image'''
        if c.pre_extracted:
            inputs, labels = data
            for i in range(len(inputs)):
                for j in range(len(inputs[i])):
                    inputs[i][j] = inputs[i][j].to(c.device)
            labels = labels.to(c.device)
        else:
            inputs, labels = data
            #print(len(inputs))
            #print(inputs[0].shape) 
            #print("Len inputs ", len(inputs))
            #print("labels ", labels)
            for type_image in range(len(inputs)):
                #print("Len inputs i ",len(inputs[type_image]))
                #print("Len inputs i 0: ",len(inputs[i][0]))
                #print("Len inputs i 1: ",len(inputs[i][1]))
                #print("inputs i 1: ",inputs[i][1])

                inputs[type_image] = inputs[type_image].to(c.device)
                inputs[type_image] = inputs[type_image].view(-1, *inputs[type_image].shape[-3:])
            labels = labels.to(c.device) 

    else:
        '''move data to device and reshape image'''
        if c.pre_extracted:
            inputs, labels = data
            for i in range(len(inputs)):
                inputs[i] = inputs[i].to(c.device)
            labels = labels.to(c.device)
        else:
            inputs, labels = data
            #print("Len inputs ", len(inputs))
            inputs, labels = inputs.to(c.device), labels.to(c.device)
            inputs = inputs.view(-1, *inputs.shape[-3:])


    return inputs, labels


class Score_Observer:
    '''Keeps an eye on the current and highest score so far'''

    def __init__(self, name):
        self.name = name
        self.max_epoch = 0
        self.max_score = None
        self.min_loss_epoch = 0
        self.min_loss_score = 0
        self.min_loss = None
        self.last = None

    def update(self, score, epoch, print_score=False):
        self.last = score
        if self.max_score == None or score > self.max_score:
            self.max_score = score
            self.max_epoch = epoch
        if print_score:
            self.print_score()

    def print_score(self):
        print('{:s}: \t last: {:.4f} \t max: {:.4f} \t epoch_max: {:d} \t epoch_loss: {:d}'.format(self.name, self.last,
                                                                                                   self.max_score,
                                                                                                   self.max_epoch,
                                                                                                   self.min_loss_epoch))
# G-Mean = sqrt(Sensitivity * Specificity)
# maximise this to obtain the best threshold for AUROC
def gmeans(true_positive_rate, false_positive_rate):
    return np.sqrt(true_positive_rate * (1-false_positive_rate))

# locate the index of the largest g-mean
def find_threshold_auroc(thresholds, true_positive_rate, false_positive_rate):
    gmeans_array = gmeans(true_positive_rate,false_positive_rate)
    index = np.argmax(gmeans_array)
    return thresholds[index], gmeans_array[index], true_positive_rate[index], false_positive_rate[index]

def get_name_from_label(image_set, index_number):
    if c.use_sidelight:
        img_names = [p for p, l in image_set.diffuse.samples]
    else:
        img_names = [p for p, l in image_set.samples]

    return img_names[i]

def original_name_from_patch(name_no_side):
    return name_no_side[:-23] + name_no_side[-17:]

def patch_coordinates_from_name(name_no_side):
    hor = name_no_side[-20]
    ver = name_no_side[-18]
    return int(hor), int(ver) 

class PatchGroupCollector():
    def __init__(self, image_set):
        self.image_set = image_set
        if c.use_sidelight:
            self.img_names = [p for p, l in image_set.diffuse.samples]
        else:
            self.img_names = [p for p, l in image_set.samples]
        self._create_dictionary()
        self.patch_groups = self._create_list_patches_index()

    def _create_dictionary(self):
        self.img_index_dictionary = {}
        for i in range(len(self.img_names)):
            name = self.img_names[i]
            #to remove P_0_1_
            if c.use_sidelight:
                name_no_side = name[:-8] + name[-4:]
                original_img_name = original_name_from_patch(name_no_side)
            else:
                original_img_name = original_name_from_patch(name)
            #remove folders
            original_img_name = original_img_name.split('/')[-1]
            if original_img_name not in self.img_index_dictionary:
                self.img_index_dictionary[original_img_name] = list()
            self.img_index_dictionary[original_img_name].append(i)

    def _create_list_patches_index(self):
        patch_groups = []
        for key in self.img_index_dictionary:
            patch_groups.append(self.img_index_dictionary[key])
        return patch_groups

def patch_grouped_evaluation(scores, labels, patch_groups):
    anomaly_indexes = []
    for i in range(len(labels)):
        if labels[i] > 0:
            anomaly_indexes.append(i)
    
    labels_patched = []
    for group in patch_groups:
        if bool(set(group) & set(anomaly_indexes)):
            labels_patched.append(1)
        else:
            labels_patched.append(0)

    scores_patched = []
    for group in patch_groups:
        scores_group = []
        for index in group:
            scores_group.append(scores[index])
        # I take the max of the scores of the patches
        scores_patched.append(max(scores_group))

    return scores_patched, labels_patched

def patch_separate_evaluation(scores, labels, img_names):
    
    score_separate = np.empty((c.num_horizontal_patches,c.num_vertical_patches),dtype=object)
    label_separate = np.empty((c.num_horizontal_patches,c.num_vertical_patches),dtype=object)
    order = [] #will contain tuple (hor,ver) for each i
    for hor, ver in itertools.product(range(c.num_horizontal_patches), range(c.num_vertical_patches)):
        score_separate[hor,ver] = []
        label_separate[hor,ver] = []
    for i in range(len(img_names)):
        name = img_names[i]
        #print(name)
        #remove sidelight number
        '''
        if c.use_sidelight:
            name = name[:-8] + name[-4:]
        '''
        #get patch string _P_0_0
        patch_string = name[-23:-17]
        hor = int(patch_string[-3])
        ver = int(patch_string[-1])
        score_separate[hor,ver].append(scores[i])
        label_separate[hor,ver].append(labels[i])
        order.append([hor,ver]) 
    return score_separate, label_separate, order






def plot_array_images(maps, labels, img_paths, tag_file, n_col=8, subsample=4, max_figures=-1):
    map_export_dir = os.path.join('./image_validation/maps/', c.modelname)
    os.makedirs(map_export_dir, exist_ok=True)
    upscale_mode = 'bilinear'
    plt.clf()
    fig, subplots = plt.subplots(3, n_col)

    fig_count = -1
    col_count = -1
    for i in range(len(maps)):
        if i % subsample != 0:
            continue

        if labels[i] == 0:
            continue

        col_count = (col_count + 1) % n_col
        if col_count == 0:
            if fig_count >= 0:
                plt.savefig(os.path.join(map_export_dir, str(fig_count) + '.jpg'), bbox_inches='tight', pad_inches=0)
                plt.close()
            fig, subplots = plt.subplots(3, n_col, figsize=(22, 8))
            fig_count += 1
            if fig_count == max_figures:
                return

        anomaly_description = img_paths[i].split('/')[-2]
        image = PIL.Image.open(img_paths[i]).convert('RGB')
        image = np.array(image)
        map = t2np(F.interpolate(maps[i][None, None], size=image.shape[:2], mode=upscale_mode, align_corners=False))[
            0, 0]
        subplots[1][col_count].imshow(map)
        subplots[1][col_count].axis('off')
        subplots[0][col_count].imshow(image)
        subplots[0][col_count].axis('off')
        subplots[0][col_count].set_title(c.class_name + ":\n" + anomaly_description)
        subplots[2][col_count].imshow(image)
        subplots[2][col_count].axis('off')
        subplots[2][col_count].imshow(map, cmap='viridis', alpha=0.3)
    for i in range(col_count, n_col):
        subplots[0][i].axis('off')
        subplots[1][i].axis('off')
        subplots[2][i].axis('off')
    if col_count >= 0:
        path_image = os.path.join(map_export_dir, str(fig_count) + '_' + str(tag_file) + '.jpg')
        plt.savefig(path_image, bbox_inches='tight', pad_inches=0)
        img = PIL.Image.open(path_image)
        convert_tensor = transforms.ToTensor()
        return convert_tensor(img)
    return
