import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve
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

def get_hist(all_maps, i, subplots, col_count):
    num_bins = 20
    flatted_tensor = all_maps[i].cpu()
    flatted_tensor = np.reshape(flatted_tensor, -1)
    flatted_tensor = np.array(flatted_tensor)
    min_value = 0.0
    max_value = 0.1
    subplots[col_count][4].hist(flatted_tensor, num_bins,range= (min_value,max_value),histtype='bar', facecolor='blue', alpha=0.5)
    subplots[col_count][4].set_ylim([0,300])

    return

def find_types_of_classification(labels, classified_as, anomaly_score, img_paths, verbose= False, create_file=False, tag= None):
    tp = []
    fp = []
    tn = []
    fn = []
    for i in range(len(labels)):
        filename = img_paths[i]
        if labels[i] == 1 and classified_as[i] == True:
            tp.append([filename, anomaly_score[i]])
        elif labels[i] == 0 and classified_as[i] == True:
            fp.append([filename, anomaly_score[i]])
        elif labels[i] == 0 and classified_as[i] == False:
            tn.append([filename, anomaly_score[i]])
        elif labels[i] == 1 and classified_as[i] == False:
            fn.append([filename, anomaly_score[i]])
    if verbose:
        print("True Positive: ", tp)
        print("False Positive: ", fp)
        print("True Negative: ", tn)
        print("False Negative: ", fn)
    if create_file:
        tp_df = pd.DataFrame(tp)
        fp_df = pd.DataFrame(fp)
        tn_df = pd.DataFrame(tn)
        fn_df = pd.DataFrame(fn)
        
        if tag is not None:
            tp_df.to_csv('study_elements/tp'+tag+'.csv', index=False)
            fp_df.to_csv('study_elements/fp'+tag+'.csv', index=False)
            tn_df.to_csv('study_elements/tn'+tag+'.csv', index=False)
            fn_df.to_csv('study_elements/fn'+tag+'.csv', index=False)
            #shutil.copy('study_elements/tp'+tag+'.csv', "data/")
            #shutil.copy('study_elements/fp'+tag+'.csv', "data/")
            #shutil.copy('study_elements/tn'+tag+'.csv', "data/")
            #shutil.copy('study_elements/fn'+tag+'.csv', "data/")
        else:
            tp_df.to_csv('study_elements/tp.csv', index=False)
            fp_df.to_csv('study_elements/fp.csv', index=False)
            tn_df.to_csv('study_elements/tn.csv', index=False)
            fn_df.to_csv('study_elements/fn.csv', index=False)
            #shutil.copy('study_elements/tp.csv', "data/")
            #shutil.copy('study_elements/fp.csv', "data/")
            #shutil.copy('study_elements/tn.csv', "data/")
            #shutil.copy('study_elements/fn.csv', "data/")
    return [x[0] for x in tp], [x[0] for x in fp], [x[0] for x in tn], [x[0] for x in fn]


def pixel_auroc(maps,img_paths):
    upscale_mode = 'bilinear'
    pixel_labels = list()
    pixel_scores = list()
    print("Computing pixel AUROC...")
    for i in tqdm(range(len(maps))):
        mask_image_location = os.path.join("data_splitted/copertoni/tire_split/masks", '/'.join(img_paths[i].split('/')[-2:]))
        if c.use_patches:
            hor, ver = patch_coordinates_from_name(mask_image_location)
            mask_image_location = original_name_from_patch(mask_image_location)
            try:
                mask_image = get_mask_patch(mask_image_location, hor, ver, c.num_horizontal_patches, c.num_vertical_patches)
            except FileNotFoundError:
                #this patch comes from an anomaly image, but it doesn't contain the anomaly
                mask_image_location = mask_image_location.replace("good", "anomaly")
                mask_image = get_mask_patch(mask_image_location, hor, ver, c.num_horizontal_patches, c.num_vertical_patches)
        else:
            mask_image = PIL.Image.open(mask_image_location)
        pixel_label = np.array(mask_image)
        pixel_label = pixel_label.flatten()
        image = PIL.Image.open(img_paths[i]).convert('RGB')
        image = np.array(image)
        map = t2np(F.interpolate(maps[i][None, None], size=image.shape[:2], mode=upscale_mode, align_corners=False))[
            0, 0]
        map = map.flatten()
        map = map.tolist()
        
        for label in pixel_label:
            if str(label) == '0' or str(label) == '3':
                pixel_labels.append(0)
            else:
                pixel_labels.append(1)
        pixel_scores.extend(map)
        '''
        for score in map:
            pixel_scores.append(score)
        '''
    pixel_labels = np.array(pixel_labels)
    pixel_scores = np.array(pixel_scores)
    print(pixel_labels)
    print(pixel_scores)
    #pixel_labels = pixel_labels.flatten()
    #pixel_scores = pixel_scores.flatten()
    print("pixel AUROC: ", roc_auc_score(pixel_labels, pixel_scores))
    print("pixel AUPRC: ", average_precision_score(pixel_labels, pixel_scores))





def plot_elements(maps, labels, img_paths, classificated_as, anomaly_score, tag_file, accepted=None, n_col=11, subsample=2, max_figures=-1):
    map_export_dir = os.path.join('./study_elements')
    os.makedirs(map_export_dir, exist_ok=True)
    upscale_mode = 'bilinear'
    plt.clf()
    fig, subplots = plt.subplots(5, n_col)
    #fig, subplots = plt.subplots(n_col,5)

    fig_count = -1
    col_count = -1
    print("Len maps: ", len(maps))
    for i in range(len(maps)):
        if i % subsample != 0:
            continue

        #if labels[i] == 0:
            #continue

        if accepted is not None and (img_paths[i] not in accepted):
            continue

        if col_count == n_col -1:
            break


        col_count = (col_count + 1) % n_col
        print("Col count ", col_count)
        if col_count == 0:
            if fig_count >= 0:
                plt.savefig(os.path.join(map_export_dir, str(fig_count) + '.jpg'), bbox_inches='tight', pad_inches=0)
                plt.close()
            fig, subplots = plt.subplots(n_col, 5, figsize=(22, 88))
            fig_count += 1
            if fig_count == max_figures:
                return
        print("pass the ", img_paths[i], "\t anomaly score: ", anomaly_score[i])


        anomaly_description = img_paths[i].split('/')[-2]
        image = PIL.Image.open(img_paths[i]).convert('RGB')
        filename = img_paths[i].split('\\')[-1].replace(".png","")
        first_name = filename.split('_')[0]
        second_name = filename.split('_')[1:-1]
        third_name = filename.split('_')[-1]
        image = np.array(image)
        map = t2np(F.interpolate(maps[i][None, None], size=image.shape[:2], mode=upscale_mode, align_corners=False))[
            0, 0]
        #import mask of the image
        #try:
        mask_image_location = os.path.join("data_splitted/copertoni/tire_split/masks", '/'.join(img_paths[i].split('/')[-2:]))
        if c.use_patches:
            hor, ver = patch_coordinates_from_name(mask_image_location)
            mask_image_location = original_name_from_patch(mask_image_location)
            try:
                mask_image = get_mask_patch(mask_image_location, hor, ver, c.num_horizontal_patches, c.num_vertical_patches)
            except FileNotFoundError:
                #this patch comes from an anomaly image, but it doesn't contain the anomaly
                mask_image_location = mask_image_location.replace("good", "anomaly")
                mask_image = get_mask_patch(mask_image_location, hor, ver, c.num_horizontal_patches, c.num_vertical_patches)
        else:
            mask_image = PIL.Image.open(mask_image_location)

        #except:
         #   print("Can't find mask of ", mask_image_location)


        subplots[col_count][0].imshow(image)
        subplots[col_count][0].axis('off')
        subplots[col_count][0].set_title(#str(first_name) +
                                        "\n" + str(second_name) +
                                         "\n" + third_name
                                         +"\n label: " + str(labels[i]) +
                                         "\n output: " + str(classificated_as[i])
                                         )
        subplots[col_count][1].imshow(map)
        subplots[col_count][1].axis('off')
        subplots[col_count][1].set_title("\n anomaly score: " + str(anomaly_score[i]))
        subplots[col_count][2].imshow(image)
        subplots[col_count][2].axis('off')
        subplots[col_count][2].imshow(map, cmap='viridis', alpha=0.3)
        subplots[col_count][3].axis('off')
        subplots[col_count][3].imshow(mask_image)
        subplots[col_count][3].set_title("\n ground truth")
        #subplots[3][col_count].axis('off')
        get_hist(maps,i, subplots, col_count)
    for i in range(col_count, n_col):
        subplots[i][0].axis('off')
        subplots[i][1].axis('off')
        subplots[i][2].axis('off')
    if col_count >= 0:
        path_image = os.path.join(map_export_dir, str(tag_file) + '.jpg')
        plt.savefig(path_image, bbox_inches='tight', pad_inches=0)
        #Export to visible area outside docker
        #dst = "data/"
        #shutil.copy(path_image, dst)
        img = PIL.Image.open(path_image)
        convert_tensor = transforms.ToTensor()
        return convert_tensor(img)
    return


def inspect_images(model_weights_path, image_loader, image_set):
    if c.multi_model:
        multi_models, _ = load_weights_multimodel(model_weights_path)
        multi_models.set_device(c.device)
        for hor,ver in itertools.product(range(c.num_horizontal_patches), range(c.num_vertical_patches)):
            multi_models.models[hor,ver].eval()
    else:
        model, _ , _ = load_weights(model_weights_path)
        model.to(c.device)
        model.eval()
    anomaly_score = list()
    test_labels = list()
    all_maps = list()
    all_value_tensors = list()

    all_maps_0 = list()
    all_maps_1 = list()
    all_maps_2 = list()


    if not c.pre_extracted and not c.multi_model:
        fe = FeatureExtractor()
        fe.eval()
        fe.to(c.device)
        for param in fe.parameters():
            param.requires_grad = False

    if c.multi_model:
        for hor,ver in itertools.product(range(c.num_horizontal_patches), range(c.num_vertical_patches)):
            print(F"Analyzing model {hor} {ver}")
            model = multi_models.models[hor,ver]
            fe = multi_models.fes[hor,ver]
            anomaly_score = list()
            test_labels = list()
            all_maps = list()
            all_value_tensors = list()

            all_maps_0 = list()
            all_maps_1 = list()
            all_maps_2 = list()
            with torch.no_grad():
                for i, data in enumerate(tqdm(image_loader[hor,ver], disable=c.hide_tqdm_bar)):
                    inputs, labels = preprocess_batch(data)
                    if not c.pre_extracted:
                        inputs = fe(inputs)
                    z = model(inputs)
                    z_concat = t2np(concat_maps(z))
                    nll_score = np.mean(z_concat ** 2, axis=(1, 2))
                    anomaly_score.append(nll_score)
                    test_labels.append(t2np(labels))
                    z_grouped = list()
                    likelihood_grouped = list()
                    for i in range(len(z)):
                        #print("z ", str(i), "shape: ", z[i].shape)
                        z_grouped.append(z[i].view(-1, *z[i].shape[1:]))
                        likelihood_grouped.append(torch.mean(z_grouped[-1] ** 2, dim=(1,)) / c.n_feat)
                    #all_maps.extend(likelihood_grouped[0])
                    '''
                    print("likelihood 0: ", likelihood_grouped[0].shape)
                    print("likelihood 1: ", likelihood_grouped[1].shape)
                    print("likelihood 2: ", likelihood_grouped[2].shape)
                    print("likelihood 1 unsqueeze: ", torch.unsqueeze(likelihood_grouped[1], 1).shape)
                    '''
                    # I want to combine all three sizes in a single map
                    # Interpolate requires one more dimension
                    likelihood_1_upsqueezed = torch.unsqueeze(likelihood_grouped[1], 1)
                    likelihood_2_upsqueezed = torch.unsqueeze(likelihood_grouped[2], 1)
                    likelihood_1_up = nn.functional.interpolate(likelihood_1_upsqueezed, size=[c.map_size[0], c.map_size[1]])
                    likelihood_2_up = nn.functional.interpolate(likelihood_2_upsqueezed, size=[c.map_size[0], c.map_size[1]])
                    # Remove useless dimension
                    likelihood_1_up = torch.squeeze(likelihood_1_up, 1)
                    likelihood_2_up = torch.squeeze(likelihood_2_up, 1)
                    '''
                    print("likelihood 1 upscaled: ", likelihood_1_up.shape)
                    print("likelihood 2 upscaled: ", likelihood_2_up.shape)
                    '''
                    len_likel_0 = len(likelihood_grouped[0])
                    len_likel_1 = len(likelihood_1_up)
                    len_likel_2 = len(likelihood_1_up)
                    #weighted average
                    likelihood_mean = ((likelihood_grouped[0] * len_likel_0) + (likelihood_1_up * len_likel_1) + (likelihood_2_up * len_likel_2)) / (len_likel_0 + len_likel_1 + len_likel_2)
                    '''
                    print("likelihood 0: ", likelihood_grouped[0])
                    '''
                    all_maps.extend(likelihood_mean)
            anomaly_score = np.concatenate(anomaly_score)
            test_labels = np.concatenate(test_labels)
            false_positive_rate, true_positive_rate, threshold_array = roc_curve(test_labels, anomaly_score)
            print("AUROC: ", roc_auc_score(test_labels, anomaly_score))
            print("AUPRC: ", average_precision_score(test_labels, anomaly_score))
            threshold, gmean, tp_rate_best, fp_rate_best = find_threshold_auroc(threshold_array, true_positive_rate, false_positive_rate)
            print('Best threshold: {:.4f} \t Geometric mean: {:.4f}'.format(threshold, gmean))
            classificated_as = anomaly_score > threshold

            test_labels = np.array([1 if l > 0 else 0 for l in test_labels])

            if c.use_sidelight:
                img_path_to_iterate = [p for p, l in image_set[hor,ver].diffuse.samples]
            else:
                img_path_to_iterate = [p for p, l in image_set[hor,ver].samples]

            tp, fp, tn, fn = find_types_of_classification(test_labels, classificated_as, anomaly_score, img_path_to_iterate, verbose=False, create_file=True, tag=str(hor)+'_'+str(ver))

            plot_elements(all_maps, test_labels, img_path_to_iterate, classificated_as,anomaly_score, 'study'+str(hor)+'_'+str(ver), accepted=fp)


            

            #pixel_auroc(all_maps, img_path_to_iterate)

            plt.subplots(1, figsize=(10,10))
            plt.title('Receiver Operating Characteristic - cs-flow')
            plt.plot(false_positive_rate, true_positive_rate)
            plt.plot([0, 1], ls="--")
            plt.plot([0, 0], [1, 0] , c=".7"), plt.plot([1, 1] , c=".7")
            plt.ylabel('True Positive Rate')
            plt.xlabel('False Positive Rate')
            plt.savefig('study_elements/auroc'+str(hor)+'_'+str(ver)+'.jpg', bbox_inches='tight', pad_inches=0)
            shutil.copy('study_elements/auroc'+str(hor)+'_'+str(ver)+'.jpg', "data/")

            precision, recall, threshold1 = precision_recall_curve(test_labels, anomaly_score)
            plt.subplots(1, figsize=(10,10))
            plt.title('Precision Recall Characteristic - cs-flow')
            plt.plot(recall, precision)
            plt.plot([0, 1], ls="--")
            plt.plot([0, 0], [1, 0] , c=".7"), plt.plot([1, 1] , c=".7")
            plt.ylabel('Precision')
            plt.xlabel('Recall')
            plt.savefig('study_elements/auprc'+str(hor)+'_'+str(ver)+'.jpg', bbox_inches='tight', pad_inches=0)
            shutil.copy('study_elements/auprc'+str(hor)+'_'+str(ver)+'.jpg', "data/")

    else:
        with torch.no_grad():
            for i, data in enumerate(tqdm(image_loader, disable=c.hide_tqdm_bar)):
                inputs, labels = preprocess_batch(data)
                if not c.pre_extracted:
                    inputs = fe(inputs)
                z = model(inputs)
                z_concat = t2np(concat_maps(z))
                nll_score = np.mean(z_concat ** 2, axis=(1, 2))
                anomaly_score.append(nll_score)
                test_labels.append(t2np(labels))
                z_grouped = list()
                likelihood_grouped = list()
                for i in range(len(z)):
                    #print("z ", str(i), "shape: ", z[i].shape)
                    z_grouped.append(z[i].view(-1, *z[i].shape[1:]))
                    likelihood_grouped.append(torch.mean(z_grouped[-1] ** 2, dim=(1,)) / c.n_feat)
                #all_maps.extend(likelihood_grouped[0])
                '''
                print("likelihood 0: ", likelihood_grouped[0].shape)
                print("likelihood 1: ", likelihood_grouped[1].shape)
                print("likelihood 2: ", likelihood_grouped[2].shape)
                print("likelihood 1 unsqueeze: ", torch.unsqueeze(likelihood_grouped[1], 1).shape)
                '''
                # I want to combine all three sizes in a single map
                # Interpolate requires one more dimension
                likelihood_1_upsqueezed = torch.unsqueeze(likelihood_grouped[1], 1)
                likelihood_2_upsqueezed = torch.unsqueeze(likelihood_grouped[2], 1)
                likelihood_1_up = nn.functional.interpolate(likelihood_1_upsqueezed, size=[c.map_size[0], c.map_size[1]])
                likelihood_2_up = nn.functional.interpolate(likelihood_2_upsqueezed, size=[c.map_size[0], c.map_size[1]])
                # Remove useless dimension
                likelihood_1_up = torch.squeeze(likelihood_1_up, 1)
                likelihood_2_up = torch.squeeze(likelihood_2_up, 1)
                '''
                print("likelihood 1 upscaled: ", likelihood_1_up.shape)
                print("likelihood 2 upscaled: ", likelihood_2_up.shape)
                '''
                len_likel_0 = len(likelihood_grouped[0])
                len_likel_1 = len(likelihood_1_up)
                len_likel_2 = len(likelihood_1_up)
                #weighted average
                likelihood_mean = ((likelihood_grouped[0] * len_likel_0) + (likelihood_1_up * len_likel_1) + (likelihood_2_up * len_likel_2)) / (len_likel_0 + len_likel_1 + len_likel_2)
                '''
                print("likelihood 0: ", likelihood_grouped[0])
                '''
                all_maps.extend(likelihood_mean)

        anomaly_score = np.concatenate(anomaly_score)
        test_labels = np.concatenate(test_labels)
        false_positive_rate, true_positive_rate, threshold_array = roc_curve(test_labels, anomaly_score)
        print("AUROC: ", roc_auc_score(test_labels, anomaly_score))
        print("AUPRC: ", average_precision_score(test_labels, anomaly_score))
        threshold, gmean, tp_rate_best, fp_rate_best = find_threshold_auroc(threshold_array, true_positive_rate, false_positive_rate)
        print('Best threshold: {:.4f} \t Geometric mean: {:.4f}'.format(threshold, gmean))
        classificated_as = anomaly_score > threshold

        test_labels = np.array([1 if l > 0 else 0 for l in test_labels])

        if c.use_sidelight:
            img_path_to_iterate = [p for p, l in image_set.diffuse.samples]
        else:
            img_path_to_iterate = [p for p, l in image_set.samples]

        tp, fp, tn, fn = find_types_of_classification(test_labels, classificated_as, anomaly_score, img_path_to_iterate, verbose=False, create_file=True)

        plot_elements(all_maps, test_labels, img_path_to_iterate, classificated_as,anomaly_score, "study", accepted=tn)


        #pixel_auroc(all_maps, img_path_to_iterate)

        plt.subplots(1, figsize=(10,10))
        plt.title('Receiver Operating Characteristic - cs-flow')
        plt.plot(false_positive_rate, true_positive_rate)
        plt.plot([0, 1], ls="--")
        plt.plot([0, 0], [1, 0] , c=".7"), plt.plot([1, 1] , c=".7")
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.savefig('study_elements/auroc.jpg', bbox_inches='tight', pad_inches=0)
        shutil.copy('study_elements/auroc.jpg', "data/")

        precision, recall, threshold1 = precision_recall_curve(test_labels, anomaly_score)
        plt.subplots(1, figsize=(10,10))
        plt.title('Precision Recall Characteristic - cs-flow')
        plt.plot(recall, precision)
        plt.plot([0, 1], ls="--")
        plt.plot([0, 0], [1, 0] , c=".7"), plt.plot([1, 1] , c=".7")
        plt.ylabel('Precision')
        plt.xlabel('Recall')
        plt.savefig('study_elements/auprc.jpg', bbox_inches='tight', pad_inches=0)
        shutil.copy('study_elements/auprc.jpg', "data/")

        if c.multi_model:
            #create all tp0_0, tp0_1,... files
            score_separate, label_separate, order = patch_separate_evaluation(anomaly_score, test_labels, img_path_to_iterate)
            classificate_separate = np.empty((c.num_horizontal_patches,c.num_vertical_patches),dtype=object)
            for hor, ver in itertools.product(range(c.num_horizontal_patches), range(c.num_vertical_patches)):
                classificate_separate[hor,ver] = []
            for i in range(len(classificated_as)):
                hor, ver = order[i]
                classificate_separate[hor,ver].append(classificated_as[i])
            for hor, ver in itertools.product(range(c.num_horizontal_patches), range(c.num_vertical_patches)):
                #select only path that are correct
                img_path_separate = []
                for path in img_path_to_iterate:
                    if "_P_"+str(hor)+"_"+str(ver) in path:
                        img_path_separate.append(path)
                tp, fp, tn, fn = find_types_of_classification(label_separate[hor,ver], classificate_separate[hor,ver], score_separate[hor,ver], img_path_separate, verbose=False, create_file=True, tag=str(hor)+'_'+str(ver))


    return



#inspect_images("weights/tire_model5.pth",validation_loader, validation_set, threshold=1.593)
#inspect_images("exps-samele/tire_model9999_braid_multimodel_rad1_rad2_diff.pth",validation_loader, validation_set)
inspect_images("exps-samele/tire_model9999_braid_multimodel_rad1_rad2_diff.pth",test_loader, test_set)