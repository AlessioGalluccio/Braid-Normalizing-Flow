import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve
from tqdm import tqdm
import config as c
from model import *
from utils import *

import neptune.new as neptune # comment this statement if you don't use neptune
import config as c
import neptuneparams as nep_params # comment this statement if you don't use neptune
from neptune.new.types import File # comment this statement if you don't use neptune

import matplotlib.pyplot as plt

from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()
for i in range(10):
    writer.add_scalar('y=2x', i * 2, i)

# Neptune.ai set up, in order to keep track of your experiments
if c.neptune_activate:
    run = neptune.init(
        project = nep_params.project,
        api_token = nep_params.api_token,
    )  # your credentials

    run["name_dataset"] = [c.dataset_path]
    run["img_dims"] = [c.img_dims]
    run["device"] = c.device
    run["n_scales"] = c.n_scales
    run["class_name"] = [c.class_name]
    run["meta_epochs"] = c.meta_epochs
    run["sub_epochs"] = c.sub_epochs
    run["batch_size"]= c.batch_size
    run["n_coupling_blocks"] = c.n_coupling_blocks
    run["learning_rate"] = c.lr_init
    run["fc_internal"] = c.fc_internal
    run["seed"] = c.seed
    run["extractor"] = c.extractor
    run["n_layer_extractor"] = c.n_layer_extractor
    run["norm_mean"] = c.norm_mean
    run["norm_std"] = c.norm_std
    run["map_size"] = c.map_size
    run["use_sidelight"] = c.use_sidelight
    run["n_feat"] = c.n_feat
    run["use_patches"] = c.use_patches 
    run["patches_mod"] = c.patches_mod
    run["multi_model"] = c.multi_model
    run["num_vertical_patches"] = c.num_vertical_patches
    run["num_horizontal_patches"] = c.num_horizontal_patches
    run["permutation_mode"] = c.permutation_mode

    #run["test_anomaly_images"] = os.listdir(os.path.join(c.dataset_path, c.class_name, 'test','anomaly'))
    #run["test_good_images"] = os.listdir(os.path.join(c.dataset_path, c.class_name, 'test','good'))
    #run["train_good_images"] = os.listdir(os.path.join(c.dataset_path, c.class_name, 'train','good'))

    #run["train/dataset"].track_files(c.dataset_path)

def validation_multi_model(multi_models, validation_loader, validation_set, epoch, score_obs_auroc,
               score_obs_aucpr, best_thresholds, gmeans, best_tp, best_fp):
    multi_models.set_device("cpu")
    for hor, ver in itertools.product(range(c.num_horizontal_patches), range(c.num_vertical_patches)):
        multi_models.set_device(c.device, hor, ver)
        model = multi_models.models[hor][ver]
        fe = multi_models.fes[hor][ver]
        model.eval()
        if c.verbose:
            print(F'\nCompute loss and scores on validation set of model {hor} {ver}:')
        test_loss = list()
        test_z = list()
        test_labels = list()
        test_loss_good = list()
        test_loss_anomaly = list()

        z_collected = []
        all_maps = []

        with torch.no_grad():
            for i, data in enumerate(tqdm(validation_loader[hor][ver], disable=c.hide_tqdm_bar)):
                #print(F"Validation loader {hor} {ver} len: ", len(validation_loader[hor][ver]))
                #print("Len data", len(data[0]))
                #print("i: ", i)
                inputs, labels = preprocess_batch(data)
                if not c.pre_extracted:
                    inputs = fe(inputs)

                z, jac = nf_forward(model, inputs)
                loss = get_loss(z, jac)

                z_collected.append(z)
                z_concat = t2np(concat_maps(z))
                score = np.mean(z_concat ** 2, axis=(1, 2))
                test_z.append(score)
                test_loss.append(t2np(loss))
                test_labels.append(t2np(labels))

                labels_np_array = np.array(t2np(labels), dtype=bool)
                # since loss is a mean of the batch, I add this loss to the anomaly ones if there is at least one True (anomaly) in it
                # It's not a great problem, since data is ordered by label, so only one batch will have mixed labels at max
                if True in labels_np_array:
                    test_loss_anomaly.append(t2np(loss))
                else:
                    test_loss_good.append(t2np(loss))

                z_grouped = list()
                likelihood_grouped = list()
                for i in range(len(z)):
                    z_grouped.append(z[i].view(-1, *z[i].shape[1:]))
                    likelihood_grouped.append(torch.mean(z_grouped[-1] ** 2, dim=(1,)) / c.n_feat)
                all_maps.extend(likelihood_grouped[0])
        test_loss = np.mean(np.array(test_loss))
        if c.verbose:
            print('Epoch: {:d} \t validation_loss: {:.4f}'.format(epoch, test_loss))

        test_loss_good = np.mean(np.array(test_loss_good))
        test_loss_anomaly = np.mean(np.array(test_loss_anomaly))
        

        test_labels = np.concatenate(test_labels)
        is_anomaly = np.array([0 if l == 0 else 1 for l in test_labels])

        anomaly_score = np.concatenate(test_z, axis=0)
        score_obs_auroc[hor][ver].update(roc_auc_score(is_anomaly, anomaly_score), epoch,
                            print_score=c.verbose or epoch == c.meta_epochs - 1)
        score_obs_aucpr[hor][ver].update(average_precision_score(is_anomaly, anomaly_score), epoch,
                            print_score=c.verbose or epoch == c.meta_epochs - 1)

        false_positive_rate_t, true_positive_rate_t, threshold_t = roc_curve(is_anomaly, anomaly_score)
        threshold, gmean, tp_rate_best, fp_rate_best = find_threshold_auroc(threshold_t, true_positive_rate_t, false_positive_rate_t)
        best_thresholds[hor][ver].append(threshold)
        gmeans[hor][ver].append(gmean)
        best_tp[hor][ver].append(tp_rate_best)
        best_fp[hor][ver].append(fp_rate_best)
        if c.verbose:
            print('Best threshold: {:.4f} \t Geometric mean: {:.4f}'.format(best_thresholds[hor][ver][-1], gmeans[hor][ver][-1]))

        if c.neptune_activate:
            model_tag = str(hor) + '_' + str(ver)
            run["train/" + model_tag + "anomaly_score"].log(anomaly_score.tolist())
            run["train/" + model_tag + "is_anomaly"].log(is_anomaly.tolist())
            run["train/" + model_tag + "auroc"].log(score_obs_auroc[hor][ver].last)
            run["train/" + model_tag + "aucpr"].log(score_obs_aucpr[hor][ver].last)
            run["train/" + model_tag + "test_loss"].log(test_loss)
            run["train/" + model_tag + "best_thresholds"].log(best_thresholds[hor][ver][-1])
            run["train/" + model_tag + "best_gmeans"].log(gmeans[hor][ver][-1])
            run["train/" + model_tag + "test_loss_good"].log(test_loss_good)
            run["train/" + model_tag + "test_loss_anomaly"].log(test_loss_anomaly)
        
        multi_models.set_device("cpu", hor, ver)

    '''
    # patch auroc
    if c.use_patches:
        patchGroupCollector = PatchGroupCollector(validation_set)
        scores_patched, labels_patched = patch_grouped_evaluation(anomaly_score, is_anomaly, patchGroupCollector.patch_groups)
        auroc_patch = roc_auc_score(labels_patched, scores_patched)
        auprc_patch = average_precision_score(labels_patched, scores_patched)
        if c.verbose:
            print('Epoch: {:d} \t PATCH_validation_auroc: {:.4f}'.format(epoch, auroc_patch))
            print('Epoch: {:d} \t PATCH_validation_auprc: {:.4f}'.format(epoch, auprc_patch))
        if c.neptune_activate:
            run["train/patch_auroc"].log(auroc_patch)
            run["train/patch_auprc"].log(auprc_patch)
    '''


def validation(model, fe, validation_loader, validation_set, epoch, score_obs_auroc,
               score_obs_aucpr, best_thresholds, gmeans, best_tp, best_fp):
    # evaluate
    model.eval()
    if c.verbose:
        print('\nCompute loss and scores on validation set:')
    test_loss = list()
    test_z = list()
    test_labels = list()
    test_loss_good = list()
    test_loss_anomaly = list()

    z_collected = []
    all_maps = []

    with torch.no_grad():
        for i, data in enumerate(tqdm(validation_loader, disable=c.hide_tqdm_bar)):
            inputs, labels = preprocess_batch(data)
            if not c.pre_extracted:
                inputs = fe(inputs)

            z, jac = nf_forward(model, inputs)
            loss = get_loss(z, jac)

            z_collected.append(z)
            z_concat = t2np(concat_maps(z))
            score = np.mean(z_concat ** 2, axis=(1, 2))
            test_z.append(score)
            test_loss.append(t2np(loss))
            test_labels.append(t2np(labels))

            labels_np_array = np.array(t2np(labels), dtype=bool)
            # since loss is a mean of the batch, I add thhis loss to the anomaly ones if there is at least one True (anomaly) in it
            # It's not a great problem, since data is ordered by label, so only one batch will have mixed labels at max
            if True in labels_np_array:
                test_loss_anomaly.append(t2np(loss))
            else:
                test_loss_good.append(t2np(loss))

            z_grouped = list()
            likelihood_grouped = list()
            for i in range(len(z)):
                z_grouped.append(z[i].view(-1, *z[i].shape[1:]))
                likelihood_grouped.append(torch.mean(z_grouped[-1] ** 2, dim=(1,)) / c.n_feat)
            all_maps.extend(likelihood_grouped[0])

    test_loss = np.mean(np.array(test_loss))
    if c.verbose:
        print('Epoch: {:d} \t validation_loss: {:.4f}'.format(epoch, test_loss))

    test_loss_good = np.mean(np.array(test_loss_good))
    test_loss_anomaly = np.mean(np.array(test_loss_anomaly))
    

    test_labels = np.concatenate(test_labels)
    is_anomaly = np.array([0 if l == 0 else 1 for l in test_labels])

    anomaly_score = np.concatenate(test_z, axis=0)
    score_obs_auroc.update(roc_auc_score(is_anomaly, anomaly_score), epoch,
                           print_score=c.verbose or epoch == c.meta_epochs - 1)
    score_obs_aucpr.update(average_precision_score(is_anomaly, anomaly_score), epoch,
                           print_score=c.verbose or epoch == c.meta_epochs - 1)

    false_positive_rate_t, true_positive_rate_t, threshold_t = roc_curve(is_anomaly, anomaly_score)
    threshold, gmean, tp_rate_best, fp_rate_best = find_threshold_auroc(threshold_t, true_positive_rate_t, false_positive_rate_t)
    best_thresholds.append(threshold)
    gmeans.append(gmean)
    best_tp.append(tp_rate_best)
    best_fp.append(fp_rate_best)
    if c.verbose:
        print('Best threshold: {:.4f} \t Geometric mean: {:.4f}'.format(best_thresholds[-1], gmeans[-1]))

    # patch auroc
    if c.use_patches:
        patchGroupCollector = PatchGroupCollector(validation_set)
        scores_patched, labels_patched = patch_grouped_evaluation(anomaly_score, is_anomaly, patchGroupCollector.patch_groups)
        auroc_patch = roc_auc_score(labels_patched, scores_patched)
        auprc_patch = average_precision_score(labels_patched, scores_patched)
        if c.verbose:
            print('Epoch: {:d} \t PATCH_validation_auroc: {:.4f}'.format(epoch, auroc_patch))
            print('Epoch: {:d} \t PATCH_validation_auprc: {:.4f}'.format(epoch, auprc_patch))
        if c.neptune_activate:
            run["train/patch_auroc"].log(auroc_patch)
            run["train/patch_auprc"].log(auprc_patch)
        #patch separate
        score_separate, label_separate, _ = patch_separate_evaluation(anomaly_score, is_anomaly, patchGroupCollector.img_names)
        for hor, ver in itertools.product(range(c.num_horizontal_patches), range(c.num_vertical_patches)):
            auroc_patch = roc_auc_score(label_separate[hor,ver], score_separate[hor,ver])
            auprc_patch = average_precision_score(label_separate[hor,ver], score_separate[hor,ver])
            if c.verbose:
                model_tag = str(hor) + '_' + str(ver)
                print('Epoch: {:d} \t PATCH_validation_auroc {}: {:.4f}'.format(epoch, model_tag, auroc_patch))
                print('Epoch: {:d} \t PATCH_validation_auprc {}: {:.4f}'.format(epoch, model_tag, auprc_patch))
            if c.neptune_activate:
                model_tag = str(hor) + '_' + str(ver)
                run["train/" + model_tag + "auroc"].log(auroc_patch)
                run["train/" + model_tag + "aucpr"].log(auprc_patch)



    '''
    try:
        fig = plot_array_images(all_maps, test_labels, [p for p, l in validation_set.samples], epoch)
        writer.add_image('validation_image', fig, epoch)
    except:
        print("Can't do plot_array_image")
    '''


    if c.neptune_activate:
        run["train/anomaly_score"].log(anomaly_score.tolist())
        run["train/is_anomaly"].log(is_anomaly.tolist())
        run["train/auroc"].log(score_obs_auroc.last)
        run["train/aucpr"].log(score_obs_aucpr.last)
        run["train/test_loss"].log(test_loss)
        run["train/best_thresholds"].log(best_thresholds[-1])
        run["train/best_gmeans"].log(gmeans[-1])
        run["train/test_loss_good"].log(test_loss_good)
        run["train/test_loss_anomaly"].log(test_loss_anomaly)

def test(model, fe, test_loader, test_set, epoch, score_obs_auroc,
         score_obs_aucpr, best_thresholds, gmeans, best_tp, best_fp):
    # evaluate
    model.eval()
    if c.verbose:
        print('\nCompute loss and scores on test set:')
    test_loss = list()
    test_z = list()
    test_labels = list()

    z_collected = []
    all_maps = []

    with torch.no_grad():
        for i, data in enumerate(tqdm(test_loader, disable=c.hide_tqdm_bar)):
            inputs, labels = preprocess_batch(data)
            if not c.pre_extracted:
                inputs = fe(inputs)

            z, jac = nf_forward(model, inputs)
            loss = get_loss(z, jac)

            z_collected.append(z)
            z_concat = t2np(concat_maps(z))
            score = np.mean(z_concat ** 2, axis=(1, 2))
            test_z.append(score)
            test_loss.append(t2np(loss))
            test_labels.append(t2np(labels))

            z_grouped = list()
            likelihood_grouped = list()
            for i in range(len(z)):
                z_grouped.append(z[i].view(-1, *z[i].shape[1:]))
                likelihood_grouped.append(torch.mean(z_grouped[-1] ** 2, dim=(1,)) / c.n_feat)
            all_maps.extend(likelihood_grouped[0])

    test_loss = np.mean(np.array(test_loss))
    if c.verbose:
        print('TEST \t test_loss: {:.4f}'.format(test_loss))

    test_labels = np.concatenate(test_labels)
    is_anomaly = np.array([0 if l == 0 else 1 for l in test_labels])

    anomaly_score = np.concatenate(test_z, axis=0)
    score_obs_auroc.update(roc_auc_score(is_anomaly, anomaly_score), epoch,
                           print_score=c.verbose or epoch == c.meta_epochs - 1)
    score_obs_aucpr.update(average_precision_score(is_anomaly, anomaly_score), epoch,
                           print_score=c.verbose or epoch == c.meta_epochs - 1)

    false_positive_rate_t, true_positive_rate_t, threshold_t = roc_curve(is_anomaly, anomaly_score)
    threshold, gmean, tp_rate_best, fp_rate_best = find_threshold_auroc(threshold_t, true_positive_rate_t, false_positive_rate_t)
    best_thresholds.append(threshold)
    gmeans.append(gmean)
    best_tp.append(tp_rate_best)
    best_fp.append(fp_rate_best)
    if c.verbose:
        print('Best threshold: {:.4f} \t Geometric mean: {:.4f}'.format(best_thresholds[-1], gmeans[-1]))

    fig = plot_array_images(all_maps, test_labels, [p for p, l in test_set.samples], epoch)
    writer.add_image('test_image', fig, epoch)

    if c.neptune_activate:
        run["test/anomaly_score"].log(anomaly_score.tolist())
        run["test/is_anomaly"].log(is_anomaly.tolist())
        run["test/auroc"].log(score_obs_auroc.last)
        run["test/aucpr"].log(score_obs_aucpr.last)
        run["test/test_loss"].log(test_loss)
        run["test/best_thresholds"].log(best_thresholds[-1])
        run["test/best_gmeans"].log(gmeans[-1])

def train_multi_model(train_loader, test_loader, test_set, validation_loader, validation_set):
    if c.load_previous_weight:
        multi_models, starting_epoch = load_weights_multimodel(c.weight_location)
        print("Loaded weights of epoch ", starting_epoch)
    else:
        multi_models = MultiModel()
    if torch.cuda.device_count() > 1:
        print("Models will use", torch.cuda.device_count(), "GPUs")
        multi_models.set_multi_device()
    #I set all the models to cpu, then I will set each one to device when needed
    multi_models.set_device("cpu")
    score_obs_auroc = np.empty((c.num_horizontal_patches,c.num_vertical_patches),dtype=object)
    score_obs_aucpr = np.empty((c.num_horizontal_patches,c.num_vertical_patches),dtype=object)
    best_thresholds = np.empty((c.num_horizontal_patches,c.num_vertical_patches),dtype=object)
    best_tp = np.empty((c.num_horizontal_patches,c.num_vertical_patches),dtype=object)
    best_fp = np.empty((c.num_horizontal_patches,c.num_vertical_patches),dtype=object)
    gmeans = np.empty((c.num_horizontal_patches,c.num_vertical_patches),dtype=object)
    for hor, ver in itertools.product(range(c.num_horizontal_patches), range(c.num_vertical_patches)):
        score_obs_auroc[hor][ver] = Score_Observer('AUROC')
        score_obs_aucpr[hor][ver] = Score_Observer('AUCPR')
        best_thresholds[hor][ver] = []
        best_tp[hor][ver] = []
        best_fp[hor][ver] = []
        gmeans[hor][ver] = []


    for epoch in range(c.meta_epochs):
        if c.load_previous_weight:
            epoch += starting_epoch + 1
        if c.verbose:
                print(F'\nTrain epoch {epoch}')
        for hor, ver in itertools.product(range(c.num_horizontal_patches), range(c.num_vertical_patches)):
            #set only the single model to device
            multi_models.set_device(c.device, hor, ver)

            model = multi_models.models[hor][ver]
            optimizer = multi_models.optimizers[hor][ver]
            fe = multi_models.fes[hor][ver]
            model.train()
            if c.verbose:
                print(F'\nModel hor {hor} ver {ver}')
            for sub_epoch in range(c.sub_epochs):
                train_loss = list()
            #print("Train loader len: ", len(train_loader))
            for i, data in enumerate(tqdm(train_loader[hor][ver], disable=c.hide_tqdm_bar)):
                #print(F"Train loader {hor} {ver} len: ", len(train_loader[hor][ver]))
                optimizer.zero_grad()

                inputs, labels = preprocess_batch(data)  # move to device and reshape
                #print("outside fe: ", len(inputs))
                #print("labels: ", labels)
                #print("data", len(data[0][0][0]))
                #print(train_loader)
                available_gpus = [torch.cuda.device(i) for i in range(torch.cuda.device_count())]
                #print("Devices: ", available_gpus)
                #print("Memory before fe in device 0: ", torch.cuda.mem_get_info(device=torch.device("cuda:0")))
                #print("Memory before fe in device 1: ", torch.cuda.mem_get_info(device=torch.device("cuda:1")))
                if not c.pre_extracted:
                    inputs = fe(inputs)
                #print("Memory after fe in device 0: ", torch.cuda.mem_get_info(device=0))
                #print("Memory after fe in device 1: ", torch.cuda.mem_get_info(device=1))

                z, jac = nf_forward(model, inputs)

                loss = get_loss(z, jac)
                train_loss.append(t2np(loss))

                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), c.max_grad_norm)
                #print("Memory before optimizer in device 0: ", torch.cuda.mem_get_info(device=0))
                #print("Memory before optimizer) in device 1: ", torch.cuda.mem_get_info(device=1))
                optimizer.step()
                #set single model back to cpu, in order to free gpu
            mean_train_loss = np.mean(train_loss)
            if c.verbose and epoch == 0 and sub_epoch % 4 == 0:
                print('Epoch: {:d}.{:d} \t train loss: {:.4f}'.format(epoch, sub_epoch, mean_train_loss))
            if c.neptune_activate:
                run["train/"+ str(hor) + '_' + str(ver) + "/train_loss"].log(mean_train_loss)
            multi_models.set_device("cpu", hor, ver)

        if validation_set is not None and validation_loader is not None:
            validation_multi_model(multi_models, validation_loader, validation_set, epoch, score_obs_auroc,
                    score_obs_aucpr, best_thresholds, gmeans, best_tp, best_fp)
        else:
            validation(multi_models, test_loader, test_set, epoch, score_obs_auroc,
                       score_obs_aucpr, best_thresholds, gmeans, best_tp, best_fp)
        #save weights
        if c.save_model:
            multi_models.set_device('cpu')
            if(c.save_only_bests == False):
                #save_weights(model, optimizer, c.modelname, epoch)
                save_weights_multimodel(multi_models, c.modelname, 9999)
            elif(c.save_only_bests and (score_obs_auroc.max_epoch == epoch)):
                save_weights_multimodel(multi_models, c.modelname, epoch)
            multi_models.set_device("cpu")



def train(train_loader, test_loader, test_set, validation_loader, validation_set):
    #ry before model in device 0: ", torch.cuda.mem_get_info(device=0))
    #print("Memory before model in device 1: ", torch.cuda.mem_get_info(device=1))
    if c.load_previous_weight:
        model, optimizer , starting_epoch = load_weights(c.weight_location)
    else:
        model = get_cs_flow_model()
        optimizer = torch.optim.Adam(model.parameters(), lr=c.lr_init, eps=1e-04, weight_decay=1e-5)
        #optimizer = torch.optim.SGD(model.parameters(), lr=c.lr_init, weight_decay=1e-5)
    if torch.cuda.device_count() > 1:
        print("Model will use", torch.cuda.device_count(), "GPUs")
        #model = InvertibleNetworkDataParallel(model)
        model = nn.DataParallel(model)
    model.to(c.device)
    #print("Memory occupied by model (no feat extr) in device 0: ", torch.cuda.mem_get_info(device=0))
    #print("Memory occupied by model (no feat extr) in device 1: ", torch.cuda.mem_get_info(device=1))
    if not c.pre_extracted:
        fe = FeatureExtractor()
        fe.eval()
        if torch.cuda.device_count() > 1:
            print("Feature extractor will use", torch.cuda.device_count(), "GPUs")
            fe = nn.DataParallel(fe)
        fe.to(c.device)
        for param in fe.parameters():
            param.requires_grad = False
    

    score_obs_auroc = Score_Observer('AUROC')
    score_obs_aucpr = Score_Observer('AUCPR')
    best_thresholds = []
    best_tp =[]
    best_fp = []
    gmeans =[]


    for epoch in range(c.meta_epochs):
        if c.load_previous_weight:
            epoch += starting_epoch + 1
        # train some epochs
        model.train()
        if c.verbose:
            print(F'\nTrain epoch {epoch}')
        for sub_epoch in range(c.sub_epochs):
            train_loss = list()
            for i, data in enumerate(tqdm(train_loader, disable=c.hide_tqdm_bar)):
                optimizer.zero_grad()

                inputs, labels = preprocess_batch(data)  # move to device and reshape
                #print("outside fe: ", len(inputs))
                #print("labels: ", labels)
                #print("data", len(data[0][0][0]))
                #print(train_loader)
                available_gpus = [torch.cuda.device(i) for i in range(torch.cuda.device_count())]
                #print("Devices: ", available_gpus)
                #print("Memory before fe in device 0: ", torch.cuda.mem_get_info(device=torch.device("cuda:0")))
                #print("Memory before fe in device 1: ", torch.cuda.mem_get_info(device=torch.device("cuda:1")))
                if not c.pre_extracted:
                    inputs = fe(inputs)
                #print("Memory after fe in device 0: ", torch.cuda.mem_get_info(device=0))
                #print("Memory after fe in device 1: ", torch.cuda.mem_get_info(device=1))

                z, jac = nf_forward(model, inputs)

                loss = get_loss(z, jac)
                train_loss.append(t2np(loss))

                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), c.max_grad_norm)
                #print("Memory before optimizer in device 0: ", torch.cuda.mem_get_info(device=0))
                #print("Memory before optimizer) in device 1: ", torch.cuda.mem_get_info(device=1))
                optimizer.step()

            mean_train_loss = np.mean(train_loss)
            if c.verbose and epoch == 0 and sub_epoch % 4 == 0:
                print('Epoch: {:d}.{:d} \t train loss: {:.4f}'.format(epoch, sub_epoch, mean_train_loss))
            if c.neptune_activate:
                run["train/train_loss"].log(mean_train_loss)

        if validation_set is not None and validation_loader is not None:
            validation(model, fe, validation_loader, validation_set, epoch, score_obs_auroc,
                    score_obs_aucpr, best_thresholds, gmeans, best_tp, best_fp)
        else:
            validation(model, fe, test_loader, test_set, epoch, score_obs_auroc,
                       score_obs_aucpr, best_thresholds, gmeans, best_tp, best_fp)
        #save weights
        if c.save_model:
            model.to('cpu')
            if(c.save_only_bests == False):
                #save_weights(model, optimizer, c.modelname, epoch)
                save_weights(model, optimizer, c.modelname, 9999)
            elif(c.save_only_bests and (score_obs_auroc.max_epoch == epoch)):
                save_weights(model, optimizer, c.modelname, epoch)
            model.to(c.device)


    if c.save_model:
        model.to('cpu')
        save_model(model, c.modelname)

    #final test of the model
    #I put the device because after saving the model, it is on cpu
    model.to(c.device)
    test(model, fe, test_loader, test_set, c.sub_epochs*c.meta_epochs + 1, score_obs_auroc,
                   score_obs_aucpr, best_thresholds, gmeans, best_tp, best_fp)

    return score_obs_auroc.max_score, score_obs_auroc.last, score_obs_auroc.min_loss_score, score_obs_aucpr.max_score, score_obs_aucpr.last, score_obs_aucpr.min_loss_score
