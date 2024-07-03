import shutil
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve, confusion_matrix, precision_score, accuracy_score, recall_score
import itertools
import numpy as np
from utils import *
import itertools
import sys

thres_computed = [  [1.081576 ,1.2097787,1.1954566,1.2057903,1.2090586,1.1940093],
                    [ 1.0529448,1.0411555,1.05305,1.1042804,1.1325476,1.0724558 ]]
def read_CSV(path):
    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        #create empty dataframe
        df = pd.DataFrame()
        print(path, "is empy")
    return df

def flexibleauroc(tag=None, input_thr=None):
    folder = 'study_elements/'
    scores = []
    labels = []
    if tag is None:
        tp_df = read_CSV(folder + 'tp.csv')
        fp_df = read_CSV(folder + 'fp.csv')
        tn_df = read_CSV(folder + 'tn.csv')
        fn_df = read_CSV(folder + 'fn.csv')
    else:
        tp_df = read_CSV(folder + 'tp'+tag+'.csv')
        fp_df = read_CSV(folder + 'fp'+tag+'.csv')
        tn_df = read_CSV(folder + 'tn'+tag+'.csv')
        fn_df = read_CSV(folder + 'fn'+tag+'.csv')

    #ipcode_to_remove = ['20410', '21373', '24787', '25016', '29847']
    #ipcode_to_remove = ['35800']
    #ipcode_to_remove = ['20410', '24787', '35800', '21373']
    ipcode_to_remove = []

    order_filenames = []
    for index, row in itertools.chain(tp_df.iterrows(), fn_df.iterrows()):
        filename = row[0].split('/')[-1].replace(".png","")
        ipcode = filename.split('_')[0]
        if(ipcode in ipcode_to_remove):
            continue
        else:
            scores.append(row[1])
            labels.append(1)
        order_filenames.append(filename)
    for index, row in itertools.chain(tn_df.iterrows(), fp_df.iterrows()):
        filename = row[0].split('/')[-1].replace(".png","")
        ipcode = filename.split('_')[0]
        if(ipcode in ipcode_to_remove):
            continue
        else:
            scores.append(row[1])
            labels.append(0)
        order_filenames.append(filename)
    anomaly_score = scores
    test_labels = labels


    false_positive_rate, true_positive_rate, threshold = roc_curve(test_labels, anomaly_score)
    threshold, gmean, tp_rate_best, fp_rate_best = find_threshold_auroc(threshold, true_positive_rate, false_positive_rate)

    auroc_score = roc_auc_score(test_labels, anomaly_score)
    if input_thr == None:
        print("AUROC: ", auroc_score)
        print("AUPRC: ", average_precision_score(test_labels, anomaly_score))
        print("Threshold: ", threshold, "\n true positive rate best: ", tp_rate_best, "\n false positive rate best: ", fp_rate_best, "\n gmean: ", gmean)
    else:
        old_threshold = threshold
        threshold = input_thr
    print("Accuracy: ", accuracy_score(test_labels, np.array(anomaly_score) > threshold))
    print("Precision: ", precision_score(test_labels, np.array(anomaly_score) > threshold))
    print("Recall: ", recall_score(test_labels, np.array(anomaly_score) > threshold))
    print("Confusion matrix: \n", confusion_matrix(test_labels, np.array(anomaly_score) > threshold))
    print("")

    if input_thr is not None:
        threshold = old_threshold

    return test_labels, anomaly_score, threshold, order_filenames, auroc_score

sys.stdout = open("study_elements/flexible_report.txt", "w")
if c.multi_model or c.use_patches:
    output = np.empty((c.num_horizontal_patches,c.num_vertical_patches),dtype=object)
    labels = np.empty((c.num_horizontal_patches,c.num_vertical_patches),dtype=object)
    order = np.empty((c.num_horizontal_patches,c.num_vertical_patches),dtype=object)
    auroc = []
    for hor,ver in itertools.product(range(c.num_horizontal_patches), range(c.num_vertical_patches)):
        print(F"Analyzing {hor} {ver}")
        tag = str(hor) + '_' + str(ver)
        test_labels, anomaly_score, threshold, order_filenames, auroc_temp = flexibleauroc(tag, input_thr=None)
        output[hor,ver] = anomaly_score > threshold
        labels[hor,ver] = test_labels
        order[hor,ver] = order_filenames
        auroc.append(auroc_temp)

    #Aggregation
    print("Start aggregation of patch evaluations")
    output_aggregated = []
    label_aggregated = []
    for filename in order[0,0]:
        temp_valuation = 0
        temp_label = 0
        for hor,ver in itertools.product(range(c.num_horizontal_patches), range(c.num_vertical_patches)):
            temp_filename = filename.replace('_0_0_', '_'+str(hor)+'_'+str(ver)+'_')
            #print(order[hor,ver])
            #print(F"start {hor} {ver}")
            index = order[hor,ver].index(temp_filename)
            if output[hor,ver][index] > 0:
                temp_valuation = 1
            if labels[hor,ver][index] > 0:
                temp_label = 1
        output_aggregated.append(temp_valuation)
        label_aggregated.append(temp_label)
    avg_auroc = sum(auroc) / float(len(auroc))
    print("Confusion matrix AGGREGATED: \n", confusion_matrix(label_aggregated, output_aggregated))
    print("Accuracy AGGREGATED: ", accuracy_score(label_aggregated, output_aggregated))
    print("Precision AGGREGATED: ", precision_score(label_aggregated, output_aggregated))
    print("Recall AGGREGATED: ", recall_score(label_aggregated, output_aggregated))
    print("Auroc AVERAGE: ", avg_auroc)


else:
    flexibleauroc()


sys.stdout.close()




