import numpy as np
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import pylab as plt
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib; matplotlib.use('TkAgg')
#---自己按照公式实现
def auc_calculate(labels,preds,n_bins=100):
    postive_len = sum(labels)
    negative_len = len(labels) - postive_len
    total_case = postive_len * negative_len
    pos_histogram = [0 for _ in range(n_bins)]
    neg_histogram = [0 for _ in range(n_bins)]
    bin_width = 1.0 / n_bins
    for i in range(len(labels)):
        nth_bin = int(preds[i]/bin_width)
        if labels[i]==1:
            pos_histogram[nth_bin] += 1
        else:
            neg_histogram[nth_bin] += 1
    accumulated_neg = 0
    satisfied_pair = 0
    for i in range(n_bins):
        satisfied_pair += (pos_histogram[i]*accumulated_neg + pos_histogram[i]*neg_histogram[i]*0.5)
        accumulated_neg += neg_histogram[i]

    return satisfied_pair / float(total_case)

if __name__ == '__main__':

    datafile = ['data/result/independent_set_predict/Independent_GATNet--result_new_labels_0_10.csv', 'data/result/independent_set_predict/Independent_GCNNet--result_new_labels_0_10.csv',
                'data/result/independent_set_predict/Independent_DeepSynergy--result_new_labels_0_10.csv',
                'data/result/independent_set_predict/RF--ROC--new_labels_0_10.csv',
                'data/result/independent_set_predict/SVM--ROC--new_labels_0_10.csv',
                'data/result/MLP/MLP_validation--ROC--new_labels_0_10.csv',
                'data/result/GBM/GBM_validation--ROC--new_labels_0_10.csv',
                'data/result/XGB/XGB_validation--ROC--new_labels_0_10.csv',
                'data/result/AdaBoost/AdaBoost_validation--ROC--new_labels_0_10.csv']
    methods = ['DeepDDS-GAT', 'DeepDDS-GCN',
                'DeepSynergy',
                'RF',
                'SVM',
               'MLP',
               'GBM',
               'XGB',
               'AdaBoost']
    for it in range(len(datafile)):
        tem = pd.read_csv(datafile[it], header=None)
        tem = np.array(tem)
        print(tem[0].shape)
        print(tem[2].shape)
        print(type(tem[0][0]))
        print(type(tem[2][0]))

        fpr, tpr, thresholds = roc_curve(tem[0], tem[2], pos_label=1)
        print("-----sklearn:", auc(fpr, tpr))
        plt.plot(fpr, tpr, label=methods[it] + '(AUC = %0.3f)' % auc(fpr, tpr))

    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.title('Predictive performance on novel drug combinations')
    plt.rcParams['savefig.dpi'] = 1000 #图片像素
    plt.savefig('independence.png')
    plt.show()