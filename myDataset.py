import random
from utils_test import *
from sklearn.model_selection import StratifiedKFold, KFold
import torch
import numpy as np
import pandas as pd
import csv

t, f = 0, 0
with open('D:\GraphDTA-master\data\independent_set\independent_set_newcombinatioin.csv', 'r') as f:
    next(f)
    reader = csv.reader(f)
    for row in reader:
        if float(row[11]) > 20:
            t = 1 + int(t)

print('t', t)

























# i = 0
# datafile = 'new_labels_0_10'
# model_file = 'GAT_GCN'
# last_best = []
#
# for enpo in range(5):
#     data = []
#     file_AUCs = 'data/result/0_10/' + model_file + '(shared weight)' + str(i) + '--AUCs--' + datafile + '.txt'
#     with open(file_AUCs, 'r') as f:
#         data = f.read().strip().split('\n')
#     best_auc = 0
#     best_data = []
#     print('len(data[1:])', len(data[1:]))
#     for pot in data[1:]:
#         Epoch, AUC_dev, PR_AUC, ACC, BACC, PREC, TPR, KAPPA, RECALL = pot.strip().split()
#         if float(AUC_dev) > best_auc:
#             best_auc = float(AUC_dev)
#             best_data = [Epoch, AUC_dev, PR_AUC, ACC, BACC, PREC, TPR, KAPPA, RECALL]
#     i += 1
#     print('best_data', best_data)
#     last_best.append(best_data)
# print('last_best', last_best)
# AUC_dev0, PR_AUC0, ACC0, BACC0, PREC0, TPR0, KAPPA0, RECALL0 = 0, 0, 0, 0, 0, 0, 0, 0
# for da in last_best:
#     AUC_dev0 += float(da[1])
#     PR_AUC0 += float(da[2])
#     ACC0 += float(da[3])
#     BACC0 += float(da[4])
#     PREC0 += float(da[5])
#     TPR0 += float(da[6])
#     KAPPA0 += float(da[7])
#     RECALL0 += float(da[8])
#
# gapAUC_dev, gapPR_AUC, gapACC, gapBACC, gapPREC, gapTPR, gapKAPPA, gapRECALL = 110, 110, 110, 110, 110, 110, 110, 110
#
# for da in last_best:
#     if abs(AUC_dev0/5 - float(da[1])) < gapAUC_dev :
#         gapAUC_dev = abs(AUC_dev0/5 - float(da[1]))
#
#     if abs(PR_AUC0/5 - float(da[2])) < gapPR_AUC :
#         gapPR_AUC = abs(PR_AUC0/5 - float(da[1]))
#
#     if abs(ACC0/5 - float(da[3])) < gapACC :
#         gapACC = abs(ACC0/5 - float(da[1]))
#
#     if abs(BACC0/5 - float(da[4])) < gapBACC :
#         gapBACC = abs(BACC0/5 - float(da[1]))
#
#     if abs(PREC0/5 - float(da[5])) < gapPREC :
#         gapPREC = abs(PREC0/5 - float(da[1]))
#
#     if abs(TPR0/5 - float(da[6])) < gapTPR :
#         gapTPR = abs(TPR0/5 - float(da[1]))
#
#     if abs(KAPPA0/5 - float(da[7])) < gapKAPPA :
#         gapKAPPA = abs(KAPPA0/5 - float(da[1]))
#
#     if abs(RECALL0/5 - float(da[8])) < gapRECALL :
#         gapRECALL = abs(RECALL0/5 - float(da[1]))
#
# print(datafile + model_file,
#       '\nAUC_dev0', AUC_dev0 / 5, gapAUC_dev,
#       '\nPR_AUC0', PR_AUC0 / 5, gapPR_AUC,
#       '\nACC0', ACC0 / 5, gapACC,
#       '\nBACC0', BACC0 / 5, gapBACC,
#       '\nPREC0', PREC0 / 5, gapPREC,
#       '\nTPR0', TPR0 / 5, gapTPR,
#       '\nKAPPA0', KAPPA0 / 5, gapKAPPA,
#       '\nRECALL0', RECALL0 / 5, gapRECALL)