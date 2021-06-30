import matplotlib.pyplot as plt
import numpy as np
import csv
import numpy as np
from numpy import random
import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns


def get_map(AZD6244_weight, file):
    confusion = AZD6244_weight
    # print(confusion.shape)
    # 热度图，后面是指定的颜色块，可设置其他的不同颜色
    plt.figure(figsize=(10, 8), dpi=1000)

    plt.imshow(confusion, cmap=plt.cm.Blues)
    # ticks 坐标轴的坐标点
    # label 坐标轴标签说明
    indices = range(len(confusion[0]))
    # print(indices)
    # 第一个是迭代对象，表示坐标的显示顺序，第二个参数是坐标轴显示列表
    #plt.xticks(indices, [0, 1, 2])
    #plt.yticks(indices, [0, 1, 2])
    idx = range(len(confusion[0]))
    # print(idx)
    plt.xticks(indices, idx, fontsize=25)
    plt.yticks(indices, idx, fontsize=25)

    plt.colorbar()

    font = {'family' : 'Times New Roman',
    'weight' : 'normal',
    'size'   : 30,
    }

    plt.xlabel('Predict', font)
    plt.ylabel('True', font)
    plt.title(file, font)

    # plt.rcParams两行是用于解决标签不能显示汉字的问题
    plt.rcParams['font.sans-serif']=['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    # 显示数据
    # for first_index in range(len(confusion)):    #第几行
    #     for second_index in range(len(confusion[first_index])):    #第几列
    #         plt.text(first_index, second_index, confusion[second_index][first_index], fontsize=30)
    # 在matlab里面可以对矩阵直接imagesc(confusion)
    # 显示
    plt.savefig('fig/' + file + '.png')
    plt.close()
    # plt.show()
if __name__ == '__main__':

    tem = np.array(([[126, 62], [289, 191]]))
    get_map(tem, 'DeepSynergy')

    # AZD6244_weight = []
    # csvFile = open('data/result/AZD2014_begin_x1.csv', 'r')
    # reader = csv.reader(csvFile)
    # for it in reader:
    #     for num in range(len(it)):
    #         it[num] = round(float(it[num]), 3)
    #     AZD6244_weight.append(it)
    # AZD6244_weight = np.array(AZD6244_weight)
    #
    # AZD2014_weight = []
    # csvFile = open('data/result/AZD2014_fin_x1.csv', 'r')
    # reader = csv.reader(csvFile)
    # for it in reader:
    #     for num in range(len(it)):
    #         it[num] = round(float(it[num]), 3)
    #     AZD2014_weight.append(it)
    # AZD2014_weight = np.array(AZD2014_weight)
    #
    # print(AZD6244_weight)
    # print(AZD2014_weight)
    #
    # dt = pd.DataFrame(AZD6244_weight.T)
    # dt = np.array(dt.corr(method='pearson'))  # 默认为'pearson'检验，可选'kendall','spearman'
    # dt = np.around(dt, decimals=2)
    # get_map(dt, '1')
    #
    # dt = pd.DataFrame(AZD2014_weight.T)
    # dt = np.array(dt.corr(method='pearson'))  # 默认为'pearson'检验，可选'kendall','spearman'
    # dt = np.around(dt, decimals=2)
    # get_map(dt, '2')

    # datafile = 'data/independent_set/independent_cell.csv'
    # independent_cell = pd.read_csv(datafile)
    # independent_cell = np.array(independent_cell)[:, 0]
    # print(independent_cell)
    #
    # datafile = 'data/independent_set/independent_drug.csv'
    # independent_drug = pd.read_csv(datafile)
    # independent_drug = np.array(independent_drug)[:, 0]
    # print(independent_drug)
    #
    # lc = len(independent_cell)
    # ld = len(independent_drug)
    #
    # tem = []
    # for i in range(ld):
    #     for j in range(ld):
    #         for k in range(lc):
    #             if i != j:
    #                 tem.append([independent_drug[i], independent_drug[j], independent_cell[k]])
    #
    # df = pd.DataFrame(tem)
    # df.to_csv('data/independent_set/independent_allin.csv', header=None, index=None)
