"""
File Name: main_without_evaluation.py
Author: 难民营
Complete Date: 2024/5/27
"""

import copy
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import BicScore


# 结构转换函数：用于计算bic的过程
def nodege(x, data):
    init_xx = []
    blinput = copy.deepcopy(init_xx)
    x = x.reshape(data.shape[1], data.shape[1])
    for i in range(data.shape[1]):
        for j in range(data.shape[1]):
            if x[i][j] == 1:
                cname = data.columns.to_list()
                a = cname[i]
                b = cname[j]
                blinput += [[a, b]]
    return blinput


# bic评分函数
def func(x, data, colname):
    # x 输入粒子位置
    # y 粒子适应度值,bic
    x = x.reshape(data.shape[1], data.shape[1])
    bic = BicScore(data)
    blinput = nodege(x, data)
    model = BayesianNetwork(blinput)
    model.add_nodes_from(colname)
    score = bic.score(model)
    y = score
    return y
