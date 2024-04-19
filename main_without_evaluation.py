"""
File Name: main_without_evaluation.py
Author: 难民营
Complete Date: 2024/5/27
"""

import pandas as pd
import numpy as np

from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator, BayesianEstimator
from pgmpy.inference import VariableElimination

from sklearn import preprocessing
from sklearn.metrics import confusion_matrix, precision_score, recall_score

from src.pc import pc
from src.initialize import initpopfit
from src.pso import getinitbest
from src.utils import plot
from src.pscore import min_frequent_item, min_ass_subgraph, min_max_fre_itemsets, priori_structrue_PC, \
    cal_black_priori
from src.PSO_BN import main_PSOBN

# data input and PC
if __name__ == '__main__':
    file_path = 'data/asia1.csv'  # 数据路径
    image_path = 'data/result.png'  # 最终结果图保存路径

    data_ori = pd.read_csv(file_path)
    data_ori = data_ori.iloc[0:3500, :]  # 用于黑白名单的获取，为整数数据
    colname = data_ori.columns.to_list()

    # 数据标准化处理
    minmax_scaler = preprocessing.MinMaxScaler()
    data = minmax_scaler.fit_transform(data_ori)
    data = pd.DataFrame(data, columns=colname)  # 用于PSO，对数据进行的处理

    data = data.iloc[0:3500, :]  # 数据切片
    n_nodes = data.shape[1]
    labels = data.columns.to_list()
    # PC
    p = pc(suff_stat={
        "C": data.corr().values,
        "n": data.shape[0]
    },
        verbose=True)

# Initialize

# 设置算法参数
# 种群大小
sizepop = 15
# 最大迭代次数
maxgen = 100
# 速度权重
'''
1）当 ω 过大会时有利于全局搜素，当 ω 过小时有利于局部搜索，在算法中会随着迭代次数的增加线性下降
2）pb，gb是控制学习因子是调节粒子自身经验和群体经验的权重，主要反应粒子之间的交流，影响粒子的运动轨迹．过大过小都会对结果造成不利影响
3）sum(w, pb, gb) == 1
4）mr为变异概率，mr越大，可以提高算法跳出局部最优解的能力；但过大会影响函数的收敛性
'''
w = 0.4
pb = 0.3
gb = 0.3
mr = 0.05

# 初始化种群
pop, fitness = initpopfit(p, sizepop, data, colname)
result = np.zeros(maxgen)  # 用于储蓄迭代过程中的适应度
print("---initpop")
print(pop)

# 生成先验知识
# 参数
min_s = 0.1
min_c = 0.5

# 计算黑白名单
frequent_itemsets = min_frequent_item(data_ori, min_s)
ass_subgraph = min_ass_subgraph(frequent_itemsets, min_c)
max_fre_itemsets = min_max_fre_itemsets(frequent_itemsets)
# white
priori_subgraph = priori_structrue_PC(max_fre_itemsets, ass_subgraph, data)
# black
black_priori_subgraph = cal_black_priori(max_fre_itemsets, priori_subgraph,
                                         ass_subgraph)

# PSO-BN
# 初始状态个体最优和种群最优，用于储存迭代过程中的输出
gbestpop, gbestfitness, pbestpop, pbestfitness = getinitbest(fitness, pop)

result, gbestpop = main_PSOBN(data, colname, sizepop, maxgen, w, pb, gb, mr,
                              pop, fitness, result, gbestpop, gbestfitness,
                              pbestpop, pbestfitness, 'p_bic', priori_subgraph,
                              black_priori_subgraph)

# Show
# 输出参数
print("---result:")
print(result)
print("---gbestpop:")
print(gbestpop.reshape(data.shape[1], data.shape[1]))
# 输出图
p_end = gbestpop.reshape(data.shape[1], data.shape[1])
plot(p_end, labels, image_path)

# 参数推断
# 邻接矩阵转化为pgmpy识别的贝叶斯网络结构
edges = []
n = p_end.shape[0]
for i in range(n):
    for j in range(n):
        if p_end[i][j] == 1:
            edges.append((colname[i], colname[j]))

model = BayesianNetwork(edges)
raw_data = data_ori
test_data = pd.read_csv(file_path).iloc[3500:5000, :]
mle = MaximumLikelihoodEstimator(model, raw_data)
model.fit(raw_data, estimator=MaximumLikelihoodEstimator)

# cpds = model.get_cpds()
#
# for cpd in cpds:
#     print("CPD of {variable}:".format(variable=cpd.variable))
#     print(cpd)

# 预测性能评估
infer = VariableElimination(model)
correct_count = 0  # 计算准确度
y_true = []  # 存储真实标签
y_pred = []  # 存储预测标签
for _, data in test_data.iterrows():
    evidence = data.drop('Dyspnea')
    evidence_dict = evidence.to_dict()
    pred = infer.map_query(variables=['Dyspnea'], evidence=evidence_dict)
    pred_value = pred['Dyspnea']  # 获取最可能的值

    if pred_value == data['Dyspnea']:
        correct_count += 1

    y_true.append(data['Dyspnea'])
    y_pred.append(pred_value)

accuracy = correct_count / len(test_data)
cm = confusion_matrix(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)

print('Accuracy: ', accuracy)
print('Confusion Matrix: ')
print(cm)
print('Precision: ', precision)
print('Recall: ', recall)
