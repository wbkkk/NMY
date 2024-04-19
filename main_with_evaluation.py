"""
File Name: main_with_evaluation.py
Author: 难民营
Complete Date: 2024/5/27
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import copy

from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination

from sklearn import svm, preprocessing
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split

from lightgbm import LGBMClassifier
import xgboost as xgb

from src.pc import pc
from src.initialize import initpopfit
from src.pso import getinitbest
from src.utils import plot
from src.pscore import min_frequent_item, min_ass_subgraph, min_max_fre_itemsets, priori_structrue_PC, \
    cal_black_priori
from src.PSO_BN import main_PSOBN

# data input and PC
if __name__ == '__main__':
    file_path = 'data/Miami_clean.csv'  # 数据路径
    image_path = 'data/result.png'  # 最终结果图保存路径

    df = pd.read_csv(file_path)  # 数据导入
    index_target = 0  # 目标变量选取

    # 训练集xy和测试集xy划分
    X = df.iloc[:, index_target + 1:]
    y = df.iloc[:, index_target]
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=0.2,
                                                        random_state=42)
    n_train = X_train.shape[0]

    # 训练集
    data_ori = df.iloc[:n_train, :]  # 用于黑白名单的获取
    raw_data = copy.deepcopy(data_ori)  # 用于分类器
    colname = data_ori.columns.to_list()

    # 测试集
    test_data = df.iloc[n_train:, :]

    # 数据标准化处理
    minmax_scaler = preprocessing.MinMaxScaler()
    data = minmax_scaler.fit_transform(data_ori)
    data = pd.DataFrame(data, columns=colname)  # 用于Apri-PSO

    data = data.iloc[:n_train, :]  # 数据切片
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

result, p_end = main_PSOBN(data, colname, sizepop, maxgen, w, pb, gb, mr, pop,
                           fitness, result, gbestpop, gbestfitness, pbestpop,
                           pbestfitness, 'p_bic', priori_subgraph,
                           black_priori_subgraph)

# Show
# 输出参数
print("---result---")
print(result)
print("---gbestpop---")
print(p_end)

# 输出图
plot(p_end, labels, image_path)

# 参数推断
# 邻接矩阵转化为pgmpy识别的贝叶斯网络结构
edges = []
null_node = []
n = p_end.shape[0]
for i in range(n):
    if p_end[i, :].any() == 0 and p_end[:, i].any() == 0:
        null_node.append(i)
    for j in range(n):
        if p_end[i][j] == 1:
            edges.append((colname[i], colname[j]))

print(edges)

model = BayesianNetwork(edges)

if null_node:
    for i in null_node:
        raw_data = raw_data.drop(labels=colname[i], axis=1)
        test_data = test_data.drop(labels=colname[i], axis=1)
        X_train = X_train.drop(labels=colname[i], axis=1)
        X_test = X_test.drop(labels=colname[i], axis=1)

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
    evidence = data.drop(colname[index_target])
    evidence_dict = evidence.to_dict()
    pred = infer.map_query(variables=[colname[index_target]],
                           evidence=evidence_dict)
    pred_value = pred[colname[index_target]]  # 获取最可能的值

    y_true.append(data[colname[index_target]])
    y_pred.append(pred_value)

# 分类器比较
# 二分类用binary
# 多分类用micro/macro

# SVM
svm_model = svm.SVC(probability=True)
svm_model.fit(X_train, y_train)
y_pred_svm = svm_model.predict(X_test)

# Random Forest
rf_model = RandomForestClassifier(n_estimators=100)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

# Naive Bayes
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)
y_pred_nb = nb_model.predict(X_test)

# Lightgbm
lg_model = LGBMClassifier()
lg_model.fit(X_train, y_train)
y_pred_lg = lg_model.predict(X_test)

# XGboost
xgb_model = xgb.XGBClassifier()
xgb_model.fit(X_train, y_train)
y_pred_xgb = nb_model.predict(X_test)

# accuracy
ap_accuracy = accuracy_score(y_true, y_pred)
svm_accuracy = accuracy_score(y_test, y_pred_svm)
rf_accuracy = accuracy_score(y_test, y_pred_rf)
nb_accuracy = accuracy_score(y_test, y_pred_nb)
lg_accuracy = accuracy_score(y_test, y_pred_lg)
xgb_accuracy = accuracy_score(y_test, y_pred_xgb)

# precision
ap_precision = precision_score(y_true, y_pred, average='micro')
svm_precision = precision_score(y_test, y_pred_svm, average='micro')
rf_precision = precision_score(y_test, y_pred_rf, average='micro')
nb_precision = precision_score(y_test, y_pred_nb, average='micro')
lg_precision = precision_score(y_test, y_pred_lg, average='micro')
xgb_precision = precision_score(y_test, y_pred_xgb, average='micro')

# recall
ap_recall = recall_score(y_true, y_pred, average='micro')
svm_recall = recall_score(y_test, y_pred_svm, average='micro')
rf_recall = recall_score(y_test, y_pred_rf, average='micro')
nb_recall = recall_score(y_test, y_pred_nb, average='micro')
lg_recall = recall_score(y_test, y_pred_lg, average='micro')
xgb_recall = recall_score(y_test, y_pred_xgb, average='micro')

# F1 score
# 二分类用binary
# 多分类用weighted
ap_f1score = f1_score(y_true, y_pred, average='weighted')
svm_f1score = f1_score(y_test, y_pred_svm, average='weighted')
rf_f1score = f1_score(y_test, y_pred_rf, average='weighted')
nb_f1score = f1_score(y_test, y_pred_nb, average='weighted')
lg_f1score = f1_score(y_test, y_pred_lg, average='weighted')
xgb_f1score = f1_score(y_test, y_pred_xgb, average='weighted')

models = [
    'SVM', 'Random Forest', 'Naive Bayes', 'Lightgbm', 'XGboost', 'Apri-PSO'
]
accuracies = [
    svm_accuracy, rf_accuracy, nb_accuracy, lg_accuracy, xgb_accuracy,
    ap_accuracy
]
precisions = [
    svm_precision, rf_precision, nb_precision, lg_precision, xgb_precision,
    ap_precision
]
recalls = [svm_recall, rf_recall, nb_recall, lg_recall, xgb_recall, ap_recall]
f1scores = [
    svm_f1score, rf_f1score, nb_f1score, lg_f1score, xgb_f1score, ap_f1score
]

# 分类器结果（table）
result_classification = pd.DataFrame({
    'Models': models,
    'Accuracy': accuracies,
    'Precision': precisions,
    'Recall': recalls,
    'F1 score': f1scores
})
print(result_classification)

# 分类器结果（bar）
plt.figure(figsize=(16, 8), dpi=100)
plt.subplot(2, 2, 1)
plt.bar(models, accuracies)
plt.xlabel('Models')
plt.ylabel('Accuracy')
plt.title('Accuracy Comparison of Different Models')

plt.subplot(2, 2, 2)
plt.bar(models, precisions)
plt.xlabel('Models')
plt.ylabel('Precision')
plt.title('Precision Comparison of Different Models')

plt.subplot(2, 2, 3)
plt.bar(models, recalls)
plt.xlabel('Models')
plt.ylabel('Recall')
plt.title('Recall Comparison of Different Models')

plt.subplot(2, 2, 4)
plt.bar(models, f1scores)
plt.xlabel('Models')
plt.ylabel('F1 score')
plt.title('F1score Comparison of Different Models')

plt.show()
