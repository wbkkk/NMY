"""
File Name: pso.py
Author: 难民营
Complete Date: 2024/5/27
"""

import numpy as np
import random
import copy
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import BicScore


# 寻优
def getinitbest(fitness, pop):
    # 群体最优的粒子位置及其适应度值
    gbestpop, gbestfitness = pop[fitness.argmax()].copy(), fitness.max()
    # 个体最优的粒子位置及其适应度值,使用copy()使得对pop的改变不影响pbestpop,pbestfitness类似
    pbestpop, pbestfitness = pop.copy(), fitness.copy()

    return gbestpop, gbestfitness, pbestpop, pbestfitness


'''
对贝叶斯网络图的基本操作有：加边、减边、反转
基于上述操作可以在现有网路图中进行变异操作，使算法能跳出局部最优解
'''


# 减边
def cut(p, r2, r3):
    ini = p[r2][r3]
    if ini == 1:
        p[r2][r3] = 0
    else:
        r = 0
        while ini != 1:
            r += 1
            row = random.randint(1, p.shape[0]) - 1
            col = random.randint(1, p.shape[1]) - 1
            ini = p[row][col]
            if ini == 1:
                p[row][col] = 0
            if r == 100:
                break
    return p


# 反转
def reverse(p, r2, r3):
    ini1 = p[r2][r3]
    ini2 = p[r3][r2]
    if ini1 == 1 and ini2 == 0:
        p[r2][r3] = 0
        p[r3][r2] = 1
    elif ini1 == 0 and ini2 == 1:
        p[r2][r3] = 1
        p[r3][r2] = 0
    return p


# 变异
def mutation(p, data, colname):
    r1 = random.randint(1, 3)
    row = p.shape[0]
    col = p.shape[1]
    # 减边变异
    if r1 == 1:
        BIC = dict()
        bic = BicScore(data)
        for i in range(len(p)):
            for j in range(len(p)):
                if p[i][j] == 1:
                    model = BayesianNetwork([[colname[i], colname[j]]])
                    Score = bic.score(model)
                    BIC.update({(i, j): Score})
        if BIC:
            min_bic_key = min(BIC, key=lambda k: BIC[k])
            r2, r3 = min_bic_key
            p = cut(p, r2, r3)

    # 反转变异
    elif r1 == 2:
        value = []
        for i in range(row):
            for j in range(col):
                if p[i][j] == 1:
                    value.append((i, j))
        if len(value) < 1:
            p = p
        else:
            rand = random.randint(1, len(value)) - 1
            r2 = value[rand][0]
            r3 = value[rand][1]
            p = reverse(p, r2, r3)

    # 加边变异
    elif r1 == 3:
        BIC = dict()
        bic = BicScore(data)
        for i in range(len(p)):
            for j in range(len(p)):
                if p[i][j] == 0 and i != j and p[j][i] != 1:
                    model = BayesianNetwork([[colname[i], colname[j]]])
                    Score = bic.score(model)
                    BIC.update({(i, j): Score})
        if BIC:
            max_bic_key = max(BIC, key=lambda k: BIC[k])
            r2, r3 = max_bic_key
            p[r2][r3] = 1
    p = mix(p)
    return p


'''
贝叶斯网路图中不能出现环结构
在初始化与迭代过程中可能出现结构图有环的情况，通过修复函数解决这一问题
'''


# 闭包矩阵
def Warshall(p):
    p_t = copy.deepcopy(p)
    n = len(p_t)
    for j in range(n):
        for i in range(n):
            if int(p_t[i][j]) == 1:
                for k in range(n):
                    p_t[i][k] = logical_add(int(p_t[i][k]), int(p_t[j][k]))
    return p_t


# 逻辑加
def logical_add(a, b):
    if a == 0 and b == 0:
        return 0
    else:
        return 1


# 修复
def mix(p):
    check = Warshall(p)
    sum = 0
    circle = []  # 储存所有闭环节点的索引值(0,1,2...)
    line = []  # 储存某闭环节点的所有父节点的索引值(0,1,2...)
    for i in range(len(check)):
        sum += check[i][i]
        if sum != 0:  # 说明结构不合法,即有闭环
            break
    while sum != 0:
        for i in range(len(check)):
            if check[i][i] != 0:
                circle.append(i)  # 保留主对角线上非0元素对应的节点
        rand1 = random.randint(1, len(circle)) - 1
        point1 = circle[rand1]
        for i in range(len(p)):
            if p[i][point1] == 1:
                line.append(i)  # 任取环内一节点，保存该节点的所有父节点
        rand2 = random.randint(1, len(line)) - 1
        point2 = line[rand2]
        rand3 = random.randint(1, 2)
        if rand3 == 1:  # 减边操作
            p = cut(p, point2, point1)
        else:  # 反转操作
            p = reverse(p, point2, point1)

        check = Warshall(p)
        sum = 0
        circle = []  # 储存所有闭环节点的索引值(0,1,2...)
        line = []  # 储存某闭环节点的所有父节点的索引值(0,1,2...)
        for i in range(len(check)):
            sum += check[i][i]
    return p


'''
基于SPSO，在忽略速度项的同时，引入GA算法的染色体交叉思想
通过交叉实现对解空间的搜索
'''


# 交叉
def cross(p, pbestpop, gbestpop, pbest_genetic_percent, gbest_genetic_percent,
          data):
    # 此处p是一维
    cross_p = copy.deepcopy(p)

    # 交叉概率
    c_1 = random.random()  # 个体最优交叉
    c_2 = random.random()  # 群体最优交叉

    ## 随机选取头尾变异
    # 个体最优交叉操作
    if c_1 < pbest_genetic_percent:
        point_0 = random.randint(0, len(p))
        point_1 = point_0 + random.randint(0, len(p) - point_0)
        for i in range(point_0, point_1):
            cross_p[i] = pbestpop[i]

    else:
        cross_p = cross_p

    # 群体最优交叉操作
    if c_2 < gbest_genetic_percent:
        point_0 = random.randint(0, len(p))
        point_1 = point_0 + random.randint(0, len(p) - point_0)
        for i in range(point_0, point_1):
            cross_p[i] = gbestpop[i]

    else:
        cross_p = cross_p

    cross_p_re = cross_p.reshape(data.shape[1], data.shape[1])
    cross_p_mix = mix(cross_p_re).flatten()
    return cross_p_mix


'''
PSO算法本身适用于连续型问题的求解，但是贝叶斯网络结构采用邻接矩阵表示，属于离散型问题
故原PSO中的速度计算公式此处不适用，针对邻接矩阵的特点编写了新的速度更新公式
'''


# 速度参数更新
def para_update(num1, num2, num3, i, times):
    # pra更新
    pra = num1 - i * 0.5 / times
    # prb更新
    prb = num2 + i * 0.25 / times
    # prc更新
    prc = num3 + i * 0.25 / times

    pr = (pra, prb, prc)
    return pr


# 随机产生va
def getva(data):
    data_0 = np.zeros((data.shape[1], data.shape[1]), dtype=np.int32)
    data_cop = data_0
    for k in range(data.shape[1]):
        X = np.random.randint(1, data.shape[1], size=1)
        Y = np.random.randint(1, data.shape[1], size=1)
        int_ra = np.random.randint(-1, 2, size=1)
        data_cop[X, Y] = int_ra
    result = data_cop.flatten()
    return result


# SPSOBN线性调节变异和交叉概率
def liner_iteration_pos(w_init, w_end, pb_init, pb_end, gb_init, gb_end, i,
                        times):
    # w线性更新
    w_it = w_init - ((w_init - w_end) * (i / times))

    # pb线性更新
    pb_it = pb_init - ((pb_init - pb_end) * (i / times))

    # w线性更新
    gb_it = gb_init - ((gb_init - gb_end) * (i / times))

    return w_it, pb_it, gb_it
