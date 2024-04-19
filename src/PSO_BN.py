"""
File Name: PSO_BN.py
Author: 难民营
Complete Date: 2024/5/27
"""

import numpy as np
import random
import copy
from src.bic import func
from src.pso import getinitbest, mutation, mix, para_update, getva
from src.pscore import final_Score, min_frequent_item, min_ass_subgraph, min_max_fre_itemsets, priori_structrue_PC, \
    cal_black_priori


def main_PSOBN(data, colname, sizepop, maxgen, w, pb, gb, mr, pop, fitness,
               result, gbestpop, gbestfitness, pbestpop, pbestfitness, method,
               priori_subgraph, black_priori_subgraph):
    # 迭代寻优
    for i in range(maxgen):
        # 概率的初值
        w_init = w
        pb_init = pb
        gb_init = gb

        # 速度参数更新
        pr = para_update(w_init, pb_init, gb_init, i, maxgen)

        # 速度更新
        v = np.zeros((sizepop, data.shape[1]**2))

        for j in range(sizepop):
            v[j] = pr[0] * getva(data) + pr[1] * (
                pbestpop[j] - pop[j]) + pr[2] * (gbestpop - pop[j])

        for x in range(v.shape[0]):
            for y in range(v.shape[1]):
                if v[x][y] <= -0.5:
                    v[x][y] = -1
                elif -0.5 < v[x][y] < 0.5:
                    v[x][y] = 0
                elif v[x][y] >= 0.5:
                    v[x][y] = 1

        # 粒子位置更新
        for j in range(sizepop):
            # 移动操作
            pop[j] = pop[j] + v[j]
            pop[j][pop[j] < 0] = 0  # 范围限制
            pop[j][pop[j] > 1] = 1
            pop[j] = mix(pop[j].reshape(data.shape[1], data.shape[1])).flatten()

            # 变异操作
            if random.random() < mr:
                p = pop[j]
                p_node = p.reshape(data.shape[1], data.shape[1])
                p_mutation = mutation(p_node, data, colname)
                pop[j] = p_mutation.flatten()
        # 存储a+b评分函数的bic值
        ori_fitness = copy.deepcopy(fitness)
        ori_pbestfitness = copy.deepcopy(pbestfitness)
        ori_gbestfitness = copy.deepcopy(gbestfitness)

        # 适应度更新
        if method == 'p_bic':
            for j in range(sizepop):
                fitness[j], ori_fitness[j] = final_Score(
                    pop[j], data, colname, priori_subgraph,
                    black_priori_subgraph)  # 计算适应度
                if fitness[j] > pbestfitness[j]:  # 个体最优更新
                    pbestfitness[j] = fitness[j]
                    ori_pbestfitness[j] = ori_fitness[j]
                    pbestpop[j] = pop[j].copy()

            if pbestfitness.max() > gbestfitness:  # 群体最优
                gbestfitness = pbestfitness.max()
                h = pbestfitness.argmax()
                ori_gbestfitness = ori_pbestfitness[h]
                gbestpop = pop[pbestfitness.argmax()].copy()
            result[i] = ori_gbestfitness

        elif method == 'bic':
            for j in range(sizepop):
                fitness[j] = func(pop[j], data, colname)  # 计算适应度

                if fitness[j] > pbestfitness[j]:  # 个体最优更新
                    pbestfitness[j] = fitness[j]

                    pbestpop[j] = pop[j].copy()

            if pbestfitness.max() > gbestfitness:  # 群体最优
                gbestfitness = pbestfitness.max()
                gbestpop = pop[pbestfitness.argmax()].copy()
            result[i] = gbestfitness

    return result, gbestpop
