"""
File Name: initialize.py
Author: 难民营
Complete Date: 2024/5/27
"""

import numpy as np
from src.pso import mix
from src.bic import func


# 生成初始种群
def init_pop(data, sizepop):
    data_dimension = data.shape[1]
    result = np.zeros((sizepop, data.shape[1]**2), dtype=np.int32)

    for i in range(sizepop):
        data_cop = data

        for j in range(data_dimension):
            X = np.random.randint(1, data_dimension, size=1)
            Y = np.random.randint(1, data_dimension, size=1)
            int_ra = np.random.randint(0, 2, size=1)
            data_cop[X, Y] = int_ra

        data_code = mix(data_cop).flatten()  # 修复随机生成的初始种群
        result[i] = data_code

    return result


# 计算初始适应度
def initpopfit(p, sizepop, data, colname):
    pop = np.zeros((sizepop, data.shape[1]**2))
    fitness = np.zeros(sizepop)

    pop = init_pop(p, sizepop)
    for i in range(sizepop):
        fitness[i] = func(pop[i], data, colname)
    return pop, fitness
