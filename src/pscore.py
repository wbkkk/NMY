"""
File Name: pscore.py
Author: 难民营
Complete Date: 2024/5/27
"""

import itertools
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
from pgmpy.estimators import PC
from pgmpy.estimators import StructureScore
import copy
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import BicScore

weight = 0.6


# 布尔值转换
def bin_classification(data):
    for index, row in data.iteritems():
        if data[index].max() > 1:
            count0 = 0
            count1 = 0
            for indexrow, row in data.iterrows():
                # print(row)
                if row[index] < data[index].max() / 2:
                    row[index] = 0
                    count0 = count0 + 1
                else:
                    row[index] = 1
                    count1 = count1 + 1
        else:
            count0 = 0
            count1 = 0
            for indexrow, row in data.iterrows():
                if row[index] == 0:
                    count0 = count0 + 1
                else:
                    count1 = count1 + 1
    return data


class PunishScore(StructureScore):

    def __init__(self,
                 data,
                 priori_subgraph,
                 black_priori_subgraph,
                 weight,
                 equivalent_sample_size=10,
                 **kwargs):
        """
        BNSL-FIM: add prior result in BDeu Score
        """
        self.equivalent_sample_size = equivalent_sample_size
        self.priori_subgraph = priori_subgraph
        self.black_priori_subgraph = black_priori_subgraph
        self.weight = weight
        super(PunishScore, self).__init__(data, **kwargs)

    def local_score(self, variable, parents):
        s = 0
        white = 0
        black = 0
        for parent in parents:
            edge = (parent, variable)
            if edge in self.priori_subgraph:
                white += 1
                s += self.weight
            elif edge in self.black_priori_subgraph:
                black += 1
                s -= self.weight
        return s


# white
def priori_structrue_PC(max_fre_itemsets, ass_subgraph, data):
    priori_subgraph = []
    for i in range(0, len(max_fre_itemsets)):
        node = []
        for x in max_fre_itemsets[i]:
            node.append(x)
        df_node = data[node]
        c = PC(df_node)
        best_model = c.estimate()
        for edge in best_model.edges():
            if not edge in priori_subgraph:
                if edge in ass_subgraph:
                    priori_subgraph.append(edge)
    return priori_subgraph


# black
def cal_black_priori(max_fre_itemsets, priori_subgraph, ass_subgraph):
    black_priori_subgraph = []
    for max_fre_itemset in max_fre_itemsets:
        for i in itertools.permutations(max_fre_itemset, 2):
            j = (i[1], i[0])
            if not i in priori_subgraph:
                if not i in ass_subgraph:
                    black_priori_subgraph.append(i)
    black_priori_subgraph = list(set(black_priori_subgraph))
    return black_priori_subgraph


# max_fre_itemsets
def min_max_fre_itemsets(frequent_itemsets):
    frequent_itemsets_new = frequent_itemsets[frequent_itemsets['length'] > 1]
    max_fre_itemsets = []
    flag = False
    frequent_itemsets_new = frequent_itemsets_new.sort_values(by="length",
                                                              ascending=False)
    for index, itemset in frequent_itemsets_new.iterrows():
        for max_fre_itemset in max_fre_itemsets:
            if itemset['itemsets'].issubset(max_fre_itemset):
                flag = True
                break
        if not flag:
            max_fre_itemsets.append(itemset['itemsets'])
        else:
            flag = False
    return max_fre_itemsets


# ass_subgraph
def min_ass_subgraph(frequent_itemsets, min_c):
    frequent_itemsets_ass = frequent_itemsets[frequent_itemsets['length'] <= 2]
    rules = association_rules(frequent_itemsets_ass, min_threshold=0.7)
    rules = rules[rules['confidence'] >= min_c]
    ass_subgraph = []
    for index, rule in rules.iterrows():
        x = (list(rule['antecedents'])[0], list(rule['consequents'])[0])
        ass_subgraph.append(x)
    return ass_subgraph


# frequent_itemsets
def min_frequent_item(data, min_s):
    data_ori = bin_classification(data)
    frequent_itemsets = apriori(data_ori, min_support=min_s, use_colnames=True)
    frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(
        lambda x: len(x))
    return frequent_itemsets


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


def final_Score(x, data, colname, priori_subgraph, black_priori_subgraph):
    x = x.reshape(data.shape[1], data.shape[1])

    Pscore = PunishScore(data, priori_subgraph, black_priori_subgraph, weight)
    bic = BicScore(data)
    blinput = nodege(x, data)
    model = BayesianNetwork(blinput)
    model.add_nodes_from(colname)

    b_score = bic.score(model)
    p_score = Pscore.score(model)
    score = p_score + b_score

    return score, b_score
