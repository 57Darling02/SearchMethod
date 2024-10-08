import random
from copy import deepcopy
from collections import Counter

import numpy as np
from tqdm import tqdm

import graph_process

originGraph ,optimizedGraph = graph_process.read_graph('original_graph.xlsx')
start = 4
take_out = [
    # [4,0],
    [5,14],
    # [18,32],
    # [25,36],
    # [40,27],
    [11,14],
    [6,8],
    [6,8],
    # [7,8],
    # [8,40],
    [9,8],
    # [10,22],
    # [11,22],
    # [12,36],
    # [13,36],
    # [14,36],
    # [29,36],
    # [35,36],
]
alldata = []
for i in take_out:
    alldata.append(i[0])
    alldata.append(i[1])
element_counts = Counter(alldata)
# Convert to dictionary
element_counts_dict = dict(element_counts)
print(element_counts_dict)

VARY = 0.05  # 变异几率
class Gene:
    def __init__(self, name='Gene', data=None):
        self.name = name
        self.length =2*len(take_out)
        if data is None:
            self.data = self._getGene()
        else:
            self.data = data
        self.giggity = None
        self.fit = self.getFit()
        self.chooseProb = 0  # 选择概率

    def _getGene(self):
        data = []
        for i in take_out:
            data.append(i[0])
            data.append(i[1])
        random.shuffle(data)
        return data

    # return fitness
    def getFit(self):
        fit = 0
        order_list = take_out.copy()
        get_order = []
        self.giggity = []
        for i in self.data:
            tempLog = []
            tempLog.append(["到达", i])
            poplist = []
            for j in range(len(order_list)):
                if i == order_list[j][0]:
                    poplist.append(j)
                    tempLog.append(["装货", order_list[j]])
            for j in range(len(poplist)):
                get_order.append(order_list.pop(poplist[j]-j))
            poplist.clear()
            for j in range(len(get_order)):
                if i == get_order[j][1]:
                    poplist.append(j)
                    tempLog.append(["卸货", get_order[j]])
            for j in range(len(poplist)):
                get_order.pop(poplist[j]-j)
            if len(tempLog) != 1:
                self.giggity.append(tempLog)


        if len(order_list) != 0:
            self.giggity = 'error'
        elif len(get_order) != 0:
            fit += np.inf
            self.giggity = 'error'

        dist = []  # from this to next
        # calculate distance
        i = 1
        while i < len(self.data):
            v1 = self.data[i - 1]
            v2 = self.data[i]
            dist.append(optimizedGraph[v1][v2])
            i += 1

        # distance cost
        distCost = sum(dist)+optimizedGraph[start][self.data[0]]

        fit += distCost
        return 1/fit

    def updateChooseProb(self, sumFit):
        self.chooseProb = self.fit / sumFit

    def moveRandSubPathLeft(self):
        path = random.randrange(len(self.data))  # choose a path index
        swapped_list = self.data[path:] + self.data[:path]
        self.data = swapped_list

# return a bunch of random genes
def getRandomGenes(size):
    genes = []
    for i in range(size):
        genes.append(Gene("Gene "+str(i)))
    return genes

def getSumFit(genes):
    sumFit = 0
    for gene in genes:
        sumFit += gene.fit
    return sumFit

def updateChooseProb(genes):
    sumFit = getSumFit(genes)
    for gene in genes:
        gene.updateChooseProb(sumFit)

# 计算累计概率
def getSumProb(genes):
    sum = 0
    for gene in genes:
        sum += gene.chooseProb
    return sum

# 选择复制，选择前 1/3
def choose(genes):
    num = int(geneNum/6) * 2    # 选择偶数个，方便下一步交叉
    # sort genes with respect to chooseProb
    key = lambda gene: gene.chooseProb
    genes.sort(reverse=True, key=key)
    # return shuffled top 1/3
    return genes[0:num]

# 交叉一对
def crossPair(gene1, gene2, crossedGenes):
    gene1.moveRandSubPathLeft()
    gene2.moveRandSubPathLeft()
    key = lambda gene: gene.fit

    possible = []
    for _ in range(10):
        newGene1 = []
        p1= random.randrange(len(gene1.data))
        p2 = 0
        tempdict = element_counts_dict.copy()
        while p2 != p1:
            ele = gene1.data[p2]
            newGene1.append(ele)
            tempdict[ele] -= 1
            p2 += 1
        # copy data not exits in father gene
        for pos in gene2.data:
            if tempdict[pos] > 0:
                tempdict[pos] -= 1
                newGene1.append(pos)
        possible.append(Gene(data=newGene1))
    possible.sort(reverse=True, key=key)
    crossedGenes.append(possible[0])
    possible = []
    for _ in range(10):
        newGene2 = []
        p1 = random.randrange(len(gene2.data))
        p2 = 0
        tempdict = element_counts_dict.copy()
        while p2 != p1:
            ele = gene1.data[p2]
            newGene2.append(ele)
            tempdict[ele] -= 1
            p2 += 1
        # copy data not exits in father gene
        for pos in gene1.data:
            if tempdict[pos] > 0:
                tempdict[pos] -= 1
                newGene2.append(pos)
        possible.append(Gene(data=newGene2))
    possible.sort(reverse=True, key=key)
    crossedGenes.append(possible[0])


# 交叉
def cross(genes):
    crossedGenes = []
    for i in range(0, len(genes), 2):
        crossPair(genes[i], genes[i+1], crossedGenes)
    return crossedGenes

# 合并
def mergeGenes(genes, crossedGenes):
    # sort genes with respect to chooseProb
    key = lambda gene: gene.chooseProb
    genes.sort(reverse=True, key=key)
    pos = geneNum - 1
    for gene in crossedGenes:
        genes[pos] = gene
        pos -= 1
    return  genes

# 变异一个
def varyOne(gene):
    varyNum = 10
    variedGenes = []
    for i in range(varyNum):
        p1, p2 = random.choices(list(range(1,len(gene.data)-2)), k=2)
        newGene = gene.data.copy()
        newGene[p1], newGene[p2] = newGene[p2], newGene[p1] # 交换
        variedGenes.append(Gene(data=newGene.copy()))
    key = lambda gene: gene.fit
    variedGenes.sort(reverse=True, key=key)
    return variedGenes[0]

# 变异
def vary(genes):
    for index, gene in enumerate(genes):
        # 精英主义，保留前三十
        if index < 30:
            continue
        if random.random() < VARY:
            genes[index] = varyOne(gene)
    return genes

def real_path_len(path):
    length = 0
    for i in range(len(path)-1):
        length += optimizedGraph[path[i]][path[i+1]]
    return length

if __name__ == "__main__" :

    geneNum = 100
    generationNum = 3000  # 迭代次数
    genes = getRandomGenes(geneNum) # 初始种群
    # 迭代
    for i in tqdm(range(generationNum)):
        updateChooseProb(genes)
        # sumProb = getSumProb(genes)
        chosenGenes = choose(deepcopy(genes))   # 选择
        crossedGenes = cross(chosenGenes)   # 交叉
        genes = mergeGenes(genes, crossedGenes) # 复制交叉至子代种群
        genes = vary(genes) # under construction
    # sort genes with respect to chooseProb
    key = lambda gene: gene.fit
    genes.sort(reverse=True, key=key)   # 以fit对种群排序
    print('\r\n')
    print('data:', genes[0].data)
    print('fit:', genes[0].fit)
    print(f'从{start}到出发')
    for i in genes[0].giggity:
        for j in i:
            if j[0] != '到达':
                index = take_out.index(j[1])
                print(f'{j[0]}:任务编号{index}')
            else:
                print(f'{j[0]}:{j[1]}')
    data = genes[0].data
    data.insert(0, start)
    print(f'real path length:{real_path_len(data)}')
    real_path_len([4, 11, 6, 6, 5, 9, 8, 8, 8, 14, 14])
