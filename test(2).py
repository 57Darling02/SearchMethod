import random, math, sys
import matplotlib.pyplot as plt # 画图
from copy import deepcopy

from numpy.core.defchararray import center
from tqdm import *  # 进度条
DEBUG = True

def print_DEBUG(msg):
    if DEBUG:
        print(msg)

railway_gap = 5
train_box_length = 16
railway_num = 2
max_train_box_num = 9

X = [i*train_box_length for _ in range (0, railway_num+1) for i in range(0, max_train_box_num-1)]
Y = [i*railway_gap  for i in range(0, railway_num+1) for _ in range (0, max_train_box_num-1)]


# insert Center_pos as the first element of check_spot
Center_pos = (-1,-1)
X.insert(0,Center_pos[0])
Y.insert(0,Center_pos[1])

check_spot = {i: (X[i], Y[i]) for i in range(len(X))}
#print_DEBUG(f"check_spot:{check_spot}")
# 进入a条铁路有b个车厢，停留时间为c-d分钟
# (a,b,c,d)
train1 = (0,9,0,60)
train2 = (1,5,0,60)
train3 = (0,8,70,150)
alltrains = [train1, train2, train3]
check_request_index = 1
# check_task_dist[0] is the no limited check_task, it need to use when generate gene
check_task_dist = {0:[(0,10000),0]}
for train in alltrains:
    a,b,c,d = train
    for i in range(1,b):
        check_task_dist[check_request_index]=[(c,d),(a*(max_train_box_num-1))+i]
        check_task_dist[check_request_index+1]=[(c,d),((a+1)*(max_train_box_num-1))+i]
        check_request_index += 2
#print_DEBUG(f'checktask:{check_task_dist}') # check_task_dist[num from 0] = [(c,d),check_spot_number]

geneNum = 100
generationNum = 3000  # 迭代次数
CENTER = 0  # 配送中心
HUGE = 99999
VARY = 0.05  # 变异几率
n = len(check_task_dist)-1  # 检查任务数量 减去第0起点任务
k = 5  # 车辆数量
epu = 20  # 早到惩罚成本
lpu = HUGE  # 晚到惩罚成本
speed = 40  # 速度，km/h
costPerKilo = 1  # 油价
Q = 0
m = 0  # 换电站数量
t = 5
dis = 0

class Gene:
    def __init__(self, name='Gene', data=None):
        self.name = name
        self.length = n + m + 1
        if data is None:
            self.data = self._getGene(self.length)
        else:
            # print(f"gene data:{data}")
            assert(self.length+k == len(data))
            self.data = data
        self.fit = self.getFit()
        self.chooseProb = 0  # 选择概率

    # randomly choose a gene
    def _generate(self, length):
        data = [i for i in range(1, length)]
        random.shuffle(data)
        data.insert(0, CENTER)
        data.append(CENTER)
        return data

    # insert zeors at proper positions
    # use average task num to insert zeros ensure gene will be divided into k robots
    def _insertZeros(self, data):
        sum = 0
        newData = []
        average_task_num = int(check_request_index/k)+1
        for index, pos in enumerate(data):
            sum += 1
            if sum > average_task_num:
                newData.append(CENTER)
                sum = 1
            newData.append(pos)
        return newData

    # return a random gene with proper center assigned
    def _getGene(self, length):
        data = self._generate(length)
        data = self._insertZeros(data)
        #print(f"1:{data}")
        return data

    # return fitness
    def getFit(self):
        fit = distCost = timeCost = overloadCost = fuelCost = 0
        dist = []  # from this to next

        # calculate distance
        i = 1
        while i < len(self.data):
            # use manhattan distance
            calculateDist = lambda x1, y1, x2, y2: math.sqrt(((x1 - x2) ** 2)) + math.sqrt((y1 - y2) ** 2)
            # print(self.data)
            check_spot_index = check_task_dist[self.data[i]][1]
            check_spot_index2 = check_task_dist[self.data[i-1]][1]
            x1 = check_spot[check_spot_index][0]
            y1 = check_spot[check_spot_index][1]
            x2 = check_spot[check_spot_index2][0]
            y2 = check_spot[check_spot_index2][1]
            dist.append(calculateDist(x1, y1, x2, y2))
            i += 1

        # distance cost
        distCost = sum(dist) * costPerKilo

        # time cost
        timeSpent = 0
        for i, pos in enumerate(self.data):
            # skip first center
            if i == 0:
                continue
            # new car
            elif pos == CENTER:
                timeSpent = 0
            # update time spent on road
            timeSpent += (dist[i - 1] / speed)
            # arrive early
            if timeSpent < check_task_dist[pos][0][0]:
                timeCost += ((check_task_dist[pos][0][0] - timeSpent) * epu)
                timeSpent = check_task_dist[pos][0][0]
            # arrive late
            elif timeSpent > check_task_dist[pos][0][1]:
                timeCost += ((timeSpent - check_task_dist[pos][0][1]) * lpu)
            # update time
            timeSpent += t

        # overload cost and out of fuel cost
        load = 0
        distAfterCharge = 0
        for i, pos in enumerate(self.data):
            # skip first center
            if i == 0:
                continue
            # charge here
            if pos >= n:
                distAfterCharge = 0
            # at center, re-load
            elif pos == CENTER:
                load = 0
                distAfterCharge = 0
            # normal
            else:
                load += t
                distAfterCharge += dist[i - 1]
                # update load and out of fuel cost
                overloadCost += (HUGE * (load > Q))
                fuelCost += (HUGE * (distAfterCharge > dis))

        fit = distCost + timeCost + overloadCost + fuelCost
        # print(f"fit:{fit}")
        return 1/fit

    def updateChooseProb(self, sumFit):
        self.chooseProb = self.fit / sumFit

    def moveRandSubPathLeft(self):
        #print(f"beforemovegene{self.data}")
        path = random.randrange(k)  # choose a path index
        index = self.data.index(CENTER, path+1) # move to the chosen index
        # move first CENTER
        locToInsert = 0
        self.data.insert(locToInsert, self.data.pop(index))
        index += 1
        locToInsert += 1
        # move data after CENTER
        while self.data[index] != CENTER:
            self.data.insert(locToInsert, self.data.pop(index))
            index += 1
            locToInsert += 1
        #print(f"aftermovegene{self.data}")
        assert(self.length+k == len(self.data))

    # plot this gene in a new window
    def plot(self):
        Xorder = [check_spot[check_task_dist[i][1]][0] for i in self.data]
        Yorder = [check_spot[check_task_dist[i][1]][1] for i in self.data]

        colors = ['black', 'red', 'blue', 'cyan', 'purple']  # 定义几种颜色
        color_index = 0
        start_index = 0

        for i in range(len(Xorder)):
            if Xorder[i] == -1 and Yorder[i] == -1:
                plt.plot(Xorder[start_index:i+1], Yorder[start_index:i+1], c=colors[color_index % len(colors)], zorder=1)
                start_index = i
                color_index += 1

        # 绘制最后一段
        plt.plot(Xorder[start_index:], Yorder[start_index:], c=colors[color_index % len(colors)], zorder=1)

        plt.scatter(X, Y, zorder=2)
        plt.scatter([X[0]], [Y[0]], marker='o', zorder=3)
        plt.scatter(X[-m:], Y[-m:], marker='^', zorder=3)
        plt.title(self.name)
        plt.show()


def getSumFit(genes):
    sum = 0
    for gene in genes:
        sum += gene.fit
    return sum


# return a bunch of random genes
def getRandomGenes(size):
    genes = []
    for i in range(size):
        genes.append(Gene("Gene "+str(i)))
    return genes


# 计算适应度和
def getSumFit(genes):
    sumFit = 0
    for gene in genes:
        sumFit += gene.fit
    return sumFit


# 更新选择概率
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
    #print(f"gene1{gene1.data}")
    gene1.moveRandSubPathLeft()
    gene2.moveRandSubPathLeft()
    #print(f"gene1{gene1.data}")
    # print(f"gene2{gene2.data}")
    newGene1 = []
    newGene2 = []
    # copy first paths
    centers = 0
    firstPos1 = 1
    for pos in gene1.data:
        firstPos1 += 1
        centers += (pos == CENTER)
        newGene1.append(pos)
        if centers >= 2:
            break
    # print(f"newGene1{newGene1}")
    centers = 0
    firstPos2 = 1
    for pos in gene2.data:
        firstPos2 += 1
        centers += (pos == CENTER)
        newGene2.append(pos)
        if centers >= 2:
            break
    # copy data not exits in father gene

    for pos in gene2.data:
        if pos not in newGene1:
            newGene1.append(pos)
    # print(f"crossnewGene1{newGene1}")
    for pos in gene1.data:
        if pos not in newGene2:
            newGene2.append(pos)
    # add center at end
    newGene1.append(CENTER)
    newGene2.append(CENTER)
    # 计算适应度最高的
    key = lambda gene: gene.fit
    possible = []
    # print(f"gene1{gene1.data}")
    #print(f"newgene1{newGene1}")
    #print(f"firstpos1:{firstPos1}")
    '''
    while gene1.data[firstPos1] != CENTER:
        newGene = newGene1.copy()
        newGene.insert(firstPos1, CENTER)
        newGene = Gene(data=newGene.copy())
        possible.append(newGene)
        firstPos1 += 1
    '''
    # Calculate the number of gaps between the second and last zero in newGene1
    second_zero_index = newGene1.index(CENTER, 1)
    last_zero_index = len(newGene1) - 1

    # Find all possible positions to insert a zero
    possible_positions = [i for i in range(second_zero_index + 2, last_zero_index)]

    # Randomly shuffle the positions and insert zeros ensuring no two zeros are adjacent
    #for _ in range(math.comb(len(possible_positions), k - 2)):
    for _ in range(10):
        tempGene = newGene1.copy()
        #print(f'newGene1{newGene1}')
        random.shuffle(possible_positions)
        #print(f'possible_positions{possible_positions}')
        inserted_zeros = 0
        temp_positions=possible_positions.copy()
        for i in range(len(possible_positions)):
            if inserted_zeros >= k - 2:
                break
            if tempGene[temp_positions[i] - 1] != CENTER and tempGene[temp_positions[i]] != CENTER:
                tempGene.insert(temp_positions[i], CENTER)
                for n in range(len(temp_positions)):
                    if temp_positions[n] > temp_positions[i]:
                        temp_positions[n] += 1
                inserted_zeros += 1
        #print(f"tempGene1{tempGene}")
        possible.append(Gene(data=tempGene.copy()))
    possible.sort(reverse=True, key=key)
    assert(possible)
    crossedGenes.append(possible[0])
    key = lambda gene: gene.fit
    possible = []
    '''
    while gene2.data[firstPos2] != CENTER:
        newGene = newGene2.copy()
        newGene.insert(firstPos2, CENTER)
        newGene = Gene(data=newGene.copy())
        possible.append(newGene)
        firstPos2 += 1
    '''
    # Calculate the number of gaps between the second and last zero in newGene1
    second_zero_index = newGene2.index(CENTER, 1)
    last_zero_index = len(newGene2) - 1

    # Find all possible positions to insert a zero
    possible_positions = [i for i in range(second_zero_index + 2, last_zero_index)]

    # Randomly shuffle the positions and insert zeros ensuring no two zeros are adjacent
    #for _ in range(math.comb(len(possible_positions), k - 2)):
    for _ in range(10):
        tempGene = newGene2.copy()
        #print(f"newGene2{newGene2}")
        random.shuffle(possible_positions)
        inserted_zeros = 0
        #print(f"possible_positions{possible_positions}")
        temp_positions=possible_positions.copy()
        for i in range(len(possible_positions)):
            if inserted_zeros >= k - 2:
                break
            if tempGene[temp_positions[i] - 1] != CENTER and tempGene[temp_positions[i]] != CENTER:
                tempGene.insert(temp_positions[i], CENTER)
                for n in range(len(temp_positions)):
                    if temp_positions[n] > temp_positions[i]:
                        temp_positions[n] += 1                
                inserted_zeros += 1
        #print(f"tempGene2{tempGene}")
        possible.append(Gene(data=tempGene.copy()))
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

if __name__ == "__main__" :
    genes = getRandomGenes(geneNum) # 初始种群
    # 迭代
    for i in tqdm(range(generationNum)):
        # print(i)
        # print('\n')
        # for gene in genes:
        #     print(f"genes{gene.data}")
        updateChooseProb(genes)
        sumProb = getSumProb(genes)
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
    genes[0].plot() # 画出来

