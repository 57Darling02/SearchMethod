import random, math
from copy import deepcopy
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')  # 或其他适合的后端
from tqdm import *  # 进度条
DEBUG = True

def print_DEBUG(msg):
    if DEBUG:
        print(msg)

railway_gap = 5
train_box_length = 16
railway_num = 3
max_train_box_num = 9

X = [i*train_box_length for _ in range (0, railway_num+1) for i in range(0, max_train_box_num-1)]
Y = [i*railway_gap  for i in range(0, railway_num+1) for _ in range (0, max_train_box_num-1)]
Xcenter= [i*train_box_length/2 for _ in range (0, railway_num) for i in range(-1, 2*(max_train_box_num-1),2)]
Ycenter = [railway_gap/2*(1+2*i)  for i in range(0, railway_num) for _ in range (-1, 2*(max_train_box_num-1),2)]
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
train3 = (2,8,70,130)
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
generationNum = 30  # 迭代次数
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
        temp_counter = 0
        index = 0
        path = random.randrange(k)  # choose a path index
        if path == 0:
            return
        while temp_counter != path:
            index = self.data.index(CENTER, index+1) # move to the chosen index
            temp_counter += 1
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
        assert(self.length+k == len(self.data))

    # plot this gene in a new window
    def plot(self):

        Xorder = [check_spot[check_task_dist[i][1]][0] for i in self.data]
        Yorder = [check_spot[check_task_dist[i][1]][1] for i in self.data]
        animate_plot(Xorder, Yorder, pause_time=1)



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
    gene1.moveRandSubPathLeft()
    gene2.moveRandSubPathLeft()
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

    # Calculate the number of gaps between the second and last zero in newGene1
    second_zero_index = newGene1.index(CENTER, 1)
    last_zero_index = len(newGene1) - 2

    # Find all possible positions to insert a zero
    possible_positions = [i for i in range(second_zero_index + 2, last_zero_index)]
    if len(possible_positions) < k - 2:
        return

    # Randomly shuffle the positions and insert zeros ensuring no two zeros are adjacent
    #for _ in range(math.comb(len(possible_positions), k - 2)):

    for _ in range(10):
        tempGene = newGene1.copy()
        temp_position = random.sample(possible_positions, len(possible_positions))
        temp_positions = temp_position[:k-2]
        temp_positions.sort()
        for i in range(len(temp_positions)):
            tempGene.insert(temp_positions[i]+i, CENTER)
        possible.append(Gene(data=tempGene.copy()))
    possible.sort(reverse=True, key=key)
    assert(possible)
    crossedGenes.append(possible[0])
    key = lambda gene: gene.fit
    possible = []
    # Calculate the number of gaps between the second and last zero in newGene2
    second_zero_index = newGene2.index(CENTER, 1)
    last_zero_index = len(newGene2) - 2
    # Find all possible positions to insert a zero
    possible_positions = [i for i in range(second_zero_index + 2, last_zero_index)]

    if len(possible_positions) < k - 2:
        return
    # Randomly shuffle the positions and insert zeros ensuring no two zeros are adjacent
    #for _ in range(math.comb(len(possible_positions), k - 2)):
    for _ in range(10):
        tempGene = newGene2.copy()
        temp_position = random.sample(possible_positions, len(possible_positions))
        temp_positions = temp_position[:k - 2]
        temp_positions.sort()
        # print(f"temp_position:{temp_positions}")
        # print(f"tempGene:{tempGene}")
        for i in range(len(temp_positions)):
            tempGene.insert(temp_positions[i] + i, CENTER)
            # print(f'tempGeneinsert:{tempGene}')
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

# def animate_plot(Xorder, Yorder, pause_time=1):
#     # 启用交互模式
#     plt.ion()

#     # 创建一个新的绘图
#     fig, ax = plt.subplots()

#     # 画出所有点，初始为红色
#     scat = ax.scatter(Xorder, Yorder, color='red')

#     # 设置图形的范围（可选，根据数据调整）
#     ax.set_xlim(min(Xorder) - 1, max(Xorder) + 1)
#     ax.set_ylim(min(Yorder) - 1, max(Yorder) + 1)

#     # 显示图形并刷新
#     plt.show()
#     plt.pause(0.1)  # 短暂暂停以确保图形窗口显示

#     # 遍历点并逐步连接
#     for i in range(1, len(Xorder)):
#         # 将上一个点变为绿色
#         ax.scatter(Xorder[i - 1], Yorder[i - 1], color='green')

#         # 连接当前点和上一个点
#         ax.plot([Xorder[i - 1], Xorder[i]], [Yorder[i - 1], Yorder[i]], color='blue')

#         # 刷新图形以显示更改
#         plt.draw()
#         plt.pause(1)  # 暂停1秒

#     # 将最后一个点变为绿色
#     ax.scatter(Xorder[-1], Yorder[-1], color='green')

#     # 最终刷新并保持图形窗口打开
#     plt.draw()
#     plt.pause(0.1)  # 短暂暂停以确保最后一个点变色
#     plt.ioff()  # 关闭交互模式
#     plt.show()

import matplotlib.pyplot as plt
import matplotlib.patches as patches

def animate_plot(Xorder, Yorder, pause_time=1):
    # 启用交互模式
    plt.ion()

    # 创建一个新的绘图
    fig, ax = plt.subplots()

    # 画出所有点，初始为红色
    scat = ax.scatter(Xorder, Yorder, color='red')

    # 设置图形的范围（可选，根据数据调整）
    ax.set_xlim(min(Xorder) - 20, max(Xorder) + 20)
    y_min = min(Yorder)
    y_max = max(Yorder)
    y_avarage = (min(Ycenter) + max(Ycenter)) / 2
    # 选择 y_min 和 y_max 中离 y_avarage 较远的那个
    if abs(y_min - y_avarage) > abs(y_max - y_avarage):
        y_range = abs(y_min - y_avarage)
    else:
        y_range = abs(y_max - y_avarage)
    scale_factor = 4  # 比例因子，可以根据需要调整
    ax.set_ylim(y_avarage - y_range * scale_factor, y_avarage + y_range * scale_factor)

    # 显示图形并刷新
    plt.show()
    plt.pause(0.1)  # 短暂暂停以确保图形窗口显示

    # 定义长方形中心位置的列表
    rectangle_centers = []
    for i in range(len(Xcenter)):
        rectangle_centers.append((Xcenter[i], Ycenter[i]))

    # 遍历中心位置并生成所有长方形
    for center_x, center_y in rectangle_centers:
        # 创建一个长为15宽为4的长方形表示火车车厢
        rect = patches.Rectangle((center_x - (train_box_length-3)/2, center_y - (railway_gap-1)/2), (train_box_length-3), (railway_gap-1), linewidth=1, edgecolor='brown', facecolor='brown')
        ax.add_patch(rect)

    # 用另一种颜色的线将中心纵坐标相等的相邻矩形连接
    for i in range(len(rectangle_centers) - 1):
        if rectangle_centers[i][1] == rectangle_centers[i + 1][1]:
            center_y = rectangle_centers[i][1]
            center_x1 = rectangle_centers[i][0] + (train_box_length-3)/2  # 左边长方形右边那条边的中间
            center_x2 = rectangle_centers[i + 1][0] - (train_box_length-3)/2  # 右边长方形左边那条边的中间
            ax.plot([center_x1, center_x2], [center_y, center_y], color='brown', linestyle='-', linewidth=2)

    # 刷新图形以显示所有长方形
    plt.draw()
    plt.pause(0.1)  # 短暂暂停以确保长方形显示

    # 遍历点并逐步连接
    for i in range(1, len(Xorder)):
        # 将上一个点变为绿色
        ax.scatter(Xorder[i - 1], Yorder[i - 1], color='green')

        # 连接当前点和上一个点
        ax.plot([Xorder[i - 1], Xorder[i]], [Yorder[i - 1], Yorder[i]], color='blue')

        # 刷新图形以显示更改
        plt.draw()
        plt.pause(pause_time)  # 暂停指定时间

    # 将最后一个点变为绿色
    ax.scatter(Xorder[-1], Yorder[-1], color='green')

    # 最终刷新并保持图形窗口打开
    plt.draw()
    plt.pause(0.1)  # 短暂暂停以确保最后一个点变色
    plt.ioff()  # 关闭交互模式
    plt.show()

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

