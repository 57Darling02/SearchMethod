import random, math
from copy import deepcopy
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.patches as patches
import numpy as np

matplotlib.use('TkAgg')
from tqdm import *  # 进度条
DEBUG = True

def print_DEBUG(msg):
    if DEBUG:
        print(msg)

interval = 0.1 # 每一步仿真的时间
beilv = 10 # 仿真加速倍率
railway_gap = 5
train_box_length = 16
railway_num = 3
max_train_box_num = 9

X = [i*train_box_length for _ in range (0, railway_num+1) for i in range(0, max_train_box_num-1)]
Y = [i*railway_gap  for i in range(0, railway_num+1) for _ in range (0, max_train_box_num-1)]
Xcenter= [i*train_box_length/2 for _ in range (0, railway_num) for i in range(-1, 2*(max_train_box_num-1),2)]
Ycenter = [railway_gap/2*(1+2*i)  for i in range(0, railway_num) for _ in range (-1, 2*(max_train_box_num-1),2)]
# insert Center_pos as the first element of check_spot
Center_pos = (-20,-1)
X.insert(0,Center_pos[0])
Y.insert(0,Center_pos[1])

check_spot = {i: (X[i], Y[i]) for i in range(len(X))}
print_DEBUG(f"check_spot:{check_spot}")
# 进入a条铁路有b个车厢，停留时间为c-d分钟
# (a,b,c,d)
train1 = (0,9,0,60)
train2 = (1,5,0,60)
train3 = (0,8,70,130)
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
print_DEBUG(f'checktask:{check_task_dist}') # check_task_dist[num from 0] = [(c,d),check_spot_number]

geneNum = 100
generationNum = 300  # 迭代次数
CENTER = 0  # 配送中心
HUGE = 100000000  # 无穷大
VARY = 0.05  # 变异几率
n = len(check_task_dist)-1  # 检查任务数量 减去第0起点任务
k = 5  # 车辆数量
epu = 20  # 早到惩罚成本
lpu = HUGE  # 晚到惩罚成本
speed = 40  # 速度，km/h
costPerKilo = 1  # 油价
Q = 0
m = 0  # 换电站数量
t = 0.5
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
    @staticmethod
    def _generate(length):
        data = [i for i in range(1, length)]
        random.shuffle(data)
        data.insert(0, CENTER)
        data.append(CENTER)
        return data

    # insert zeors at proper positions
    # use average task num to insert zeros ensure gene will be divided into k robots
    @staticmethod
    def _insertZeros(data):
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
        fit = timeCost = max_time_spent = 0
        dist = []  # from this to next
        timespentlist = []
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
                if timeSpent > max_time_spent:
                    max_time_spent = timeSpent
                    timespentlist.append(timeSpent)
                timeSpent = 0
            # update time spent on road
            timeSpent += (dist[i - 1] / (speed/3.6))
            # arrive early
            if timeSpent < check_task_dist[pos][0][0]:
                timeCost += ((check_task_dist[pos][0][0] - timeSpent) * epu)
                timeSpent = check_task_dist[pos][0][0]
            # arrive late
            elif timeSpent > check_task_dist[pos][0][1]:
                timeCost += HUGE
            # update time
            timeSpent += t

        maxtimespentCost = max_time_spent * 100

        fit = maxtimespentCost + timeCost + distCost + (max(timespentlist)-min(timespentlist))*100 +(max(dist)-min(dist))*100
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
        animate_plot_funcanimation(Xorder, Yorder)



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


def animate_plot_funcanimation(Xorder, Yorder, interval=10):
    """
    使用 FuncAnimation 实现机器人移动的动画仿真。
    将废弃此段，重新设计仿真

    参数：
    - Xorder: 点的X坐标列表
    - Yorder: 点的Y坐标列表
    - base_steps_per_unit: 每单位距离的基础步数，用于控制移动速度
    - interval: 每帧之间的时间间隔（毫秒）
    """



    # 计算每段的距离和对应的步数
    distances = [math.hypot(Xorder[i] - Xorder[i-1], Yorder[i] - Yorder[i-1]) for i in range(1, len(Xorder))]
    steps_per_move = [max(int(distance/(speed*3.6) *1000/interval ), 1) for distance in distances]

    # 创建一个新的绘图
    fig, ax = plt.subplots()

    # 画出所有点，初始为红色
    scat = ax.scatter(Xorder, Yorder, color='red')

    # 设置图形的范围
    ax.set_xlim(min(Xorder) - 20, max(Xorder) + 20)
    y_min = min(Yorder)
    y_max = max(Yorder)
    y_avarage = (min(Ycenter) + max(Ycenter)) / 2
    # 选择 y_min 和 y_max 中离 y_avarage 较远的那个
    if abs(y_min - y_avarage) > abs(y_max - y_avarage):
        y_range = abs(y_min - y_avarage)
    else:
        y_range = abs(y_max - y_avarage)
    scale_factor = 2  # 比例因子，可以根据需要调整
    ax.set_ylim(y_avarage - y_range * scale_factor, y_avarage + y_range * scale_factor)


    # 初始化机器人的位置，开始于第一个点
    robot_marker, = ax.plot([Xorder[0]], [Yorder[0]], marker='o', markersize=10, color='green')

    # 初始化路径线条
    path_line, = ax.plot([], [], color='blue')
    path_x, path_y = [Xorder[0]], [Yorder[0]]

    # 初始化颜色列表
    colors = ['red'] * len(Xorder)
    colors[0] = 'green'
    scat.set_color(colors)

    # Precompute all positions
    positions = []
    # 定义车厢中心位置的列表
    rectangle_centers = []
    for i in range(len(Xcenter)):
        rectangle_centers.append((Xcenter[i], Ycenter[i]))

    # 遍历中心位置并生成车厢
    for center_x, center_y in rectangle_centers:
        # 创建一个长为15宽为4的长方形表示火车车厢
        rect = patches.Rectangle((center_x - (train_box_length-3)/2, center_y - (railway_gap-1)/2), (train_box_length-3), (railway_gap-1), linewidth=1, edgecolor='brown', facecolor='brown')
        ax.add_patch(rect)

    # 用另一种颜色的线将两个相邻车厢连接
    for i in range(len(rectangle_centers) - 1):
        if rectangle_centers[i][1] == rectangle_centers[i + 1][1]:
            center_y = rectangle_centers[i][1]
            center_x1 = rectangle_centers[i][0] + (train_box_length-3)/2  # 左边长方形右边那条边的中间
            center_x2 = rectangle_centers[i + 1][0] - (train_box_length-3)/2  # 右边长方形左边那条边的中间
            ax.plot([center_x1, center_x2], [center_y, center_y], color='brown', linestyle='-', linewidth=2)

    # 刷新图形以显示所有长方形
    plt.draw()
    plt.pause(0.1)  # 短暂暂停以确保长方形显示

    # 遍历点并逐步连接，这里是计算机器人的移动路径
    for i in range(1, len(Xorder)):
        x_start, y_start = Xorder[i - 1], Yorder[i - 1]
        x_end, y_end = Xorder[i], Yorder[i]

        # 计算x和y方向的步数
        x_step = int(abs(x_end - x_start) / (speed) * 1000 / interval)
        y_step = int(abs(y_end - y_start) / (speed) * 1000 / interval)

        # 初始化增量
        delta_x = delta_y = 0
        if x_step != 0:
            delta_x = (x_end - x_start) / x_step
        if y_step != 0:
            delta_y = (y_end - y_start) / y_step

        current_x, current_y = x_start, y_start

        # 添加x方向的中间位置
        for step in range(x_step):
            current_x = x_start + delta_x * step
            positions.append((current_x, current_y))

        # 添加y方向的中间位置
        for step in range(y_step):
            current_y = y_start + delta_y * step
            positions.append((current_x, current_y))

        # 确保最终位置是精确的
        positions.append((x_end, y_end))
        colors[i] = 'green'  # 标记点为已访问

    # Define the update function for FuncAnimation
    def update(frame):
        if frame >= len(positions):
            return robot_marker, path_line

        current_x, current_y = positions[frame]
        robot_marker.set_data([current_x], [current_y])

        # Update the path
        path_x.append(current_x)
        path_y.append(current_y)
        path_line.set_data(path_x, path_y)

        # Update colors if at the end of a move
        # Determine which point is reached
        move_index = 0
        cumulative_steps = 0
        for i, steps in enumerate(steps_per_move):
            if frame < cumulative_steps + steps:
                break
            cumulative_steps += steps
            move_index += 1
            if move_index < len(colors):
                colors[move_index] = 'green'
                scat.set_color(colors)

        return robot_marker, path_line

    ani = animation.FuncAnimation(fig, update, frames=len(positions), interval=interval, blit=True, repeat=False)

    plt.show()

class robot:
    def __init__(self,data,color = 'gray' ):
        self.timestamp = 0
        self.color = color
        self.data = data
        self.check = False
        # load the destination
        self.destinations = []
        self.start_time_list = []
        for i in data:
            self.destinations.append(np.array(check_spot[check_task_dist[i][1]], dtype=np.float64))
            self.start_time_list.append(check_task_dist[i][0])
        # initialize robot position at the first destination (0)
        self.position = np.array(self.destinations.pop(0), dtype=np.float64)  # 车辆的位置
        self.velocity = np.array((0, 0), dtype=np.float64)  # 车辆的速度
    def find_neighbors(self, all_vehicles, communication_range):
        """
        查找在通信范围内的邻居车辆
        """
        neighbors = []
        for vehicle in all_vehicles:
            if vehicle != self:
                distance = np.linalg.norm(self.position - vehicle.position)
                if distance <= communication_range:
                    neighbors.append(vehicle)
        return neighbors
    def update_state(self, neighbors , alpha=0.05):
        if not self.destinations: # 完成任务后不再更新状态
            return
        delta_v = np.array((0.0, 0.0))# initialize delta_v
        if not np.array_equal(self.position, self.destinations[0]):
            delta_v =  self.destinations[0] - self.position
        else:
            if self.check == False:
                self.check = True
                self.temp_timestamp = self.timestamp
                self.color = 'blue'
                return
            else:
                if self.timestamp - self.temp_timestamp >= t * 60 * interval :
                    self.color = 'gray'
                    self.check = False
                    self.destinations.pop(0)
                else:
                    delta_v = self.position
                    self.timestamp += interval
                    return
        # 机器人势场防撞机制
        # repulsion_force = np.array((0.0, 0.0))
        # if neighbors:
        #     for neighbor in neighbors:
        #         repulsion_force -= alpha * (self.position - neighbor.position) /max(1, (np.linalg.norm(self.position - neighbor.position))**2)
        if delta_v[0] != 0 and self.position[1] in Y:
            self.position[0] += delta_v[0] if abs(delta_v[0])<speed / 3.6 * interval else delta_v[0]/abs(delta_v[0]) * speed / 3.6 * interval
        else:
            self.position[1] += delta_v[1] if abs(delta_v[1])<speed/2 / 3.6 * interval else delta_v[1]/abs(delta_v[1]) * speed/2 / 3.6 * interval
        self.timestamp += interval

def run_simulation(robots, communication_range = 0, steps=1000):
    fig, ax = plt.subplots(figsize=(10, 8))
    # 定义车厢中心位置的列表
    rectangle_centers = []
    for i in range(len(Xcenter)):
        rectangle_centers.append((Xcenter[i], Ycenter[i]))



    for step in range(steps):
        ax.clear()  # 清除上一步的绘图
        Xorder ,Yorder = [], []
        for i in check_spot:
            Xorder.append(check_spot[i][0])
            Yorder.append(check_spot[i][1])
        # 画出所有点，初始为红色
        scat = ax.scatter(Xorder, Yorder, color='red')
        # 设置图形的范围
        ax.set_xlim(min(Xorder) - 20, max(Xorder) + 20)
        y_min = min(Yorder)
        y_max = max(Yorder)
        y_avarage = (min(Ycenter) + max(Ycenter)) / 2
        # 选择 y_min 和 y_max 中离 y_avarage 较远的那个
        if abs(y_min - y_avarage) > abs(y_max - y_avarage):
            y_range = abs(y_min - y_avarage)
        else:
            y_range = abs(y_max - y_avarage)
        scale_factor = 2  # 比例因子，可以根据需要调整
        ax.set_ylim(y_avarage - y_range * scale_factor, y_avarage + y_range * scale_factor)
        # 遍历中心位置并生成车厢
        for center_x, center_y in rectangle_centers:
            # 创建一个长为15宽为4的长方形表示火车车厢
            rect = patches.Rectangle((center_x - (train_box_length - 3) / 2, center_y - (railway_gap - 1) / 2),
                                     (train_box_length - 3), (railway_gap - 1), linewidth=1, edgecolor='brown',
                                     facecolor='brown')
            ax.add_patch(rect)

        # 用另一种颜色的线将两个相邻车厢连接
        for i in range(len(rectangle_centers) - 1):
            if rectangle_centers[i][1] == rectangle_centers[i + 1][1]:
                center_y = rectangle_centers[i][1]
                center_x1 = rectangle_centers[i][0] + (train_box_length - 3) / 2  # 左边长方形右边那条边的中间
                center_x2 = rectangle_centers[i + 1][0] - (train_box_length - 3) / 2  # 右边长方形左边那条边的中间
                ax.plot([center_x1, center_x2], [center_y, center_y], color='brown', linestyle='-', linewidth=2)



        for robot in robots:
            robot.update_state(None)


        for i, robot in enumerate(robots):
            ax.plot(robot.position[0], robot.position[1], marker='o', color=robot.color, markersize=6)
        plt.pause(interval/beilv)

    plt.show()


def generate_robots():
    global robots, i
    robots = []
    datas = [0]
    i = 1
    zerocounter = 1
    while i < len(genes[0].data):
        datas.append(genes[0].data[i])
        if genes[0].data[i] == CENTER:
            zerocounter += 1
            if zerocounter == 2:
                robot1 = robot(datas)
                robots.append(robot1)
                datas = [0]
                zerocounter = 1
                print(f'robot1:{robot1.data}')
        i += 1
    return robots



if __name__ == "__main__" :
    genes = getRandomGenes(geneNum) # 初始种群
    # 迭代
    for i in tqdm(range(generationNum)):
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
    robots = generate_robots()
    run_simulation(robots)

