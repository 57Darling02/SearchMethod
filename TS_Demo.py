# Function: Tabu Search Demo
import random
import numpy as np
import graph_process
from tqdm import tqdm

# 输入条件
# input condition
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


# setting parameters
length_tabulist = 40
candidate_number = 300
loop_times = 3000
tabu_list = []
solution_record = []
originGraph ,optimizedGraph = graph_process.read_graph('original_graph.xlsx')

class Solution:
    def __init__(self, data = None , action = None):
        if data is None:
            self.data = self._get_original_salution()
        else:
            self.data = data
        self.cost = self.get_cost()
        self.action = action
        self.giggity = [] #记录操作流程
    def get_cost(self):
        cost = 0
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
            cost += np.inf
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

        cost += distCost
        return cost

    def _get_original_salution(self):
        data = []
        for i in take_out:
            data.append(i[0])
            data.append(i[1])
        random.shuffle(data)
        return data


    def reverse(self, a, b):
        # 翻转索引a到b-1之间的城市
        self.data = self.data[:a] + self.data[a:b][::-1] + self.data[b:]
        return (0, a, b)
    def swap(self, a, b):
        # 交换两个城市的位置
        self.data[a], self.data[b] = self.data[b], self.data[a]
        return (1, a, b)
    def move(self, a, b):
        # 将城市a移动到城市b之后
        self.data.insert(b+1, self.data.pop(a))
        return (2, a, b)
    def get_neighborhood(self):
        # 随机选择两个城市
        a, b = random.sample(range(len(self.data)), 2)
        option = random.randint(0, 2)
        return (option, a, b)
    def generate_candidate(self , action):
        # 生成候选解
        option , a, b = action
        if option == 0:
            self.reverse(a, b)
        elif option == 1:
            self.swap(a, b)
        else:
            self.move(a, b)
        self.action = action
        self.cost = self.get_cost()



def optimize_solution(best_solution:Solution):
    global tabu_list ,solution_record
    current_solution = best_solution
    posible_solution = [current_solution]
    if len(tabu_list) > length_tabulist:
        for i in range(int(len(tabu_list)/5)):
            tabu_list.pop(i)
    for _ in range(candidate_number):
        new_solution = Solution(current_solution.data)
        neighbor = new_solution.get_neighborhood()
        if neighbor not in tabu_list:
            new_solution.generate_candidate(neighbor)
            if new_solution.cost < current_solution.cost:
                current_solution = new_solution
                posible_solution.append(current_solution)
            continue
    # choose the best solution form the posible_solution
    choose_solution = min(posible_solution, key=lambda x: x.cost)
    if choose_solution.action is not None:
        tabu_list.append(choose_solution.action)
        solution_record.append(choose_solution)
    else:
        print('No neighbor found')
    return choose_solution

if __name__ == "__main__" :
    best_solution =Solution()
    for i in tqdm(range(loop_times)):
        best_solution = optimize_solution(best_solution)
    print(best_solution.data)
    print(best_solution.cost)
    print(best_solution.giggity)
    for i in best_solution.giggity:
        for j in i:
            if j[0] != '到达':
                index = take_out.index(j[1])
                print(f'{j[0]}:任务编号{index}')
            else:
                print(f'{j[0]}:{j[1]}')
