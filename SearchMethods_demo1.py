import networkx as nx
import numpy as np

import graph_process

originGraph ,optimizedGraph = graph_process.read_graph('original_graph.xlsx')
print(f"originGraph:{originGraph}")

def real_path_len(path):
    length = 0
    for i in range(len(path)-1):
        length += originGraph[path[i]][path[i+1]]
    return length
class Node:
    def __init__(self, id , position = None):
        self.id = id
        self.name = f"v{id+1}"
        self.position = position
        self.neighbor = []
        self.inhabitant = []
        if id == 36:
            self.inhabitant.append("A")
        self.f = 0
    def search(self,name):
        if name in self.inhabitant:
            return True
        else:
            return False
    def set_position(self, position):
        self.position = position


Nodes = {}
a = [0,1,1,2,2,1,1,1]
b = [0,1,2,2,1,1,1,1,1]
c = [[1,3],[4,8],[9,12],[13,17],[18,23],[24,31],[32,36],[37,41]]
d= [[4,18,24],[19,25,32,37],[5,13],[1,9,14,26,33,38],[10,15,27,34,39],[2,6,11,16,20,28,35,40],[3,7,12,17,21,29],[22,30,36,41],[8,23,31]]
def getpos(i):
    x = None
    y = None
    for j in range(len(c)):
        if i <= c[j][1] and i >= c[j][0]:
            x = a[j]
    for j in range(len(d)):
        if i in d[j]:
            y = b[j]
    if x==None or y == None:
        print(f"error:{i},x{x},y{y}")
    return (x,y)
for i in range(len(originGraph)):
    node = Node(i, getpos(i+1))
    for j in range(len(originGraph)):
        if originGraph[i][j] != float('inf'):
            node.neighbor.append(j)
    Nodes[i] = node
start_node = 0




# we can use BFS to find the end_node
def BFS(start, target):
    # create a queue
    queue = []
    # create a list to store the visited nodes and the node which is planned to visit
    visited = [start]
    # create a list to store the path
    path = []
    # add the start node to the queue
    queue.append(start)
    # while the queue is not empty
    while queue:
        # get the first element in the queue
        node = queue.pop(0)
        # add the node to the path
        path.append(node)
        # if the node is the end node
        # search current node
        if Nodes[node].search(target):
            return path
        # get the neighbors of the node
        for neighbor_node in Nodes[node].neighbor:
            # if the node is not visited and the distance between the node and the neighbor are not infinite
            if neighbor_node not in visited:
                # add the neighbor to the queue
                queue.append(neighbor_node)
                visited.append(neighbor_node)
    return path # 得出搜索路径
path = BFS(start_node, "A")
print(f"BFS search_path:{path} \nlen:{len(path)}")

def DFS_loop(start, target):
    # create a stack
    stack = []
    # create a list to store the visited nodes and the node which is planned to visit
    visited = [start]
    # create a list to store the path
    path = []
    # add the start node to the stack
    stack.append(start)
    # while the stack is not empty
    while stack:
        # get the last element in the stack
        # if method_type is 0 ,we use pre-order
        node = stack.pop()
        # add the node to the path
        path.append(node)
        # if the node is the end node
        if Nodes[node].search(target):
            return path
        # get the neighbors of the node
        for neighbor_node in Nodes[node].neighbor:
            # if the node is not visited and the distance between the node and the neighbor are not infinite
            if neighbor_node not in visited:
                # add the neighbor to the stack
                stack.append(neighbor_node)
                visited.append(neighbor_node)
    return path
path = DFS_loop(start_node, "A")
print(f"DFS_loop search_path:{path} \nlen:{len(path)}")

def DFS_recur(start, target, visited=None, path=None):
    if path is None:
        path = []
    if visited is None:
        visited = []
    visited.append(start)
    path.append(start)
    if Nodes[start].search(target):
        return path
    for neighbor_node in Nodes[start].neighbor:
        if neighbor_node not in visited:
            result = DFS_recur(neighbor_node, target, visited, path)
            if result:
                return result
    return None
path = DFS_recur(start_node, "A")
print(f"DFS_recur_preorder search_path:{path} \nlen:{len(path)}")

def DFS_recur_post_order(start, target, visited=None, path=None):
    if path is None:
        path = []
    if visited is None:
        visited = []
    visited.append(start)
    for neighbor_node in Nodes[start].neighbor:
        if neighbor_node not in visited:
            result = DFS_recur_post_order(neighbor_node, target, visited, path)
            if result:
                return result
    path.append(start)
    if Nodes[start].search(target):
        return path
    return None
path = DFS_recur_post_order(start_node, "A")
print(f"DFS_recur_post_order search_path:{path} \nlen:{len(path)}")

# we can use Dijkstra to find the path
def UniformCostSearch(ori_graph, start, target):
    dist={} # record the distance from start to the node
    # record the previous node
    pred={}
    search_path = [start] # search_path use to record the order which the node is real visited
    if Nodes[start].search(target):
        path = [start]
        return search_path , path
    dist[start]=0
    while True:
        min = np.inf
        min_index = -1
        for visited_node in search_path:
            for neighbor_node in Nodes[visited_node].neighbor:
                if neighbor_node not in search_path and dist[visited_node] + ori_graph[visited_node][neighbor_node] < dist.get(neighbor_node, np.inf):
                    dist[neighbor_node] = dist[visited_node] + ori_graph[visited_node][neighbor_node]
                    pred[neighbor_node] = visited_node
                if neighbor_node not in search_path and dist[neighbor_node] <= min:
                    min = dist[neighbor_node]
                    min_index = neighbor_node
        search_path.append(min_index)
        if Nodes[min_index].search(target):
            print('find the end node')
            break
        if len(search_path) == len(Nodes):
            print('can not find the A')
            return search_path, []

    current = search_path[-1]
    path =[current] # real path form start to end
    while(current!=start):
        current = pred[current]
        path.insert(0,current)
    return path ,search_path
# run code
path ,search_path = UniformCostSearch(originGraph, 0, 'A')
print(f'Dijkstra_Path:{path}\nuniform cost search_path:{search_path}\nlen:{len(search_path)}')
print(f"real path length:{real_path_len(path)}")

def detect(target):
    return 36

def heuristic(node, target):
    # 使用欧几里得距离作为启发式函数
    return np.linalg.norm(np.array(Nodes[node].position) - np.array(Nodes[target].position))
def heuristic_manhattan(node, target):
    # 使用曼哈顿距离作为启发式函数
    return np.sum(np.abs(np.array(Nodes[node].position) - np.array(Nodes[target].position)))


def trynext(graph, current_node, try_node, target_node):
    g_score = graph[current_node][try_node]
    h_score = heuristic(try_node, target_node)
    f_score = g_score + h_score
    return f_score

def A_star_search(graph, start, target):
    current_node = start
    path = [current_node]
    search_path = [current_node]
    while not Nodes[current_node].search(target):
        f_score = np.inf
        next_node = None
        for neighbor_node in Nodes[current_node].neighbor:
            if neighbor_node not in path:
                search_path.append(neighbor_node)
                f = trynext(graph, current_node, neighbor_node, detect(target))
                if f < f_score:
                    f_score = f
                    next_node = neighbor_node
        if next_node is None:
            print('can not find the A')
            break
        path.append(next_node)
        current_node = next_node
    return path,search_path

path , search_path = A_star_search(originGraph, start_node, 'A')
print(f'path:{path}\nlen:{len(path)}\nsearchpath{search_path}\nlen:{len(search_path)}')
print(f"real path length:{real_path_len(path)}")
