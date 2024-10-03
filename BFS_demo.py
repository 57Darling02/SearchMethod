import numpy as np
import graph_process
originGraph ,optimizedGraph = graph_process.read_graph('original_graph.xlsx')

print(f"originGraph:{originGraph}")

# we can use BFS to find the end_node
def BFS(graph, start, end):
    # create a queue
    queue = []
    # create a list to store the visited nodes
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
        if node == end:
            return path
        # get the neighbors of the node
        for x in range(len(graph)):
            # if the node is not visited and the distance between the node and the neighbor is not infinite
            if x not in visited and graph[node][x] != float('inf'):
                # add the neighbor to the queue
                queue.append(x)
                visited.append(x)
    return path # 得出搜索路径



def DFS_in_loop(graph, start, end):
    # create a stack
    stack = []
    # create a list to store the visited nodes
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
        if node == end:
            return path
        # get the neighbors of the node
        for x in range(len(graph)):
            # if the node is not visited and the distance between the node and the neighbor is not infinite
            if x not in visited and graph[node][x] != float('inf'):
                # add the neighbor to the stack
                stack.append(x)
                visited.append(x)
    return path # 得出搜索路径 没找到


def DFS_in_recursion(graph, start, end, method_type):
    # create a list to store the path
    path = []
    # if method_type is 0 ,we use pre-order
    if method_type == 0:
        def dfs(graph, node, visited=None):
            nonlocal path
            if visited is None:
                visited = set()  # 初始化已访问节点集合
            if node not in visited:
                visited.add(node)  # 将当前节点标记为已访问
                path.append(node)  # 访问当前节点（你可以在这里进行其他处理）
                if node == end:
                    return True
                # get the neighbors of the node
                for x in range(len(graph)):
                    # if the node is not visited and the distance between the node and the neighbor is not infinite
                    if x not in visited and graph[node][x] != float('inf'):
                        if dfs(graph, x, visited):
                            return True
            return False
    # if method_type is 1 ,we use post-order
    else:
        def dfs(graph, node, visited=None):
            nonlocal path
            if visited is None:
                visited = set()  # 初始化已访问节点集合
            if node not in visited:
                visited.add(node)  # 将当前节点标记为已访问
                # get the neighbors of the node
                for x in range(len(graph)):
                    # if the node is not visited and the distance between the node and the neighbor is not infinite
                    if x not in visited and graph[node][x] != float('inf'):
                        if dfs(graph, x, visited):
                            return True
                path.append(node)  # 访问当前节点（你可以在这里进行其他处理）
                if node == end:
                    return True
            return False
    dfs(graph, start)
    return path # 得出搜索路径


def UniformCostSearch(graph, start, end):

    pass

# we can use Dijkstra to find the path
def Dijkstra(ori_graph, start, end):
    search_path = []
    n = len(ori_graph)
    dist=np.full(n, np.inf)
    # record the previous node
    pred=np.full(n,-1)
    visitd = []
    dist[start]=0
    for i in range(n-1):
        min = np.inf
        min_index = -1
        for j in range(n):
            if j not in visitd and dist[j] <= min:
                min = dist[j]
                min_index = j
        visitd.append(min_index)
        for j in range(n):
            if ori_graph[min_index][j] !=np.inf and (j not in visitd) and dist[min_index]!=np.inf and dist[min_index] + ori_graph[min_index][j] < dist[j]:
                dist[j]=dist[min_index] + ori_graph[min_index][j]
                pred[j]=min_index
        search_path.append(min_index)
        if min_index == end:
            print('Dijkstra find the end node')
            break
    current = end
    path =[]
    while(current!=start):
        current = pred[current]
        path.insert(0,current)
    return path ,search_path


# we assume that the start point is 0 (v1)
# and the end point is 36 (v37)
path = BFS(originGraph, 0, 36)
print(f"BFS search_path:{path} \nlen:{len(path)}")


path = DFS_in_loop(originGraph, 0, 36)
print(f"DFS_in_loop search_path:{path} \nlen:{len(path)}")
a = len(set(path))
path = DFS_in_recursion(originGraph, 0, 36, 0)
print(f"DFS_in_recursion0 search_path:{path} \nlen:{len(path)}")

path = DFS_in_recursion(originGraph, 0, 36, 1)
print(f"DFS_in_recursion1 search_path:{path} \nlen:{len(path)}")
#
path ,search_path = Dijkstra(originGraph, 0, 36)
print(f'Dijkstra_Path:{path}\nsearch_path:{search_path}\nlen:{len(search_path)}')


