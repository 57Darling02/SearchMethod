import math
import numpy as np
M = math.inf
import pandas as pd
# ues floydWarshall algorithm to calculate the shortest path for each node
def floydWarshall(graph):
    # 获取顶点的数量
    n = len(graph)
    # 初始化距离矩阵和路径矩阵
    dist = [row[:] for row in graph]
    # 执行Floyd-Warshall算法
    for k in range(n):  # 对于每一个顶点作为中间顶点
        for i in range(n):  # 遍历每一个起始顶点
            for j in range(n):  # 遍历每一个终点
                if dist[i][j] > dist[i][k] + dist[k][j]:
                    dist[i][j] = dist[i][k] + dist[k][j]
    return dist


def read_graph(filename='data.xlsx'):
    # 读取Excel文件
    df = pd.read_excel(filename, sheet_name=0, header=None)
    # 将DataFrame中的所有'M'替换为numpy.inf
    df.replace('M', M, inplace=True)
    # 如果需要将数据转换为二维数组（numpy数组）
    graph = np.copy(df.values)
    dist = floydWarshall(df.values)
    np.savetxt("opt_graph.txt", dist)
    np.savetxt("ori_graph.txt", graph)
    return np.array(graph), np.array(dist)