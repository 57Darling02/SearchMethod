import heapq
import graph_process
originGraph ,optimizedGraph = graph_process.read_graph('original_graph.xlsx')
print(f"originGraph:{originGraph}")


class Node:
    def __init__(self, id, parent=None):
        self.id = id
        self.name = f"v{id+1}"
        self.parent = parent
        self.g = 0
        self.h = 0
        # f = g + h
        self.f = 0

def heuristic(current, goal):
    # 曼哈顿距离作为启发函数
    return abs(current.position[0] - goal.position[0]) + abs(current.position[1] - goal.position[1])

def astar(maze, start, end):
    open_list = []
    closed_list = set()
    heapq.heappush(open_list, (0, start))

    while open_list:
        current_cost, current_node = heapq.heappop(open_list)

        if current_node == end:
            path = []
            while current_node:
                path.append(current_node.position)
                current_node = current_node.parent
            return path[::-1]

        closed_list.add(current_node)

        for next_pos in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
            new_position = (current_node.position[0] + next_pos[0], current_node.position[1] + next_pos[1])

            if new_position[0] < 0 or new_position[0] >= len(maze) or \
                new_position[1] < 0 or new_position[1] >= len(maze[0]) or \
                maze[new_position[0]][new_position[1]] == 1:
                continue

            new_node = Node(new_position, current_node)
            new_node.g = current_node.g + 1
            new_node.h = heuristic(new_node, end)
            new_node.f = new_node.g + new_node.h

            if new_node in closed_list:
                continue

            heapq.heappush(open_list, (new_node.f, new_node))

    return None

# 示例迷宫
maze = [
    [0, 0, 0, 0, 0],
    [0, 1, 1, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 1, 1, 1, 0],
    [0, 0, 0, 0, 0]
]

start_node = Node((0, 0))
end_node = Node((4, 4))

path = astar(maze, start_node, end_node)
print("A* Path:", path)
