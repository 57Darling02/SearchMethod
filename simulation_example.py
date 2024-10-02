import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
colors = plt.get_cmap('cool', 5)
dt = 0.05  # 时间步长
lane_width = 4  # 每条车道的宽度
lane_centers_y = [2, 6, 10, 14, 18]  # 多车道中心位置
vehicle_width = 1  # 车辆宽度的一半


# 定义车辆类
class Vehicle:
    def __init__(self, position, velocity, delta):
        self.position = np.array(position, dtype=np.float64)  # 车辆的位置
        self.velocity = np.array(velocity, dtype=np.float64)  # 车辆的速度
        self.delta = np.array(delta, dtype=np.float64)  # 车辆在编队中的相对位置

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

    def update_state(self, neighbors, leader, alpha=0.05, beta=0.1, repulsion_force=2.5, lane_attraction_force=0.005):
        """
        更新车辆状态，加入多车道吸引力和碰撞避免
        """
        delta_v = np.array([0.0, 0.0])

        # 势场法 - 保持在最近的车道中心
        closest_lane_center = min(lane_centers_y, key=lambda center: abs(center - self.position[1]))  # 最近车道中心
        lane_center_force = np.array([0, (closest_lane_center - self.position[1]) ** 2])

        # 这里还要修改一下保持在道路中心的函数，不然会飘出去！

        delta_v += lane_attraction_force * lane_center_force

        # 势场法 - 车道边界排斥力，防止车辆越出车道
        if self.position[1] - vehicle_width < min(lane_centers_y) - lane_width / 2 + 0.5:
            delta_v += repulsion_force * np.array(
                [0, (min(lane_centers_y) - lane_width / 2) - (self.position[1] - vehicle_width)])
        elif self.position[1] + vehicle_width > max(lane_centers_y) + lane_width / 2 - 0.5:
            delta_v += repulsion_force * np.array(
                [0, (max(lane_centers_y) + lane_width / 2) - (self.position[1] + vehicle_width)])


        if neighbors:
            for neighbor in neighbors:
                delta_v -= alpha * (self.position - neighbor.position - self.delta + neighbor.delta)
                delta_v -= beta * (self.velocity - neighbor.velocity)

        if leader:
            delta_v -= alpha * (self.position - leader.position - self.delta) * 2
            delta_v -= beta * (self.velocity - leader.velocity) * 2

        # 限制速度的变化量
        if np.linalg.norm(delta_v) > 0.05:
            delta_v = delta_v * 0.05 / np.linalg.norm(delta_v)

        self.velocity += delta_v
        if np.linalg.norm(self.velocity) > 20:
            self.velocity = self.velocity * 20 / np.linalg.norm(self.velocity)
        if np.linalg.norm(self.velocity) < 10:
            self.velocity = self.velocity * 10 / np.linalg.norm(self.velocity)

        self.position += self.velocity * dt


# 虚拟车辆
class VirtualVehicle(Vehicle):
    def __init__(self, vehicles, offset=2.5):
        max_x = np.max([vehicle.position[0] for vehicle in vehicles])
        avg_y = np.mean([vehicle.position[1] for vehicle in vehicles])
        des_y = min(lane_centers_y, key=lambda center: abs(center - avg_y))
        position = np.array([max_x + offset, des_y])
        velocity = np.array([15, 0])
        delta = np.array([0, 0])
        super().__init__(position, velocity, delta)

    def update_virtual_vehicle(self, vehicles):
        self.position += self.velocity * dt


# 初始化车辆和虚拟车
vehicles = [
    Vehicle([6, 6], [14, 0.0], [2, -4]),
    Vehicle([0.5, 10], [15, 0.0], [-4, 4]),
    Vehicle([15, 13], [13, 0.0], [-4, -4]),
    Vehicle([6, 17], [16, 0.0], [2, 4]),
    Vehicle([25, 2], [14, 0.0], [8, 0])
]

virtual_vehicle = VirtualVehicle(vehicles)
communication_range = 50.0


# 执行一致性算法
def run_consensus(vehicles, communication_range, steps=1000):
    fig, ax = plt.subplots(figsize=(10, 8))

    for step in range(steps):
        ax.clear()  # 清除上一步的绘图

        # 获取所有车辆的最小和最大 x, y 值，以调整坐标轴
        all_x = [vehicle.position[0] for vehicle in vehicles]
        all_y = [vehicle.position[1] for vehicle in vehicles]
        min_x, max_x = min(all_x), max(all_x)
        min_y, max_y = min(all_y), max(all_y)

        # 设置动态的坐标轴范围，留出一定的边距
        ax.set_xlim(min_x - 10, max_x + 10)
        ax.set_ylim(-2, 22)  # 固定纵坐标范围

        # 绘制每条车道的边界
        for lane_center in lane_centers_y:
            road_y_min = lane_center - lane_width / 2
            road_y_max = lane_center + lane_width / 2
            ax.plot([min_x - 10, max_x + 10], [road_y_min, road_y_min], 'k--')  # 道路下边界
            ax.plot([min_x - 10, max_x + 10], [road_y_max, road_y_max], 'k--')  # 道路上边界

        if step < 100:
            for vehicle in vehicles:
                vehicle.update_state(None, None)
        else:
            for vehicle in vehicles:
                neighbors = vehicle.find_neighbors(vehicles, communication_range)
                vehicle.update_state(neighbors, virtual_vehicle)

        virtual_vehicle.update_virtual_vehicle(vehicles)

        for i, vehicle in enumerate(vehicles):
            ax.plot(vehicle.position[0], vehicle.position[1], marker='o', color=colors(i), markersize=6)

        plt.pause(0.01)

    plt.show()


# 运行一致性算法
run_consensus(vehicles, communication_range, steps=800)
