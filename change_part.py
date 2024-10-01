X = [i*train_box_length for _ in range (0, railway_num+1) for i in range(0, max_train_box_num-1)]
Y = [i*railway_gap  for i in range(0, railway_num+1) for _ in range (0, max_train_box_num-1)]
Xcenter= [i*train_box_length/2 for _ in range (0, railway_num) for i in range(-1, 2*(max_train_box_num-1),2)]
Ycenter = [railway_gap/2*(1+2*i)  for i in range(0, railway_num) for _ in range (-1, 2*(max_train_box_num-1),2)]

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