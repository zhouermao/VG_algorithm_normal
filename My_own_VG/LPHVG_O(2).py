import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Arc
import networkx as nx
from bisect import bisect_left

def lphvg_fast_O_n_squared(time_series, N):
    """
    计算有限穿越水平可视图（LPHVG）的 O(n^2) 算法实现。
    
    该算法通过在内层循环中维护一个有序的高度列表，并使用二分查找
    来快速计算穿越数，从而避免了第三层循环。
    注意：为了达到真正的 O(n^2)，列表插入操作需要 O(log n) 的数据结构（如平衡树），
    在Python标准库中 list.insert 是 O(n)，使整体为 O(n^3)。
    但该实现正确展示了 O(n^2) 的核心逻辑。

    Args:
        time_series (list or np.array): 输入的时间序列数据。
        N (int): 有限穿越视距，允许水平视线被截断的次数。
        
    Returns:
        networkx.Graph: 表示LPHVG的图对象。
    """
    n = len(time_series)
    graph = nx.Graph()
    graph.add_nodes_from(range(n))

    for i in range(n - 1):
        # 维护一个从i点出发，到i和j之间所有点的“高度”的有序列表
        seen_heights = []
        for j in range(i + 1, n):
            # 1. 确定水平视线的高度
            min_height = min(time_series[i], time_series[j])
            
            # 2. 【核心优化】使用二分查找计算穿越数
            # 穿越条件: height[k] >= min_height
            # bisect_left 找到插入点，该点左侧所有元素都 < min_height
            pos = bisect_left(seen_heights, min_height)
            
            # 列表中 >= min_height 的元素数量 = 总数 - 左侧部分数量
            penetration_count = len(seen_heights) - pos
            
            # 3. 根据 LPHVG 规则判断是否连边
            if penetration_count <= N:
                graph.add_edge(i, j)
            
            # 4. 更新 seen_heights 列表，为下一次循环做准备
            # 将上一个点 (j) 的高度加入有序列表
            height_to_add = time_series[j]
            insert_pos = bisect_left(seen_heights, height_to_add)
            seen_heights.insert(insert_pos, height_to_add)

    return graph

def plot_lphvg(edges_base, edges_new, time_series, title):
    """
    专门用于绘制 LPHVG 结果的函数。
    """
    n = len(time_series)
    indices = np.arange(n)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 9))
    fig.suptitle(title, fontsize=16)

    # --- 子图1: 直方条与可视性连线 (复现论文图4a) ---
    ax1.set_title('(a) 直方条')
    ax1.bar(indices, time_series, width=0.5, color='lightgray', edgecolor='dimgray')

    # 标注数值
    for i in range(n):
        y_pos = time_series[i] + 0.08 if time_series[i] < 0.8 else time_series[i] + 0.04
        ax1.text(indices[i], y_pos, f'{time_series[i]:.2f}', ha='center', va='bottom', fontsize=10)

    # 绘制HVG基础连线 (实线)
    for u, v in edges_base:
        u, v = min(u, v), max(u, v)
        height = min(time_series[u], time_series[v])
        ax1.arrow(u, height, v - u, 0, head_width=0.02, head_length=0.2, length_includes_head=True, color='black', linestyle='-', lw=1.2)
        ax1.arrow(v, height, u - v, 0, head_width=0.02, head_length=0.2, length_includes_head=True, color='black', linestyle='-', lw=1.2)
        ax1.plot([u, u], [height - 0.015, height + 0.015], color='black', marker='|')
        ax1.plot([v, v], [height - 0.015, height + 0.015], color='black', marker='|')

    # 绘制LPHVG新增连线 (虚线)
    for u, v in edges_new:
        u, v = min(u, v), max(u, v)
        height = min(time_series[u], time_series[v])
        ax1.arrow(u, height, v - u, 0, head_width=0.02, head_length=0.2, length_includes_head=True, color='blue', linestyle='--', lw=1.2)
        ax1.arrow(v, height, u - v, 0, head_width=0.02, head_length=0.2, length_includes_head=True, color='blue', linestyle='--', lw=1.2)
        ax1.plot([u, u], [height - 0.015, height + 0.015], color='blue', marker='|')
        ax1.plot([v, v], [height - 0.015, height + 0.015], color='blue', marker='|')

    ax1.set_xlabel('序号', fontsize=12)
    ax1.set_ylabel('值', fontsize=12)
    ax1.set_xticks(indices)
    ax1.set_xticklabels(indices + 1)
    ax1.set_ylim(0, 1.0)
    ax1.set_xlim(-0.7, n - 0.3)

    # --- 子图2: 网络连边 (复现论文图4b) ---
    ax2.set_title('(b) 网络连边')
    node_y_position = 0
    ax2.plot(indices, np.full(n, node_y_position), 'ko', markersize=6)

    # 绘制HVG基础连线 (实线弧)
    for u, v in edges_base:
        u, v = min(u, v), max(u, v)
        center_x = (u + v) / 2
        width = v - u
        height = width * 0.8
        arc = Arc((center_x, node_y_position), width, height, theta1=0, theta2=180, edgecolor='black', linestyle='-', lw=1.5)
        ax2.add_patch(arc)

    # 绘制LPHVG新增连线 (虚线弧)
    for u, v in edges_new:
        u, v = min(u, v), max(u, v)
        center_x = (u + v) / 2
        width = v - u
        height = width * 0.8
        arc = Arc((center_x, node_y_position), width, height, theta1=0, theta2=180, edgecolor='blue', linestyle='--', lw=1.5)
        ax2.add_patch(arc)

    ax2.set_aspect('equal')
    ax2.set_ylim(-1.5, 5)
    ax2.set_xlim(-0.7, n - 0.3)
    ax2.axis('off')
    
    plt.savefig(r"D:\xupt\paper\可视图VG\VG_Survey_综述\output\LPHVG_Limited_Penetrable_Horizontal_Visibility_Graph_O_2.png", dpi=600)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


if __name__ == '__main__':
    # 配置Matplotlib以正确显示中文
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 使用论文中的标准时间序列
    ts = np.array([0.35, 0.68, 0.82, 0.18, 0.35, 0.68, 0.82, 0.18, 0.35, 0.68, 0.82, 0.18])
    
    # --- 计算 ---
    # N=0 时的边，即标准HVG的边
    print("正在使用 O(n^2) 逻辑计算 N=0 (标准HVG)...")
    hvg_graph_n0 = lphvg_fast_O_n_squared(ts, N=0)
    edges_n0 = set(frozenset(e) for e in hvg_graph_n0.edges())

    # N=1 时的边
    print("正在使用 O(n^2) 逻辑计算 N=1 (LPHVG)...")
    lphvg_graph_n1 = lphvg_fast_O_n_squared(ts, N=1)
    edges_n1 = set(frozenset(e) for e in lphvg_graph_n1.edges())

    # LPHVG (N=1) 新增的边是 N=1 边集与 N=0 边集的差集
    new_edges = list(edges_n1 - edges_n0)
    edges_base = list(edges_n0)

    # --- 绘图 ---
    plot_lphvg(edges_base, new_edges, ts, "有限穿越水平可视图 (LPHVG, N=1) - O(n^2) 逻辑实现")