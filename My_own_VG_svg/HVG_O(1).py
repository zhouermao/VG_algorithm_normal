import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Arc
import networkx as nx

def hvg_fast_O_n(time_series):
    """
    使用基于栈的单调结构计算水平可视图（HVG）。
    该算法的时间复杂度为 O(n)，是最高效的实现。

    Args:
        time_series (list or np.array): 输入的时间序列数据。
        
    Returns:
        networkx.Graph: 表示HVG的图对象。
    """
    n = len(time_series)
    graph = nx.Graph()
    graph.add_nodes_from(range(n))
    
    stack = [] # 栈中存储节点的索引

    # 从左到右遍历每个节点
    for i, y_i in enumerate(time_series):
        # 当前节点y_i比栈顶节点更高, 循环弹出所有比当前节点矮的栈顶节点
        while stack and time_series[stack[-1]] < y_i:
            j = stack.pop()
            graph.add_edge(j, i)
        
        # 此时的栈顶节点（如果存在）必然比当前节点高或相等
        if stack:
            j = stack[-1]
            graph.add_edge(j, i)
            
        # 当前节点入栈，等待后续连接
        stack.append(i)
        
    return graph

def plot_hvg(graph, time_series, title):
    """
    一个通用的函数，用于绘制 HVG 的两种标准图形。
    Args:
        graph (nx.Graph): networkx 图对象。
        time_series (np.array): 原始时间序列数据。
        title (str): 图形的主标题。
    """
    n = len(time_series)
    indices = np.arange(n)
    edges = list(graph.edges())

    # 创建一个包含两个子图的画布
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 9))
    fig.suptitle(title, fontsize=16)

    # --- 子图1: 直方条与可视性连线 (复现论文图2a) ---
    ax1.set_title('(a) 直方条')
    ax1.bar(indices, time_series, width=0.5, color='lightgray', edgecolor='dimgray')

    # 在直方条顶部标注数值
    for i in range(n):
        y_pos = time_series[i] + 0.08 if time_series[i] < 0.8 else time_series[i] + 0.04
        ax1.text(indices[i], y_pos, f'{time_series[i]:.2f}', ha='center', va='bottom', fontsize=10)

    # 绘制水平可视性连线（带箭头）
    for u, v in edges:
        u, v = min(u,v), max(u,v)
        height = min(time_series[u], time_series[v])
        ax1.arrow(u, height, v - u, 0, head_width=0.02, head_length=0.2, length_includes_head=True, color='dimgray', lw=1.2)
        ax1.arrow(v, height, u - v, 0, head_width=0.02, head_length=0.2, length_includes_head=True, color='dimgray', lw=1.2)
        ax1.plot([u, u], [height - 0.015, height + 0.015], color='black', marker='|')
        ax1.plot([v, v], [height - 0.015, height + 0.015], color='black', marker='|')

    ax1.set_xlabel('序号', fontsize=12)
    ax1.set_ylabel('值', fontsize=12)
    ax1.set_xticks(indices)
    ax1.set_xticklabels(indices + 1)
    ax1.set_ylim(0, 1.0)
    ax1.set_xlim(-0.7, n - 0.3)

    # --- 子图2: 网络连边 (复现论文图2b) ---
    ax2.set_title('(b) 网络连边')
    node_y_position = 0
    ax2.plot(indices, np.full(n, node_y_position), 'ko', markersize=6)
    for u, v in edges:
        u, v = sorted((u, v))
        center_x = (u + v) / 2
        width = v - u
        height = width * 0.8
        arc = Arc((center_x, node_y_position), width, height, theta1=0, theta2=180, edgecolor='black', lw=1.5)
        ax2.add_patch(arc)
    ax2.set_aspect('equal')
    ax2.set_ylim(-1.5, 5)
    ax2.set_xlim(-0.7, n - 0.3)
    ax2.axis('off')


    save_path = r"D:\xupt\paper\可视图VG\VG_Survey_综述\VG_algorightm\output_svg\HVG_Horizontal_Visibility_Graph_O_1.svg"
    plt.tight_layout(rect=[0, 0, 1, 0.95]) # 这一行其实可以放在savefig之前
    plt.savefig(save_path, format='svg', bbox_inches='tight')
    plt.show()




if __name__ == '__main__':
    # 配置Matplotlib以正确显示中文
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    # 使用论文中的标准时间序列
    ts = np.array([0.35, 0.68, 0.82, 0.18, 0.35, 0.68, 0.82, 0.18, 0.35, 0.68, 0.82, 0.18])

    # 使用 O(n) 的快速算法计算可视图
    print("正在使用 O(n) 快速算法计算 HVG...")
    hvg_graph = hvg_fast_O_n(ts)

    # 绘制结果
    plot_hvg(hvg_graph, ts, "水平可视图 (HVG) - O(n) 快速算法实现")