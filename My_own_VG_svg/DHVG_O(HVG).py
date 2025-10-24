import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import networkx as nx

def directed_hvg_fast_O_n(time_series):
    """
    使用 O(n) 线性时间复杂度的栈方法计算定向水平可视图 (DHVG)。

    Args:
        time_series (list or np.array): 输入的时间序列数据。
        
    Returns:
        networkx.DiGraph: 表示DHVG的有向图对象。
    """
    n = len(time_series)
    # 使用有向图 DiGraph
    graph = nx.DiGraph()
    graph.add_nodes_from(range(n))
    
    stack = [] # 栈中存储节点的索引

    # 从左到右遍历每个节点
    for i, y_i in enumerate(time_series):
        # 循环弹出所有比当前节点矮的栈顶节点
        # 由于 j < i，所以是从 j 到 i 的有向边
        while stack and time_series[stack[-1]] < y_i:
            j = stack.pop()
            graph.add_edge(j, i)
        
        # 与此时的栈顶节点（如果存在）建立连接
        if stack:
            j = stack[-1]
            graph.add_edge(j, i)
            
        # 当前节点入栈
        stack.append(i)
        
    return graph

def plot_dhvg(graph, time_series, title):
    """
    一个专门用于绘制 DHVG 结果的函数，图中会使用箭头表示方向。
    """
    n = len(time_series)
    indices = np.arange(n)
    edges = list(graph.edges())

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 9))
    fig.suptitle(title, fontsize=16)

    # --- 子图1: 直方条与有向可视性连线 (复现论文图5a) ---
    ax1.set_title('(a) 直方条')
    ax1.bar(indices, time_series, width=0.5, color='lightgray', edgecolor='dimgray')

    # 标注数值
    for i in range(n):
        y_pos = time_series[i] + 0.08 if time_series[i] < 0.8 else time_series[i] + 0.04
        ax1.text(indices[i], y_pos, f'{time_series[i]:.2f}', ha='center', va='bottom', fontsize=10)

    # 绘制有向水平可视性连线（带箭头）
    for u, v in edges:
        height = min(time_series[u], time_series[v])
        # 绘制从 u 指向 v 的箭头
        ax1.arrow(u, height, v - u, 0, head_width=0.03, head_length=0.25, length_includes_head=True, color='dimgray', lw=1.2)

    ax1.set_xlabel('序号', fontsize=12)
    ax1.set_ylabel('值', fontsize=12)
    ax1.set_xticks(indices)
    ax1.set_xticklabels(indices + 1)
    ax1.set_ylim(0, 1.0)
    ax1.set_xlim(-0.7, n - 0.3)

    # --- 子图2: 网络连边 (复现论文图5b) ---
    ax2.set_title('(b) 网络连边')
    node_y_position = 0
    node_positions = {i: (i, node_y_position) for i in indices}
    ax2.plot(indices, np.full(n, node_y_position), 'ko', markersize=6)

    # 使用 FancyArrowPatch 绘制带箭头的弧线
    for u, v in edges:
        # connectionstyle='arc3,rad=0.3' 创建一个弯曲的弧线
        arrow = FancyArrowPatch(
            posA=node_positions[u], 
            posB=node_positions[v],
            arrowstyle='->',
            connectionstyle='arc3,rad=0.4',
            color='black',
            mutation_scale=15, # 控制箭头大小
            lw=1.5
        )
        ax2.add_patch(arrow)

    ax2.set_aspect('equal')
    ax2.set_ylim(-1.5, 5)
    ax2.set_xlim(-0.7, n - 0.3)
    ax2.axis('off')



    # plt.savefig(r"D:\xupt\paper\可视图VG\VG_Survey_综述\output\DHVG_Directed_Horizontal_Visibility_Graph_O_HVG.png", dpi=600)
    save_path = r"D:\xupt\paper\可视图VG\VG_Survey_综述\VG_algorightm\output_svg\DHVG_Directed_Horizontal_Visibility_Graph_O_HVG"
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    plt.savefig(save_path, format='svg', bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    # 配置Matplotlib以正确显示中文
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 使用论文中的标准时间序列
    ts = np.array([0.35, 0.68, 0.82, 0.18, 0.35, 0.68, 0.82, 0.18, 0.35, 0.68, 0.82, 0.18])
    
    # 使用 O(n) 的快速算法计算有向图
    print("正在使用 O(n) 快速算法计算 DHVG...")
    dhvg_graph = directed_hvg_fast_O_n(ts)

    # 绘制结果
    plot_dhvg(dhvg_graph, ts, "定向水平可视图 (DHVG) - O(n) 快速算法实现")