import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Arc
import networkx as nx
import math

def plot_whvg(graph, time_series, title):
    """
    一个专门用于绘制加权水平可视图（WHVG）的函数。
    边的颜色表示其权重。
    """
    n = len(time_series)
    indices = np.arange(n)
    edges = list(graph.edges(data=True))

    if not edges: # 处理没有边的情况
        weights = []
        min_w, max_w = 1, 10 # 设定一个默认范围
    else:
        weights = [data['weight'] for _, _, data in edges]
        min_w, max_w = min(weights), max(weights)
        
    # 设置颜色映射 (viridis 是一个很好的连续色图)
    cmap = plt.cm.viridis 
    norm = plt.Normalize(vmin=min_w, vmax=max_w)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    fig.suptitle(title, fontsize=16)

    # --- 子图1: 带权重的直方条图 ---
    ax1.set_title('(a) 直方条')
    ax1.bar(indices, time_series, width=0.5, color='lightgray', edgecolor='dimgray')

    for u, v, data in edges:
        u, v = min(u,v), max(u,v)
        color = cmap(norm(data['weight']))
        height = min(time_series[u], time_series[v])
        ax1.plot([u, v], [height, height], color=color, linewidth=2)

    ax1.set_xlabel('序号', fontsize=12)
    ax1.set_ylabel('值', fontsize=12)
    ax1.set_xticks(indices)
    ax1.set_xticklabels(indices + 1)
    
    # 添加颜色条作为图例
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = fig.colorbar(sm, ax=ax1, orientation='vertical')
    cbar.set_label('权重', fontsize=12)

    # --- 子图2: 带权重的网络连边图 ---
    ax2.set_title('(b) 网络连边')
    node_y_position = 0
    ax2.plot(indices, np.full(n, node_y_position), 'ko', markersize=6)
    for u, v, data in edges:
        u, v = sorted((u, v))
        color = cmap(norm(data['weight']))
        center_x = (u + v) / 2
        width = v - u
        height = width * 0.8
        arc = Arc((center_x, node_y_position), width, height, theta1=0, theta2=180, edgecolor=color, lw=2)
        ax2.add_patch(arc)
    ax2.set_aspect('equal')
    ax2.set_ylim(-1.5, 5)
    ax2.set_xlim(-0.7, n - 0.3)
    ax2.axis('off')
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


def weighted_hvg_fast_O_n(time_series):
    """
    使用 O(n) 栈方法计算加权水平可视图 (WHVG)。
    """
    n = len(time_series)
    graph = nx.Graph()
    graph.add_nodes_from(range(n))
    
    stack = [] # 栈中存储节点的索引

    for i, y_i in enumerate(time_series):
        # 循环弹出所有比当前节点矮的栈顶节点
        while stack and time_series[stack[-1]] < y_i:
            j = stack.pop()
            # 计算权重并添加
            weight = abs((time_series[j] - y_i) * (j - i)) + 1
            graph.add_edge(j, i, weight=weight)
        
        # 与此时的栈顶节点（如果存在）建立连接
        if stack:
            j = stack[-1]
            weight = abs((time_series[j] - y_i) * (j - i)) + 1
            graph.add_edge(j, i, weight=weight)
            
        # 当前节点入栈
        stack.append(i)
        
    return graph


if __name__ == '__main__':
    # 配置Matplotlib以正确显示中文
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 使用论文中的标准时间序列
    ts = np.array([0.35, 0.68, 0.82, 0.18, 0.35, 0.68, 0.82, 0.18, 0.35, 0.68, 0.82, 0.18])
    
    print("正在使用 O(n) 算法计算 WHVG...")
    whvg_graph = weighted_hvg_fast_O_n(ts)

    print(f"\n计算完成。节点数: {whvg_graph.number_of_nodes()}, 边数: {whvg_graph.number_of_edges()}")
    # 打印其中几条边和它们的权重
    print("部分边的权重示例:")
    for u, v, data in list(whvg_graph.edges(data=True))[:5]:
        print(f"  边({u}, {v}): 权重 = {data['weight']:.4f}")

    # 绘制结果
    plot_whvg(whvg_graph, ts, "加权水平可视图 (WHVG) - O(n) 算法实现")