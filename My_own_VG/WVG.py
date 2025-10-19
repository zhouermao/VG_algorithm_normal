import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Arc
import networkx as nx
import math

def plot_wvg(graph, time_series, title):
    """
    一个专门用于绘制加权可视图（WVG）的函数。
    边的颜色表示其权重。
    """
    n = len(time_series)
    indices = np.arange(n)
    edges = list(graph.edges(data=True))

    weights = [data['weight'] for _, _, data in edges]
    
    # 设置颜色映射
    if not weights: # 处理没有边的情况
        min_w, max_w = -1, 1
    else:
        min_w, max_w = min(weights), max(weights)
        
    cmap = plt.cm.coolwarm # 冷色调代表负权重（向下），暖色调代表正权重（向上）
    norm = plt.Normalize(vmin=min_w, vmax=max_w)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    fig.suptitle(title, fontsize=16)

    # --- 子图1: 带权重的直方条图 ---
    ax1.set_title('(a) 直方条')
    ax1.bar(indices, time_series, width=0.5, color='lightgray', edgecolor='dimgray')

    for u, v, data in edges:
        color = cmap(norm(data['weight']))
        ax1.plot([indices[u], indices[v]], [time_series[u], time_series[v]], color=color, linewidth=1.5, zorder=0)
    
    ax1.set_xlabel('序号', fontsize=12)
    ax1.set_ylabel('值', fontsize=12)
    ax1.set_xticks(indices)
    ax1.set_xticklabels(indices + 1)
    
    # 添加颜色条作为图例
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = fig.colorbar(sm, ax=ax1, orientation='vertical')
    cbar.set_label('权重 (弧度)', fontsize=12)

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


# --- 修正点 1: 修改函数签名，增加 full_time_series 参数 ---
def _wvg_dc_recursive(ts_slice, original_indices, graph, full_time_series):
    """
    用于WVG的分治法递归辅助函数，会计算并添加权重。
    """
    if len(ts_slice) <= 1:
        return

    max_local_idx = np.argmax(ts_slice)
    max_val = ts_slice[max_local_idx]
    max_original_idx = original_indices[max_local_idx]

    # 从最大值点向左连接
    max_slope_left = -np.inf
    for i in range(max_local_idx - 1, -1, -1):
        current_slope_val = (max_val - ts_slice[i]) / (max_local_idx - i)
        if current_slope_val > max_slope_left:
            u, v = original_indices[i], max_original_idx
            # --- 修正点 2: 使用 full_time_series 计算权重 ---
            weight = math.atan((full_time_series[v] - full_time_series[u]) / (v - u))
            graph.add_edge(u, v, weight=weight)
            max_slope_left = current_slope_val

    # 从最大值点向右连接
    max_slope_right = -np.inf
    for i in range(max_local_idx + 1, len(ts_slice)):
        current_slope_val = (ts_slice[i] - max_val) / (i - max_local_idx)
        if current_slope_val > max_slope_right:
            u, v = max_original_idx, original_indices[i]
            # --- 修正点 2: 使用 full_time_series 计算权重 ---
            weight = math.atan((full_time_series[v] - full_time_series[u]) / (v - u))
            graph.add_edge(u, v, weight=weight)
            max_slope_right = current_slope_val
            
    # --- 修正点 3: 在递归调用时传递 full_time_series ---
    _wvg_dc_recursive(ts_slice[:max_local_idx], original_indices[:max_local_idx], graph, full_time_series)
    _wvg_dc_recursive(ts_slice[max_local_idx+1:], original_indices[max_local_idx+1:], graph, full_time_series)


def weighted_visibility_graph(time_series):
    """
    使用 O(n log n) 分治法计算加权可视图 (WVG) 的主函数。
    """
    n = len(time_series)
    graph = nx.Graph() # WVG 是无向图
    graph.add_nodes_from(range(n))
    
    initial_indices = np.arange(n)
    # --- 修正点 3: 初始调用时传递 full_time_series (即它自身) ---
    _wvg_dc_recursive(time_series, initial_indices, graph, time_series)
    
    return graph


if __name__ == '__main__':
    # 配置Matplotlib以正确显示中文
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 使用论文中的标准时间序列
    ts = np.array([0.35, 0.68, 0.82, 0.18, 0.35, 0.68, 0.82, 0.18, 0.35, 0.68, 0.82, 0.18])
    
    print("正在使用 O(n log n) 算法计算 WVG...")
    wvg_graph = weighted_visibility_graph(ts)

    print(f"\n计算完成。节点数: {wvg_graph.number_of_nodes()}, 边数: {wvg_graph.number_of_edges()}")
    # 打印其中几条边和它们的权重
    print("部分边的权重示例:")
    for u, v, data in list(wvg_graph.edges(data=True))[:5]:
        print(f"  边({u}, {v}): 权重 = {data['weight']:.4f}")

    # 绘制结果
    plot_wvg(wvg_graph, ts, "加权可视图 (WVG) - O(n log n) 算法实现")