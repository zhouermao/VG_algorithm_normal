import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import networkx as nx
from bisect import bisect_left

def directed_lpvg_fast_O_n_squared(time_series, N):
    """
    计算有向有限穿越可视图（DLPVG）的 O(n^2) 算法实现。
    逻辑与 LPVG 相同，但创建的是有向图（DiGraph）。
    """
    n = len(time_series)
    # 使用有向图 DiGraph
    graph = nx.DiGraph()
    graph.add_nodes_from(range(n))

    for i in range(n - 1):
        seen_slopes = []
        for j in range(i + 1, n):
            current_slope = (time_series[j] - time_series[i]) / (j - i)
            
            pos = bisect_left(seen_slopes, current_slope)
            penetration_count = len(seen_slopes) - pos
            
            if penetration_count <= N:
                # 添加从 i 指向 j 的有向边
                graph.add_edge(i, j)
            
            if j > i + 1:
                prev_point_slope = (time_series[j-1] - time_series[i]) / (j - 1 - i)
                insert_pos = bisect_left(seen_slopes, prev_point_slope)
                seen_slopes.insert(insert_pos, prev_point_slope)

    return graph

def plot_dlpvg(edges_base, edges_new, time_series, title):
    """
    专门用于绘制 DLPVG 结果的函数，使用箭头和不同线型。
    """
    n = len(time_series)
    indices = np.arange(n)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 9))
    fig.suptitle(title, fontsize=16)

    # --- 子图1: 直方条与有向可视性连线 (复现论文图6a) ---
    ax1.set_title('(a) 直方条')
    ax1.bar(indices, time_series, width=0.5, color='lightgray', edgecolor='dimgray')

    # 标注数值
    for i in range(n):
        y_pos = time_series[i] + 0.08 if time_series[i] < 0.8 else time_series[i] + 0.04
        ax1.text(indices[i], y_pos, f'{time_series[i]:.2f}', ha='center', va='bottom', fontsize=10)

    # 绘制基础有向连线 (实线箭头)
    for u, v in edges_base:
        ax1.arrow(u, time_series[u], v - u, time_series[v] - time_series[u], 
                  head_width=0.15, head_length=0.2, length_includes_head=True, 
                  color='black', linestyle='-', lw=1.2)

    # 绘制新增有向连线 (虚线箭头)
    for u, v in edges_new:
        ax1.arrow(u, time_series[u], v - u, time_series[v] - time_series[u], 
                  head_width=0.15, head_length=0.2, length_includes_head=True, 
                  color='blue', linestyle='--', lw=1.2)

    ax1.set_xlabel('序号', fontsize=12)
    ax1.set_ylabel('值', fontsize=12)
    ax1.set_xticks(indices)
    ax1.set_xticklabels(indices + 1)
    ax1.set_ylim(0, 1.0)
    ax1.set_xlim(-0.7, n - 0.3)

    # --- 子图2: 网络连边 (复现论文图6b) ---
    ax2.set_title('(b) 网络连边')
    node_y_position = 0
    node_positions = {i: (i, node_y_position) for i in indices}
    ax2.plot(indices, np.full(n, node_y_position), 'ko', markersize=6)

    # 绘制基础有向弧线 (实线)
    for u, v in edges_base:
        arrow = FancyArrowPatch(posA=node_positions[u], posB=node_positions[v],
                                arrowstyle='->', connectionstyle='arc3,rad=0.4',
                                color='black', mutation_scale=15, lw=1.5, linestyle='-')
        ax2.add_patch(arrow)
        
    # 绘制新增有向弧线 (虚线)
    for u, v in edges_new:
        arrow = FancyArrowPatch(posA=node_positions[u], posB=node_positions[v],
                                arrowstyle='->', connectionstyle='arc3,rad=0.4',
                                color='blue', mutation_scale=15, lw=1.5, linestyle='--')
        ax2.add_patch(arrow)

    ax2.set_aspect('equal')
    ax2.set_ylim(-1.5, 5)
    ax2.set_xlim(-0.7, n - 0.3)
    ax2.axis('off')

    save_path = r"D:\xupt\paper\可视图VG\VG_Survey_综述\VG_algorightm\output_svg\DLPVG_Limited_Penetrable_Visibility_Graph_O_LPVG.svg"
    plt.tight_layout(rect=[0, 0, 1, 0.95]) # 这一行其实可以放在savefig之前
    plt.savefig(save_path, format='svg', bbox_inches='tight')
    plt.show()




if __name__ == '__main__':
    # 配置Matplotlib以正确显示中文
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 使用论文中的标准时间序列
    ts = np.array([0.35, 0.68, 0.82, 0.18, 0.35, 0.68, 0.82, 0.18, 0.35, 0.68, 0.82, 0.18])
    
    # --- 计算 ---
    # N=0 时的边，即标准有向可视图(DVG)的边
    print("正在使用 O(n^2) 逻辑计算 N=0 (标准DVG)...")
    dvg_graph_n0 = directed_lpvg_fast_O_n_squared(ts, N=0)
    edges_n0 = set(dvg_graph_n0.edges())

    # N=1 时的边
    print("正在使用 O(n^2) 逻辑计算 N=1 (DLPVG)...")
    dlpvg_graph_n1 = directed_lpvg_fast_O_n_squared(ts, N=1)
    edges_n1 = set(dlpvg_graph_n1.edges())

    # DLPVG (N=1) 新增的边
    new_edges = list(edges_n1 - edges_n0)
    edges_base = list(edges_n0)

    # --- 绘图 ---
    plot_dlpvg(edges_base, new_edges, ts, "有向有限穿越可视图 (DLPVG, N=1) - O(n^2) 逻辑实现")