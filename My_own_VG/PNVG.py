import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import networkx as nx
import math

def parametric_natural_vg(time_series, alpha_degrees):
    """
    计算参数自然可视图 (PNVG)。
    该实现基于 O(n^2) 的经典VG逻辑。

    Args:
        time_series (np.array): 输入的时间序列数据。
        alpha_degrees (float): 视角参数 alpha (单位: 度)。
        
    Returns:
        tuple: (DiGraph, list_of_valid_edges, list_of_invalid_edges)
    """
    n = len(time_series)
    graph = nx.DiGraph()
    graph.add_nodes_from(range(n))
    
    valid_edges = []
    invalid_edges = []
    
    alpha_rad = math.radians(alpha_degrees)

    for i in range(n - 1):
        for j in range(i + 1, n):
            # 1. 检查经典VG可视性
            is_visible = True
            for k in range(i + 1, j):
                if time_series[k] >= time_series[j] + (time_series[i] - time_series[j]) * (j - k) / (j - i):
                    is_visible = False
                    break
            
            if is_visible:
                # 2. 如果可见，则计算角度
                delta_y = time_series[j] - time_series[i]
                delta_x = j - i
                
                # atan2返回与x轴正方向的夹角（弧度）
                angle_with_horizontal = math.atan2(delta_y, delta_x)
                
                # 与垂线（y轴）的夹角
                angle_with_vertical = abs(math.pi / 2 - angle_with_horizontal)
                
                # 3. 检查角度约束
                if angle_with_vertical < alpha_rad:
                    # 4. 如果满足约束，则为有效边，加入图中
                    graph.add_edge(i, j)
                    valid_edges.append((i, j))
                else:
                    # 否则为无效边
                    invalid_edges.append((i, j))
                    
    return graph, valid_edges, invalid_edges


def plot_pnvg(graph, valid_edges, invalid_edges, time_series, title):
    """
    专门用于绘制 PNVG 结果的函数。
    """
    n = len(time_series)
    indices = np.arange(n)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 9))
    fig.suptitle(title, fontsize=16)

    # --- 子图1: 直方条与两类可视性连线 (复现论文图7a) ---
    ax1.set_title('(a) 直方条')
    ax1.bar(indices, time_series, width=0.5, color='lightgray', edgecolor='dimgray')
    
    # 标注数值
    for i, val in enumerate(time_series):
        y_pos = val + 0.3
        ax1.text(indices[i], y_pos, f'{val}', ha='center', va='bottom', fontsize=10)
        
    # 绘制无效边 (蓝色虚线)
    for u, v in invalid_edges:
        ax1.plot([u, v], [time_series[u], time_series[v]], color='blue', linestyle='--', linewidth=1.5)

    # 绘制有效边 (红色实线箭头)
    for u, v in valid_edges:
        ax1.arrow(u, time_series[u], v - u, time_series[v] - time_series[u], 
                  head_width=0.15, head_length=0.2, length_includes_head=True, 
                  color='red', linestyle='-', lw=1.5, zorder=10)

    ax1.set_xlabel('序号', fontsize=12)
    ax1.set_ylabel('值', fontsize=12)
    ax1.set_xticks(indices)
    ax1.set_xticklabels(indices + 1)
    ax1.set_ylim(0, 11)
    ax1.set_xlim(-0.7, n - 0.3)

    # --- 子图2: 网络连边 (复现论文图7b) ---
    ax2.set_title('(b) 网络连边')
    node_y_position = 0
    node_positions = {i: (i, node_y_position) for i in indices}
    ax2.plot(indices, np.full(n, node_y_position), 'ko', markersize=6)

    # 只绘制最终图中的有效边
    for u, v in graph.edges():
        # 绘制红色和蓝色的弧线来模仿论文风格
        # 红色部分
        arrow_red = FancyArrowPatch(posA=node_positions[u], posB=node_positions[v],
                                    arrowstyle='->', connectionstyle='arc3,rad=0.4',
                                    color='red', mutation_scale=15, lw=1.5, linestyle='-')
        ax2.add_patch(arrow_red)
        # 蓝色部分 (模拟双色效果)
        arrow_blue = FancyArrowPatch(posA=node_positions[u], posB=node_positions[v],
                                     arrowstyle='-', connectionstyle='arc3,rad=-0.4',
                                     color='blue', mutation_scale=15, lw=1.5, linestyle='-')
        ax2.add_patch(arrow_blue)


    ax2.set_aspect('equal')
    ax2.set_ylim(-2, 3)
    ax2.set_xlim(-0.7, n - 0.3)
    ax2.axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

if __name__ == '__main__':
    # 配置Matplotlib以正确显示中文
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 使用与图7相似的示例时间序列
    ts = np.array([7.3, 5.0, 6.2, 6.6, 2.1, 5.0, 9.1])
    
    # 设置视角参数 alpha = 90 度
    ALPHA = 90
    
    # --- 计算 ---
    print(f"正在计算 PNVG (alpha={ALPHA}°)...")
    pnvg_graph, valid_edges, invalid_edges = parametric_natural_vg(ts, alpha_degrees=ALPHA)
    
    print("\n计算完成。")
    print(f"可见但因角度被拒绝的边: {invalid_edges}")
    print(f"最终图中的有效边: {valid_edges}")
    
    # --- 绘图 ---
    plot_pnvg(pnvg_graph, valid_edges, invalid_edges, ts, f"参数自然可视图 (PNVG, α={ALPHA}°)")