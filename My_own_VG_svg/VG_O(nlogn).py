import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Arc
import networkx as nx


# 配置Matplotlib以正确显示中文和负号
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定中文字体
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号


def plot_visibility_graph(graph, time_series, title):
    """
    一个通用的函数，用于绘制可视图的两种标准图形。
    Args:
        graph (nx.Graph): networkx 图对象。
        time_series (np.array): 原始时间序列数据。
        title (str): 图形的主标题。
    """
    n = len(time_series)
    indices = np.arange(n)
    edges = list(graph.edges())

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 9))
    fig.suptitle(title, fontsize=16)

    # --- 子图1: 直方条与可视性连线 ---
    ax1.set_title('(a) 直方条')
    ax1.bar(indices, time_series, width=0.5, color='lightgray', edgecolor='dimgray')
    for i in range(n):
        y_pos = time_series[i] + 0.08 if time_series[i] < 0.8 else time_series[i] + 0.04
        ax1.text(indices[i], y_pos, f'{time_series[i]:.2f}', ha='center', va='bottom', fontsize=10)
    for u, v in edges:
        ax1.plot([indices[u], indices[v]], [time_series[u], time_series[v]], color='dimgray', linewidth=1.2, zorder=0)
    ax1.set_xlabel('序号', fontsize=12)
    ax1.set_ylabel('值', fontsize=12)
    ax1.set_xticks(indices)
    ax1.set_xticklabels(indices + 1)
    ax1.set_ylim(0, 1.0)
    ax1.set_xlim(-0.7, n - 0.3)

    # --- 子图2: 网络连边 ---
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



    save_path = r"D:\xupt\paper\可视图VG\VG_Survey_综述\VG_algorightm\output_svg\VG_classic_visibility_graph_O_nlogn.svg"
    plt.tight_layout(rect=[0, 0, 1, 0.95]) # 这一行其实可以放在savefig之前
    plt.savefig(save_path, format='svg', bbox_inches='tight')
    plt.show()


def _dc_vg_recursive(ts_slice, original_indices, graph):
    """
    分治法的递归辅助函数。
    Args:
        ts_slice (np.array): 当前处理的时间序列子序列。
        original_indices (np.array): ts_slice 中每个点在原始序列中的索引。
        graph (nx.Graph): 用于添加边的图对象。
    """
    # 基本情况：如果子序列长度小于等于1，则无法继续分割，返回。
    if len(ts_slice) <= 1:
        return

    # 1. 找到子序列中的最大值及其在子序列中的索引
    max_local_idx = np.argmax(ts_slice)
    max_val = ts_slice[max_local_idx]
    
    # 获取该点在原始序列中的索引
    max_original_idx = original_indices[max_local_idx]

    # 2. 从最大值点向左连接可见点
    max_slope_left = -np.inf
    for i in range(max_local_idx - 1, -1, -1):
        current_slope = (max_val - ts_slice[i]) / (max_local_idx - i)
        if current_slope > max_slope_left:
            graph.add_edge(max_original_idx, original_indices[i])
            max_slope_left = current_slope

    # 2. 从最大值点向右连接可见点
    max_slope_right = -np.inf
    for i in range(max_local_idx + 1, len(ts_slice)):
        current_slope = (ts_slice[i] - max_val) / (i - max_local_idx)
        if current_slope > max_slope_right:
            graph.add_edge(max_original_idx, original_indices[i])
            max_slope_right = current_slope
            
    # 3. & 4. 对左右两个子序列进行递归调用
    # 处理左子序列
    _dc_vg_recursive(ts_slice[:max_local_idx], original_indices[:max_local_idx], graph)
    # 处理右子序列
    _dc_vg_recursive(ts_slice[max_local_idx+1:], original_indices[max_local_idx+1:], graph)


def vg_divide_and_conquer(time_series):
    """
    使用分治法（DC）计算经典可视图（VG）的主函数。
    时间复杂度为 O(n log n)。
    """
    n = len(time_series)
    graph = nx.Graph()
    graph.add_nodes_from(range(n))
    
    initial_indices = np.arange(n)
    _dc_vg_recursive(time_series, initial_indices, graph)
    
    return graph


if __name__ == '__main__':
    # 使用论文中的标准时间序列
    ts = np.array([0.35, 0.68, 0.82, 0.18, 0.35, 0.68, 0.82, 0.18, 0.35, 0.68, 0.82, 0.18])

    # 使用 O(n log n) 的分治算法计算可视图
    print("正在使用 O(n log n) 分治算法计算...")
    dc_graph = vg_divide_and_conquer(ts)

    # 绘制结果
    plot_visibility_graph(dc_graph, ts, "经典可视图 - O(n log n) 分治算法实现")