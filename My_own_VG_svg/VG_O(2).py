"""
@Author  : Ermao Zhou
@Contact : O(n^2) efficient algorithm demonstration
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Arc
import networkx as nx

# --- 1. 数据准备 ---
# 根据论文3.1节，使用给定的周期为4的时间序列
ts = np.array([0.35, 0.68, 0.82, 0.18, 0.35, 0.68, 0.82, 0.18, 0.35, 0.68, 0.82, 0.18])
n = len(ts)
indices = np.arange(n)

# 配置Matplotlib以正确显示中文和负号
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定中文字体
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号


# 这是优化后的 O(N^2) 算法
def fast_visibility_graph(time_series):
    series_len = len(time_series)
    graph = nx.Graph()
    graph.add_nodes_from(range(series_len))
    for i in range(series_len - 1):
        max_slope = -np.inf
        for j in range(i + 1, series_len):
            slope = (time_series[j] - time_series[i]) / (j - i)
            if slope > max_slope:
                graph.add_edge(i, j)
                max_slope = slope
    return graph



# 使用 O(N^2) 算法
vg_graph_fast = fast_visibility_graph(ts)
edges_fast = sorted([tuple(sorted(edge)) for edge in vg_graph_fast.edges()])
print(f"O(N^2) 算法找到 {len(edges_fast)} 条边: {edges_fast}")


# 使用 O(N^2) 算法的结果进行后续绘图
edges = list(vg_graph_fast.edges())

# --- 4. 绘图 (与您的代码完全相同) ---
# ... (您的绘图代码无需任何修改)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 9))
fig.suptitle('经典可视图 (Classic Visibility Graph) - O(N^2) 实现', fontsize=16)

# 子图1: 直方条与可视性连线
ax1.set_title('(a) 直方条')
ax1.bar(indices, ts, width=0.5, color='lightgray', edgecolor='dimgray')
for i in range(n):
    y_pos = ts[i] + 0.08 if ts[i] < 0.8 else ts[i] + 0.04
    ax1.text(indices[i], y_pos, f'{ts[i]}', ha='center', va='bottom', fontsize=10)
for u, v in edges:
    ax1.plot([indices[u], indices[v]], [ts[u], ts[v]], color='dimgray', linewidth=1.2, zorder=0)
ax1.set_xlabel('序号', fontsize=12)
ax1.set_ylabel('值', fontsize=12)
ax1.set_xticks(indices)
ax1.set_xticklabels(indices + 1)
ax1.set_ylim(0, 1.0)
ax1.set_xlim(-0.7, n - 0.3)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# 子图2: 网络连边
ax2.set_title('(b) 网络连边')
node_y_position = 0
node_positions = {i: (i, node_y_position) for i in indices}
ax2.plot(indices, np.full(n, node_y_position), 'ko', markersize=6)
for u, v in edges:
    center_x = (u + v) / 2
    width = v - u
    height = width * 0.8
    arc = Arc((center_x, node_y_position), width, height, theta1=0, theta2=180, edgecolor='black', lw=1.5, zorder=0)
    ax2.add_patch(arc)
ax2.set_aspect('equal')
ax2.set_ylim(-1.5, 5)
ax2.set_xlim(-0.7, n - 0.3)
ax2.axis('off')



save_path = r"D:\xupt\paper\可视图VG\VG_Survey_综述\VG_algorightm\output_svg\VG_classic_visibility_graph_O_2.svg"
plt.tight_layout(rect=[0, 0, 1, 0.95]) # 这一行其实可以放在savefig之前
plt.savefig(save_path, format='svg', bbox_inches='tight')
plt.show()