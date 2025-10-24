import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Arc
import networkx as nx
from bisect import bisect_left

# --- 1. 数据准备 ---
# [cite_start]使用论文中给定的周期为4的时间序列 [cite: 61]
ts = np.array([0.35, 0.68, 0.82, 0.18, 0.35, 0.68, 0.82, 0.18, 0.35, 0.68, 0.82, 0.18])
n = len(ts)
indices = np.arange(n)

# 配置Matplotlib以正确显示中文
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# --- 2. 高效的 O(n^2) LPVG 算法实现 ---
def lpvg_fast_O_n_squared(time_series, N):
    """
    计算有限穿越可视图（LPVG）的 O(n^2) 算法实现。
    该算法对于固定的i，在j的循环中通过维护一个有序的斜率列表，
    使用二分查找来快速计算穿越数，避免了第三层循环。

    Args:
        time_series (list or np.array): 输入的时间序列数据。
        N (int): 有限穿越视距，允许视线穿越的障碍数。
        
    Returns:
        networkx.Graph: 表示LPVG的图对象。
    """
    n = len(time_series)
    graph = nx.Graph()
    graph.add_nodes_from(range(n))

    for i in range(n - 1):
        # 维护一个从i点出发，到i和j之间所有点的斜率的有序列表
        seen_slopes = []
        for j in range(i + 1, n):
            # 当前考察的边是 (i, j)
            current_slope = (time_series[j] - time_series[i]) / (j - i)
            
            # 穿越点k的条件是 slope(i,k) >= slope(i,j)
            # 我们需要在 seen_slopes 中找到 >= current_slope 的元素数量
            # seen_slopes 是有序的，可以使用二分查找
            
            # bisect_left 找到插入点，该点左侧所有元素都 < current_slope
            pos = bisect_left(seen_slopes, current_slope)
            
            # 那么，列表中大于等于 current_slope 的元素数量就是总数减去左侧部分
            penetration_count = len(seen_slopes) - pos
            
            if penetration_count <= N:
                graph.add_edge(i, j)
            
            # 将上一个点 (j-1) 的斜率（如果存在）加入到有序列表中
            # 注意：是 j-1 的斜率，因为当考察 (i,j) 时，中间点只到 j-1
            if j > i + 1:
                prev_point_slope = (time_series[j-1] - time_series[i]) / (j - 1 - i)
                # 找到插入位置并插入，以保持列表有序
                insert_pos = bisect_left(seen_slopes, prev_point_slope)
                seen_slopes.insert(insert_pos, prev_point_slope)

    return graph


# --- 3. 生成图并获取不同类型的边 ---
# [cite_start]使用 O(n^2) 算法计算 N=0 时的边，即经典VG的边 [cite: 117]
vg_graph_n0 = lpvg_fast_O_n_squared(ts, N=0)
edges_n0 = set(frozenset(e) for e in vg_graph_n0.edges())

# 使用 O(n^2) 算法计算 N=1 时的边
lpvg_graph_n1 = lpvg_fast_O_n_squared(ts, N=1)
edges_n1 = set(frozenset(e) for e in lpvg_graph_n1.edges())

# LPVG (N=1) 新增的边是 N=1 边集与 N=0 边集的差集
new_edges = edges_n1 - edges_n0
# 转换回列表方便绘图
edges_base = list(edges_n0)
edges_new = list(new_edges)


# --- 4. 绘图 ---
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 9))
fig.suptitle('有限穿越可视图 (LPVG, N=1) - O(n^2) 算法实现', fontsize=16)

# --- 子图1: 直方条与可视性连线 (复现论文图3a) ---
ax1.set_title('(a) 直方条')
ax1.bar(indices, ts, width=0.5, color='lightgray', edgecolor='dimgray')

# 标注数值
for i in range(n):
    y_pos = ts[i] + 0.08 if ts[i] < 0.8 else ts[i] + 0.04
    ax1.text(indices[i], y_pos, f'{ts[i]}', ha='center', va='bottom', fontsize=10)

# [cite_start]绘制VG基础连线 (实线) [cite: 117]
for u, v in edges_base:
    ax1.plot([indices[u], indices[v]], [ts[u], ts[v]], color='black', linestyle='-', linewidth=1.2, zorder=0)

# [cite_start]绘制LPVG新增连线 (虚线) [cite: 117]
for u, v in edges_new:
    ax1.plot([indices[u], indices[v]], [ts[u], ts[v]], color='blue', linestyle='--', linewidth=1.2, zorder=0)

ax1.set_xlabel('序号', fontsize=12)
ax1.set_ylabel('值', fontsize=12)
ax1.set_xticks(indices)
ax1.set_xticklabels(indices + 1)
ax1.set_ylim(0, 1.0)
ax1.set_xlim(-0.7, n - 0.3)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# --- 子图2: 网络连边 (复现论文图3b) ---
ax2.set_title('(b) 网络连边')
node_y_position = 0
ax2.plot(indices, np.full(n, node_y_position), 'ko', markersize=6)

# 绘制VG基础连线 (实线弧)
for u, v in edges_base:
    u, v = sorted((u, v))
    center_x = (u + v) / 2
    width = v - u
    height = width * 0.8
    arc = Arc((center_x, node_y_position), width, height, theta1=0, theta2=180, edgecolor='black', linestyle='-', lw=1.5)
    ax2.add_patch(arc)

# 绘制LPVG新增连线 (虚线弧)
for u, v in edges_new:
    u, v = sorted((u, v))
    center_x = (u + v) / 2
    width = v - u
    height = width * 0.8
    arc = Arc((center_x, node_y_position), width, height, theta1=0, theta2=180, edgecolor='blue', linestyle='--', lw=1.5)
    ax2.add_patch(arc)

ax2.set_aspect('equal')
ax2.set_ylim(-1.5, 5)
ax2.set_xlim(-0.7, n - 0.3)
ax2.axis('off')

save_path = r"D:\xupt\paper\可视图VG\VG_Survey_综述\VG_algorightm\output_svg\LPVG_Limited_Penetrable_Visibility_Graph_O_2.svg"
plt.tight_layout(rect=[0, 0, 1, 0.95]) # 这一行其实可以放在savefig之前
plt.savefig(save_path, format='svg', bbox_inches='tight')
plt.show()

