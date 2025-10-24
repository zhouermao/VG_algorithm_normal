import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Arc
import networkx as nx

# --- 1. 数据准备 ---
# 同样使用论文中给定的时间序列
ts = np.array([0.35, 0.68, 0.82, 0.18, 0.35, 0.68, 0.82, 0.18, 0.35, 0.68, 0.82, 0.18])
n = len(ts)
indices = np.arange(n)

# 配置Matplotlib
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# --- 2. LPVG 算法实现 ---
def limited_penetrable_vg(time_series, N):
    """
    计算有限穿越可视图（Limited Penetrable Visibility Graph）的边。
    
    Args:
        time_series (list or np.array): 输入的时间序列数据。
        N (int): 有限穿越视距，允许视线穿越的障碍数。
        
    Returns:
        networkx.Graph: 表示LPVG的图对象。
    """
    series_len = len(time_series)
    graph = nx.Graph()
    graph.add_nodes_from(range(series_len))
    
    for i in range(series_len):
        for j in range(i + 1, series_len):
            penetrations = 0
            # 检查所有中间点 k
            for k in range(i + 1, j):
                # 检查点k是否阻挡了视线
                if time_series[k] >= time_series[j] + (time_series[i] - time_series[j]) * (j - k) / (j - i):
                    penetrations += 1
            
            # 如果穿越数小于等于N，则添加一条边
            if penetrations <= N:
                graph.add_edge(i, j)
                
    return graph

# --- 3. 生成图并获取不同类型的边 ---
# N=0 时的边，即经典VG的边
vg_graph_n0 = limited_penetrable_vg(ts, N=0)
edges_n0 = set(vg_graph_n0.edges())

# N=1 时的边
lpvg_graph_n1 = limited_penetrable_vg(ts, N=1)
edges_n1 = set(lpvg_graph_n1.edges())

# LPVG (N=1) 新增的边
new_edges = edges_n1 - edges_n0
# 转换回列表方便绘图
edges_base = list(edges_n0)
edges_new = list(new_edges)


# --- 4. 绘图 ---
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 9))
fig.suptitle('有限穿越可视图 (LPVG, N=1) O(3) 算法实现', fontsize=16)

# --- 子图1: 直方条与可视性连线 (复现论文图3a) ---
ax1.set_title('(a) 直方条')
ax1.bar(indices, ts, width=0.5, color='lightgray', edgecolor='dimgray')

# 标注数值
for i in range(n):
    y_pos = ts[i] + 0.08 if ts[i] < 0.8 else ts[i] + 0.04
    ax1.text(indices[i], y_pos, f'{ts[i]}', ha='center', va='bottom', fontsize=10)

# 绘制VG基础连线 (实线)
for u, v in edges_base:
    ax1.plot([indices[u], indices[v]], [ts[u], ts[v]], color='black', linestyle='-', linewidth=1.2, zorder=0)

# 绘制LPVG新增连线 (虚线)
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
    center_x = (u + v) / 2
    width = v - u
    height = width * 0.8
    arc = Arc((center_x, node_y_position), width, height, theta1=0, theta2=180, edgecolor='black', linestyle='-', lw=1.5)
    ax2.add_patch(arc)

# 绘制LPVG新增连线 (虚线弧)
for u, v in edges_new:
    center_x = (u + v) / 2
    width = v - u
    height = width * 0.8
    arc = Arc((center_x, node_y_position), width, height, theta1=0, theta2=180, edgecolor='blue', linestyle='--', lw=1.5)
    ax2.add_patch(arc)

ax2.set_aspect('equal')
ax2.set_ylim(-1.5, 5)
ax2.set_xlim(-0.7, n - 0.3)
ax2.axis('off')

save_path = r"D:\xupt\paper\可视图VG\VG_Survey_综述\VG_algorightm\output_svg\LPVG_Limited_Penetrable_Visibility_Graph_O_3.svg"
plt.tight_layout(rect=[0, 0, 1, 0.95]) # 这一行其实可以放在savefig之前
plt.savefig(save_path, format='svg', bbox_inches='tight')
plt.show()
