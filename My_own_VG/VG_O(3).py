"""
@Author  : Ermao Zhou
@Contact : average O(n^2) time complexity
            worst O(n^2) time complexity
"""



import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Arc
import networkx as nx

# --- 1. 数据准备 ---
# [cite_start]根据论文3.1节，使用给定的周期为4的时间序列 [cite: 61]
ts = np.array([0.35, 0.68, 0.82, 0.18, 0.35, 0.68, 0.82, 0.18, 0.35, 0.68, 0.82, 0.18])
n = len(ts)
indices = np.arange(n)

# 配置Matplotlib以正确显示中文和负号
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定中文字体
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号

# --- 2. 经典可视图 (VG) 算法实现 ---
def classic_visibility_graph(time_series):
    """
    计算经典可视图（Classic Visibility Graph）的边。
    
    [cite_start]可视性准则[cite: 62, 97]:
    对于任意两点 A(ta, ya) 和 B(tb, yb)，以及它们之间的任意点 C(tc, yc)，
    如果满足 yc < yb + (ya - yb) * (tb - tc) / (tb - ta)，则A和B可见。
    这等价于任何中间点C都在线段AB的下方。
    
    Args:
        time_series (list or np.array): 输入的时间序列数据。
        
    Returns:
        networkx.Graph: 表示可视图的图对象。
    """
    series_len = len(time_series)
    # 初始化一个无向图
    graph = nx.Graph()
    # 添加节点
    graph.add_nodes_from(range(series_len))
    
    # 遍历所有节点对 (i, j) 其中 i < j
    for i in range(series_len):
        for j in range(i + 1, series_len):
            is_visible = True
            # 检查所有中间点 k
            for k in range(i + 1, j):
                # [cite_start]根据论文公式(1)检查点k是否阻挡了i和j之间的视线 [cite: 97]
                # 如果点k的高度大于或等于视线在k点的高度，则视线被阻挡
                if time_series[k] >= time_series[j] + (time_series[i] - time_series[j]) * (j - k) / (j - i):
                    is_visible = False
                    break
            
            # 如果没有点阻挡视线，则添加一条边
            if is_visible:
                graph.add_edge(i, j)
                
    return graph

# --- 3. 生成图并获取边 ---
vg_graph = classic_visibility_graph(ts)
edges = list(vg_graph.edges())

# --- 4. 绘图 ---
# 创建一个包含两个子图的画布
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 9))
fig.suptitle('经典可视图 (Classic Visibility Graph) O3实现', fontsize=16)

# --- 子图1: 直方条与可视性连线 (复现论文图1a) ---
ax1.set_title('(a) 直方条')
# 绘制直方条
ax1.bar(indices, ts, width=0.5, color='lightgray', edgecolor='dimgray')

# 在直方条顶部标注数值
for i in range(n):
    # 调整文本位置以避免重叠
    y_pos = ts[i] + 0.08 if ts[i] < 0.8 else ts[i] + 0.04
    ax1.text(indices[i], y_pos, f'{ts[i]}', ha='center', va='bottom', fontsize=10)

# 绘制可视性连线
for u, v in edges:
    ax1.plot([indices[u], indices[v]], [ts[u], ts[v]], color='dimgray', linewidth=1.2, zorder=0)

ax1.set_xlabel('序号', fontsize=12)
ax1.set_ylabel('值', fontsize=12)
ax1.set_xticks(indices)
ax1.set_xticklabels(indices + 1) # 论文中序号从1开始
ax1.set_ylim(0, 1.0)
ax1.set_xlim(-0.7, n - 0.3)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)


# --- 子图2: 网络连边 (复现论文图1b) ---
ax2.set_title('(b) 网络连边')
node_y_position = 0
node_positions = {i: (i, node_y_position) for i in indices}

# 绘制节点
ax2.plot(indices, np.full(n, node_y_position), 'ko', markersize=6)

# 将边绘制为弧线
for u, v in edges:
    center_x = (u + v) / 2
    width = v - u
    # 根据边的长度调整弧线高度，使其美观
    height = width * 0.8  
    arc = Arc((center_x, node_y_position), width, height, theta1=0, theta2=180, edgecolor='black', lw=1.5, zorder=0)
    ax2.add_patch(arc)

# 样式设置，使其与论文风格一致
ax2.set_aspect('equal')
ax2.set_ylim(-1.5, 5) # 留出足够的空间显示弧线
ax2.set_xlim(-0.7, n - 0.3)
ax2.axis('off') # 隐藏坐标轴

plt.savefig(r"D:\xupt\paper\可视图VG\VG_Survey_综述\output\VG_classic_visibility_graph_O_3.png", dpi=600)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()
