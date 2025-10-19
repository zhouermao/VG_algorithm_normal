
"""
@Author  : Ermao Zhou
@Contact : average O(n) time complexity
            worst O(n^2) time complexity
"""



import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Arc
import networkx as nx

# --- 1. 数据准备 ---
# 同样使用论文中给定的时间序列
ts = np.array([0.35, 0.68, 0.82, 0.18, 0.35, 0.68, 0.82, 0.18, 0.35, 0.68, 0.82, 0.18])
n = len(ts)
indices = np.arange(n)

# 配置Matplotlib以正确显示中文
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# --- 2. 水平可视图 (HVG) 算法实现 ---
def horizontal_visibility_graph(time_series):
    """
    计算水平可视图（Horizontal Visibility Graph）的边。
    
    可视性准则:
    对于任意两点 A(ta, ya) 和 B(tb, yb)，以及它们之间的任意点 C(tc, yc)，
    如果满足 yc < min(ya, yb)，则A和B可见。
    
    Args:
        time_series (list or np.array): 输入的时间序列数据。
        
    Returns:
        networkx.Graph: 表示水平可视图的图对象。
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
            # 找到两个端点的最小高度
            min_height = min(time_series[i], time_series[j])
            
            # 检查所有中间点 k
            for k in range(i + 1, j):
                # 如果有任何中间点的高度大于或等于端点的最小高度，则视线被阻挡
                if time_series[k] >= min_height:
                    is_visible = False
                    break
            
            # 如果视线未被阻挡，则添加一条边
            if is_visible:
                graph.add_edge(i, j)
                
    return graph

# --- 3. 生成图并获取边 ---
hvg_graph = horizontal_visibility_graph(ts)
edges = list(hvg_graph.edges())

# --- 4. 绘图 ---
# 创建一个包含两个子图的画布
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 9))
fig.suptitle('水平可视图 (Horizontal Visibility Graph)', fontsize=16)

# --- 子图1: 直方条与可视性连线 (复现论文图2a) ---
ax1.set_title('(a) 直方条')
# 绘制直方条
ax1.bar(indices, ts, width=0.5, color='lightgray', edgecolor='dimgray')

# 在直方条顶部标注数值
for i in range(n):
    y_pos = ts[i] + 0.08 if ts[i] < 0.8 else ts[i] + 0.04
    ax1.text(indices[i], y_pos, f'{ts[i]}', ha='center', va='bottom', fontsize=10)

# 绘制水平可视性连线（带箭头）
for u, v in edges:
    # 水平线的高度取两端点的最小值
    height = min(ts[u], ts[v])
    # 使用带箭头的线来模仿论文中的样式
    ax1.arrow(u, height, v - u, 0, head_width=0.02, head_length=0.2, length_includes_head=True, color='dimgray', lw=1.2)
    ax1.arrow(v, height, u - v, 0, head_width=0.02, head_length=0.2, length_includes_head=True, color='dimgray', lw=1.2)
    # 在端点处添加小标记
    ax1.plot([u, u], [height - 0.015, height + 0.015], color='black', marker='|')
    ax1.plot([v, v], [height - 0.015, height + 0.015], color='black', marker='|')

ax1.set_xlabel('序号', fontsize=12)
ax1.set_ylabel('值', fontsize=12)
ax1.set_xticks(indices)
ax1.set_xticklabels(indices + 1)
ax1.set_ylim(0, 1.0)
ax1.set_xlim(-0.7, n - 0.3)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# --- 子图2: 网络连边 (复现论文图2b) ---
ax2.set_title('(b) 网络连边')
node_y_position = 0
ax2.plot(indices, np.full(n, node_y_position), 'ko', markersize=6)

# 将边绘制为弧线
for u, v in edges:
    center_x = (u + v) / 2
    width = v - u
    height = width * 0.8  
    arc = Arc((center_x, node_y_position), width, height, theta1=0, theta2=180, edgecolor='black', lw=1.5)
    ax2.add_patch(arc)

ax2.set_aspect('equal')
ax2.set_ylim(-1.5, 5)
ax2.set_xlim(-0.7, n - 0.3)
ax2.axis('off')
plt.savefig(r"D:\xupt\paper\可视图VG\VG_Survey_综述\output\HVG_Horizontal_Visibility_Graph_O_2.png", dpi=600)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()
