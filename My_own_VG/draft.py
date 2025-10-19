import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import math

# 1. 经典可视图 (VG) 算法
def create_visibility_graph(ts, directed=False):
    """
    根据给定的时间序列创建一个经典可视图 (VG).

    参数:
    ts (list or np.array): 输入的时间序列.
    directed (bool): 是否生成有向图.

    返回:
    networkx.Graph or networkx.DiGraph: 生成的可视图.
    """
    if directed:
        g = nx.DiGraph()
    else:
        g = nx.Graph()
    
    n = len(ts)
    g.add_nodes_from(range(n))

    for i in range(n):
        for j in range(i + 1, n):
            is_visible = True
            # 根据论文中的公式 (1) 进行判断
            # y_c < y_b + (y_a - y_b) * (t_b - t_c) / (t_b - t_a)
            for k in range(i + 1, j):
                if ts[k] >= ts[j] + (ts[i] - ts[j]) * (j - k) / (j - i):
                    is_visible = False
                    break
            
            if is_visible:
                if directed:
                    g.add_edge(i, j)
                else:
                    g.add_edge(i, j)
    return g

# 2. 水平可视图 (HVG) 算法
def create_horizontal_visibility_graph(ts, directed=False):
    """
    根据给定的时间序列创建一个水平可视图 (HVG).

    参数:
    ts (list or np.array): 输入的时间序列.
    directed (bool): 是否生成有向图 (用于DHVG).

    返回:
    networkx.Graph or networkx.DiGraph: 生成的水平可视图.
    """
    if directed:
        g = nx.DiGraph()
    else:
        g = nx.Graph()
        
    n = len(ts)
    g.add_nodes_from(range(n))

    for i in range(n):
        for j in range(i + 1, n):
            is_horizontally_visible = True
            # 根据论文中的公式 (2) 进行判断
            # y_c < y_a and y_c < y_b
            min_height = min(ts[i], ts[j])
            for k in range(i + 1, j):
                if ts[k] >= min_height:
                    is_horizontally_visible = False
                    break
            
            if is_horizontally_visible:
                if directed:
                    g.add_edge(i, j)
                else:
                    g.add_edge(i, j)
    return g

# 3. 有限穿越可视图 (LPVG) 算法
def create_limited_penetrable_vg(ts, n_penetrations=1):
    """
    根据给定的时间序列创建一个有限穿越可视图 (LPVG).

    参数:
    ts (list or np.array): 输入的时间序列.
    n_penetrations (int): 允许的最大穿越次数.

    返回:
    networkx.Graph: 生成的有限穿越可视图.
    """
    g = nx.Graph()
    n = len(ts)
    g.add_nodes_from(range(n))

    for i in range(n):
        for j in range(i + 1, n):
            penetrations = 0
            for k in range(i + 1, j):
                # 检查点 k 是否阻挡了 i 和 j 之间的视线
                if ts[k] >= ts[j] + (ts[i] - ts[j]) * (j - k) / (j - i):
                    penetrations += 1
            
            # 如果阻挡次数小于或等于允许的最大穿越次数，则添加边
            if penetrations <= n_penetrations:
                g.add_edge(i, j)
    return g

# 4. 有向水平可视图 (DHVG) 算法
def create_directed_horizontal_vg(ts):
    """
    根据给定的时间序列创建一个有向水平可视图 (DHVG).
    这本质上是HVG算法的有向版本。

    参数:
    ts (list or np.array): 输入的时间序列.

    返回:
    networkx.DiGraph: 生成的有向水平可视图.
    """
    return create_horizontal_visibility_graph(ts, directed=True)

# 5. 加权可视图 (WVG) 算法
def create_weighted_visibility_graph(ts):
    """
    根据给定的时间序列创建一个加权可视图 (WVG).

    参数:
    ts (list or np.array): 输入的时间序列.

    返回:
    networkx.Graph: 生成的加权可视图，边带有权重.
    """
    g = nx.Graph()
    n = len(ts)
    g.add_nodes_from(range(n))

    for i in range(n):
        for j in range(i + 1, n):
            is_visible = True
            for k in range(i + 1, j):
                if ts[k] >= ts[j] + (ts[i] - ts[j]) * (j - k) / (j - i):
                    is_visible = False
                    break
            
            if is_visible:
                # 根据论文中的公式 (6) 计算权重
                # w_ab = arctan((y_b - y_a) / (t_b - t_a))
                weight = math.atan((ts[j] - ts[i]) / (j - i))
                g.add_edge(i, j, weight=round(weight, 4))
    return g

def plot_graph_and_series(ts, g, title):
    """
    可视化时间序列及其对应的可视图.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [1, 2]})
    
    # 绘制时间序列
    ax1.plot(range(len(ts)), ts, 'o-', label='Time Series')
    ax1.set_title(f'Time Series for {title}')
    ax1.set_xlabel('Time (t)')
    ax1.set_ylabel('Value (y)')
    ax1.grid(True)
    
    # 绘制网络图
    pos = {i: (i, 0) for i in range(len(ts))} # 将节点排成一条线
    nx.draw(g, pos, with_labels=True, node_color='lightblue', node_size=500, font_size=10, ax=ax2)
    
    # 如果是加权图，显示权重
    if nx.is_weighted(g):
        edge_labels = nx.get_edge_attributes(g, 'weight')
        nx.draw_networkx_edge_labels(g, pos, edge_labels=edge_labels, ax=ax2)
    
    ax2.set_title(title)
    plt.tight_layout()
    plt.savefig(f"{title.replace(' ', '_')}.png")
    plt.close()
    print(f"可视化图表已保存为 {title.replace(' ', '_')}.png")


if __name__ == '__main__':
    # 使用论文图1中的样本时间序列
    sample_ts = np.array([0.35, 0.68, 0.82, 0.18, 0.35, 0.68, 0.82, 0.18, 0.35, 0.68, 0.82, 0.18])

    print("--- 1. 经典可视图 (VG) ---")
    vg = create_visibility_graph(sample_ts)
    print("生成的边列表 (VG):", vg.edges())
    plot_graph_and_series(sample_ts, vg, "Classic Visibility Graph (VG)")
    print("-" * 30)

    print("--- 2. 水平可视图 (HVG) ---")
    hvg = create_horizontal_visibility_graph(sample_ts)
    print("生成的边列表 (HVG):", hvg.edges())
    plot_graph_and_series(sample_ts, hvg, "Horizontal Visibility Graph (HVG)")
    print("-" * 30)
    
    print("--- 3. 有限穿越可视图 (LPVG, N=1) ---")
    lpvg = create_limited_penetrable_vg(sample_ts, n_penetrations=1)
    print("生成的边列表 (LPVG, N=1):", lpvg.edges())
    plot_graph_and_series(sample_ts, lpvg, "Limited Penetrable VG (LPVG N=1)")
    print("-" * 30)

    print("--- 4. 有向水平可视图 (DHVG) ---")
    dhvg = create_directed_horizontal_vg(sample_ts)
    print("生成的有向边列表 (DHVG):", dhvg.edges())
    plot_graph_and_series(sample_ts, dhvg, "Directed Horizontal VG (DHVG)")
    print("-" * 30)
    
    print("--- 5. 加权可视图 (WVG) ---")
    wvg = create_weighted_visibility_graph(sample_ts)
    print("生成的带权边列表 (WVG):", wvg.edges(data=True))
    plot_graph_and_series(sample_ts, wvg, "Weighted Visibility Graph (WVG)")
    print("-" * 30)