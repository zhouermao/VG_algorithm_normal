import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from bisect import bisect_left

def lpvg_fast_O_n_squared(time_series, N):
    """
    计算有限穿越可视图（LPVG）的 O(n^2) 算法实现。
    这是 CLPVG 算法的计算核心。
    """
    n = len(time_series)
    graph = nx.Graph()
    graph.add_nodes_from(range(n))

    for i in range(n - 1):
        seen_slopes = []
        for j in range(i + 1, n):
            current_slope = (time_series[j] - time_series[i]) / (j - i)
            pos = bisect_left(seen_slopes, current_slope)
            penetration_count = len(seen_slopes) - pos
            
            if penetration_count <= N:
                graph.add_edge(i, j)
            
            if j > i + 1:
                prev_point_slope = (time_series[j-1] - time_series[i]) / (j - 1 - i)
                insert_pos = bisect_left(seen_slopes, prev_point_slope)
                seen_slopes.insert(insert_pos, prev_point_slope)

    return graph

def clpvg_from_lpvg(time_series, N):
    """
    通过将序列加倍并应用LPVG算法来计算环形有限穿越可视图（CLPVG）。
    
    Args:
        time_series (list or np.array): 输入的时间序列数据。
        N (int): 有限穿越视距。
        
    Returns:
        networkx.Graph: 表示CLPVG的图对象。
    """
    n = len(time_series)
    
    # 1. 将原序列复制一份拼接到末尾，长度变为 2n
    ts_doubled = np.concatenate([time_series, time_series])
    
    # 2. 在这个 2n 长度的线性序列上运行 LPVG 算法
    # 注意：我们只关心前 n 个点发出的连接
    linear_graph = nx.Graph()
    linear_graph.add_nodes_from(range(2 * n))

    # 为了效率，可以只计算前 n 个点发出的边
    for i in range(n):
        seen_slopes = []
        # j 的范围需要覆盖一个完整的周期 n
        for j in range(i + 1, i + n):
            current_slope = (ts_doubled[j] - ts_doubled[i]) / (j - i)
            pos = bisect_left(seen_slopes, current_slope)
            penetration_count = len(seen_slopes) - pos
            
            if penetration_count <= N:
                linear_graph.add_edge(i, j)
            
            if j > i + 1:
                prev_point_slope = (ts_doubled[j-1] - ts_doubled[i]) / (j - 1 - i)
                insert_pos = bisect_left(seen_slopes, prev_point_slope)
                seen_slopes.insert(insert_pos, prev_point_slope)

    # 3. 将线性图的边映射回 n 个节点的环形图
    circular_graph = nx.Graph()
    circular_graph.add_nodes_from(range(n))
    
    for u, v in linear_graph.edges():
        # 确保边是在原始 n 个节点范围内建立的
        if u < n:
            original_u = u % n
            original_v = v % n
            # 避免添加自环
            if original_u != original_v:
                circular_graph.add_edge(original_u, original_v)
                
    return circular_graph

def plot_circular_graph(graph, time_series, title):
    """
    绘制环形布局的可视图网络。
    """
    n = len(time_series)
    plt.figure(figsize=(8, 8))
    
    # 使用 networkx 的环形布局
    pos = nx.circular_layout(graph)
    
    # 绘制节点，节点大小可以与时间序列的值相关联
    node_sizes = 50 + 400 * (time_series - np.min(time_series)) / (np.max(time_series) - np.min(time_series))
    nx.draw_networkx_nodes(graph, pos, node_color='skyblue', node_size=node_sizes)
    
    # 绘制边
    nx.draw_networkx_edges(graph, pos, edge_color='gray', alpha=0.7)
    
    # 绘制节点标签
    nx.draw_networkx_labels(graph, pos, font_size=12)
    
    plt.title(title, fontsize=16)

    # plt.savefig(r"D:\xupt\paper\可视图VG\VG_Survey_综述\output\CLPVG_Circular_Limited_Penetrable_Visibility_Graph_O_LPVG.png", dpi=600)
    # plt.axis('equal')
    # plt.show()
    # save_path = r"D:\xupt\paper\可视图VG\VG_Survey_综述\output_svg\HVG_O_2.svg"
    save_path = r"D:\xupt\paper\可视图VG\VG_Survey_综述\VG_algorightm\output_svg\CLPVG_Circular_Limited_Penetrable_Visibility_Graph_O_LPVG.svg"

    plt.tight_layout(rect=[0, 0, 1, 0.95]) # 这一行其实可以放在savefig之前

    plt.savefig(save_path, format='svg', bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    # 配置Matplotlib以正确显示中文
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 使用论文中的标准时间序列
    ts = np.array([0.35, 0.68, 0.82, 0.18, 0.35, 0.68, 0.82, 0.18, 0.35, 0.68, 0.82, 0.18])
    N = 1 # 设置最大允许穿越次数
    
    print(f"正在计算 CLPVG (N={N})...")
    clpvg_graph = clpvg_from_lpvg(ts, N)
    
    print(f"计算完成。节点数: {clpvg_graph.number_of_nodes()}, 边数: {clpvg_graph.number_of_edges()}")
    
    # 绘制环形图
    plot_circular_graph(clpvg_graph, ts, f"环形有限穿越可视图 (CLPVG, N={N})")