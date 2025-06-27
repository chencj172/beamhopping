import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform

# 设置matplotlib中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']  # 设置中文字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 读取小区位置数据
df = pd.read_csv('data_dir/cell_positions_fixed.csv')
print("cell area data:")
print(df)

# 提取坐标
positions = df[['X', 'Y', 'Z']].values
cell_indices = df['Cell_Index'].values

# 计算所有小区的中心点
center = np.mean(positions, axis=0)
print(f"\ngeometry center: {center}")

# 计算每个小区到中心的距离
distances_to_center = np.linalg.norm(positions - center, axis=1)

# 按距离排序
sorted_indices = np.argsort(distances_to_center)
print(f"\n Sort communities by distance center:")
for i, idx in enumerate(sorted_indices):
    cell_id = cell_indices[idx]
    dist = distances_to_center[idx]
    print(f"cell {cell_id}: 距离中心 {dist:.1f} ")

# 定义分层
core_cells = sorted_indices[:2]  # 最近的2个小区作为核心
inner_cells = sorted_indices[2:8]  # 接下来6个小区作为内环
middle_cells = sorted_indices[8:14]  # 再6个小区作为中环
outer_cells = sorted_indices[14:]  # 剩余小区作为外环

print(f"\n核心区域 (距离最近): {[cell_indices[i] for i in core_cells]}")
print(f"内环区域: {[cell_indices[i] for i in inner_cells]}")
print(f"中环区域: {[cell_indices[i] for i in middle_cells]}")
print(f"外环区域: {[cell_indices[i] for i in outer_cells]}")

# 计算小区间的距离矩阵
distance_matrix = squareform(pdist(positions))

# 为每个小区找到最近邻
print(f"\n小区邻接关系:")
neighbor_map = {}
for i, cell_id in enumerate(cell_indices):
    # 找到最近的3-4个邻居
    distances = distance_matrix[i]
    nearest_indices = np.argsort(distances)[1:5]  # 排除自己，取前4个最近邻
    neighbors = [cell_indices[j] for j in nearest_indices]
    neighbor_map[cell_id] = neighbors
    print(f"小区 {cell_id}: 邻居 {neighbors}")

# 可视化小区分布（2D投影到X-Y平面）
plt.figure(figsize=(12, 8))
colors = ['red', 'orange', 'yellow', 'lightblue']
labels = ['核心区域', '内环区域', '中环区域', '外环区域']

for layer_idx, cells in enumerate([core_cells, inner_cells, middle_cells, outer_cells]):
    layer_positions = positions[cells]
    plt.scatter(layer_positions[:, 0], layer_positions[:, 1], 
               c=colors[layer_idx], s=100, alpha=0.7, label=labels[layer_idx])
    
    # 添加小区编号标注
    for i, cell_idx in enumerate(cells):
        cell_id = cell_indices[cell_idx]
        plt.annotate(str(cell_id), 
                    (layer_positions[i, 0], layer_positions[i, 1]),
                    xytext=(5, 5), textcoords='offset points', fontsize=8)

plt.scatter(center[0], center[1], c='black', s=200, marker='x', label='几何中心')
plt.xlabel('X坐标 (米)')
plt.ylabel('Y坐标 (米)')
plt.title('小区分布与热点分层')
plt.legend()
plt.grid(True, alpha=0.3)
plt.axis('equal')
plt.tight_layout()
plt.savefig('cell_distribution_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# 生成Python代码用于queue.py
print(f"\n生成的热点配置代码:")
print(f"core_cells = {[cell_indices[i] for i in core_cells]}")
print(f"inner_cells = {[cell_indices[i] for i in inner_cells]}")
print(f"middle_cells = {[cell_indices[i] for i in middle_cells]}")
print(f"outer_cells = {[cell_indices[i] for i in outer_cells]}")

print(f"\nneighbor_map = {{")
for cell_id, neighbors in neighbor_map.items():
    print(f"    {cell_id}: {neighbors},")
print(f"}}")
