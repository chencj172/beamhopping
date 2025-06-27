"""
测试基于实际地理位置的热点分布
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from environment.queue import QueueModel
import numpy as np
import matplotlib.pyplot as plt

# 设置matplotlib中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']  # 设置中文字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 创建队列模型
queue_model = QueueModel(
    num_cells=19,
    traffic_intensity=0.8,
    min_arrival_rate=0.1,
    max_arrival_rate=1.0
)

# 获取到达率
arrival_rates = queue_model.arrival_rates

print("基于实际地理位置的热点分布测试:")
print("=" * 50)

# 按小区索引排序显示
for i in range(19):
    cell_id = i + 1  # 小区ID = 索引 + 1
    rate = arrival_rates[i]
    print(f"小区 {cell_id:2d} (索引 {i:2d}): 到达率 {rate:.3f}")

print("\n按到达率分层显示:")
print("=" * 30)

# 核心区域
core_cells = [0]  # 小区1
print(f"核心区域 (100%): 小区 {[i+1 for i in core_cells]} -> 平均到达率 {np.mean([arrival_rates[i] for i in core_cells]):.3f}")

# 内环区域
inner_ring = [3, 1, 2, 6, 4, 11]  # 小区4,2,3,7,5,12
print(f"内环区域 (80%):  小区 {[i+1 for i in inner_ring]} -> 平均到达率 {np.mean([arrival_rates[i] for i in inner_ring]):.3f}")

# 中环区域
middle_ring = [15, 7, 8, 9, 12, 10]  # 小区16,8,9,10,13,11
print(f"中环区域 (50%):  小区 {[i+1 for i in middle_ring]} -> 平均到达率 {np.mean([arrival_rates[i] for i in middle_ring]):.3f}")

# 外环区域
outer_ring = [13, 14, 16, 18, 17]  # 小区14,15,17,19,18
print(f"外环区域 (30%):  小区 {[i+1 for i in outer_ring]} -> 平均到达率 {np.mean([arrival_rates[i] for i in outer_ring]):.3f}")

# 其余区域
remaining_cells = [5]  # 小区6
print(f"其余区域 (25%):  小区 {[i+1 for i in remaining_cells]} -> 平均到达率 {np.mean([arrival_rates[i] for i in remaining_cells]):.3f}")

# 计算流量梯度
max_rate = max(arrival_rates)
min_rate = min(arrival_rates)
gradient = max_rate / min_rate if min_rate > 0 else float('inf')
print(f"\n流量梯度（最高/最低）: {gradient:.1f}:1")

# 可视化到达率分布
plt.figure(figsize=(12, 8))

# 创建柱状图
cell_ids = [i+1 for i in range(19)]
colors = []
for i in range(19):
    if i in core_cells:
        colors.append('red')
    elif i in inner_ring:
        colors.append('orange')
    elif i in middle_ring:
        colors.append('yellow')
    elif i in outer_ring:
        colors.append('lightblue')
    else:
        colors.append('lightgray')

bars = plt.bar(cell_ids, arrival_rates, color=colors, alpha=0.7, edgecolor='black')

# 添加数值标签
for i, (bar, rate) in enumerate(zip(bars, arrival_rates)):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
             f'{rate:.2f}', ha='center', va='bottom', fontsize=8)

# 添加图例
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='red', alpha=0.7, label='核心区域 (100%)'),
    Patch(facecolor='orange', alpha=0.7, label='内环区域 (80%)'),
    Patch(facecolor='yellow', alpha=0.7, label='中环区域 (50%)'),
    Patch(facecolor='lightblue', alpha=0.7, label='外环区域 (30%)'),
    Patch(facecolor='lightgray', alpha=0.7, label='其余区域 (25%)')
]
plt.legend(handles=legend_elements, loc='upper right')

plt.xlabel('Cell ID')
plt.ylabel('Arrival Rate (packets/slot)')
plt.title('Geographic-based Hotspot Traffic Distribution')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('geographic_hotspot_distribution.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"\n热点分布图已保存为 'geographic_hotspot_distribution.png'")
