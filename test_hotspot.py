"""
测试热点流量分布
验证修改后的队列模型是否生成了合理的中心辐射式流量分布
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# 添加当前目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from environment.queue import QueueModel

def test_hotspot_distribution():
    """测试热点分布"""
    print("测试热点流量分布...")
    
    # 创建队列模型
    queue_model = QueueModel(
        num_cells=19,
        traffic_intensity=0.8,
        min_arrival_rate=0.1,
        max_arrival_rate=1.0
    )
    
    # 获取热点信息
    hotspot_info = queue_model.get_hotspot_info()
    arrival_rates = hotspot_info['arrival_rates']
    
    print("\n各小区的到达率分布：")
    print("小区编号\t到达率\t流量等级")
    print("-" * 40)
    
    # 定义流量等级
    max_rate = np.max(arrival_rates)
    for i, rate in enumerate(arrival_rates):
        percentage = (rate / max_rate) * 100
        if percentage >= 90:
            level = "核心热点"
        elif percentage >= 70:
            level = "高流量"
        elif percentage >= 40:
            level = "中等流量"
        elif percentage >= 20:
            level = "低流量"
        else:
            level = "基础流量"
        
        print(f"小区 {i:2d}\t{rate:.3f}\t{level} ({percentage:.0f}%)")
    
    # 可视化流量分布
    plt.figure(figsize=(12, 8))
    
    # 创建热力图
    plt.subplot(2, 2, 1)
    bars = plt.bar(range(19), arrival_rates, color='skyblue', edgecolor='navy', alpha=0.7)
    plt.title('各小区流量到达率分布')
    plt.xlabel('小区编号')
    plt.ylabel('到达率 (包/时隙)')
    plt.grid(True, alpha=0.3)
    
    # 为条形图添加数值标签
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.2f}', ha='center', va='bottom', fontsize=8)
    
    # 按流量等级分组显示
    plt.subplot(2, 2, 2)
    # 定义流量等级和对应的小区
    core_cells = [8, 9]
    inner_cells = [1, 2, 6, 7, 14, 15]
    middle_cells = [0, 3, 5, 10, 13, 16]
    outer_cells = [4, 11, 12, 17, 18]
    
    levels = ['核心区', '内环', '中环', '外环']
    avg_rates = [
        np.mean([arrival_rates[i] for i in core_cells if i < len(arrival_rates)]),
        np.mean([arrival_rates[i] for i in inner_cells if i < len(arrival_rates)]),
        np.mean([arrival_rates[i] for i in middle_cells if i < len(arrival_rates)]),
        np.mean([arrival_rates[i] for i in outer_cells if i < len(arrival_rates)])
    ]
    
    colors = ['red', 'orange', 'yellow', 'lightblue']
    plt.bar(levels, avg_rates, color=colors, alpha=0.7, edgecolor='black')
    plt.title('按区域分组的平均流量')
    plt.ylabel('平均到达率')
    plt.xticks(rotation=45)
    
    # 流量分布饼图
    plt.subplot(2, 2, 3)
    total_rates = [
        sum([arrival_rates[i] for i in core_cells if i < len(arrival_rates)]),
        sum([arrival_rates[i] for i in inner_cells if i < len(arrival_rates)]),
        sum([arrival_rates[i] for i in middle_cells if i < len(arrival_rates)]),
        sum([arrival_rates[i] for i in outer_cells if i < len(arrival_rates)])
    ]
    
    plt.pie(total_rates, labels=levels, colors=colors, autopct='%1.1f%%', startangle=90)
    plt.title('各区域流量占比')
    
    # 模拟运行一段时间的队列变化
    plt.subplot(2, 2, 4)
    time_steps = 50
    queue_history = []
    
    for t in range(time_steps):
        # 生成流量
        arrivals = queue_model.generate_traffic()
        
        # 模拟数据率（简化）
        data_rates = np.random.uniform(50000, 200000, 19)  # bits/s
        
        # 更新队列
        queue_model.update(data_rates)
        
        # 记录队列长度
        queue_lengths = queue_model.get_queue_lengths()
        queue_history.append(queue_lengths)
    
    # 绘制几个关键小区的队列变化
    queue_history = np.array(queue_history)
    key_cells = [8, 9, 1, 4]  # 核心、内环、外环代表
    cell_names = ['核心-8', '核心-9', '内环-1', '外环-4']
    
    for i, cell in enumerate(key_cells):
        if cell < queue_history.shape[1]:
            plt.plot(queue_history[:, cell], label=cell_names[i], marker='o', markersize=3)
    
    plt.title('关键小区队列长度变化')
    plt.xlabel('时间步')
    plt.ylabel('队列长度')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('hotspot_distribution_test.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\n流量分布测试完成！")
    print(f"核心区域平均流量: {avg_rates[0]:.3f}")
    print(f"内环区域平均流量: {avg_rates[1]:.3f}")
    print(f"中环区域平均流量: {avg_rates[2]:.3f}")
    print(f"外环区域平均流量: {avg_rates[3]:.3f}")
    print(f"流量梯度比 (核心:外环): {avg_rates[0]/avg_rates[3]:.1f}:1")

if __name__ == "__main__":
    test_hotspot_distribution()
