"""
测试评估可视化功能
"""

import sys
import os

# 添加当前目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from train_dynamic import evaluate_all_algorithms

def test_evaluation_visualization():
    """测试评估可视化"""
    print("开始测试算法评估和可视化...")
    
    # 在简单环境下评估算法（较快）
    results = evaluate_all_algorithms(complexity_level='simple', num_episodes=5)
    
    print("评估完成！")
    print("可视化图表已保存到 evaluation_results/ 目录")

if __name__ == "__main__":
    test_evaluation_visualization()
