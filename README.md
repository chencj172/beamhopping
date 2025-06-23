# 基于强化学习的卫星跳波束图案设计与资源分配实验

## 项目概述

本项目实现了一个基于强化学习的卫星跳波束图案设计与资源分配系统。系统使用PPO算法进行训练，包含两个策略网络：一个用于跳波束图案设计（离散动作空间），另一个用于功率资源分配（连续动作空间）。

## 系统参数

- 卫星高度：1000km
- 波束个数：4
- 频段：Ka频段
- 最大覆盖小区数：19个
- 系统总带宽：200MHz
- 系统总功率：39dBW
- 频率复用方式：全频复用

## 项目结构

```
├── README.md                 # 项目说明文档
├── environment/              # 环境模拟模块
│   ├── satellite_env.py      # 卫星环境类
│   ├── channel.py            # 信道模型
│   └── queue.py              # 业务队列模型
├── rl/                       # 强化学习算法模块
│   ├── ppo.py                # PPO算法实现
│   ├── models.py             # 神经网络模型
│   └── buffer.py             # 经验回放缓冲区
├── data/                     # 数据文件
│   ├── satellite_data.py     # 卫星位置数据生成
│   ├── cell_data.py          # 小区位置数据生成
│   └── traffic_data.py       # 业务请求数据生成
├── utils/                    # 工具函数
│   ├── visualization.py      # 可视化工具
│   └── metrics.py            # 性能评估指标
└── main.py                   # 主程序入口
```

## 运行方法

```bash
# 安装依赖
pip install -r requirements.txt

# 运行实验
python main.py
```

## 系统目标

在不超过最大工作波束数量和系统总功率的约束下，最大化系统吞吐量并最小化系统延迟。