# Caltech-101图像分类实验

# 实验概述
本实验探索迁移学习在Caltech-101数据集上的应用效果，采用ResNet-18作为基础架构，比较了两种训练策略：

1.使用ImageNet预训练权重初始化并微调
2.完全从零开始训练

通过系统调整学习率等超参数，评估了预训练模型在小规模数据集上的性能优势。

# 环境与工具
​​框架​​: PyTorch + ResNet-18
​​数据集​​: Caltech-101 (101类+1背景类)
​​硬件​​: NVIDIA RTX 2080 Ti GPU (11GB显存)
​​Python​​: 3.8
​​模型权重​​: 百度网盘 (提取码: bypk)

# 代码结构
```
├── dataset.py              # 数据集加载与预处理
├── caltech101/             # 数据集
├── checkpoints/            # 训练好的模型
├── logs/                   # 训练日志
├── result/                 # 不同学习率的模型权重
├── model.py                # 模型定义
├── train.py                # 训练与验证流程
├── utils.py                # 辅助工具
└── visualize_prediction.py # 随机抽取图片查看预测效果
```

# 数据集配置
从Caltech-101官网下载数据集
解压到项目目录下的caltech101文件夹
```

└── caltech101/
    ├── accordion/
    ├── airplanes/
    ├── ...
    └── BACKGROUND_Google/
```
# 训练模型​
见train.py


使用TensorBoard监控训练过程：
```
tensorboard --logdir logs
```
