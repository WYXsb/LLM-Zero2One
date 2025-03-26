import deepspeed
import torch
import torch.nn as nn
from torchvision.models import resnet50

model = resnet50(num_classes=10)  # 默认参数为 FP32
# 初始化 DeepSpeed 引擎
model_engine, optimizer, _, _ = deepspeed.initialize(
    model=model,
    model_parameters=model.parameters(),
    config="ds_config.json"
)


for inputs, labels in dataloader:
    inputs = inputs.to(model_engine.device)  # 输入数据（FP32）
    labels = labels.to(model_engine.device)

    # 前向传播（自动使用 FP16 计算权重）
    outputs = model_engine(inputs.half())  # 输入转换为 FP16
    loss = nn.CrossEntropyLoss()(outputs, labels)

    # 反向传播（DeepSpeed 自动处理梯度缩放）
    model_engine.backward(loss)

    # 参数更新（使用 FP32 主权重）
    model_engine.step()

    # 梯度清零（针对 FP32 主权重）
    model_engine.zero_grad()
