import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# 定义数据集
X_train = torch.tensor([[0], [1], [2]], dtype=torch.float32)
y_train = torch.tensor([[0], [1], [0]], dtype=torch.float32)

# 定义多层感知机模型
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(1, 64)  # 输入层到隐藏层的全连接层
        self.fc2 = nn.Linear(64, 32)  # 隐藏层到隐藏层的全连接层
        self.fc3 = nn.Linear(32, 1)  # 隐藏层到输出层的全连接层
        self.relu = nn.ReLU()  # 激活函数

    def forward(self, x):
        x = self.relu(self.fc1(x))  # 第一层全连接 + 激活
        x = self.relu(self.fc2(x))  # 第二层全连接 + 激活
        x = torch.sigmoid(self.fc3(x))  # 第三层全连接 + sigmoid激活（用于二分类）
        return x

# 实例化模型、损失函数和优化器
model = MLP()
criterion = nn.BCELoss()  # 二分类任务使用二元交叉熵损失函数
optimizer = optim.Adam(model.parameters(), lr=0.01)  # Adam优化器

# 训练模型
num_epochs = 1000
for epoch in range(num_epochs):
    # 前向传播
    outputs = model(X_train)
    loss = criterion(outputs, y_train)

    # 反向传播和优化
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # 打印训练信息
    if (epoch+1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# 测试模型
with torch.no_grad():
    outputs = model(X_train)
    predicted = torch.round(outputs)
    print(f'Predicted: {predicted.squeeze().tolist()}')