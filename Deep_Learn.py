#!D/Application/python/python.exe
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
##################  超参

##################
# 检查是否有可用的 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 定义数据集
X_train = torch.tensor([[0,0], [1,2], [2,2]], dtype=torch.float32).to(device)
y_train = torch.tensor([[0], [1], [0]], dtype=torch.float32).to(device)

# 定义多层感知机模型
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(2, 64)  # 输入层到隐藏层的全连接层
        self.fc2 = nn.Linear(64, 32)  # 隐藏层到隐藏层的全连接层
        self.fc3 = nn.Linear(32, 1)  # 隐藏层到输出层的全连接层
        self.relu = nn.ReLU()  # 激活函数

    def forward(self, x):
        x = self.relu(self.fc1(x))  # 第一层全连接 + 激活
        x = self.relu(self.fc2(x))  # 第二层全连接 + 激活
        x = torch.sigmoid(self.fc3(x))  # 第三层全连接 + sigmoid激活（用于二分类）
        return x

# 实例化模型、损失函数和优化器
model = MLP().to(device)
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
###  绘图
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# 获取模型在训练集上的预测概率值
with torch.no_grad():
    outputs = model(X_train).cpu().numpy()
    predicted_prob = outputs.squeeze()

# 计算训练集上的 AUC
fpr, tpr, thresholds = roc_curve(y_train.cpu().numpy(), predicted_prob)
roc_auc = auc(fpr, tpr)

# 绘制 AUC 曲线
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.2f}')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.show()
    
####################################################
# 保存模型参数
torch.save(model.state_dict(), 'D:/desk/github/OE_Machine_learning/model.pth')
print("Model has been saved.")

# # 加载模型参数
# loaded_model = MLP()
# loaded_model.load_state_dict(torch.load('model.pth'))
# loaded_model.eval()  # 将模型设置为评估模式

# # 使用加载的模型进行预测
# X_new = torch.tensor([[1, 0], [0, 1]], dtype=torch.float32)
# with torch.no_grad():
#     predictions = loaded_model(X_new)
#     print("Predictions:", predictions)
