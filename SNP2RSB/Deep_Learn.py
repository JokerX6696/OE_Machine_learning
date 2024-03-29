#!D:/Application/python/python.exe
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.optim.lr_scheduler import StepLR
import pandas as pd
### 参数
x_file = 'D:/desk/github/OE_Machine_learning/SNP2RSB/data/train/human_snp460_sample1250.ped'  # 特征值
y_file = 'D:/desk/github/OE_Machine_learning/SNP2RSB/data/train/overall_pheno.xls'  # 结果
x_file_test = 'D:/desk/github/OE_Machine_learning/SNP2RSB/data/test/data2_460snp.ped'  # 特征值
y_file_test = 'D:/desk/github/OE_Machine_learning/SNP2RSB/data/test/overall_pheno.xls'  # 结果
##################  超参

##################
# 检查是否有可用的 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 定义训练数据集
x = pd.read_csv(x_file,sep='\t',header=None,index_col=0)
x = x.rename_axis('sample')
y = pd.read_csv(y_file,sep='\t',header=0,index_col=0)
y=y.rename_axis('sample')
x = x.reindex(y.index)

x.drop(columns=[1,2,3,4,5], inplace=True)
# 处理 x 转化为 数值
x = x.replace(" ","",regex=True)
x = x.replace({"00":1311,"AA":0, "AT":1, "TA":1,"AC":2, "CA":2, "AG":3, "GA":3, "TT":4, "TC":5, "CT":5, "TG":6, "GT":6, "CC":7, "CG":8, "GC":8, "GG":9},regex=True)


x_list = [] ; y_list = []
for index, row in x.iterrows():
    temp = row.to_list()
    x_list.append(temp)

def is_numeric(value):
    numeric_types = (int, float, complex)
    return isinstance(value, numeric_types)

for j in x_list:
    for k in j:
        if not is_numeric(k):
            print('特征值文件 错误 出现了缺失值！')
            exit()

ret = y[y.columns[0]].to_list()
for i in ret:
    if i >= 40:
        y_list.append([1])
    elif i < 40:
        y_list.append([0])
    else:
        print('结果文件 错误 出现了缺失值！')
        exit()
    
    
X_train = torch.tensor(x_list, dtype=torch.float32).to(device)
y_train = torch.tensor(y_list, dtype=torch.float32).to(device)
# 定义测试数据集
x2 = pd.read_csv(x_file_test,sep='\t',header=None,index_col=0)
x2 = x2.rename_axis('sample')
y2 = pd.read_csv(y_file_test,sep='\t',header=0,index_col=0,encoding='gbk')
y2.index=[i.strip() for i in y2.index.to_list()]
y2=y2.rename_axis('sample')
x2 = x2.reindex(y2.index)

x2.drop(columns=[1,2,3,4,5], inplace=True)
# 处理 x 转化为 数值
x2 = x2.replace(" ","",regex=True)
x2 = x2.replace({"00":1311,"AA":0, "AT":1, "TA":1,"AC":2, "CA":2, "AG":3, "GA":3, "TT":4, "TC":5, "CT":5, "TG":6, "GT":6, "CC":7, "CG":8, "GC":8, "GG":9},regex=True)


x2_list = [] ; y2_list = []
for index, row in x2.iterrows():
    temp = row.to_list()
    x2_list.append(temp)


for j in x2_list:
    for k in j:
        if not is_numeric(k):
            print('特征值文件 错误 出现了缺失值！')
            exit()

ret = y2[y2.columns[0]].to_list()
for i in ret:
    if i >= 40:
        y2_list.append([1])
    elif i < 40:
        y2_list.append([0])
    else:
        print('结果文件 错误 出现了缺失值！')
        exit()
X_test = torch.tensor(x2_list, dtype=torch.float32).to(device)
y_test = torch.tensor(y2_list, dtype=torch.float32).to(device)
# 定义多层感知机模型
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(460, 64)  # 输入层到隐藏层的全连接层
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
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam优化器

# 定义学习率调度器
scheduler = StepLR(optimizer, step_size=100, gamma=0.8)  # 每隔100个epoch将学习率缩小为原来的0.1倍
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
    # 更新学习率
    #scheduler.step()

    # 打印训练信息
    if (epoch+1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# 测试模型
# test = torch.tensor([[0,0], [1,1], [2,2]], dtype=torch.float32).to(device)
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
plt.savefig('D:/desk/github/OE_Machine_learning/SNP2RSB/Training_set_AUC.png')

    
####################################################
# 保存模型参数
torch.save(model.state_dict(), 'D:/desk/github/OE_Machine_learning/SNP2RSB/model.pth')
print("Model has been saved.")
# 保存整体表格
scz = []
for j in y_train:
    for k in j:
        scz.append(k)
predicted_list = predicted.cpu().squeeze().tolist()
index_list = y.index.to_list()
ret_list = ret
scz_list = [elem.item() for elem in scz]
df = {'sample': index_list, 'score': ret_list, 'class': scz_list, 'predicted': predicted_list}
df_ret = pd.DataFrame(df)
df_ret.to_csv('D:/desk/github/OE_Machine_learning/SNP2RSB/predict.xls',index=False,sep='\t')
exit()
# # 加载模型参数
# loaded_model = MLP()
# loaded_model.load_state_dict(torch.load('model.pth'))
# loaded_model.eval()  # 将模型设置为评估模式

# # 使用加载的模型进行预测
# X_new = torch.tensor([[1, 0], [0, 1]], dtype=torch.float32)
# with torch.no_grad():
#     predictions = loaded_model(X_new)
#     print("Predictions:", predictions)
