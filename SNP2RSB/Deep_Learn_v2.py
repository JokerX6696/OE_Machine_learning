#!D:/Application/python/python.exe
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.optim.lr_scheduler import StepLR
import pandas as pd
import random
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc,roc_auc_score

### 参数
wk_dir = 'D:/desk/github/OE_Machine_learning/SNP2RSB/'
x_file = wk_dir + 'data/human_snp460_sample1250.ped'  # 特征值
y_file = wk_dir + 'data/overall_pheno.xls'  # 结果
##################  超参
num_epochs = 2000  # 迭代次数
xxl = 0.0005  # 学习率
bl = 0.1 # 测试集比例
num_tzz = 48 # 特征值数量
foreach_num = 1000
##################
# 检查是否有可用的 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 定义多层感知机模型
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(num_tzz, 24)  # 输入层到隐藏层的全连接层
        self.fc2 = nn.Linear(24, 12)  # 隐藏层到隐藏层的全连接层
        self.fc3 = nn.Linear(12, 1)  # 隐藏层到输出层的全连接层
        self.relu = nn.ReLU()  # 激活函数

    def forward(self, x):
        x = self.relu(self.fc1(x))  # 第一层全连接 + 激活
        x = self.relu(self.fc2(x))  # 第二层全连接 + 激活
        x = torch.sigmoid(self.fc3(x))  # 第三层全连接 + sigmoid激活（用于二分类）
        return x
# 定义数据集
x_file = 'D:/desk/github/OE_Machine_learning/SNP2RSB/data/human_snp460_sample1250.ped'
y_file = 'D:/desk/github/OE_Machine_learning/SNP2RSB/data/overall_pheno.xls'

x = pd.read_csv(x_file,sep='\t',header=None,index_col=0)
x = x.rename_axis('sample')
y = pd.read_csv(y_file,sep='\t',header=0,index_col=0)
y=y.rename_axis('sample')
x = x.reindex(y.index)

x.drop(columns=[1,2,3,4,5], inplace=True)
# 处理 x 转化为 数值
x = x.replace(" ","",regex=True)
x = x.replace({"00":1311,"AA":0, "AT":1, "TA":1,"AC":2, "CA":2, "AG":3, "GA":3, "TT":4, "TC":5, "CT":5, "TG":6, "GT":6, "CC":7, "CG":8, "GC":8, "GG":9},regex=True)
auc_max = 0
num_xl = 0
fo = open(wk_dir + 'train_log.txt','w')
while num_xl < foreach_num:
    # 随机选取特征值 num_tzz 个
    numbers = list(range(0, 460))
    tzz_sel = random.sample(numbers, num_tzz)
    x_list = [] ; y_list = []
    for index, row in x.iterrows():
        temp = [row.to_list()[i] for i in tzz_sel]
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

    # 划分数据集为训练集和测试集（90% 训练，10% 测试）
    l = len(x_list)        
    numbers = list(range(0, l))
    sample_size = int(len(numbers) * bl)
    test_sample = random.sample(numbers, sample_size)
    train_sample =  [num for num in numbers if num not in test_sample]

        
    X_train = torch.tensor([x_list[i] for i in train_sample], dtype=torch.float32).to(device)
    y_train = torch.tensor([y_list[i] for i in train_sample], dtype=torch.float32).to(device)
    X_test = torch.tensor([x_list[i] for i in test_sample], dtype=torch.float32).to(device)
    y_test = torch.tensor([y_list[i] for i in test_sample], dtype=torch.float32).to(device)



    # 实例化模型、损失函数和优化器
    model = MLP().to(device)
    criterion = nn.BCELoss()  # 二分类任务使用二元交叉熵损失函数
    optimizer = optim.Adam(model.parameters(), lr=xxl)  # Adam优化器

    # 定义学习率调度器
    #scheduler = StepLR(optimizer, step_size=100, gamma=0.1)  # 每隔100个epoch将学习率缩小为原来的0.1倍
    # 训练模型
    losses = []
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
        # 记录损失值
        losses.append(loss.item())
        # 打印训练信息
        if (epoch+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')


    # 测试模型
    # test = torch.tensor([[0,0], [1,1], [2,2]], dtype=torch.float32).to(device)
    with torch.no_grad():
        outputs = model(X_test)
        predicted = torch.round(outputs)
        # print(f'Predicted: {predicted.squeeze().tolist()}')
    

    # 获取模型在训练集上的预测概率值
    with torch.no_grad():
        outputs = model(X_test).cpu().numpy()
        predicted_prob = outputs.squeeze()

    # 计算训练集上的 AUC
    fpr, tpr, thresholds = roc_curve(y_test.cpu().numpy(), predicted_prob)
    roc_auc = auc(fpr, tpr)
    y_test_numpy = y_test.cpu().numpy()
    auc_value = roc_auc_score(y_test_numpy, predicted_prob)
    
    
    
    if auc_value > auc_max:
        print(f"特征值位点：{tzz_sel}\nauc:{auc_value}",file=fo)
            ###  绘图
        # 绘制损失函数的收敛曲线
        plt.plot(losses, label='Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss Convergence')
        plt.legend()
        plt.savefig(wk_dir + 'loss_curve.png')
        plt.close()
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
        plt.savefig(wk_dir + 'Training_set_AUC.png')
        plt.close()
        # 保存整体表格
        scz = []
        for j in y_test:
            for k in j:
                scz.append(k)
        predicted_list = predicted.cpu().squeeze().tolist()
        index_list = y.index.to_list()
        ret_list = [ret[i] for i in test_sample]
        scz_list = [elem.item() for elem in scz]
        df = {'sample': [index_list[i] for i in test_sample], 'score': ret_list, 'class': scz_list, 'predicted': predicted_list}
        df_ret = pd.DataFrame(df)
        df_ret.to_csv(wk_dir+'predict.xls',index=False,sep='\t')
         # 保存模型参数
        torch.save(model.state_dict(), wk_dir + 'model.pth')
        print("Model has been saved.")
        auc_max = auc_value
    num_xl += 1

fo.close  
####################################################
   

