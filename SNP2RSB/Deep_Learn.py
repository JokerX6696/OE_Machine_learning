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
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
### 参数 
wk_dir = 'D:/desk/github/OE_Machine_learning/SNP2RSB'
x_file_train = 'D:/desk/github/OE_Machine_learning/SNP2RSB/data/train/human_snp460_sample1250.ped'  # 特征值
y_file_train = 'D:/desk/github/OE_Machine_learning/SNP2RSB/data/train/overall_pheno.xls'  # 结果
x_file_test = 'D:/desk/github/OE_Machine_learning/SNP2RSB/data/test/data2_460snp.ped'  # 特征值
y_file_test = 'D:/desk/github/OE_Machine_learning/SNP2RSB/data/test/overall_pheno.xls'  # 结果
##################  超参
Threshold = 40
num_epoch = 1000
lr = 0.001
snp_num = 454
num_tzz = 60
foreach_num = 500
###################
# 定义多层感知机模型
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(num_tzz, 64)  # 输入层到隐藏层的全连接层
        self.fc2 = nn.Linear(64, 32)  # 隐藏层到隐藏层的全连接层
        self.fc3 = nn.Linear(32, 1)  # 隐藏层到输出层的全连接层
        self.relu = nn.ReLU()  # 激活函数
    def forward(self, x):
        x = self.relu(self.fc1(x))  # 第一层全连接 + 激活
        x = self.relu(self.fc2(x))  # 第二层全连接 + 激活
        x = torch.sigmoid(self.fc3(x))  # 第三层全连接 + sigmoid激活（用于二分类）
        return x

##################  自定函数
# 测试数据是否合格
def is_numeric(value):  
    numeric_types = (int, float, complex)
    return isinstance(value, numeric_types)

def get_tensor(x_file,y_file):
    # 定义训练数据集
    x = pd.read_csv(x_file,sep='\t',header=None,index_col=0)
    x = x.rename_axis('sample')
    try:
        y = pd.read_csv(y_file,sep='\t',header=0,index_col=0)
    except UnicodeDecodeError:
        y = pd.read_csv(y_file,sep='\t',header=0,index_col=0,encoding='gbk')
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
    for j in x_list:
        for k in j:
            if not is_numeric(k):
                print('特征值文件 错误 出现了缺失值！')
                exit()
    ret = y[y.columns[0]].to_list()
    for i in ret:
        if i >= Threshold:
            y_list.append([1])
        elif i < Threshold:
            y_list.append([0])
        else:
            print('结果文件 错误 出现了缺失值！')
            exit()
    return x_list,y_list,x,y

    

# 检查是否有可用的 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 获取特征值
X_train,y_train,_,_ = get_tensor(x_file=x_file_train,y_file=y_file_train)
X_test,y_test,x,y = get_tensor(x_file=x_file_test,y_file=y_file_test)


# 定义学习率调度器
# scheduler = StepLR(optimizer, step_size=100, gamma=0.8)  # 每隔100个epoch将学习率缩小为原来的0.1倍
# 训练模型
auc_max = 0
num_xl = 0
fo = open(wk_dir + '/train_log.txt','w')
num_epochs = num_epoch
while num_xl < foreach_num:
    # 实例化模型、损失函数和优化器
    model = MLP().to(device)
    criterion = nn.BCELoss()  # 二分类任务使用二元交叉熵损失函数
    optimizer = optim.Adam(model.parameters(), lr=lr)  # Adam优化器
    # 随机选取特征值 num_tzz 个
    numbers = list(range(0, snp_num))
    tzz_sel = random.sample(numbers, num_tzz)
    X_train_part = [[j[i] for i in tzz_sel] for j in X_train]
    y_train_part = y_train
    X_test_part = [[j[i] for i in tzz_sel] for j in X_test]
    y_test_part = y_test
    X_train_part = torch.tensor(X_train_part, dtype=torch.float32).to(device)
    y_train_part = torch.tensor(y_train_part, dtype=torch.float32).to(device)
    X_test_part = torch.tensor(X_test_part, dtype=torch.float32).to(device)
    y_test_part = torch.tensor(y_test_part, dtype=torch.float32).to(device)
    losses = []
    for epoch in range(num_epochs):
        # 前向传播
        outputs = model(X_train_part)
        loss = criterion(outputs, y_train_part)

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
        outputs = model(X_test_part)
        predicted = torch.round(outputs)
        # print(f'Predicted: {predicted.squeeze().tolist()}')


    # 获取模型在训练集上的预测概率值
    with torch.no_grad():
        outputs = model(X_test_part).cpu().numpy()
        predicted_prob = outputs.squeeze()
    # 计算训练集上的 AUC
    fpr, tpr, thresholds = roc_curve(y_test, predicted_prob)
    roc_auc = auc(fpr, tpr)
    y_test_numpy = y_test
    auc_value = roc_auc_score(y_test_numpy, predicted_prob)

    # # 计算训练集上的 AUC
    # fpr, tpr, thresholds = roc_curve(y_test_part.cpu().numpy(), predicted_prob)
    # roc_auc = auc(fpr, tpr)
    if auc_value > auc_max:
        print(f"特征值位点：{tzz_sel}\nauc:{auc_value}",file=fo)
        # 绘制损失函数的收敛曲线
        plt.plot(losses, label='Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss Convergence')
        plt.legend()
        plt.savefig(wk_dir + '/loss_curve.png')
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
        plt.savefig('D:/desk/github/OE_Machine_learning/SNP2RSB/Training_set_AUC.png')
        plt.close()
        ####################################################
        # 保存模型参数
        torch.save(model.state_dict(), 'D:/desk/github/OE_Machine_learning/SNP2RSB/model.pth')
        print("Model has been saved.")
        #  保存整体表格
        predicted_list = predicted.cpu().squeeze().tolist()
        index_list = y.index.to_list()
        ret_list = y['心率体温综合评分'].to_list()
        scz = []
        for j in ret_list:
            if j < Threshold:
                scz.append(0)
            else:
                scz.append(1)
        
        df = {'sample': index_list, 'score': ret_list, 'class': scz, 'predicted': predicted_list}
        df_ret = pd.DataFrame(df)
        df_ret.to_csv('D:/desk/github/OE_Machine_learning/SNP2RSB/predict.xls',index=False,sep='\t')

        all_pre = len(scz)
        pos = 0
        for i in range(0,all_pre):
            if predicted_list[i] == scz[i]:
                pos += 1
        ret = pos/all_pre
        
        auc_max = auc_value
    print(f"预测准确率为 {ret}")
    print(f"AUC={auc_value}")
    num_xl += 1
fo.close  

