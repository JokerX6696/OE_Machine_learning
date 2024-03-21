import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.optim.lr_scheduler import StepLR
import pandas as pd
### 参数

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 定义数据集
x_file = 'D:/desk/github/OE_Machine_learning/data/human_snp460_sample1250.ped'
y_file = 'D:/desk/github/OE_Machine_learning/data/overall_pheno.xls'

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

# 加载模型参数
loaded_model = MLP().to(device)
loaded_model.load_state_dict(torch.load('D:/desk/github/OE_Machine_learning/model.pth'))
loaded_model.eval()  # 将模型设置为评估模式

# 使用加载的模型进行预测
X_new = torch.tensor(X_train, dtype=torch.float32).to(device)
with torch.no_grad():
    predictions = loaded_model(X_new)
    predicted_classes = (predictions > 0.5).float()
    ret = []
    for j in predicted_classes.tolist():
        for k in j:
            ret.append(int(k))
    print("Predictions:", ret)


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
df_ret.to_csv('D:/desk/github/OE_Machine_learning/predict.xls',index=False,sep='\t')