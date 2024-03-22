#!D:/Application/python/python.exe
import pandas as pd
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
x = x.replace({"00":1311,"AA":0, "AT":1, "AC":2, "AG":3, "TA":4, "TT":5, "TC":6, "TG":7, "CA":8, "CT":9, "CC":10, "CG":11, "GA":12, "GT":13, "GC":14, "GG":15},regex=True)


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
    