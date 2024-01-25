# import torch
#
# print("是否可用：", torch.cuda.is_available())        # 查看GPU是否可用
# print("GPU数量：", torch.cuda.device_count())        # 查看GPU数量
# print("torch方法查看CUDA版本：", torch.version.cuda)  # torch方法查看CUDA版本
# print("GPU索引号：", torch.cuda.current_device())    # 查看GPU索引号
# print("GPU名称：", torch.cuda.get_device_name())    # 根据索引号得到GPU名称
# print("版本：", torch.version.__version__)    # 根据索引号得到GPU名称

# import argparse
#
# p = argparse.ArgumentParser()
# p.add_argument('--sq' , type=int,default=3)
# p.add_argument('--num',type=str,default='./data.csv')
# a =p.parse_args()
# print( a.sq **2)
# print(a.num)

# import pandas as pd
# from sklearn.preprocessing import StandardScaler
# import numpy as np
#
# data_path = ".\PSM"
# data = pd.read_csv(data_path + '/train.csv')
# #取第一列
# datav = data.values[:, 1:]
# scaler = StandardScaler()
# data = np.nan_to_num(data)
# print(data[1,:])

import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)