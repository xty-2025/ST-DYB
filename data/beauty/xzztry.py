# import numpy as np
#
# with open('posandneg_edge.npy', "rb") as f:
#     a=np.load(f, allow_pickle=True)
#
# array = [None] * 16
#
# a=[]
# a.append([122, 123])
# a.append([12, 13])
# a.append([12, 133])
# array[8]=a
# print(array)
#
# jj={}
# jj['1']=[]
# jj['0']=[]
# jj['1'].append([1,2])
#
# print(jj)
#
# dictionary = {key: [] for key in range(1, 17)}
# print(dictionary)
# dictionary[1].append([2,3])
# dictionary[12].append([333,3])
# print(dictionary)
#
# import torch
#
# # 假设您有一个名为 tensor 的 tensor 对象
# tensor = torch.tensor(5.0)  # 示例值为 5.0
#
# # 将 tensor 转换为 int 类型
# integer_value = int(tensor.item())
#
# # 打印转换后的整数值
# print(type(integer_value))



import torch
tensor_2d = torch.tensor([[1, 2, 3],
                          [4, 5, 6]])
tensor_d = torch.tensor([[4, 1, 1],
                          [2, 1, 2]])
pos_score = torch.sum(tensor_2d*tensor_d, dim=1)
pos_out = torch.bmm(tensor_2d, tensor_d)
print(pos_score)
print(pos_out)
