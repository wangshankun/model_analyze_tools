
import torch
import numpy as np
np.random.seed(5)

# 随机生成输入和权重矩阵,W是对称量化，A是非对称量化
A = np.random.randint(0, 255, size=(4, 3)).astype(np.float32)
W = np.random.randint(-128, 127, size=(3, 2)).astype(np.float32)

z_a = 128
alpha_a = 0.1
z_w = 0
alpha_w = 0.05

FA = (A - z_a) * alpha_a
FW = (W - z_w) * alpha_w


Y = np.zeros((4, 2), dtype=np.float32)
for i in range(4):
    for j in range(2):
        dot_product = 0
        for k in range(3):
            dot_product += (A[i][k] - z_a) * (W[k][j] - z_w)
        Y[i][j] = dot_product * alpha_a * alpha_w


assert np.allclose(Y, np.matmul(FA, FW))


Y = np.zeros((4, 2), dtype=np.float32)
for i in range(4):
    for j in range(2):
        for k in range(3):
            #dot_product += (A[i][k] - z_a) * (W[k][j] - z_w)
            #dot_product += (A[i][k] - z_a) * (W[k][j])
            Y[i][j] += (A[i][k]*W[k][j] - z_a*W[k][j])
Y = Y * alpha_a * alpha_w
assert np.allclose(Y, np.matmul(FA, FW))



Y = np.zeros((4, 2), dtype=np.float32)
#此循环可以用Y = np.matmul(A, W)代替
for i in range(4):
    for j in range(2):
        for k in range(3):
            Y[i][j] += A[i][k]*W[k][j]

#此循环可拆解几步:(并行处理效率更高)
#1. W矩阵按列累加后变成一维矩阵 
#2. 然后将这个一维矩阵扩展到(4,2)维度
#3. 乘上一个 -z_a系数
#4. 直接和Y矩阵相加
for i in range(4):
    for j in range(2):
        for k in range(3):
            Y[i][j] -= z_a*W[k][j]
Y = Y * alpha_a * alpha_w
assert np.allclose(Y, np.matmul(FA, FW))



#########################最终效果##################################
Y = np.zeros((4, 2), dtype=np.float32)
Y = np.matmul(A, W)#uint * int

W_exp = np.zeros((4, 2), dtype=np.float32)#矩阵乘法零点补偿
for i in range(4):
    for j in range(2):
        for k in range(3):
            W_exp[i][j] += W[k][j]


W_col_sum = torch.sum(torch.from_numpy(W), dim=0, keepdim = False)#列相加
W_col_sum = W_col_sum.unsqueeze(0).expand(4, 2)#扩展到与result相同的shape
W_col_sum = W_col_sum.numpy()
assert np.array_equal(W_exp, W_col_sum) #三层循环和torch.sum后unsqueeze一样效果
Y = Y - z_a*W_col_sum
Y = Y * alpha_a * alpha_w

assert np.allclose(Y, np.matmul(FA, FW))
