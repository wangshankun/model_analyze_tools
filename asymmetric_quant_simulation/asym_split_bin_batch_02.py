
import torch
import numpy as np
np.random.seed(5)


import torch
from torch import nn
import torch.nn.functional as F

class Omac:
    def __init__(self):
        self.input_bits = 8
        #噪声数据来源https://confluence.int.lightelligence.co/display/~ydi/oMAC+model
        self.noise_sigma = 1.15#这种方案每次输出时候加上正态分布的
        self.noise_mean = 0
        self.adc_out_bits = 8
        self.opt_out_bits = 13
        self.vector_len = 32

omac = Omac()

def quant(tensor_a, sa, zp_a, tensor_b, sb, zp_b):
    B   = tensor_a.shape[0]#batch
    M   = tensor_a.shape[1]#
    K   = tensor_a.shape[2]#
    N   = tensor_b.shape[1]#


    tensor_a = torch.round(tensor_a/sa + zp_a)
    tensor_b = torch.round(tensor_b/sb + zp_b)

    b_quant_sum = torch.sum(tensor_b, dim=0, keepdim = False)#列相加
    b_quant_sum = b_quant_sum.unsqueeze(0).unsqueeze(1).expand(B, M, N)#扩展为result的shape

    tensor_a = tensor_a.type(torch.uint8)#把unit8输入拆分为8个二进制矩阵
    tensor_a_bits = [((tensor_a & (1 << i)) >> i).type(torch.uint8) for i in range(8)]
  
    total_result = torch.zeros(B, M, N)
    for i in range(len(tensor_a_bits)):
        tensor_a = tensor_a_bits[i] << 7#先放大,满足光功率
        tensor_a = tensor_a.type(torch.float32)
        result = tensor_a.matmul(tensor_b)
        result = result.type(torch.int32)
        result = result >> (7 - i)#后缩小
  
        total_result = total_result + result

    if zp_a.item() != 0 and zp_b.item() == 0:
        total_result = total_result - zp_a*b_quant_sum


    total_result = total_result*sa*sb#反量化
    ''' 
    if not torch.allclose(result, golden_res, rtol=1e-1, atol=1e-1):
        cos_sim = F.cosine_similarity(result.view(-1), golden_res.view(-1), 0).item()
        # 判断是否小于0.9
        if cos_sim < 0.9:
            print("golden_res:  ", golden_res)
            print("reult:  ", result)
            raise ValueError("Cosine similarity is below 0.9.")
    '''
    #return golden_res
    return total_result

# 随机生成输入和权重矩阵,W是对称量化，A是非对称量化
A = torch.randint(0, 255, size=(2, 4, 3))
W = torch.randint(-128, 127, size=(3, 2))

z_a = 128
alpha_a = 0.1
z_w = 0
alpha_w = 0.05

#########################浮点数结果##################################
FA = (A - z_a) * alpha_a
FW = (W - z_w) * alpha_w
float_result = FA.matmul(FW)

#########################整形计算 + 反量化##################################
Y = torch.zeros_like(float_result)
Y = A.matmul(W)#uint * int
W_col_sum = torch.sum(W, dim=0, keepdim = False)#列相加
W_col_sum = W_col_sum.unsqueeze(0).unsqueeze(1).expand(2, 4, 2)
Y = Y - z_a*W_col_sum
Y = Y * alpha_a * alpha_w
##########################伪量化逻辑(浮点变整形计算 + 反量化)#################################
quant_resutl = quant(FA, torch.tensor(alpha_a), torch.tensor(z_a),
                     FW, torch.tensor(alpha_w), torch.tensor(z_w))
'''
print(Y)
print(float_result)
print(quant_resutl)
'''
assert torch.allclose(Y, float_result, rtol=1e-3, atol=1e-3)
assert torch.allclose(quant_resutl, float_result, rtol=1e-3, atol=1e-3)
