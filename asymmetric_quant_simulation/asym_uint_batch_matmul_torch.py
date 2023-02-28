
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

    golden_res = tensor_a.matmul(tensor_b)

    if zp_a.item() != 0:
        tensor_a = torch.clamp(torch.round(tensor_a/sa + zp_a), 0, 2**omac.input_bits -1)
    else:
        tensor_a = torch.round(torch.clamp(tensor_a/sa + zp_a, -(2**(omac.input_bits-1)), 2**(omac.input_bits-1) -1))

    if zp_b.item() != 0:
        tensor_b = torch.clamp(torch.round(tensor_b/sb + zp_b), 0, 2**omac.input_bits -1)
    else:
        tensor_b = torch.round(torch.clamp(tensor_b/sb + zp_b, -(2**(omac.input_bits-1)), 2**(omac.input_bits-1) -1))

 
    b_quant_sum = torch.sum(tensor_b, dim=0, keepdim = False)#列相加
    b_quant_sum = b_quant_sum.unsqueeze(0).unsqueeze(1).expand(B, M, N)#扩展为result的shape

    result = tensor_a.matmul(tensor_b)
    if zp_a.item() != 0 and zp_b.item() == 0:
        result = result - zp_a*b_quant_sum
        pass

    result = result*sa*sb#反量化
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
    return result

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
print(quant(FA, torch.tensor(alpha_a), torch.tensor(z_a), 
            FW, torch.tensor(alpha_w), torch.tensor(z_w))
     )
'''
assert torch.allclose(Y, float_result, rtol=1e-3, atol=1e-3)
assert torch.allclose(quant_resutl, float_result, rtol=1e-3, atol=1e-3)





'''
def conv2d_quant_exe(input, input_scale, input_zp, kernel, kernel_scale, kernel_zp, bias=None, stride=1, padding=0, dilation=1, groups=1):
    stride, padding, dilation = pre_process_param(stride, padding, dilation)
    if padding > 0:
        input = F.pad(input, (padding,padding,padding,padding))
    batch_size = input.shape[0]
    input_h, input_w = input.shape[2:4]
    kernel_h, kernel_w = kernel.shape[2:4]
    out_channel, in_channel = kernel.shape[0:2]
    output_h = math.floor((input_h - kernel_h) / stride + 1)
    output_w = math.floor((input_w - kernel_w) / stride + 1)

    unfold = nn.Unfold(kernel_size=(kernel_h, kernel_w), stride=stride)
    input_vector = unfold(input)

    kernel_vector = kernel.reshape(kernel.shape[0], -1).T
    input_vector = input_vector.permute(0,2,1).contiguous()

    output = quant(input_vector, input_scale, input_zp, kernel_vector, kernel_scale, kernel_zp)#omac quant gemm

    if bias != None:
        output = output + bias
    output = output.reshape(batch_size, output_h, output_w, out_channel).permute(0,3,1,2).contiguous()
    return output
'''
