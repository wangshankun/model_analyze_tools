
#!/usr/bin/python
# -*- coding: UTF-8 -*-
import os, onnx, copy
import time
import numpy as np
import onnxruntime
import collections
from collections import OrderedDict
import PIL
import torch
import torchvision
from torch import nn

def _get_test_image_tensor():
    data_dir = os.path.join(os.path.dirname(__file__), "/root/work/MQBench/application/imagenet_example/PTQ/ptq/")
    img_path = os.path.join(data_dir, "ILSVRC2012_val_00000025.JPEG")
    input_image = PIL.Image.open(img_path)
    # Based on example from https://pytorch.org/hub/pytorch_vision_resnet/
    preprocess = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize(256),
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )
    return preprocess(input_image).unsqueeze(0)



import torch.fx
from mqbench.utils.state import enable_quantization, disable_all
quant_model = torch.load("MEALV2_ResNet50_8bit_pertensor_onnx_qnn.pth")
test_net = quant_model.eval()


activation = dict()
def get_activation(name):
    def hook(model, input, output):  # pylint: disable=unused-argument, redefined-builtin
        activation[name] = output.detach()
    return hook

def has_submodules(model):
    #通过list(model.modules())来获取模型的所有子模块，
    #并判断其长度是否大于1，若大于1，则说明模型还有子模块。
    return hasattr(model, "modules") and len(list(model.modules())) > 5


for name, layer in test_net._modules.items():
    if not has_submodules(layer):
        handle = layer.register_forward_hook(get_activation(name))
output = test_net(_get_test_image_tensor().cuda())
handle.remove() 

print(test_net)
#print(activation['layer1_0_conv2_post_act_fake_quantizer'])
for name, layer in test_net._modules.items():
    if not has_submodules(layer):
        print(name, ": ", activation[name])






'''
def get_attention(name):
    def hook(module, input, output):
        # import pdb; pdb.set_trace();
        # len(input) == 1, but input is tuple so input[0]
        # input[0].shape, torch.Size([4, 901, 768])
        x = input[0]
        B, N, C = x.shape
        qkv = (
            module.qkv(x)
            .reshape(B, N, 3, module.num_heads, C // module.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * module.scale

        attn = attn.softmax(dim=-1)  # [:,:,1,1:]
        attention[name] = attn

    return hook
'''
