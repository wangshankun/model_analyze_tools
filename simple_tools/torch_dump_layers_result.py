
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
    #data_dir = os.path.join(os.path.dirname(__file__), "/root/work/MQBench/application/imagenet_example/PTQ/ptq/")
    #img_path = os.path.join(data_dir, "ILSVRC2012_val_00000025.JPEG")
    img_path = "/root/imagenet/val/n01751748/ILSVRC2012_val_00000001.JPEG"
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
quant_model = torch.load("mqbench_qmodel_org_1205.pth")
test_net = quant_model.eval()
output = test_net(_get_test_image_tensor().cuda())
_, pred = output.topk(5, 1, True, True)
pred = pred.t()
print(pred)
print(type(quant_model), type(test_net))




def create_hook_fn(name):
    def hook_fn(module, input, output):
        print('Layer name: ', name)
        print('Module type: ', type(module))
        #print('Input shape: ', input[0].shape)
        print('Output shape: ', output.shape)
        #print(output)
        output_np = output.detach().cpu().numpy()
        np.save("torch_layer_dump_1205/" + name + '.npy', output_np)
    return hook_fn

from mqbench.fake_quantize.qdrop_quantizer import QDropFakeQuantize
for name, layer in test_net._modules.items():
    print("Layer name: ", name)
    if isinstance(layer, QDropFakeQuantize):
    #if name in ["layer1_0_conv2_post_act_fake_quantizer", "layer1_0_conv1_post_act_fake_quantizer"]:
        hook = create_hook_fn(name)
        handle = layer.register_forward_hook(hook)

output = test_net(_get_test_image_tensor().cuda())
handle.remove() 
