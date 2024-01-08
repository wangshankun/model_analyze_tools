
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
from PIL import Image



import torch.fx
from mqbench.utils.state import enable_quantization, disable_all



########################config data process#################################
preprocess = torchvision.transforms.Compose([
    torchvision.transforms.Resize(256),
    torchvision.transforms.CenterCrop(224),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
])

path_onnx_model = "./resnet50_48mix_1205.onnx"

########################config onnruntime#################################

model = onnx.load(path_onnx_model)

for node in model.graph.node:#将中间所有节点的输出都做最终输出
    for output in node.output:
        if node.name == "input_quantized":
            model.graph.output.extend([onnx.ValueInfoProto(name=output)])

ort_session = onnxruntime.InferenceSession(model.SerializeToString())
outputs = [x.name for x in ort_session.get_outputs()]

input_name = ort_session.get_inputs()[0].name
dir_path = f"verfiy_data_1205"
isExists=os.path.exists(dir_path) #判断路径是否存在，存在则返回true
if not isExists:
    os.makedirs(dir_path)

def onnx_inference_on_image(img, label, results, img_path):
    img = preprocess(img).unsqueeze(0).numpy().astype(np.float32)

    ort_outs = ort_session.run(outputs, {input_name: img})
    ort_outs = OrderedDict(zip(outputs, ort_outs))

    filename = os.path.basename(img_path)
    for (x, y) in ort_outs.items():
        if x == '1005_QuantizeLinear': 
            file_path = dir_path + f"/{filename}.npy"
            np.save(file_path, y)


#########################run imagenet val data#####################
imagenet_val_dataset = torchvision.datasets.imagenet.ImageNet(
    "/root/imagenet", split="val")


onnx_results = []
for num, (img_path, label) in enumerate(imagenet_val_dataset.imgs):
    #if num == 1:
    #    break
    if num%100 == 0:
        print(num, img_path, label)
    img = Image.open(img_path)
    if img.mode != "RGB":
        continue
 
    onnx_inference_on_image(img, label, onnx_results, img_path)

