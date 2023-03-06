
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



########################config torch runtime#################################
quant_model = torch.load("/root/work/MQBench/application/imagenet_example/mqbench_qmodel.pth")
test_net = quant_model.eval()
def torch_inference_on_image(img, label, results):
    img = preprocess(img).unsqueeze(0).cuda()

    outputs = test_net(img)
    tmp_out = outputs.detach()#_, pred = output.topk(5, 1, True, True)
    top5 = np.argsort(tmp_out.cpu().numpy(), axis=-1)[:, -5:]
    results.append((int(top5[0, -1] == label), np.sum(top5 == label)))
    #print("torch result: ", top5)
########################config onnruntime#################################
sess = onnxruntime.InferenceSession("/root/work/MQBench/application/imagenet_example/mqbench_qmodel_deploy.onnx",  providers=['CUDAExecutionProvider'])
input_name = sess.get_inputs()[0].name
def onnx_inference_on_image(img, label, results):
    img = preprocess(img).unsqueeze(0).numpy().astype(np.float32)

    outputs = sess.run(None, {input_name: img})[0]
    top5 = np.argsort(outputs, axis=-1)[:, -5:]
    results.append((int(top5[0, -1] == label), np.sum(top5 == label)))
    #print("onnx result: ", top5)

#########################run imagenet val data#####################
imagenet_val_dataset = torchvision.datasets.imagenet.ImageNet(
    "/root/imagenet", split="val")

torch_results = []
onnx_results = []
for num, (img_path, label) in enumerate(imagenet_val_dataset.imgs):
    if num%100 == 0:
        print(num, img_path, label)
    img = Image.open(img_path)
    if img.mode != "RGB":
        continue
 
    torch_inference_on_image(img, label, torch_results)
    onnx_inference_on_image(img, label, onnx_results)


print("pytorch run result:")
print("Top1:", sum(acc[0] for acc in torch_results) / len(torch_results))
print("Top5:", sum(acc[1] for acc in torch_results) / len(torch_results))
print("onnx run result:")
print("Top1:", sum(acc[0] for acc in onnx_results) / len(onnx_results))
print("Top5:", sum(acc[1] for acc in onnx_results) / len(onnx_results))
