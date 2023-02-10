
#!/usr/bin/python
# -*- coding: UTF-8 -*-
import os, onnx, copy
import time
import numpy as np
import onnxruntime
import collections
from collections import OrderedDict
'''
def benchmark(model_path):
    session = onnxruntime.InferenceSession(model_path)
    input_name = session.get_inputs()[0].name

    total = 0.0
    runs = 1
    input_data = np.zeros((1, 3, 224, 224), np.float32)
    # Warming up
    _ = session.run([], {input_name: input_data})
    for i in range(runs):
        start = time.perf_counter()
        out = session.run([], {input_name: input_data})
        #print(out)
        end = (time.perf_counter() - start) * 1000
        total += end
        print(f"{end:.2f}ms")
    total /= runs
    print(f"Avg: {total:.2f}ms")
benchmark("./MEALV2_ResNet50_8bit_pertensor_onnx_qnn_deploy.onnx")
'''
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


session = onnxruntime.InferenceSession("./MEALV2_ResNet50_8bit_pertensor_onnx_qnn_deploy.onnx")
input_name = session.get_inputs()[0].name
input_data = _get_test_image_tensor()
input_data = input_data.numpy()
x = session.run([], {input_name: input_data})
x = torch.from_numpy(x[0])
_, pred = x.topk(5, 1, True, True)
pred = pred.t()
print(pred)


model = onnx.load("./MEALV2_ResNet50_8bit_pertensor_onnx_qnn_deploy.onnx")
ori_output = copy.deepcopy(model.graph.output)
for node in model.graph.node:
    if node.name == "input_quantized" or node.name == "Conv_6_quantized" or node.name == "MaxPool_11":
        for output in node.output:
            model.graph.output.extend([onnx.ValueInfoProto(name=output)])
ort_session = onnxruntime.InferenceSession(model.SerializeToString())
ort_inputs  = {input_name: input_data}
outputs = [x.name for x in ort_session.get_outputs()]
ort_outs = ort_session.run(outputs, ort_inputs)
ort_outs = OrderedDict(zip(outputs, ort_outs))
print(ort_outs)


