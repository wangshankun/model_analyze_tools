import torch
import torch.nn as nn

# 定义模型
model = nn.Sequential(
    nn.Linear(10, 20),
    nn.ReLU(),
    nn.Linear(20, 30),
    nn.ReLU(),
    nn.Linear(30, 40),
    nn.ReLU()
)

# 定义回调函数
class IntermediateLayerGetter(nn.Module):
    def __init__(self, model, return_layers):
        super(IntermediateLayerGetter, self).__init__()
        self.model = model
        self.return_layers = set(return_layers)

    def forward(self, x):
        out = []
        for name, module in self.model._modules.items():
            x = module(x)
            if name in self.return_layers:
                out.append(x)
        return out

# 实例化回调函数
return_layers = {"0", "2"}
getter = IntermediateLayerGetter(model, return_layers)

# 获取模型中间层结果
input = torch.randn(1, 10)
outputs = getter(input)
for output in outputs:
    print(output.shape)

