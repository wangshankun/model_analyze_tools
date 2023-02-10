
## torch 两种不同网络
### 一般伪量化的模型都是fx网络结构

```
torch.nn.Module 的一个子类是 torch.nn.Sequential，它允许您定义一组有序的模块，并以一个简单的方式组合它们。因此，这种类型的模型称为“序列模型”。

torch.fx.GraphModule 则是另一种模型类型，表示一个动态图模型。与序列模型相比，动态图模型在运行时可以动态地更改数据流。这是通过在编译时动态地构建图形来实现的，并且该图形在整个模型生命周期内可以更改。

因此，在选择使用 torch.nn.Sequential 还是 torch.fx.GraphModule 时，应考虑需要的模型复杂度和需要的控制级别。如果需要简单的序列模型，则应使用 torch.nn.Sequential，而如果需要更复杂的动态图模型，则应使用 torch.fx.GraphModule。
```
