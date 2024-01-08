import onnx

# 加载模型
model = onnx.load("mqbench_qmodel_deploy_align_1226.onnx")
graph = model.graph

node_to_delete = 'Relu_3'

# 查找并重定向输入输出
for node in graph.node:
    if node.name == node_to_delete:
        # 记录要删除节点的输入和输出
        input_tensor = node.input[0]
        output_tensor = node.output[0]

        # 删除节点
        graph.node.remove(node)

        # 重定向所有后续节点的输入
        for subsequent_node in graph.node:
            for i, input_name in enumerate(subsequent_node.input):
                if input_name == output_tensor:
                    subsequent_node.input[i] = input_tensor

        break

# 保存修改后的模型
onnx.save(model, "modified.onnx")

