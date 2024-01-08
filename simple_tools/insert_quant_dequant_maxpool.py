import onnx
from onnx import helper
from onnx import TensorProto

# 加载你的ONNX模型
model_path = 'mqbench_qmodel_deploy_align.onnx'  # 替换为你的模型路径
model = onnx.load(model_path)

# 遍历图中的节点
graph = model.graph
for i, node in enumerate(graph.node):
    # 检查是否是MaxPool节点
    if node.op_type == 'MaxPool':
        # 获取MaxPool节点的输出名称
        maxpool_output = node.output[0]

        # 创建DequantizeLinear节点
        dequantize_output = maxpool_output + "_dequantized"
        dequantize_node = onnx.helper.make_node(
            'DequantizeLinear',
            inputs=[maxpool_output, 'conv1_post_act_fake_quantizer.scale', 'conv1_post_act_fake_quantizer.zero_point'],  # 输入：MaxPool的输出，量化的scale和zero_point
            outputs=[dequantize_output],
            name='Dequantize_After_MaxPool'
        )

        # 创建QuantizeLinear节点
        quantize_output = maxpool_output + "_requantized"
        quantize_node = onnx.helper.make_node(
            'QuantizeLinear',
            inputs=[dequantize_output, 'maxpool_post_act_fake_quantizer.scale', 'maxpool_post_act_fake_quantizer.zero_point'],  # 输入：Dequantize的输出，量化的scale和zero_point
            outputs=[quantize_output],
            name='Quantize_After_Dequantize'
        )

        # 将新节点插入到MaxPool节点之后
        graph.node.insert(i + 1, dequantize_node)
        graph.node.insert(i + 2, quantize_node)

        # 更新后续节点的输入为QuantizeLinear的输出
        for subsequent_node in graph.node[i+3:]:
            input_names = list(subsequent_node.input)
            if maxpool_output in input_names:
                idx = input_names.index(maxpool_output)
                subsequent_node.input[idx] = quantize_output


try:
    onnx.checker.check_model(model)
except onnx.checker.ValidationError as e:
    logger.critical('The model is invalid: %s' % e)

# 保存修改后的模型
onnx.save(model, 'mqbench_qmodel_deploy_align_requant.onnx')


