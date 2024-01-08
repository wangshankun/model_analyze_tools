import onnx
from onnx import numpy_helper

def get_initializer_value(model, initializer_name):
    """
    Get the value of an initializer from an ONNX model.

    :param model: The ONNX model
    :param initializer_name: The name of the initializer to retrieve the value for
    :return: The value of the initializer as a NumPy array
    """
    # 遍历所有的初始化器
    for initializer in model.graph.initializer:
        # 检查初始化器的名称是否匹配
        if initializer.name == initializer_name:
            # 使用 numpy_helper 将 protobuf 对象转换为 NumPy 数组
            return numpy_helper.to_array(initializer)
    # 如果找不到指定名称的初始化器，返回 None
    return None
            

def modify_initializer(model, initializer_name, modified_tensor):
    # 遍历初始化器以找到并修改常量
    for initializer in model.graph.initializer:
        # 找到要修改的常量
        if initializer.name == initializer_name:
            # 将新值转换为正确的数据类型
            modified_tensor = numpy_helper.from_array(modified_tensor, name=initializer_name)
            # 将初始化器中的旧值替换为新值
            initializer.CopyFrom(modified_tensor)
            break

# 加载ONNX模型
model = onnx.load('mqbench_qmodel_deploy.onnx')

# 遍历图中的所有节点
for node in model.graph.node:
    # 找到所有的QLinearAdd节点
    if node.op_type == 'QLinearAdd':#强制修改对应initializer数值
        fourth_input_value = get_initializer_value(model, node.input[4])
        modify_initializer(model, node.input[1], fourth_input_value)
        
        fifth_input_value = get_initializer_value(model, node.input[5])
        modify_initializer(model, node.input[2], fifth_input_value)

# 保存修改后的模型
onnx.save(model, 'mqbench_qmodel_deploy_align.onnx')
