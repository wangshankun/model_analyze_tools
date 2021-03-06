import os
import sys
import logging
import argparse
import onnx
from onnx.tools import update_model_dims
import numpy as np
import onnx.helper as helper
from onnx import shape_inference, TensorProto

ONNX_DTYPE = {
    0: TensorProto.UNDEFINED,
    1: TensorProto.FLOAT,
    2: TensorProto.UINT8,
    3: TensorProto.INT8,
    4: TensorProto.UINT16,
    5: TensorProto.INT16,
    6: TensorProto.INT32,
    7: TensorProto.INT64,
    8: TensorProto.STRING,
    9: TensorProto.BOOL,
    10: TensorProto.FLOAT16,
    11: TensorProto.DOUBLE,
    12: TensorProto.UINT32,
    13: TensorProto.UINT64,
    14: TensorProto.COMPLEX64,
    15: TensorProto.COMPLEX128,
    16: TensorProto.BFLOAT16,
}
def main(argv):
    parser = argparse.ArgumentParser(description="ModelTrans && SubgraphFuse.")
    parser.add_argument("--src_model", required=True, help="Onnx model file path.")
    parser.add_argument("--dst_model", required=True, help="Output mode name.")
    parser.add_argument('--inputs', required=True, nargs='*', 
                       help="""Input tensors info: "input_name;elem_type;dim0;dim1;dim2;dim3" 
                               \r\n Example: "in0;1;1;224;224;3" "in1;1;1;32;32;128" """)

    args = parser.parse_args(argv)
    #解析input信息
    inputs_info = {}
    for in_tensor_str in args.inputs:
        info = in_tensor_str.split(';')
        inputs_info[info[0]] = info[1:]

    model = onnx.load(args.src_model)
    graph = model.graph
    #根据input信息设置模型输入graph shape
    for inx, input_tensor in enumerate(graph.input):
        if input_tensor.name in inputs_info:
            input_info = inputs_info[input_tensor.name]
            input_info = list(map(int, input_info))
            input_tensor_new = onnx.helper.make_tensor_value_info(
                                   name      = input_tensor.name, 
                                   elem_type = input_info[0], 
                                   shape     = input_info[1:])
            graph.input.remove(input_tensor)
            graph.input.insert(inx, input_tensor_new)
        else:
            print("input name is incorrect!")
            return 

    # append all tensor infos to graph input
    weight_infos = []
    tensors = graph.initializer
    for i, tensor in enumerate(tensors):
        value_info = helper.make_tensor_value_info(tensor.name, ONNX_DTYPE[tensor.data_type], tensor.dims)
        weight_infos.append(value_info)
        graph.input.insert(i+1, value_info) # because 0 is for placeholder, so start index is 1

    inferred_model = shape_inference.infer_shapes(model)
    onnx.checker.check_model(inferred_model)
    onnx.save(inferred_model, args.dst_model)
    inferred_graph = inferred_model.graph
    inferred_value_info = inferred_graph.value_info
    print(inferred_value_info)


if __name__ == "__main__":
    main(sys.argv[1:])
