# 获取onnx模型中间层的输入输出shape信息

* 命令行例子，--inputs 多个输入顺序需要和模型本身的input node顺序一致
``
 python main.py --src_model bertsquad-12-sim.onnx --dst_model bertsquad-12-sim-shape.onnx --inputs "unique_ids_raw_output___9:0;7;1" "segment_ids:0;7;1;256" "input_mask:0;7;1;256" "input_ids:0;7;1;256"
``
