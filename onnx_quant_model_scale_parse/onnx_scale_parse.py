import sys
import onnx
from onnx.helper import get_attribute_value
import collections
import numpy as np
import logging
from matplotlib import pyplot as plt

class ParseConvScale:
    def __init__(self):
        self.model = None
        self.node_dict = {}
        self.init_dict = {}
        self.scale_dict = {}
        self.node_by_input = collections.defaultdict(list)
        self.node_by_output = collections.defaultdict(list)

    def _get_first_match_node_from_input(self, begin_node, type):#从input向上溯源第一个符合要求的node
            hit_nodes = []
            #sys.setrecursionlimit(16)#修改递归深度，量化节点紧邻卷积
            def _recursion(n, type):
                #print(n.name, n.op_type, n.input)
                if n.op_type == type:
                    hit_nodes.append(n)
                    return
                if n.op_type == 'Div':#遇到div提前返回
                    return
                for i in n.input:
                    for o in self.node_by_output[i]:
                        try:
                            _recursion(o, type)
                        except:#递归爆栈
                            print(begin_node, n.name, type)
                            return
            _recursion(self.node_dict[begin_node], type)
            return hit_nodes

    def _get_first_match_node_from_output(self, begin_node, type):#从output向下第一个符合要求的node
            hit_nodes = []
            def _recursion(n, type):
                if n.op_type == type:
                    hit_nodes.append(n)
                    return
                for o in n.output:
                    for i in self.node_by_input[o]:
                        _recursion(i, type)
            _recursion(self.node_dict[begin_node], type)
            return hit_nodes

    def parse(self, model):
        self._update_model(model)
        for n in self.model.graph.node:
            if n.op_type == "Conv":
                a_node = self._get_first_match_node_from_input(n.name, "Mul")
                w_node = self._get_first_match_node_from_input(n.name, "FixedPerTensorAffine")
                if len(w_node) == 0:
                    w_node = self._get_first_match_node_from_input(n.name, "FixedPerChannelAffine")
                z_node = self._get_first_match_node_from_output(n.name, "Div")

                if a_node and w_node and z_node:
                    a_scale = a_node[0].input[1]#第二个参数
                    a_scale = np.array(self.init_dict[a_scale])
                    w_scale = w_node[0].input[1]#第二个参数
                    w_scale = np.array(self.init_dict[w_scale])
                    z_scale = z_node[0].input[1]#第二个参数
                    z_scale = np.array(self.init_dict[z_scale])
                    m_scale = a_scale * w_scale / z_scale

                    self.scale_dict[n.name] = {'a_scale':[], 'w_scale':[], 'z_scale':[], 'm_scale':[]}
                    self.scale_dict[n.name]['a_scale'] = a_scale
                    self.scale_dict[n.name]['w_scale'] = w_scale
                    self.scale_dict[n.name]['z_scale'] = z_scale
                    self.scale_dict[n.name]['m_scale'] = m_scale
                else:
                    print('Conv {} no quantization parameter'.format(n.name))

    def get_scale_dict(self):
        return self.scale_dict

    def _onnx_datatype_to_npType(self, data_type):
        if data_type == 1:
            return np.float32
        elif data_type == 6:
            return np.int32
        elif data_type == 7:
            return np.int64
        elif data_type == 11:
            return np.float64
        else:
            print(data_type)
            raise TypeError("don't support data type")

    def _parser_initializer(self, initializer):
        name = initializer.name
        dtype = initializer.data_type
        weights = np.frombuffer(initializer.raw_data, dtype=self._onnx_datatype_to_npType(dtype))
        self.init_dict[name] = weights

    def _parser_graph_initializers(self):
        initializers = self.model.graph.initializer
        for initializer in initializers:
            self._parser_initializer(initializer)

    def _convert_constant_to_init(self):
        '''
        Convert constant node to initializer
        '''
        remove_nodes = [n for n in self.model.graph.node if n.op_type == "Constant"]
        for n in remove_nodes:
            val = get_attribute_value(n.attribute[0])
            val.name = n.output[0]
            self.model.graph.initializer.append(val)
            self.model.graph.node.remove(n)

    def _update_model(self, model):
        self.model = model
        # clean
        self.node_dict = {}
        self.init_dict = {}
        self.node_by_input = collections.defaultdict(list)
        self.node_by_output = collections.defaultdict(list)

        self._convert_constant_to_init()
        # update
        self.node_dict = {n.name: n for n in self.model.graph.node}
        self._parser_graph_initializers()

        for n in self.model.graph.node:
            for i in n.input:
                self.node_by_input[i].append(n)
            for o in n.output:
                self.node_by_output[o].append(n)

def dump_data_to_histogram_jpg(data, title):
    data = np.array(data)
    data = data.flatten()
    first_edge, last_edge = data.min(), data.max()
    n_equal_bins = 100 #分箱个数
    bin_edges = np.linspace(start=first_edge, stop=last_edge, num=n_equal_bins + 1, endpoint=True)
    plt.hist(data, bins = bin_edges) 
    plt.title(title) 
    plt.xlabel("min:%f --- max:%f" % (first_edge, last_edge), fontsize=10)
    #plt.show()
    pic_name = title + ".jpg"
    plt.savefig(pic_name)
    

if __name__ == '__main__':
    #model_path = "./resnet34-ssd-whole-4bit-quant.onnx"
    #model_path = "ssd-r34-w8a8-s-tensor-quant.onnx"
    model_path = "MEALV2_ResNet50_4w4a_s_pertensor.onnx"
    #model_path = "MEALV2_ResNet50_4w4a_s_perchannel.onnx"
    model = onnx.load(model_path)
    run = ParseConvScale()
    run.parse(model)
    scale_dict = run.get_scale_dict()
    #print(scale_dict)
    m_scale = []
    for it in scale_dict.values():
        if len(list(it['m_scale'])) > 1:
            m_scale.extend(list(it['m_scale'].flatten()))
        else:
            m_scale.append(it['m_scale'])

    #dump_data_to_histogram_jpg(weight_scale, "resnet34-ssd-whole-4bit-weight_scale-histogram")
    #dump_data_to_histogram_jpg(active_scale, "resnet34-ssd-whole-4bit-active_scale-histogram")
    #dump_data_to_histogram_jpg(m_scale, "MEALV2_ResNet50_4w4a_s_perchannel-m_scale-histogram")
    dump_data_to_histogram_jpg(m_scale, "MEALV2_ResNet50_4w4a_s_pertensor-m_scale-histogram")
