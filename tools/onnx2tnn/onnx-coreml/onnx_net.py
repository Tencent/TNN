from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import argparse
import sys
import time

import struct
import json

import onnx
from onnx import helper, shape_inference
from onnx import AttributeProto, TensorProto, GraphProto
from onnx import numpy_helper

def main():
    # parser = argparse.ArgumentParser()
    # parser.add_argument('coreml_folder', help='Input coreml model folder')
    # args = parser.parse_args()
    # coreml_folder = args.coreml_folder
    #
    # coreml_net = coreml_folder + '/model.espresso.net'
    # coreml_shape = coreml_folder + '/model.espresso.shape'
    # coreml_weights = coreml_folder + '/model.espresso.weights'

    # print(net_layer_data[1])
    # tensor_shape = [16]
    # tensor = helper.make_tensor('1', TensorProto.FLOAT, tensor_shape, net_layer_data[1])
    # print(tensor)

    #构建onnx
    #创建输入 (ValueInfoProto)
    net_inputes = []
    net_input = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 13, 4, 4])
    net_inputes.append(net_input)

    #创建输出 (ValueInfoProto)
    net_output = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 13, 4, 4])
    net_outputes = [net_output]

    onnx_blob_shapes = [net_input, net_output]

    #创建nodes NodeProto
    onnx_net_nodes = []
    onnx_net_weights = []

    node_inputs = ["input"]
    node_outputs = ["output"]
    layer_node = helper.make_node("ThresholdedRelu",  # node type
                                  node_inputs,  # inputs
                                  node_outputs,  # outputs
                                  alpha = 1.5
                                  )
    onnx_net_nodes.append(layer_node)

    #创建graph GraphProto
    graph_def = helper.make_graph(
        onnx_net_nodes,
        'onnx-model',
        net_inputes,
        net_outputes,
        initializer = onnx_net_weights,
        value_info = onnx_blob_shapes,
    )

    #创建model (ModelProto)
    # onnx_model = helper.make_model(graph_def, producer_name='YouTu Tencent')
    onnx_model = helper.make_model(graph_def, producer_name='YouTu Tencent', opset_imports=[helper.make_operatorsetid("", 12)])

    # print('The model is:\n{}'.format(onnx_model))
    onnx.checker.check_model(onnx_model)
    print('Before shape inference, the shape info of Y is:\n{}'.format(onnx_model.graph.value_info))

    # Apply shape inference on the model
    inferred_model = shape_inference.infer_shapes(onnx_model)
    onnx.checker.check_model(inferred_model)
    print('After shape inference, the shape info of Y is:\n{}'.format(inferred_model.graph.value_info))
    onnx.save(inferred_model, 'onnx_net.onnx')


if __name__ == '__main__':
    main()
