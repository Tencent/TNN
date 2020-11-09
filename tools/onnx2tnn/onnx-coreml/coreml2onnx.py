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
    parser = argparse.ArgumentParser()
    parser.add_argument('coreml_folder', help='Input coreml model folder')
    args = parser.parse_args()
    coreml_folder = args.coreml_folder

    layer_bytes = []
    net_layer_data = {}
    coreml_net = coreml_folder + '/model.espresso.net'
    coreml_shape = coreml_folder + '/model.espresso.shape'
    coreml_weights = coreml_folder + '/model.espresso.weights'

    with open(coreml_net, encoding='utf-8') as f:
        net_dict = json.load(f)
        net_layers = net_dict['layers']

    # print(net_layers[1])

    with open(coreml_shape, encoding='utf-8') as f:
        net_dict = json.load(f)
        net_layer_shapes = net_dict['layer_shapes']

    # print(net_layer_shapes[net_layers[1]['bottom']])

    # coreml_weights
    with open(coreml_weights, 'rb') as f:
        # First byte of the file is an integer with how many
        # sections there are.  This lets us iterate through each section
        # and get the map for how to read the rest of the file.
        num_layers = struct.unpack('<i', f.read(4))[0]
        # print("num_layers: " + str(num_layers))

        f.read(4)  # padding bytes

        # The next section defines the number of bytes each layer contains.
        # It has a format of
        # | Layer Number | <padding> | Bytes in layer | <padding> |
        while len(layer_bytes) < num_layers:
            layer_num, _, num_bytes, _ = struct.unpack('<iiii', f.read(16))
            layer_bytes.append((layer_num, num_bytes))

        # print("layer_num: " + str(layer_num))
        # print("num_bytes: " + str(num_bytes))
        # print("each layer:\n")

        # Read actual layer weights.  Weights are floats as far as I can tell.
        for layer_num, num_bytes in layer_bytes:
            # print("layer_num: " + str(layer_num))
            # print("count: " + str(num_bytes / 4))
            data = struct.unpack("f" * int((num_bytes / 4)), f.read(num_bytes))
            net_layer_data[layer_num] = data

    # print(net_layer_data[1])
    # tensor_shape = [16]
    # tensor = helper.make_tensor('1', TensorProto.FLOAT, tensor_shape, net_layer_data[1])
    # print(tensor)

    #构建onnx
    #创建输入 (ValueInfoProto)
    net_inputes = []
    net_input_names = net_layers[0]['bottom'].split(',')
    for net_input_name in net_input_names:
        net_input_shape_dict = net_layer_shapes[net_input_name]
        net_input = helper.make_tensor_value_info(net_input_name, TensorProto.FLOAT,
                                                  [net_input_shape_dict['n'], net_input_shape_dict['k'],
                                                   net_input_shape_dict['h'], net_input_shape_dict['w']])
        net_inputes.append(net_input)

    #创建输出 (ValueInfoProto)
    net_output_shape_dict = net_layer_shapes[net_layers[-1]['top']]
    net_output = helper.make_tensor_value_info(net_layers[-1]['top'], TensorProto.FLOAT, [net_output_shape_dict['n'], net_output_shape_dict['k'], net_output_shape_dict['h'], net_output_shape_dict['w']])
    net_outputes = [net_output]

    # check the model.espresso.net file to adjust the index #
    print('check the model.espresso.net file to adjust the index')
    # net_output_name = net_layers[-2]['top']
    net_output_name = net_layers[-1]['top']
    if net_output_name.isdigit() != True:
        net_output2_shape_dict = net_layer_shapes[net_output_name]
        net_output2 = helper.make_tensor_value_info(net_output_name, TensorProto.FLOAT, [net_output2_shape_dict['n'], net_output2_shape_dict['k'], net_output2_shape_dict['h'], net_output2_shape_dict['w']])
        net_outputes.append(net_output2)

    onnx_blob_shapes = []
    for blob_name, blob_shape_dict in net_layer_shapes.items():
        onnx_blob_shapes.append(helper.make_tensor_value_info(blob_name, TensorProto.FLOAT,
                                                              [blob_shape_dict['n'],
                                                               blob_shape_dict['k'],
                                                               blob_shape_dict['h'],
                                                               blob_shape_dict['w']]))

    #创建nodes NodeProto
    onnx_net_nodes = []
    onnx_net_weights = []

    # check the model.espresso.net file to adjust the index #
    print('check the model.espresso.net file to adjust the index')
    # layer_info = net_layers[1]
    layer_info = net_layers[0]

    for layer_info in net_layers:
        print(layer_info['type'])
        if layer_info['type'] == 'convolution':
            stride_x = 1
            if  ('stride_x' in layer_info):
                stride_x = layer_info['stride_x']

            stride_y = 1
            if  ('stride_y' in layer_info):
                stride_y = layer_info['stride_y']

            node_inputs = layer_info['bottom'].split(',')
            if  ('blob_weights' in layer_info):
                node_inputs.append(str(layer_info['blob_weights']))
            if  ('blob_biases' in layer_info):
                node_inputs.append(str(layer_info['blob_biases']))

            node_conv_outputs = layer_info['top'].split(',')
            node_relu_outputs = []
            if layer_info['fused_relu'] == 1:
                node_relu_outputs = node_conv_outputs
                node_conv_outputs = []
                for temp_output in node_relu_outputs:
                    conv_output_blob_name = 'conv_'+temp_output
                    node_conv_outputs.append(conv_output_blob_name)
                    blob_shape_dict = net_layer_shapes[temp_output]
                    onnx_blob_shapes.append(helper.make_tensor_value_info(conv_output_blob_name, TensorProto.FLOAT,
                                                                          [blob_shape_dict['n'],
                                                                           blob_shape_dict['k'],
                                                                           blob_shape_dict['h'],
                                                                           blob_shape_dict['w']]))

            conv_group_num = layer_info['n_groups']
            layer_node = helper.make_node('Conv', # node type
                                          node_inputs, # inputs
                                          node_conv_outputs, # outputs
                                          kernel_shape = [layer_info['Nx'], layer_info['Ny']],
                                          strides = [stride_x, stride_y],
                                          pads = [layer_info['pad_l'], layer_info['pad_t'], layer_info['pad_r'], layer_info['pad_b']],
                                          group = conv_group_num,
                                          dilations = [1, 1])
            onnx_net_nodes.append(layer_node)

            #weights
            weights_shape = [layer_info['C'], int(layer_info['K']/conv_group_num), layer_info['Nx'], layer_info['Ny']]
            onnx_weights_tensor = helper.make_tensor(str(layer_info['blob_weights']), TensorProto.FLOAT, weights_shape, net_layer_data[layer_info['blob_weights']])
            onnx_net_weights.append(onnx_weights_tensor)

            #bias
            if  ('blob_biases' in layer_info):
                bias_shape = [layer_info['C']]
                onnx_bias_tensor = helper.make_tensor(str(layer_info['blob_biases']), TensorProto.FLOAT, bias_shape, net_layer_data[layer_info['blob_biases']])
                onnx_net_weights.append(onnx_bias_tensor)

            if layer_info['fused_relu'] == 1:
                layer_node = helper.make_node('Relu', # node type
                                              node_conv_outputs, # inputs
                                              node_relu_outputs, # outputs
                                              )
                onnx_net_nodes.append(layer_node)
        elif layer_info['type'] == 'pool':
            stride_x = 1
            if  ('stride_x' in layer_info):
                stride_x = layer_info['stride_x']

            stride_y = 1
            if  ('stride_y' in layer_info):
                stride_y = layer_info['stride_y']

            node_type = 'MaxPool'
            if  layer_info['avg_or_max'] == 0:
                node_type = 'AveragePool'

            node_inputs = layer_info['bottom'].split(',')
            node_outputs = layer_info['top'].split(',')
            layer_node = helper.make_node(node_type, # node type
                                          node_inputs, # inputs
                                          node_outputs, # outputs
                                          kernel_shape = [layer_info['size_x'], layer_info['size_y']],
                                          strides = [stride_x, stride_y],
                                          pads = [layer_info['pad_l'], layer_info['pad_t'], layer_info['pad_r'], layer_info['pad_b']])
            onnx_net_nodes.append(layer_node)
        elif layer_info['type'] == 'elementwise':
            node_inputs = layer_info['bottom'].split(',')
            node_type = ''
            node_inputs_extra = []
            if  layer_info['operation'] == 0:
                # check
                node_type = 'Add'
            elif layer_info['operation'] == 1:
                # check 注意如果输如只有1个，需要像取[layer_info['alpha']]值
                node_type = 'Mul'
                if len(node_inputs) == 1:
                    # scales
                    scales_tensor_name = 'elementwise_' + layer_info['top']
                    node_inputs_extra.append(scales_tensor_name)
                    scales = [layer_info['alpha']]
                    onnx_scales_tensor = helper.make_tensor(scales_tensor_name, TensorProto.FLOAT, [1], scales)
                    onnx_net_weights.append(onnx_scales_tensor)
            elif layer_info['operation'] == 2:
                node_type = 'Sub'
            elif layer_info['operation'] == 3:
                # check
                node_type = 'Mul'
                if len(node_inputs) == 1:
                    # scales
                    scales_tensor_name = 'elementwise_' + layer_info['top']
                    node_inputs_extra.append(scales_tensor_name)
                    scales = [layer_info['alpha']]
                    onnx_scales_tensor = helper.make_tensor(scales_tensor_name, TensorProto.FLOAT, [1], scales)
                    onnx_net_weights.append(onnx_scales_tensor)
            elif layer_info['operation'] == 10:
                # check，求倒数，y=1/x
                node_type = 'Div'
                # scales
                scales_tensor_name = 'elementwise_' + layer_info['top']
                node_inputs_extra.append(scales_tensor_name)
                scales = [layer_info['alpha']]
                onnx_scales_tensor = helper.make_tensor(scales_tensor_name, TensorProto.FLOAT, [1], scales)
                onnx_net_weights.append(onnx_scales_tensor)
            elif layer_info['operation'] == 24:
                # check
                node_type = 'Abs'
            else:
                print('Error: unsupported elementwise operation: ' + str(layer_info['operation']))
                assert(0)

            node_inputs = layer_info['bottom'].split(',')
            node_inputs.extend(node_inputs_extra)
            node_outputs = layer_info['top'].split(',')
            layer_node = helper.make_node(node_type, # node type
                                          node_inputs, # inputs
                                          node_outputs, # outputs
                                          )
            onnx_net_nodes.append(layer_node)
        elif layer_info['type'] == 'upsample':
            node_type = 'Upsample'
            mode = 0
            if  layer_info['mode'] != 0:
                print('Error: unsupported upsample mode: ' + str(layer_info['mode']))
                assert(0)

            scales_tensor_name = 'upsample_'+layer_info['top']

            node_inputs = layer_info['bottom'].split(',')

            if node_inputs[0].isdigit() != True:
                node_input_shape_dict = net_layer_shapes[node_inputs[0]]
                node_input_tensor = helper.make_tensor_value_info(node_inputs[0], TensorProto.FLOAT,
                                                                  [node_input_shape_dict['n'], node_input_shape_dict['k'],
                                                                   node_input_shape_dict['h'], node_input_shape_dict['w']])
                net_inputes.append(node_input_tensor)


            node_inputs.append(scales_tensor_name)
            node_outputs = layer_info['top'].split(',')
            layer_node = helper.make_node(node_type, # node type
                                          node_inputs, # inputs
                                          node_outputs, # outputs
                                          mode = 'nearest',
                                          )
            onnx_net_nodes.append(layer_node)

            # scales
            scales = [1.0, 1.0, layer_info['scaling_factor_x'], layer_info['scaling_factor_y']]
            onnx_scales_tensor = helper.make_tensor(scales_tensor_name, TensorProto.FLOAT, [4], scales)
            onnx_net_weights.append(onnx_scales_tensor)
        elif layer_info['type'] == 'concat':
            node_type = 'Concat'

            node_inputs = layer_info['bottom'].split(',')
            node_outputs = layer_info['top'].split(',')
            layer_node = helper.make_node(node_type,  # node type
                                          node_inputs,  # inputs
                                          node_outputs,  # outputs
                                          axis = 1,
                                          )
            onnx_net_nodes.append(layer_node)
        elif layer_info['type'] == 'activation':
            node_inputs = layer_info['bottom'].split(',')
            node_outputs = layer_info['top'].split(',')

            activation_mode = layer_info['mode']
            if activation_mode == 0:
                layer_node = helper.make_node('Relu',  # node type
                 node_inputs,  # inputs
                  node_outputs,  # outputs
                  )
            elif activation_mode == 1:
                layer_node = helper.make_node('Tanh',  # node type
                 node_inputs,  # inputs
                  node_outputs,  # outputs
                  )
            elif activation_mode == 2:
                layer_node = helper.make_node('LeakyRelu',  # node type
                 node_inputs,  # inputs
                  node_outputs,  # outputs
                  )
            elif activation_mode == 3:
                layer_node = helper.make_node('Sigmoid',  # node type
                 node_inputs,  # inputs
                  node_outputs,  # outputs
                  )
            elif activation_mode == 4:
                layer_node = helper.make_node('PRelu',  # node type
                 node_inputs,  # inputs
                  node_outputs,  # outputs
                  )
            elif activation_mode == 8:
                layer_node = helper.make_node('Elu',  # node type
                 node_inputs,  # inputs
                  node_outputs,  # outputs
                  )
            elif activation_mode == 9:
                layer_node = helper.make_node('ThresholdedRelu',  # node type
                 node_inputs,  # inputs
                  node_outputs,  # outputs
                  alpha = layer_info['alpha']
                  )
            elif activation_mode == 10:
                layer_node = helper.make_node('Softplus',  # node type
                 node_inputs,  # inputs
                  node_outputs,  # outputs
                  )
            elif activation_mode == 12:
                layer_node = helper.make_node('Softsign',  # node type
                 node_inputs,  # inputs
                  node_outputs,  # outputs
                  )
            else:
                print('Error: unsupported activation mode: ' + str(activation_mode))
                assert(0)

            onnx_net_nodes.append(layer_node)
        elif layer_info['type'] == 'load_constant':
            # constant_blob
            print('constant_blob: ' + str(layer_info['constant_blob']))
        else:
            print('Error: unsupported layer type: ' + layer_info['type'])
            assert(0)

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
    onnx.save(inferred_model, coreml_folder+'/model.onnx')


if __name__ == '__main__':
    main()
