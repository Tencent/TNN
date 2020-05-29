import numpy as np
import src.c2oObject as Node
import math
import copy


def get_pool_pads(layer):
    pad = layer.pooling_param.pad
    if pad != 0:
        pad_h = pad_w = pad
    else:
        if layer.pooling_param.pad_h != 0 and layer.pooling_param.pad_w != 0:
            pad_h = layer.pooling_param.pad_h
            pad_w = layer.pooling_param.pad_w
        else:
            pad_h = pad_w = 0
    pads = [0, 0, pad_h, pad_w, 0, 0, pad_h, pad_w]

    return pads


def calculate_pad_output_shape(input_shape, pads):
    pad_h = pads[2]
    pad_w = pads[3]
    output_shape = copy.deepcopy(input_shape[0])

    output_shape[2] = output_shape[2] + 2 * pad_h
    output_shape[3] = output_shape[3] + 2 * pad_w
    return [output_shape]


def create_pad_node(layer, node_name, input_name, output_name, input_shape):
    pads = get_pool_pads(layer)
    attributes = {"mode": "constant"}
    pad_input_name = input_name
    pad_output_name = output_name
    pad_output_shape = calculate_pad_output_shape(input_shape, pads)

    node = Node.c2oNode(layer, node_name, 'Pad', pad_input_name, pad_output_name, input_shape, pad_output_shape,
                        attributes)

    return node


def get_pool_attributes(layer, pool_type, input_shape):
    number = input_shape[0][0]
    channel = input_shape[0][1]
    height = input_shape[0][2]
    weight = input_shape[0][3]
    kernel_size = layer.pooling_param.kernel_size
    pad = layer.pooling_param.pad
    stride = layer.pooling_param.stride

    if pool_type == 'GlobalMaxPool' or pool_type == 'GlobalAveragePool':
        global_pooling = True
    else:
        global_pooling = False
    # pass kernel_shape
    if global_pooling:
        kernel_h = height
        kernel_w = weight
    else:
        if kernel_size != 0:
            kernel_h = kernel_w = kernel_size
        elif layer.pooling_param.kernel_h != 0 and layer.pooling_param.kernel_w != 0:
            kernel_h = layer.pooling_param.kernel_h
            kernel_w = layer.pooling_param.kernel_w
        else:
            kernel_h = 1
            kernel_w = 1
    kernel_shape = [kernel_h, kernel_w]
    # pass pad
    if pad != 0:
        pad_h = pad_w = pad
    else:
        if layer.pooling_param.pad_h != 0 and layer.pooling_param.pad_w != 0:
            pad_h = layer.pooling_param.pad_h
            pad_w = layer.pooling_param.pad_w
        else:
            pad_h = pad_w = 0
    pads = [pad_h, pad_w, pad_h, pad_w]
    # 由于 caffe 与 onnx 的 pad 的计算的原因，将 pad 属性，单独创建一个节点
    pads = [0, 0, 0, 0]
    # pass strides
    stride_h = stride_w = 1
    if stride != 1:
        stride_h = stride_w = stride
    else:
        if layer.pooling_param.stride_h != 0 and layer.pooling_param.stride_w != 0:
            stride_h = layer.pooling_param.stride_h
            stride_w = layer.pooling_param.stride_w
        else:
            stride_h = stride_w = 1
    strides = [stride_h, stride_w]

    # pass round_mode
    # caffe 上默认是使用 ceil 的，但是在 onnx 默认使用 floor
    # caffe definition
    #   enum RoundMode {
    #     CEIL = 0;
    #     FLOOR = 1;
    #   }
    # default Ceil = 0
    # onnx ceil_mode floor = 0, ceil = 1, default: floor = 0
    round_mode_ceil = 0
    round_mode_floor = 1
    round_mode = 0
    if layer.pooling_param.round_mode == 0:
        round_mode = round_mode_ceil
    elif layer.pooling_param.round_mode == 1:
        round_mode = round_mode_floor
    else:
        # wrong condition
        exit(-1)
    if round_mode == round_mode_ceil:
        ceil_mode = 1
    else:
        ceil_mode = 0

    attributes = {"kernel_shape": kernel_shape,
                  "strides": strides,
                  "pads": pads,
                  "ceil_mode": ceil_mode
                  }
    return attributes


# 计算输出维度
def get_pooling_output_shape(input_shape, layer, attributes, with_indices=False):
    number = input_shape[0][0]
    channel = input_shape[0][1]
    kernel_shape = attributes["kernel_shape"]
    kernel_h = kernel_shape[0]
    kernel_w = kernel_shape[1]
    pads = attributes["pads"]
    strides = attributes["strides"]
    stride_h = strides[0]
    stride_w = strides[1]
    ceil_mode = attributes["ceil_mode"]
    pad_h = pads[2]
    pad_w = pads[3]
    height = input_shape[0][2]
    width = input_shape[0][3]

    if ceil_mode == 1:
        # ceil
        pooled_height = int(math.ceil((height + 2 * pad_h - kernel_h) / stride_h)) + 1
        pooled_width = int(math.ceil((width + 2 * pad_h - kernel_w) / stride_w)) + 1
    else:
        # floor
        pooled_height = int(math.floor((height + 2 * pad_h - kernel_h) / stride_h)) + 1
        pooled_width = int(math.floor((width + 2 * pad_h - kernel_w) / stride_w)) + 1

    if pad_h != 0 or pad_w != 0:
        if ((pooled_height - 1) * stride_h) >= (height + pad_h):
            pooled_height = pooled_height - 1
        if ((pooled_width - 1) * stride_w) >= (width + pad_w):
            pooled_width = pooled_width - 1
    if kernel_h == 0:
        kernel_h = kernel_w = 1
    if with_indices:
        output_shape = [[number, channel, pooled_height, pooled_width],
                        [number, channel, pooled_height, pooled_width]]
    else:
        output_shape = [[number, channel, pooled_height, pooled_width]]
    return output_shape


def pooling_type(layer):
    pool_value = layer.pooling_param.pool
    global_value = layer.pooling_param.global_pooling
    if pool_value == 0 and global_value is True:
        return 'GlobalMaxPool'
    elif pool_value == 1 and global_value is True:
        return 'GlobalAveragePool'
    elif pool_value == 0 and global_value is False:
        return 'MaxPool'
    elif pool_value == 1 and global_value is False:
        return 'AveragePool'
    else:
        print("unsupport pooling!")
        exit(-1)


# 构建节点
def create_pooling_node(layer, nodename, inname, outname, input_shape):
    pool_type = pooling_type(layer)
    node = None
    attributes = get_pool_attributes(layer, pool_type, input_shape)
    with_indices = True if len(outname) == 2 else False
    output_shape = get_pooling_output_shape(input_shape, layer, attributes, with_indices=with_indices)

    # 判断是池化种类,最大池化、平均池化
    if pool_type == 'GlobalMaxPool':
        node = Node.c2oNode(layer, nodename, "GlobalMaxPool", inname, outname, input_shape, output_shape, dict={})
    elif pool_type == 'MaxPool':
        node = Node.c2oNode(layer, nodename, "MaxPool", inname, outname, input_shape, output_shape, dict=attributes)
    elif pool_type == 'GlobalAveragePool':
        node = Node.c2oNode(layer, nodename, "GlobalAveragePool", inname, outname, input_shape, output_shape,
                            dict={})
    elif pool_type == 'AveragePool':
        node = Node.c2oNode(layer, nodename, "AveragePool", inname, outname, input_shape, output_shape,
                            dict=attributes)
    # Layers[i].pooling_param.pool==2为随机池化
    assert (node is not None)
    return node
