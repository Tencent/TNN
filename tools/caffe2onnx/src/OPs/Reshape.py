import src.c2oObject as Node
import numpy as np
from typing import *
from operator import mul
from functools import reduce

# 计算输出维度
def getReshapeOutShape(layer, input_shape: List) -> List:
    if layer.type == 'InnerProduct':
        dims = input_shape[0]
        in_prod = 1
        for i in range(1, len(dims)):
            in_prod = in_prod * dims[i]
        output_shape = [dims[0], in_prod]
        return [output_shape]

    elif layer.type == 'ShuffleChannel':
        ## change [N, C, H, W] -> [N, G, C', H, W] tensor
        group = layer.shuffle_channel_param.group
        n, g, c, h, w = input_shape[0][0], group, int(input_shape[0][1] / group), input_shape[0][2], input_shape[0][3]
        out_shape = [[n, g, c, h, w]]
        return out_shape

    elif layer.type == 'DeReshape':
        n, c, h, w = input_shape[0][0], input_shape[0][1] * input_shape[0][2], input_shape[0][3], input_shape[0][4]
        out_shape  = [[n, c, h, w]]
        return out_shape

    elif layer.type == 'Flatten':

        axis = layer.flatten_param.axis
        assert axis == 1, "Flatten: not support axis not equal 1"
        # return [[0, -1]]
        shape = input_shape[0]
        input_prod = 1
        for i in range(axis, len(shape)):
            input_prod = input_prod * shape[i]
        output_shape = [shape[0:axis] + [input_prod]]
        return output_shape

    elif layer.type == 'Scale':
        return input_shape

    elif layer.type == 'Reshape':
        shape = input_shape[0]
        re_shape = layer.reshape_param.shape.dim
        new_shape_list = []
        for j in range(len(re_shape)):
            if re_shape[j] == 0:
                # if value = 0 ; then use original
                new_shape_list.append(shape[j])
            else:
                new_shape_list.append(re_shape[j])
        if -1 in new_shape_list:
            index = new_shape_list.index(-1)
            if index == 0:
                prod = reduce(mul, new_shape_list[1:], 1)
            elif index == (len(new_shape_list) -1):
                prod = reduce(mul, new_shape_list[0:index])
            else:
                prod = reduce(mul, new_shape_list[0:index]) * reduce(mul, new_shape_list[index+1:], 1)
            new_shape_list[index] = int(reduce(mul, shape, 1) / prod)
        output_shape = [new_shape_list]
        return output_shape


def get_reshape_param(layer, input_shape: List[int]) -> List[int]:
    re_shape = layer.reshape_param.shape.dim
    return re_shape


# 构建节点
def createReshape(layer, node_name, input_name, output_name, input_shape, output_shape={}):
    # 获取output_shape
    if layer.type == 'Scale' and output_shape != {}:
        node = Node.c2oNode(layer, node_name, "Reshape", input_name, output_name, input_shape, output_shape)
        return node
    else:
        output_shape = getReshapeOutShape(layer, input_shape)

    # 构建node
    node = Node.c2oNode(layer, node_name, "Reshape", input_name, output_name, input_shape, output_shape)
    return node
