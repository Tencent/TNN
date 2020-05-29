import src.c2oObject as Node
from typing import List
import copy


def get_concat_attributes(layer):
    ##轴
    axis = layer.concat_param.axis
    attributes = {"axis": axis}
    return attributes


# 计算输出维度
def get_concat_outshape(layer, input_shape: List) -> List:
    bottom = input_shape[0]
    axis = layer.concat_param.axis

    output_shape = copy.deepcopy(bottom)

    assert (axis < len(bottom))

    for i in range(1, len(input_shape)):
        output_shape[axis] = output_shape[axis] + input_shape[i][axis]
    return [output_shape]
    #
    # if len(bottom) == 2:
    #     n, c = bottom[0], 0
    #     for i in range(len(input_shape)):
    #         c = c + input_shape[i][1]
    #     output_shape = [[n, c]]
    #     return output_shape
    #
    # elif len(bottom) == 3:
    #     n, c = bottom[0], 0
    #     for i in range(len(input_shape)):
    #         c = c + input_shape[i][1]
    #     output_shape = [[n, c]]
    #     return output_shape
    #
    # elif len(bottom) == 4:
    #     n, c, w, h = input_shape[0][0], 0, input_shape[0][2], input_shape[0][3]
    #     for i in range(len(input_shape)):
    #         c = c + input_shape[i][1]
    #     output_shape = [[n, c, w, h]]
    #     return output_shape


# 构建节点
def createConcat(layer, nodename, inname, outname, input_shape):
    attributes = get_concat_attributes(layer)

    output_shape = get_concat_outshape(layer, input_shape)

    node = Node.c2oNode(layer, nodename, "Concat", inname, outname, input_shape, output_shape, attributes)

    return node
