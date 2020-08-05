import src.c2oObject as Node
import numpy as np


def get_InstanceNorm_param(layer, input_shape):
    scale = []
    bias = []
    for i in range(input_shape[0][1]):
        scale.append(1)
        bias.append(0)
    return scale, bias


def create_InstanceNorm_attributes(layer):
    epsilon: float = layer.mvn_param.eps
    if not epsilon:
        epsilon = 1e-05

    attributes = {"epsilon": epsilon}
    return attributes


def get_InstanceNorm_output_shape(input_shape):
    output_shape = input_shape
    return output_shape


def create_InstanceNorm_op(layer, node_name, input_name, output_name, input_shape):
    output_shape = get_InstanceNorm_output_shape(input_shape)
    attributes = create_InstanceNorm_attributes(layer)
    node = Node.c2oNode(layer, node_name, "InstanceNormalization",
                        input_name, output_name,
                        input_shape,output_shape,attributes)
    return node
