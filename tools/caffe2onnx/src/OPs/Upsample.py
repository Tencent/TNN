import src.c2oObject as Node
import numpy as np


# 获取超参数
def get_upsample_attri(layer):
    # scale = layer.upsample_param.scale
    # scales = [1.0,1.0,scale,scale]
    # dict = {"scales":scales,"mode":"nearest"}#Upsample将scales放入参数里面了
    # dict = {"width_scale": scale,"height_scale":scale, "mode": "nearest"}#在OpenVINO读onnx的时候要求用width_scale和height_scale
    scale = layer.upsample_param.scale
    scales = [1.0, 1.0, scale, scale]

    attributes = {"mode": "linear",
                  'scales': scales}

    return attributes


def get_upsample_outputshape(input_shape, layer):
    scale = layer.upsample_param.scale
    scales = [1.0, 1.0, scale, scale]
    output_shape = [np.multiply(np.array(scales, dtype=np.int), np.array(input_shape[0])).tolist()]
    return output_shape


def create_upsample_node(layer, node_name, input_name, output_name, input_shape):
    attributes = get_upsample_attri(layer)
    output_shape = get_upsample_outputshape(input_shape, layer)

    # print(output_shape)
    node = Node.c2oNode(layer, node_name, "Upsample", input_name, output_name, input_shape, output_shape, attributes)
    return node
