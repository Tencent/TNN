import numpy as np
import src.c2oObject as Node
import math


# 获取超参数
def getConvAttri(layer, input_shape):
    # 膨胀系数 dilations
    dilations = [1, 1]
    if layer.convolution_param.dilation != []:
        dilation = layer.convolution_param.dilation[0]
        dilations = [dilation, dilation]
    ##填充pads
    pads = [0, 0, 0, 0]  # 默认为0
    if layer.convolution_param.pad != []:  # 若存在pad,则根据pad赋值
        pads = np.array([layer.convolution_param.pad] * 4).reshape(1, -1)[0].tolist()
    elif layer.convolution_param.pad_h != 0 or layer.convolution_param.pad_w != 0:  # 若存在pad_w,pad_h则根据其赋值
        pads = [layer.convolution_param.pad_h, layer.convolution_param.pad_w, layer.convolution_param.pad_h,
                layer.convolution_param.pad_w]
    ##步长strides
    strides = [1, 1]  # 默认为1
    if layer.convolution_param.stride != []:
        strides = np.array([layer.convolution_param.stride] * 2).reshape(1, -1)[0].tolist()

    elif layer.convolution_param.stride_h != 0 and layer.convolution_param.stride_w != 0:
        strides = [layer.convolution_param.stride_h, layer.convolution_param.stride_w]

    ##卷积核尺寸kernel_shape
    kernel_shape = np.array([layer.convolution_param.kernel_size] * 2).reshape(1, -1)[0].tolist()
    if layer.convolution_param.kernel_size == []:
        kernel_shape = [layer.convolution_param.kernel_h, layer.convolution_param.kernel_w]
    ##分组group
    group = 1
    if layer.type == "ConvolutionDepthwise":
        group = input_shape[0][1]
    else:
        group = layer.convolution_param.group

    # 超参数字典
    dict = {
        #"auto_pad":"NOTSET",
        "dilations": dilations,
        "group": group,
        "kernel_shape": kernel_shape,
        "pads": pads,
        "strides": strides
    }
    return dict


# 计算输出维度
def getConvOutShape(input_shape, layer, dict):
    dilations = dict["dilations"]
    kernel_shape = dict["kernel_shape"]
    pads = dict["pads"]
    strides = dict["strides"]
    ##卷积核数量kernel_num
    kernel_num = layer.convolution_param.num_output
    # reference the caffe source code
    kernel_extent_h = dilations[0] * (kernel_shape[0] - 1) + 1
    output_shape_h = math.floor((input_shape[0][2] + 2 * pads[0] - kernel_extent_h) / strides[0]) + 1

    kernel_extent_w = dilations[1] * (kernel_shape[1] - 1) + 1
    output_shape_w = math.floor((input_shape[0][3] + 2 * pads[1] - kernel_extent_w) / strides[1]) + 1

    output_shape = [[input_shape[0][0], kernel_num, output_shape_h, output_shape_w]]
    return output_shape


# 构建节点
def createConv(layer, node_name, input_name, output_name, input_shape):
    attributes = getConvAttri(layer, input_shape)
    output_shape = getConvOutShape(input_shape, layer, attributes)
    # 构建node
    node = Node.c2oNode(layer, node_name, "Conv", input_name, output_name, input_shape, output_shape, attributes)
    return node
