import numpy as np
import src.c2oObject as Node
##---------------------------------------------------ConvTranspose层-------------------------------------------------------##
#获取超参数
def getConvTransposeAttri(layer):
    ##膨胀系数dilations
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
    group = layer.convolution_param.group


    # 超参数字典
    dict = {  # "auto_pad":"NOTSET",
        "dilations": dilations,
        "group": group,
        "kernel_shape": kernel_shape,
        "pads": pads,
        "strides": strides
    }
    return dict
#计算输出维度
def getConvTransposeOutShape(input_shape, layer,dict):
    dilations = dict["dilations"]
    kernel_shape = dict["kernel_shape"]
    pads = dict["pads"]
    strides = dict["strides"]
    ##卷积核数量kernel_num
    kernel_num = layer.convolution_param.num_output

    def get_output_shape(i, k, p, s):
        return (i-1)*s + k -2*p

    h = get_output_shape(input_shape[0][2], kernel_shape[0], pads[0], strides[0])
    w = get_output_shape(input_shape[0][3], kernel_shape[1], pads[1], strides[1])

    output_shape = [[input_shape[0][0], kernel_num, h, w]]

    return output_shape
#构建节点
def createConvTranspose(layer, nodename, inname, outname, input_shape):
    dict = getConvTransposeAttri(layer)
    output_shape = getConvTransposeOutShape(input_shape, layer, dict)
    #构建node
    node = Node.c2oNode(layer, nodename, "ConvTranspose", inname, outname, input_shape, output_shape, dict)
    return node
