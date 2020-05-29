import src.c2oObject as Node


# 获取超参数
def getReluAttri(layer):
    attributes = {}
    if layer.relu_param.negative_slope != 0:
        attributes = {"alpha": layer.relu_param.negative_slope}
    return attributes


# 计算输出维度
def getReluOutShape(input_shape):
    # 获取output_shape
    output_shape = input_shape
    return output_shape


# 构建节点
def createRelu(layer, nodename, inname, outname, input_shape):
    attributes = getReluAttri(layer)
    output_shape = getReluOutShape(input_shape)

    if attributes == {}:
        node = Node.c2oNode(layer, nodename, "Relu", inname, outname, input_shape, output_shape)
    else:
        node = Node.c2oNode(layer, nodename, "LeakyRelu", inname, outname, input_shape, output_shape, dict=attributes)

    return node
