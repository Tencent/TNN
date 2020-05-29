import src.c2oObject as Node
##---------------------------------------------softmax层--------------------------------------------------------------##
#获取超参数
def getSoftmaxAttri(layer):
    ##轴
    axis = layer.softmax_param.axis
    #超参数字典
    dict = {"axis": axis}
    return dict
#计算输出维度
def getSoftmaxOutShape(input_shape):
    #计算输出维度output_shape
    output_shape = input_shape#与输入维度一样
    return output_shape
#构建节点
def createSoftmax(layer, nodename, inname, outname, input_shape):
    dict = getSoftmaxAttri(layer)
    output_shape = getSoftmaxOutShape(input_shape)
    #构建node
    node = Node.c2oNode(layer, nodename, "Softmax", inname, outname, input_shape, output_shape, dict)
    return node
