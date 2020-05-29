import src.c2oObject as Node

##-----------------------------BatchNormalization层 = BatchNorm + Scale-------------------------------------##
#获取超参数
def getBNAttri(layer):
    #超参数字典
    eps = layer.batch_norm_param.eps
    momentum = layer.batch_norm_param.moving_average_fraction
    dict = {"epsilon": eps,  # 滑动系数
            "momentum": momentum
            }
    return dict
#计算输出维度
def getBNOutShape(input_shape):
    output_shape = input_shape
    return output_shape
#构建节点
def createBN(layer, nodename, inname, outname, input_shape):
    dict = getBNAttri(layer)
    #计算output_shape,输出维度等于输入维度
    output_shape = getBNOutShape(input_shape)

    #构建node
    node = Node.c2oNode(layer, nodename, "BatchNormalization", inname, outname, input_shape, output_shape,dict)
    return node