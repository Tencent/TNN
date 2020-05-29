import src.c2oObject as Node
##-------------------------------------------------LRN层-------------------------------------------------------------##
#获取超参数
def getLRNAttri(layer):
    # 获取超参数
    ##尺寸
    size = layer.lrn_param.local_size
    ##alpha
    alpha = layer.lrn_param.alpha
    ##beta
    beta = layer.lrn_param.beta

    # 超参数字典
    dict = {"alpha":alpha,
            "beta":beta,
            "bias":1.0,
            "size": size}
    return dict
#计算输出维度
def getLRNOutShape(input_shape):
    # 计算输出维度output_shape
    output_shape = input_shape  # 与输入维度一样
    return output_shape
#构建节点
def createLRN(layer,nodename, inname,outname,input_shape):
    dict = getLRNAttri(layer)
    output_shape = getLRNOutShape(input_shape)

    #构建node
    node = Node.c2oNode(layer, nodename, "LRN", inname, outname, input_shape, output_shape, dict)
    return node
