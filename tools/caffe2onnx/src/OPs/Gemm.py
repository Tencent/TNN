import src.c2oObject as Node
##-----------------------------------------------------Gemm层-------------------------------------------------------##
#获取超参数
def getGemmAttri(layer):
    #超参数字典
    dict = {"alpha": 1.0,
            "beta": 1.0,
            "transA": 0,
            "transB": 1}
    return dict
#计算输出维度
def getGemmOutShape(input_shape,num_output):
    output_shape = [[input_shape[0][0], num_output]]
    return output_shape
#构建节点
def createGemm(layer, nodename, inname, outname, input_shape, num_output):
    dict = getGemmAttri(layer)
    output_shape = getGemmOutShape(input_shape,num_output)
    #构建node
    node = Node.c2oNode(layer, nodename, "Gemm", inname, outname, input_shape, output_shape, dict)
    return node
