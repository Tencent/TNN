import src.c2oObject as Node
##-------------------------------------------------eltwise层----------------------------------------------------------##
def createEltwise(layer, nodename, inname, outname, input_shape):
    #判断算子类型
    if layer.eltwise_param.operation == 0:
        node = __createMul(layer, nodename, inname, outname, input_shape)#按元素相乘

    elif layer.eltwise_param.operation == 1:
        node = __createAdd(layer, nodename, inname, outname, input_shape)#按元素相加

    elif layer.eltwise_param.operation == 2:
        node = __createMax(layer, nodename, inname, outname, input_shape)#按元素求最大值

    return node



##----------------------------------------------Mul层,对应Prod-----------------------------------------------##
def __createMul(layer, nodename, inname, outname, input_shape):
    output_shape = input_shape[0]
    node = Node.c2oNode(layer, nodename, "Mul", inname, outname, input_shape, output_shape)
    return node

##---------------------Add层,可能是两个中间层输出相加，也可能是一个输出加一个bias这种------------------------##
def __createAdd(layer, nodename, inname, outname, input_shape):
    output_shape = [input_shape[0]]
    node = Node.c2oNode(layer, nodename, "Add", inname, outname, input_shape, output_shape)
    return node
##----------------------------------------------Max层-------------------------------------------------------------##
def __createMax(layer, nodename, inname, outname, input_shape):
    output_shape = input_shape
    node = Node.c2oNode(layer, nodename, "Max", inname, outname, input_shape, output_shape)
    return node
