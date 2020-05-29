import src.c2oObject as Node


def getPReluOutShape(input_shape):
    output_shape = input_shape
    return output_shape


def createPRelu(layer, nodename, inname, outname, input_shape):
    output_shape = getPReluOutShape(input_shape)
    node = Node.c2oNode(layer, nodename, "PRelu", inname, outname, input_shape, output_shape)
    return node
