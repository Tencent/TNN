#coding:utf-8
import src.c2oObject as Node


def getOutShape(input_shape):
    # 获取output_shape
    return input_shape

# 构建节点


def createTanh(layer, nodename, inname, outname, input_shape):
    output_shape = getOutShape(input_shape)
    node = Node.c2oNode(layer, nodename, "Tanh", inname,
                        outname, input_shape, output_shape)
    return node
