import numpy as np
import src.c2oObject as Node
##-----------------------------------------------------UnPooling层--------------------------------------------------##
#获取超参数
def getUnPoolingAttri(layer):
    # ##池化核尺寸
    # kernel_shape = np.array([layer.pooling_param.kernel_size]*2).reshape(1,-1)[0].tolist()
    # if layer.pooling_param.kernel_size == []:
    #     kernel_shape = [layer.pooling_param.kernel_h,layer.pooling_param.kernel_w]
    # ##步长
    # strides = [1, 1]#默认为1
    # if layer.pooling_param.stride != []:
    #     strides = np.array([layer.pooling_param.stride]*2).reshape(1,-1)[0].tolist()
    # ##填充
    # pads = [0, 0, 0, 0]#默认为0
    # # 这里与卷积时一样,有pad,就按其值设置
    # if layer.pooling_param.pad != []:
    #     pads = np.array([layer.pooling_param.pad] * 4).reshape(1, -1)[0].tolist()
    # elif layer.pooling_param.pad_h != 0 or layer.pooling_param.pad_w != 0:
    #     pads = [layer.pooling_param.pad_h,layer.pooling_param.pad_w,layer.pooling_param.pad_h,layer.pooling_param.pad_w]

    #超参数字典
    dict = {"kernel_shape": [2, 2],
            "strides": [2, 2],
            "pads": [0, 0, 0, 0]
            }
    return dict
#计算输出维度
def getUnPoolingOutShape(input_shape,layer,dict):
    kernel_shape = dict["kernel_shape"]
    pads = dict["pads"]
    strides = dict["strides"]

    #计算输出维度,与卷积一样,若为非整数则向上取整
    # h = (input_shape[0][2] - kernel_shape[0] + 2 * pads[0])/strides[0] + 1
    # if h > int(h):
    #     output_shape_h = int(h) + 1
    #     pads = [0,0,1,1]
    # else:
    #     output_shape_h = int(h)
    # output_shape = [[input_shape[0][0],input_shape[0][1],output_shape_h,output_shape_h]]

    output_shape = [[input_shape[0][0], input_shape[0][1], input_shape[0][2]*2, input_shape[0][3]*2]]

    return output_shape
#构建节点
def createUnPooling(layer,nodename,inname,outname,input_shape):
    dict = getUnPoolingAttri(layer)
    output_shape = getUnPoolingOutShape(input_shape,layer,dict)

    node = Node.c2oNode(layer, nodename, "MaxUnpool", inname, outname, input_shape, output_shape, dict=dict)

    return node
