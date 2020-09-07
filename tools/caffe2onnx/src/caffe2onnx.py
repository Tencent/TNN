import src.OPs as op
from src.c2oObject import *
from onnx import helper
import copy
import numpy as np
from src.op_layer_info import *
import random
import sys
from typing import *
import onnx


class Caffe2Onnx():
    def __init__(self, net, model, onnxname):
        # 初始化一个c2oGraph对象
        self.onnxmodel = c2oGraph(onnxname)
        # 网络和参数
        self.netLayerCaffe = self.GetNetLayerCaffe(net)
        self.netModelCaffe = self.GetNetModelCaffe(model)

        # 模型的输入名和输入维度
        self.model_input_name = []
        self.model_input_shape = []

        # 节点列表
        self.onnxNodeList = []

        # 获取层列表
        LayerList = self.AddInputsTVIAndGetLayerList(net)
        self.GenerateOnnxNodeList(LayerList)
        self.AddOutputsTVIAndValueInfo()

    # 获取网络层
    def GetNetLayerCaffe(self, net):
        if len(net.layer) == 0 and len(net.layers) != 0:
            return net.layers
        elif len(net.layer) != 0 and len(net.layers) == 0:
            return net.layer
        else:
            print("prototxt layer error")
            return -1

    # 获取参数层
    def GetNetModelCaffe(self, model):
        if len(model.layer) == 0 and len(model.layers) != 0:
            return model.layers
        elif len(model.layer) != 0 and len(model.layers) == 0:
            return model.layer
        else:
            print("caffemodel layer error")
            return -1

    # 将模型输入信息添加到Inputs中并获取后续层列表
    def AddInputsTVIAndGetLayerList(self, net):
        # 如果第一个layer的类型为Input,且没有net.input存在
        if net.input == [] and self.netLayerCaffe[0].type == "Input":
            layer_list = []
            # 考虑到整个网络会有多输入情况
            for lay in self.netLayerCaffe:
                if lay.type == "Input":
                    if len(lay.top) == 1 and lay.top[0] != lay.name:
                        input_layer_name = lay.top[0]
                    else:
                        input_layer_name = lay.name

                    in_tvi = helper.make_tensor_value_info(
                        input_layer_name + "_input", TensorProto.FLOAT,
                        lay.input_param.shape[0].dim)

                    self.model_input_name.append(input_layer_name + "_input")
                    self.model_input_shape.append(lay.input_param.shape[0].dim)
                    self.onnxmodel.addInputsTVI(in_tvi)
                else:
                    layer_list.append(lay)
            return layer_list

        # 如果存在net.input
        elif net.input != []:
            in_tvi = helper.make_tensor_value_info("input", TensorProto.FLOAT,
                                                   net.input_dim)
            self.model_input_name.append("input")
            self.model_input_shape.append(net.input_dim)
            self.onnxmodel.addInputsTVI(in_tvi)
            return self.netLayerCaffe

        # 以上情况都不是,则该caffe模型没有输入,存在问题
        else:
            raise ValueError("the caffe model has no input")

    # 得到layer的参数shape
    def GetParamsShapeAndData(self, layer):
        ParamShape = []
        ParamData = []
        # 根据这个layer名找出对应的caffemodel中的参数
        for model_layer in self.netModelCaffe:
            if layer.name == model_layer.name:
                Params = copy.deepcopy(model_layer.blobs)
                ParamShape = [p.shape.dim for p in Params]
                ParamData = [p.data for p in Params]
                if layer.type == "BatchNorm" or layer.type == "BN":
                    if len(ParamShape) == 3:
                        # 如果是bn层，则不用最后一层的滑动系数
                        ParamShape = ParamShape[:-1]
                        ParamData = ParamData[:-1]
                    elif len(ParamShape) == 2 and len(ParamShape[0]) != 1:
                        ParamShape = [[ParamShape[0][1]], [ParamShape[1][1]]]
                        ParamData = ParamData
        return ParamShape, ParamData

    def get_param_shape(self, params):
        shapes = []
        for p in params:
            if p.shape.dim != []:
                shape = p.shape.dim
                shapes.append(shape)
            else:
                shape = [p.num, p.channels, p.height, p.width]
                shapes.append(shape)
        return shapes

    # 将参数添加到Inputs中,并生成tensor存储数据
    def AddInputsTVIFromParams(self, layer, ParamName, ParamType):
        ParamShape = []
        ParamData = []
        # 根据这个layer名找出对应的caffemodel中的参数
        for model_layer in self.netModelCaffe:
            if layer.name == model_layer.name:
                Params = copy.deepcopy(model_layer.blobs)
                #ParamShape = [p.shape.dim for p in Params]
                ParamShape = self.get_param_shape(Params)
                ParamData = [p.data for p in Params]
                if layer.type == "BatchNorm" or layer.type == "BN":
                    if len(ParamShape) == 3:
                        # 如果是bn层，params为[mean, var, s]，则需要把mean和var除以滑动系数s
                        ParamShape = ParamShape[:-1]
                        ParamData = [
                            [q / (Params[-1].data[0])
                             for q in p.data] if i == 0 else
                            [q / (Params[-1].data[0] + 1e-5) for q in p.data]
                            for i, p in enumerate(Params[:-1])
                        ]  # with s
                    elif len(ParamShape) == 2 and len(ParamShape[0]) == 4:
                        ParamShape = [[ParamShape[0][1]], [ParamShape[1][1]]]
                        ParamData = [[q / 1. for q in p.data] if i == 0 else
                                     [q / (1. + 1e-5) for q in p.data]
                                     for i, p in enumerate(Params)]
                if layer.type == "Reshape":
                    ParamShape = [[len(model_layer.reshape_param.shape.dim)]]
                    ParamData = [model_layer.reshape_param.shape.dim]
                if layer.type == "Convolution" or layer.type == "ConvolutionDepthwise":
                    if len(ParamShape) == 2:
                        ParamShape[1] = [ParamShape[0][0]]
                if layer.type == "InnerProduct":
                    if len(ParamShape[0]) > 2:
                        ParamShape[0] = [ParamShape[0][2], ParamShape[0][3]]
                    if len(ParamShape) == 2:
                        if len(ParamShape[1]) > 2:
                            ParamShape[1] = [ParamShape[1][2], ParamShape[1][3]]
                if layer.type == "Normalize":
                    if len(ParamShape) == 1:
                        ParamShape[0] = [1, ParamShape[0][0], 1, 1]

                # comment it for tvm because tvm use broadcast at prelu layer
                # 个人感觉如果不用 tvm，就不需要使用 Prelu
                # if layer.type == 'PReLU':
                #     ParamShape = [[ParamShape[0][0], 1, 1]]

                break
        # 判断是否有Param
        if ParamShape != []:
            ParamName = ParamName[0:len(ParamShape)]
            ParamType = ParamType[0:len(ParamShape)]
            for i in range(len(ParamShape)):
                ParamName[i] = layer.name + ParamName[i]
                p_tvi = helper.make_tensor_value_info(ParamName[i],
                                                      ParamType[i],
                                                      ParamShape[i])
                p_t = helper.make_tensor(ParamName[i], ParamType[i],
                                         ParamShape[i], ParamData[i])
                self.onnxmodel.addInputsTVI(p_tvi)
                self.onnxmodel.addInitTensor(p_t)
                # print("添加参数" + ParamName[i] + "输入信息和tensor数据")
        if layer.type == "BatchNorm" or layer.type == "BN" or layer.type == "Scale":
            return ParamName, ParamShape
        return ParamName

    # 手动将参数添加到输入信息中,并生成tensor存储数据
    def AddInputsTVIMannul(self, layer, param_names, param_types, param_shapes,
                           param_data):
        node_names = copy.deepcopy(param_names)
        for i in range(len(param_shapes)):
            node_names[i] = layer.name + param_names[i]
            p_tvi = helper.make_tensor_value_info(node_names[i],
                                                  param_types[i],
                                                  param_shapes[i])
            p_t = helper.make_tensor(node_names[i], param_types[i],
                                     param_shapes[i], param_data[i])
            self.onnxmodel.addInputsTVI(p_tvi)
            self.onnxmodel.addInitTensor(p_t)
        return node_names
        # # 由于 Slice 的 input 情况特殊，所以需要特殊处理
        # if layer.type == 'Slice':
        #     for i in range(len(ParamShape)):
        #         p_tvi = helper.make_tensor_value_info(Param_Name[i], ParamType[i], ParamShape[i])
        #         p_t = helper.make_tensor(Param_Name[i], ParamType[i], ParamShape[i], ParamData[i])
        #         self.onnxmodel.addInputsTVI(p_tvi)
        #         self.onnxmodel.addInitTensor(p_t)
        #     return Param_Name
        # else:
        #     for i in range(len(ParamShape)):
        #         Param_Name[i] = layer.name + ParamName[i]
        #         p_tvi = helper.make_tensor_value_info(Param_Name[i], ParamType[i], ParamShape[i])
        #         p_t = helper.make_tensor(Param_Name[i], ParamType[i], ParamShape[i], ParamData[i])
        #         self.onnxmodel.addInputsTVI(p_tvi)
        #         self.onnxmodel.addInitTensor(p_t)
        #     return Param_Name

    # 获取上一层的输出名(即当前层的输入)
    def GetLastLayerOutNameAndShape(self, layer):
        output_name = []
        outshape = []
        # flag is True: 模型的输入没有被覆盖
        # flag is False: 模型的输入已经被覆盖
        flag = True

        # 如果结点列表为空，或者当前层的bottom在input_name中，那么上一层输入一定是 Input
        if self.onnxNodeList == []:
            output_name += self.model_input_name
            outshape += self.model_input_shape

        else:
            for i in range(len(layer.bottom)):

                # 因为prototxt中存在top和bottom同名的情况，但是layer.bottom只能对应一个node，所以对每个layer.bottom，找到最末的那个同名节点作为上一层节点
                name = None
                shape = None
                for node in self.onnxNodeList:
                    for j in range(len(node.top) if node.node.op_type != "MaxPool" else 1):
                        if layer.bottom[i] == node.top[j]:
                            name = node.outputs_name[j]
                            shape = node.outputs_shape[j]
                        for k in range(len(node.bottom)):
                            if node.top[j] == node.bottom[k]:
                                for w in range(len(self.model_input_name)):
                                    if node.top[j] + '_input' == self.model_input_name[w]:
                                        flag = False

                for j in range(len(self.model_input_name)):
                    if layer.bottom[i] + '_input' == self.model_input_name[j] and flag:
                        output_name.append(self.model_input_name[j])
                        outshape.append(self.model_input_shape[j])

                if name:
                    output_name.append(name)
                    outshape.append(shape)

        try:
            assert output_name, "Failed at layer %s, layer's bottom not detected ..." % (layer.name)
        except:
            print("Failed at layer %s, layer's bottom not detected ..." % (layer.name))
            exit(-1)
        return output_name, outshape

    # 获取当前层的输出名，即layername
    def GetCurrentLayerOutName(self, layer):
        # return [layer.name]
        # 考虑有多个输出的情况
        # # TODO: 为什么要使用 layer.name 进行替代呢?
        if layer.top == layer.bottom and len(layer.top) == 1:
            return [layer.name]
        return [out for out in layer.top]


    def GenerateOnnxNodeList(self, Layers):
        for i in range(len(Layers)):
            print("convert layer: " + Layers[i].name)
            # Convolution
            if Layers[i].type == "Convolution" or Layers[i].  type == Layer_CONVOLUTION:
                # 1.获取节点输入名、输入维度、输出名、节点名
                input_name, input_shape = self.GetLastLayerOutNameAndShape(Layers[i])
                output_name = self.GetCurrentLayerOutName(Layers[i])
                node_name = Layers[i].name

                # 2.生成节点参数tensor value info,并获取节点参数名,将参数名加入节点输入名列表
                conv_pname = self.AddInputsTVIFromParams(Layers[i], op_pname["Conv"], op_ptype["Conv"])
                input_name.extend(conv_pname)

                # 3.构建conv_node
                conv_node = op.createConv(Layers[i], node_name, input_name, output_name, input_shape)

                # 4.添加节点到节点列表
                self.onnxNodeList.append(conv_node)

            elif Layers[i].type == "ConvolutionDepthwise" or Layers[i].type == Layer_CONVOLUTION:
                # 1.获取节点输入名、输入维度、输出名、节点名
                input_name, input_shape = self.GetLastLayerOutNameAndShape(Layers[i])
                output_name = self.GetCurrentLayerOutName(Layers[i])
                node_name = Layers[i].name

                # 2.生成节点参数tensor value info,并获取节点参数名,将参数名加入节点输入名列表
                conv_pname = self.AddInputsTVIFromParams(Layers[i], op_pname["Conv"], op_ptype["Conv"])
                input_name.extend(conv_pname)

                # 3.构建conv_node
                conv_node = op.createConv(Layers[i], node_name, input_name, output_name, input_shape)

                # 4.添加节点到节点列表
                self.onnxNodeList.append(conv_node)

            # BatchNorm+Scale
            elif Layers[i].type == "BatchNorm" or Layers[i].type == "BN":
                # 1.获取节点输入名、输入维度、输出名、节点名
                input_name, input_shape = self.GetLastLayerOutNameAndShape(Layers[i])  # 获取输入名列表和输入形状
                output_name = self.GetCurrentLayerOutName(Layers[i])  # 获取输出名列表
                node_name = Layers[i].name

                # 2.生成节点参数tensor value info,并获取节点参数名,将参数名加入节点输入名列表
                if i < len(Layers) - 1 and Layers[i + 1].type == "Scale":
                    scale_pname, scale_pshape = self.AddInputsTVIFromParams(Layers[i + 1], op_pname["Scale"],
                                                                            op_ptype["Scale"])
                    bn_pname, bn_pshape = self.AddInputsTVIFromParams(Layers[i], op_pname["BatchNorm"],
                                                                      op_ptype["BatchNorm"])
                    assert bn_pshape == scale_pshape, "BatchNorm and Scale params should share the same shape"
                    input_name.extend(scale_pname)
                    input_name.extend(bn_pname)
                else:
                    bn_pshape, _ = self.GetParamsShapeAndData(Layers[i])
                    custom_params = [np.ones(shape=bn_pshape[0], dtype=np.float),
                                     0.001 + np.zeros(shape=bn_pshape[1], dtype=np.float)]
                    scale_pname = self.AddInputsTVIMannul(Layers[i], op_pname["Scale"], op_ptype["Scale"], bn_pshape,
                                                          custom_params)
                    bn_pname, bn_pshape = self.AddInputsTVIFromParams(Layers[i], op_pname["BatchNorm"],
                                                                      op_ptype["BatchNorm"])
                    input_name.extend(scale_pname)
                    input_name.extend(bn_pname)

                # 3.构建bn_node
                bn_node = op.createBN(Layers[i], node_name, input_name, output_name, input_shape)

                # 4.添加节点到节点列表
                self.onnxNodeList.append(bn_node)

            elif Layers[i].type == "Scale":
                if i > 0 and (Layers[i - 1].type == "BatchNorm" or Layers[i - 1].type == "BN"):
                    # bn + scale
                    continue
                # signal scale
                input_name, input_shape = self.GetLastLayerOutNameAndShape(Layers[i])  # 获取输入名列表和输入形状
                output_name = self.GetCurrentLayerOutName(Layers[i])  # 获取输出名列表
                # node_name = Layers[i].name + random.choice('1234567890abcdefghijklmnopqrst')
                node_name = Layers[i].name
                has_two_input: bool = False
                if len(input_name) > 1:
                    has_two_input = True

                if has_two_input and op.need_add_reshape(input_shape):
                    reshape_layer = copy.deepcopy(Layers[i])
                    # add reshape layer
                    reshape_node_name =  input_name[1] + '_reshap_' + random.choice('1234567890abcdefghijklmnopqrst')

                    reshape_input_name = input_name[1]
                    reshape_input_shape = input_shape[1]

                    reshape_shape_data = op.get_param_shape(input_shape)
                    reshape_shape_shape = np.shape(reshape_shape_data)

                    reshape_params = self.AddInputsTVIMannul(Layers[i], [reshape_node_name + 'shape'], [TensorProto.INT64],
                                                             [reshape_shape_shape], [reshape_shape_data])

                    reshape_output_name = [reshape_input_name + '_output_name']


                    reshape_node = op.createReshape(reshape_layer, reshape_node_name, [reshape_input_name, reshape_params[0]],
                                                    reshape_output_name, reshape_input_shape,  output_shape=[reshape_shape_data])

                    self.onnxNodeList.append(reshape_node)

                    # add mul node
                    input_name[1] = reshape_output_name[0]
                    input_shape[1] = reshape_shape_data
                    mul_node = op.create_mul_node(Layers[i], node_name, input_name, output_name, input_shape)

                    self.onnxNodeList.append(mul_node)
                else:
                    param_shape, param_data = self.GetParamsShapeAndData(Layers[i])
                    # Scale = Mul + Add
                    if len(param_shape) == 2:
                        # create mul
                        param_scale_shape = [1, param_shape[0][0], 1, 1]
                        param_scale_data = param_data[0]
                        param_scale_name = self.AddInputsTVIMannul(Layers[i], ["_scale"], [TensorProto.FLOAT], [param_scale_shape], [param_scale_data])

                        mul_node_name = node_name + "_mul"
                        mul_input_name = [input_name[0], param_scale_name[0]]
                        mul_output_name = [output_name[0] + "_mul"]
                        mul_input_shape = [input_shape[0], param_scale_shape]

                        mul_node = op.create_mul_node(Layers[i], mul_node_name, mul_input_name, mul_output_name, mul_input_shape)
                        self.onnxNodeList.append(mul_node)

                        param_bias_shape = [1, param_shape[1][0], 1, 1]
                        param_bias_data = param_data[1]
                        param_bias_name = self.AddInputsTVIMannul(Layers[i], ["_bias"], [TensorProto.FLOAT], [param_bias_shape], [param_bias_data])

                        add_node_name = node_name + "_add"
                        add_input_name = [mul_output_name[0], param_bias_name[0]]
                        add_output_name = output_name
                        add_input_shape = [input_shape[0], param_bias_shape]
                        add_node = op.create_add_node(Layers[i], add_node_name, add_input_name, add_output_name, add_input_shape)
                        self.onnxNodeList.append(add_node)
                    # Scale = Mul
                    if len(param_shape) == 1:
                        # create mul
                        param_scale_shape = [1, param_shape[0][0], 1, 1]
                        param_scale_data = param_data[0]
                        param_scale_name = self.AddInputsTVIMannul(
                            Layers[i], ["_scale"], [TensorProto.FLOAT],
                            [param_scale_shape], [param_scale_data])

                        mul_input_name = [input_name[0], param_scale_name[0]] 
                        mul_input_shape = [input_shape[0], param_scale_shape]

                        mul_node = op.create_mul_node(Layers[i], node_name,
                                                      mul_input_name,
                                                      output_name,
                                                      mul_input_shape)
                        self.onnxNodeList.append(mul_node)

            # Pooling
            elif Layers[i].type == "Pooling" or Layers[i].type == Layer_POOLING:
                # TODO:
                # Pooling <= Pad + Pool
                # NOTE： 由于 Caffe 和 ONNX 对 AveragePool 的处理的方式的不同，所以需要在pool node 之前添加 Pad node
                # 1.获取节点输入名、输入维度、输出名、节点名
                input_name, input_shape = self.GetLastLayerOutNameAndShape(Layers[i])  # 获取输入名列表和输入形状
                output_name = self.GetCurrentLayerOutName(Layers[i])  # 获取输出名列表
                node_name = Layers[i].name
                # create pad node
                pads = op.get_pool_pads(Layers[i])
                pads_shape = [np.shape(pads)]
                pads_name = node_name + "_output"
                pads_output_name = [node_name + "_output"]
                pad_output_shape = op.calculate_pad_output_shape(input_shape, pads)
                pads_param = self.AddInputsTVIMannul(Layers[i], ["_pad"], [TensorProto.INT64], pads_shape, [pads])
                input_name.extend(pads_param)

                pool_type = op.pooling_type(Layers[i])
                if pool_type == "GlobalMaxPool" or pool_type == "MaxPool":
                    constant_value = [-sys.float_info.max]
                    constant_shape = [np.shape(constant_value)]

                    constant_value_param = self.AddInputsTVIMannul(Layers[i], ["_constant_value"], [TensorProto.FLOAT],
                                                                   constant_shape, [constant_value])
                    input_name.extend(constant_value_param)

                pad_node = op.create_pad_node(Layers[i], pads_name, input_name, pads_output_name, input_shape)
                self.onnxNodeList.append(pad_node)

                # 2.构建pool_node
                pool_node = op.create_pooling_node(Layers[i], node_name, pads_output_name, output_name,
                                                   pad_output_shape)

                # 3.添加节点到节点列表
                self.onnxNodeList.append(pool_node)


            # MaxUnPool
            elif Layers[i].type == "MaxUnpool":
                # 1.获取节点输入名、输入维度、输出名、节点名
                input_name, input_shape = self.GetLastLayerOutNameAndShape(Layers[i])  # 获取输入名列表和输入形状
                output_name = self.GetCurrentLayerOutName(Layers[i])  # 获取输出名列表
                node_name = Layers[i].name

                # 2.构建unpool_node
                unpool_node = op.createUnPooling(Layers[i], node_name, input_name, output_name, input_shape)

                # 3.添加节点到节点列表
                self.onnxNodeList.append(unpool_node)


            # Eltwise
            elif Layers[i].type == "Eltwise" or Layers[i].type == Layer_ELTWISE:
                # 1.获取节点输入名、输入维度、输出名、节点名
                output_name = self.GetCurrentLayerOutName(Layers[i])  # 获取输出名列表
                input_name, input_shape = self.GetLastLayerOutNameAndShape(Layers[i])  # 获取输入名列表和输入形状

                node_name = Layers[i].name

                # 2.构建eltwise_node
                eltwise_node = op.createEltwise(Layers[i], node_name, input_name, output_name, input_shape)

                # 3.添加节点到节点列表
                self.onnxNodeList.append(eltwise_node)


            # Softmax
            elif Layers[i].type == "Softmax" or Layers[i].type == Layer_SOFTMAX:
                # 1.获取节点输入名、输入维度、输出名、节点名
                input_name, input_shape = self.GetLastLayerOutNameAndShape(Layers[i])  # 获取输入名列表和输入形状
                output_name = self.GetCurrentLayerOutName(Layers[i])  # 获取输出名列表
                node_name = Layers[i].name

                # 2.构建softmax_node
                softmax_node = op.createSoftmax(Layers[i], node_name, input_name, output_name, input_shape)

                # 3.添加节点到节点列表
                self.onnxNodeList.append(softmax_node)


            # Relu
            elif Layers[i].type == "ReLU" or Layers[i].type == Layer_RELU:
                # 1.获取节点输入名、输入维度、输出名、节点名
                input_name, input_shape = self.GetLastLayerOutNameAndShape(Layers[i])  # 获取输入名列表和输入形状
                output_name = self.GetCurrentLayerOutName(Layers[i])  # 获取输出名列表
                node_name = Layers[i].name
                # letters = '1234567890abcdefghijklmnopqrst'
                # length = random.randrange(5, 16)
                # randstr = ''.join(random.choice(letters) for _ in range(length))
                # node_name = node_name
                # for i in range(len(output_name)):
                #     output_name[i] = output_name[i] + random.choice('1234567890abcdef')
                #print(output_name)


                # 2.构建relu_node
                relu_node = op.createRelu(Layers[i], node_name, input_name, output_name, input_shape)

                # 3.添加节点到节点列表
                self.onnxNodeList.append(relu_node)
            # PRelu
            elif Layers[i].type == "PReLU":
                # 1.获取节点输入名、输入维度、输出名、节点名
                input_name, input_shape = self.GetLastLayerOutNameAndShape(Layers[i])
                output_name = self.GetCurrentLayerOutName(Layers[i])
                node_name = Layers[i].name

                # 2.生成节点参数tensor value info,并获取节点参数名,将参数名加入节点输入名列表
                pname = self.AddInputsTVIFromParams(Layers[i], op_pname["PRelu"], op_ptype["PRelu"])
                input_name.extend(pname)

                # 3.构建PRelu_node
                PRelu_node = op.createPRelu(Layers[i], node_name, input_name, output_name, input_shape)

                # 4.添加节点到节点列表
                self.onnxNodeList.append(PRelu_node)
            # relu6
            elif Layers[i].type == 'ReLU6':
                # relu6 = clip(0, 6)
                # add relu node
                input_name, input_shape = self.GetLastLayerOutNameAndShape(Layers[i])
                output_name = self.GetCurrentLayerOutName(Layers[i])
                node_name = Layers[i].name

                min_value = np.float(0)
                max_value = np.float(6)
                shape = np.shape([min_value])
                min_param = self.AddInputsTVIMannul(Layers[i], ["_min"],
                                                    [TensorProto.FLOAT], [shape],
                                                    [[min_value]])
                input_name.extend(min_param)
                max_param = self.AddInputsTVIMannul(Layers[i], ['_max'],
                                                    [TensorProto.FLOAT], [shape],
                                                    [[max_value]])
                input_name.extend(max_param)
                relu6_node = op.create_clip_node(Layers[i], node_name, input_name, output_name, input_shape)

                self.onnxNodeList.append(relu6_node)

            elif Layers[i].type == "Sigmoid":
                # 1.获取节点输入名、输入维度、输出名、节点名
                input_name, input_shape = self.GetLastLayerOutNameAndShape(Layers[i])  # 获取输入名列表和输入形状
                output_name = self.GetCurrentLayerOutName(Layers[i])  # 获取输出名列表
                node_name = Layers[i].name

                # 2.构建relu_node
                sigmoid_node = op.createSigmoid(Layers[i], node_name, input_name, output_name, input_shape)

                # 3.添加节点到节点列表
                self.onnxNodeList.append(sigmoid_node)
            elif Layers[i].type == 'Log':
                input_name, input_shape = self.GetLastLayerOutNameAndShape(Layers[i])  # 获取输入名列表和输入形状
                output_name = self.GetCurrentLayerOutName(Layers[i])  # 获取输出名列表
                node_name = Layers[i].name

                log_node = op.create_log_node(Layers[i], node_name, input_name, output_name, input_shape)

                self.onnxNodeList.append(log_node)
            # LRN
            elif Layers[i].type == "LRN" or Layers[i].type == Layer_LRN:
                # 1.获取节点输入名、输入维度、输出名、节点名
                input_name, input_shape = self.GetLastLayerOutNameAndShape(Layers[i])
                output_name = self.GetCurrentLayerOutName(Layers[i])
                node_name = Layers[i].name

                # 2.构建LRN_node
                LRN_node = op.createLRN(Layers[i], node_name, input_name, output_name, input_shape)

                # 3.添加节点到节点列表
                self.onnxNodeList.append(LRN_node)


            # Dropout
            elif Layers[i].type == "Dropout" or Layers[i].type == Layer_DROPOUT:
                # 1.获取节点输入名、输入维度、输出名、节点名
                input_name, input_shape = self.GetLastLayerOutNameAndShape(Layers[i])
                output_name = self.GetCurrentLayerOutName(Layers[i])
                node_name = Layers[i].name

                # 2.构建Dropout_node
                Dropout_node = op.createDropout(Layers[i], node_name, input_name, output_name, input_shape)

                # 3.添加节点到节点列表
                self.onnxNodeList.append(Dropout_node)


            # Upsample
            elif Layers[i].type == "Upsample" or Layers[i].type == Layer_UPSAMPLE:
                # 1.获取节点输入名、输入维度、输出名、节点名
                input_name, input_shape = self.GetLastLayerOutNameAndShape(Layers[i])
                output_name = self.GetCurrentLayerOutName(Layers[i])
                node_name = Layers[i].name

                # 2.生成节点参数tensor value info,并获取节点参数名,将参数名加入节点输入名列表
                # add roi input

                # add scales input
                paramshape = [[8, 1],
                              [4, 1]]
                paramdata = [[1, 1, 1, 1, 2, 2, 2, 2],
                             [1.0, 1.0, Layers[i].upsample_param.scale, Layers[i].upsample_param.scale]]

                pname = self.AddInputsTVIMannul(Layers[i], op_pname["Upsample"], op_ptype["Upsample"], paramshape,
                                               paramdata)

                input_name.extend(pname)

                # 3.构建Upsample_node
                Upsample_node = op.create_resize_node(Layers[i], node_name, input_name, output_name, input_shape)

                # 4.添加节点到节点列表
                self.onnxNodeList.append(Upsample_node)

            elif Layers[i].type == 'Interp':
                input_name, input_shape = self.GetLastLayerOutNameAndShape(Layers[i])
                output_name = self.GetCurrentLayerOutName(Layers[i])
                node_name = Layers[i].name

                interp_node = op.create_interp_node(Layers[i], node_name, input_name, output_name, input_shape)

                self.onnxNodeList.append(interp_node)

            # Concat
            elif Layers[i].type == "Concat" or Layers[i].type == Layer_CONCAT:
                # 1.获取节点输入名、输入维度、输出名、节点名
                input_name, input_shape = self.GetLastLayerOutNameAndShape(Layers[i])
                output_name = self.GetCurrentLayerOutName(Layers[i])
                node_name = Layers[i].name

                # 2.构建Concat_node
                Concat_node = op.createConcat(Layers[i], node_name, input_name, output_name, input_shape)

                # 3.添加节点到节点列表
                self.onnxNodeList.append(Concat_node)

            elif Layers[i].type == 'Slice':
                # 1. 获取节点书输入名，输入维度，输出名，节点名
                input_name, input_shape = self.GetLastLayerOutNameAndShape(Layers[i])
                output_name_list = self.GetCurrentLayerOutName(Layers[i])
                node_name = Layers[i].name

                starts, ends, axes = op.analyzeLayer(Layers[i], input_shape)

                SliceLayer = copy.deepcopy(Layers[i])

                for i in range(len(output_name_list)):
                    # 放在这里的原因是
                    slice_name = copy.deepcopy(input_name)
                    # starts ends axes 的 shape 是相同的
                    shape = [np.shape([1])]

                    starts_param = self.AddInputsTVIMannul(SliceLayer, ['_starts' + str(i)],
                                                           [TensorProto.INT64], shape,
                                                           [[starts[i]]])
                    ends_param = self.AddInputsTVIMannul(SliceLayer, ['_ends' + str(i)],
                                                         [TensorProto.INT64], shape,
                                                         [[ends[i]]])
                    axes_param = self.AddInputsTVIMannul(SliceLayer, ['_axes' + str(i)],
                                                         [TensorProto.INT64], shape,
                                                         [[axes[i]]])
                    slice_name.extend(starts_param)
                    slice_name.extend(ends_param)
                    slice_name.extend(axes_param)

                    Slice_node = op.createSlice(SliceLayer, output_name_list[i], slice_name, [output_name_list[i]],
                                                input_shape, starts[i], ends[i])
                    # 3. 添加节点到节点列表
                    self.onnxNodeList.append(Slice_node)
            # Reshape
            elif Layers[i].type == "Reshape":
                # 1.获取节点输入名、输入维度、输出名、节点名
                input_name, input_shape = self.GetLastLayerOutNameAndShape(Layers[i])
                output_name = self.GetCurrentLayerOutName(Layers[i])
                node_name = Layers[i].name

                # 2.生成节点参数tensor value info,并获取节点参数名,将参数名加入节点输入名列表
                reshape_param = op.get_reshape_param(Layers[i], input_shape)
                reshape_param_shape = [np.shape(reshape_param)]
                pname = self.AddInputsTVIMannul(Layers[i], op_pname["Reshape"], op_ptype["Reshape"], reshape_param_shape,
                                                [reshape_param])
                input_name.extend(pname)

                # 3.构建reshape节点
                reshape_node = op.createReshape(Layers[i], node_name, input_name, output_name, input_shape)

                # 4.添加点到节点列表
                self.onnxNodeList.append(reshape_node)

            # InnerProduct
            # 由于onnx中没有全连接层，因此需要拆分，拆分有两种方法(Reshape+Gemm,Reshape+MatMul+Add)
            elif Layers[i].type == "InnerProduct" or Layers[i].type == Layer_INNER_PRODUCT:
                node_layer = copy.deepcopy(Layers[i])  # 深拷贝
                node_input_name, node_input_shape = self.GetLastLayerOutNameAndShape(node_layer)  # 获取输入名列表和输入形状

                reshape_outname = ""
                reshape_output_shape = op.getReshapeOutShape(Layers[i], node_input_shape)
                need_reshape = 0 if reshape_output_shape[0] == node_input_shape[0] else 1

                if need_reshape:
                    ####一、reshape
                    # 1.获取节点输入名、输入维度、输出名、节点名
                    reshape_outname = [node_layer.name + "_Reshape"]
                    reshape_nodename = node_layer.name + "_Reshape"

                    # 2.生成节点参数tensor value info,并获取节点参数名, 将参数名加入节点输入名列表
                    paramshape = [[2]]
                    reshape_pname = self.AddInputsTVIMannul(node_layer, op_pname["Reshape"], op_ptype["Reshape"],
                                                            paramshape, reshape_output_shape)
                    node_input_name.extend(reshape_pname)
                    # 3.构建reshape_node
                    reshape_node = op.createReshape(node_layer, reshape_nodename, node_input_name, reshape_outname,
                                                    node_input_shape)

                    # 4.添加节点到节点列表
                    self.onnxNodeList.append(reshape_node)

                # import ipdb; ipdb.set_trace()

                ####二、Gemm 最后一个node输出保持原名称
                gemm_layer = copy.deepcopy(Layers[i])  # 深拷贝
                # 1.获取节点输入名、输入维度、输出名、节点名
                gemm_inname = reshape_outname if need_reshape == 1 else node_input_name
                gemm_input_shape = reshape_output_shape if need_reshape == 1 else node_input_shape
                gemm_outname = [gemm_layer.name]
                gemm_nodename = gemm_layer.name

                # 2.生成节点参数tensor value info,并获取节点参数名,将参数名加入节点输入名列表
                gemm_pname = self.AddInputsTVIFromParams(gemm_layer, op_pname["InnerProduct"], op_ptype[
                    "InnerProduct"])  # 获取输入参数，对于add来说blobs[1]里存放的是bias不需要,所以直接获取blobs[0]
                gemm_inname.extend(gemm_pname)

                # 3.构建gemm_node
                matmul_node = op.createGemm(gemm_layer, gemm_nodename, gemm_inname, gemm_outname, gemm_input_shape,
                                            gemm_layer.inner_product_param.num_output)

                # 4.添加节点到节点列表
                self.onnxNodeList.append(matmul_node)

            elif Layers[i].type == 'ShuffleChannel':
                # TODO support ShuffleChannel
                # reshape  [N, C, H, W] tensor to [N, G, C', H, W]
                node_layer = copy.deepcopy(Layers[i])  # 深拷贝
                node_input_name, node_input_shape = self.GetLastLayerOutNameAndShape(node_layer)  # 获取输入名列表和输入形状

                reshape_outname = ""
                reshape_output_shape = op.getReshapeOutShape(Layers[i], node_input_shape)
                need_reshape = 0 if reshape_output_shape[0] == node_input_shape[0] else 1

                if need_reshape:
                    # 一. reshape  [N, C, H, W] tensor to [N, G, C', H, W]
                    # 1.获取节点输入名、输入维度、输出名、节点名
                    reshape_outname = [node_layer.name + "_Reshape"]
                    reshape_nodename = node_layer.name + "_Reshape"

                    # 2.生成节点参数tensor value info,并获取节点参数名,将参数名加入节点输入名列表
                    param_data = op.getReshapeOutShape(node_layer, node_input_shape)
                    param_shape = np.array([1, 2, 3, 4, 5], np.int).shape
                    reshape_pname = self.AddInputsTVIMannul(node_layer, op_pname["Reshape"], op_ptype["Reshape"],
                                                            [param_shape], param_data)

                    node_input_name.extend(reshape_pname)
                    # 这里不用对输入进行拓展，因为输入没有增加
                    # node_input_name.extend(reshape_pname)
                    # 3.构建reshape_node
                    reshape_node = op.createReshape(node_layer,
                                                    reshape_nodename,
                                                    node_input_name,
                                                    reshape_outname,
                                                    node_input_shape)

                    # 4.添加节点到节点列表
                    self.onnxNodeList.append(reshape_node)

                # 2. transpose  [N, C', G, H, W]
                transpose_layer = copy.deepcopy(Layers[i])  # 深拷贝
                # 1.获取节点输入名、输入维度、输出名、节点名
                transpose_input_name = reshape_outname if need_reshape == 1 else node_input_name
                transpose_input_shape = reshape_output_shape if need_reshape == 1 else node_input_shape
                transpose_output_name = [node_layer.name + "_Transpose"]
                transpose_node_name = node_layer.name + "_Transpose"

                # 2.生成节点参数tensor value info,并获取节点参数名,将参数名加入节点输入名列表
                # 获取输入参数，对于add来说blobs[1]里存放的是bias不需要,所以直接获取blobs[0]

                # TODO 这地方为什么要选择使用AddInputsTVIMannul？取决于什么？
                # ANSWER: 取决于要转换的 onnx 的类型
                # TODO param_date 是什么？为什么要设置这个变量
                param_data = [[2]]
                # transpose_pname = self.AddInputsTVIMannul(transpose_layer,
                #                                      op_pname["Transpose"],
                #                                      op_ptype['Transpose'],
                #                                      param_data,
                #                                      transpose_input_shape)
                # transpose_input_name.extend(transpose_pname)
                # 3.
                transpose_node = op.createTranspose(transpose_layer,
                                                    transpose_node_name,
                                                    transpose_input_name,
                                                    transpose_output_name,
                                                    transpose_input_shape)
                # 4.添加节点到节点列表
                self.onnxNodeList.append(transpose_node)

                # 三、 Reshape [N, C', G, H, W] tensor to [N, C, H, W]
                #
                end_layer = copy.deepcopy(Layers[i])
                end_layer.type = "DeReshape"
                # 最后的输出的节点要保持原名称，这是为了生成该节点，保持链路畅通
                end_output_name = [end_layer.name]
                end_node_name = end_layer.name

                # 上一层的输出是这一层的输入
                end_input_name = transpose_node.outputs_name
                end_input_shape = transpose_node.outputs_shape
                # 最后保持输出和输入的形状是一致的
                end_output_shape = [[node_input_shape[0][0], -1, node_input_shape[0][2], node_input_shape[0][3]]]
                param_shape = [np.array([1, 2, 3, 4], dtype=np.int).shape]
                end_pname = self.AddInputsTVIMannul(node_layer, op_pname["DouReshape"], op_ptype["DouReshape"],
                                                    param_shape, end_output_shape)

                end_input_name.extend(end_pname)
                # 构建
                end_node = op.createReshape(end_layer,
                                            end_node_name,
                                            end_input_name,
                                            end_output_name,
                                            end_input_shape)

                self.onnxNodeList.append(end_node)

            # Deconvolution
            elif Layers[i].type == "Deconvolution":
                # 1.获取节点输入名、输入维度、输出名、节点名
                input_name, input_shape = self.GetLastLayerOutNameAndShape(Layers[i])
                output_name = self.GetCurrentLayerOutName(Layers[i])
                node_name = Layers[i].name

                # 2.生成节点参数tensor value info,并获取节点参数名,将参数名加入节点输入名列表

                conv_pname = self.AddInputsTVIFromParams(Layers[i], op_pname["ConvTranspose"],
                                                         op_ptype["ConvTranspose"])
                input_name.extend(conv_pname)

                # 3.构建conv_node
                conv_node = op.createConvTranspose(Layers[i], node_name, input_name, output_name, input_shape)
                # if True:
                #     self.__print_debug_info(node_name, input_name, output_name, input_shape, conv_node.outputs_shape)

                # 4.添加节点到节点列表
                self.onnxNodeList.append(conv_node)

            # Flatten
            elif Layers[i].type == "Flatten":
                # 1.获取节点输入名、输入维度、输出名、节点名
                input_name, input_shape = self.GetLastLayerOutNameAndShape(Layers[i])
                output_name = self.GetCurrentLayerOutName(Layers[i])
                node_name = Layers[i].name

                # 由于后面 Flatten 的优化有问题,所以目前先将 Flatten -> reshape
                # flatten_node = op.create_flatten_node(layers[i], node_name, input_name,
                #                                       output_name, input_shape)
                # self.onnxnodelist.append(flatten_nodelatten_node)
                # continue

                # Flatten -> Reshape
                # import ipdb; ipdb.set_trace()
                # # 2.生成节点参数tensor value info,并获取节点参数名,将参数名加入节点输入名列表
                paramshape = [[2]]
                paramdata = op.getReshapeOutShape(Layers[i], input_shape)
                reshape_pname = self.AddInputsTVIMannul(Layers[i], op_pname["Reshape"], op_ptype["Reshape"], paramshape,
                                                        paramdata)
                input_name.extend(reshape_pname)

                # 3.构建reshape_node
                reshape_node = op.createReshape(Layers[i], node_name, input_name, output_name, input_shape)
                # 4.添加节点到节点列表
                self.onnxNodeList.append(reshape_node)

            elif Layers[i].type == "Permute":
                # Permute -> Transpose
                input_name, input_shape = self.GetLastLayerOutNameAndShape(Layers[i])
                output_name = self.GetCurrentLayerOutName(Layers[i])
                node_name = Layers[i].name

                transpose_node = op.createTranspose(Layers[i], node_name, input_name, output_name, input_shape)
                self.onnxNodeList.append(transpose_node)
            elif Layers[i].type == "PriorBox":
                input_name, input_shape = self.GetLastLayerOutNameAndShape(Layers[i])
                output_name = self.GetCurrentLayerOutName(Layers[i])
                node_name = Layers[i].name

                priorbox_node = op.create_priorbox_node(Layers[i], node_name, input_name, output_name, input_shape)

                self.onnxNodeList.append(priorbox_node)

            elif Layers[i].type == "DetectionOutput":
                input_name, input_shape = self.GetLastLayerOutNameAndShape(Layers[i])
                output_name = self.GetCurrentLayerOutName(Layers[i])
                node_name = Layers[i].name

                detection_output_node = op.create_detection_output(Layers[i], node_name, input_name, output_name, input_shape)
                self.onnxNodeList.append(detection_output_node)
            elif Layers[i].type == "Axpy":
                # axpy = mul + add
                # top = bottom[0] * bottom[1] + bottom[2]
                input_name, input_shape = self.GetLastLayerOutNameAndShape(Layers[i])
                output_name = self.GetCurrentLayerOutName(Layers[i])
                node_name = Layers[i].name
                # create mul node
                mul_node = op.create_axpy_mul_node(Layers[i], node_name, input_name, output_name, input_shape)
                self.onnxNodeList.append(mul_node)

                # create add node
                add_node = op.create_axpy_add_node(Layers[i], node_name, input_name, output_name, input_shape)
                self.onnxNodeList.append(add_node)
            elif Layers[i].type == "Normalize":
                input_name, input_shape = self.GetLastLayerOutNameAndShape(Layers[i])
                output_name = self.GetCurrentLayerOutName(Layers[i])
                node_name = Layers[i].name
                lp_normalization_output_name = [output_name[0] + "_lp"]
                lp_normalization_node = op.create_Lp_Normalization(Layers[i], node_name, input_name,
                                                                   lp_normalization_output_name, input_shape)
                self.onnxNodeList.append(lp_normalization_node)
                # get Normalize
                scale_shape, scale_data = self.GetParamsShapeAndData(Layers[i])
                scale_shape = [1, scale_shape[0][0], 1, 1]
                scale_input = self.AddInputsTVIFromParams(Layers[i], ["_scale"], [TensorProto.FLOAT])
                mul_input_name = [lp_normalization_output_name[0], node_name + "_scale"]
                mul_input_shape = [input_shape[0], scale_shape]
                mul_node = op.create_mul_node(Layers[i], node_name + "_mul", mul_input_name, output_name,
                                              mul_input_shape)
                self.onnxNodeList.append(mul_node)
            elif Layers[i].type == "Power":
                # Power: Mul + Add + Pow
                # create mul node
                input_name, input_shape = self.GetLastLayerOutNameAndShape(Layers[i])
                output_name = self.GetCurrentLayerOutName(Layers[i])
                node_name = Layers[i].name
                power, scale, shift = op.get_power_param(Layers[i])
                scale_node_name = self.AddInputsTVIMannul(Layers[i], ["_scale"], [TensorProto.FLOAT], [np.shape(scale)], [scale])
                mul_input_name = [input_name[0], scale_node_name[0]]
                mul_node = op.create_mul_node(Layers[i], node_name + "_mul", mul_input_name, [output_name[0] + "_mul"],
                                              [input_shape[0], np.shape(power)])
                self.onnxNodeList.append(mul_node)
                # create Add node
                shift_param_name = self.AddInputsTVIMannul(Layers[i], ["_shift"], [TensorProto.FLOAT], [np.shape(scale)],
                                                        [shift])
                add_input_name = [output_name[0] + "_mul", shift_param_name[0]]
                add_node = op.create_add_node(Layers[i], node_name + "_add", add_input_name, [output_name[0] + "_add"], [input_shape[0], np.shape(shift)])
                self.onnxNodeList.append(add_node)

                # create Pow
                power_param_name = self.AddInputsTVIMannul(Layers[i], ["_param_power"], [TensorProto.FLOAT], [np.shape(power)],[power])
                power_input_name = [output_name[0] + "_add", power_param_name[0]]
                power_node = op.create_power_node(Layers[i], node_name + "_power", power_input_name, output_name,
                                                  [input_shape[0], np.shape(power)])
                self.onnxNodeList.append(power_node)

            elif Layers[i].type == "TanH":
                # 1.获取节点输入名、输入维度、输出名、节点名
                input_name, input_shape = self.GetLastLayerOutNameAndShape(
                    Layers[i])  # 获取输入名列表和输入形状
                output_name = self.GetCurrentLayerOutName(Layers[i])  # 获取输出名列表
                node_name = Layers[i].name

                # 2.构建tanh_node
                tanh_node = op.createTanh(
                    Layers[i], node_name, input_name, output_name, input_shape)

                # 3.添加节点到节点列表
                self.onnxNodeList.append(tanh_node)
                
            elif Layers[i].type == "Crop":
                # Crop: Slice
                # create Slice node
                input_name, input_shape = self.GetLastLayerOutNameAndShape(Layers[i])
                output_name = self.GetCurrentLayerOutName(Layers[i])
                node_name = Layers[i].name

                starts, ends, axes = op.get_crop_param(Layers[i],input_shape)
                
                Crop_name=[]
                Crop_name.append(input_name[0])
                
                starts_param = self.AddInputsTVIMannul(Layers[i],
                                                       ['_starts' + str(i)],
                                                       [TensorProto.INT64],
                                                       [np.shape(starts)],
                                                       [starts])
                ends_param = self.AddInputsTVIMannul(Layers[i],
                                                     ['_ends' + str(i)],
                                                     [TensorProto.INT64],
                                                     [np.shape(ends)], [ends])
                axes_param = self.AddInputsTVIMannul(Layers[i],
                                                     ['_axes' + str(i)],
                                                     [TensorProto.INT64],
                                                     [np.shape(axes)], [axes])
           
                Crop_name.extend(starts_param)
                Crop_name.extend(ends_param)
                Crop_name.extend(axes_param)
                crop_node = op.create_crop_node(Layers[i], node_name, Crop_name, output_name,
                                                  input_shape)
                self.onnxNodeList.append(crop_node)

            # MVN
            elif Layers[i].type == "MVN":
                # MVN: InstanceNormalization
                # create InstanceNormalization
                if Layers[i].mvn_param.normalize_variance  == False or Layers[i].mvn_param.across_channels  == True:
                               print("Failed type not support: " + Layers[i].type)
                               exit(-1)
                              

                input_name, input_shape = self.GetLastLayerOutNameAndShape(
                    Layers[i])
                output_name = self.GetCurrentLayerOutName(Layers[i])
                node_name = Layers[i].name

                MVN_name = []
                MVN_name.append(input_name[0])
                scale, bias = op.get_InstanceNorm_param(Layers[i],input_shape)

                scale_param = self.AddInputsTVIMannul(Layers[i],
                                                       ['_scale' + str(i)],
                                                       [TensorProto.FLOAT],
                                                       [np.shape(scale)],
                                                       [scale])
                bias_param = self.AddInputsTVIMannul(Layers[i],
                                                     ['_bias' + str(i)],
                                                     [TensorProto.FLOAT],
                                                     [np.shape(bias)], [bias])

                MVN_name.extend(scale_param)
                MVN_name.extend(bias_param)
                MVN_node = op.create_InstanceNorm_op(Layers[i], node_name,
                                                MVN_name, output_name,
                                                input_shape)
                self.onnxNodeList.append(MVN_node)
            else:
                print("Failed type not support: " + Layers[i].type)
                exit(-1)

    # 判断当前节点是否是输出节点
    def JudgeOutput(self, current_node, nodelist):
        for output_name in current_node.outputs_name:
            for node in nodelist:
                if output_name in node.inputs_name:
                    return False
        return True

    # 添加模型输出信息和中间节点信息
    def AddOutputsTVIAndValueInfo(self):
        for i in range(len(self.onnxNodeList)):
            if self.JudgeOutput(self.onnxNodeList[i], self.onnxNodeList):  # 构建输出节点信息
                lastnode = self.onnxNodeList[i]
                for j in range(len(lastnode.outputs_shape)):
                    output_tvi = helper.make_tensor_value_info(lastnode.outputs_name[j], TensorProto.FLOAT,
                                                               lastnode.outputs_shape[j])
                    self.onnxmodel.addOutputsTVI(output_tvi)
            else:  # 构建中间节点信息
                innernode = self.onnxNodeList[i]
                for k in range(len(innernode.outputs_shape)):
                    hid_out_tvi = helper.make_tensor_value_info(innernode.outputs_name[k], TensorProto.FLOAT,
                                                                innernode.outputs_shape[k])
                    self.onnxmodel.addValueInfoTVI(hid_out_tvi)

    # 创建模型
    def createOnnxModel(self):
        node_def = [Node.node for Node in self.onnxNodeList]
        graph_def = helper.make_graph(
            node_def,
            self.onnxmodel.name,
            self.onnxmodel.in_tvi,
            self.onnxmodel.out_tvi,
            self.onnxmodel.init_t,
            value_info=self.onnxmodel.hidden_out_tvi
        )
        model_def = helper.make_model(graph_def, producer_name='Tencent YouTu')
        print("2.onnx模型转换完成")
        return model_def
