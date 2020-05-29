# Tencent is pleased to support the open source community by making TNN available.
#
# Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.
#
# Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
#
# https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software distributed
# under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.


import src.c2oObject as Node


# 计算输出维度
def getReshapeOutShape(layer, input_shape):
    try:
        # 获取layer的reshape param
        re_shape = layer.reshape_param.shape.dim
    except Exception as e:
        re_shape = []

    # 计算input shape所有维度之积
    in_prod = 1
    for dim in input_shape[0]:
        in_prod = in_prod * dim
    if re_shape == []:
        output_shape = [[1, in_prod]]
    else:
        output_shape = re_shape
        for i in range(len(re_shape)):
            if re_shape[i] == 0:
                output_shape[i] = input_shape[0][i]

        for j in range(len(output_shape)):
            if output_shape[j] == -1:
                for d in output_shape:
                    in_prod = in_prod / d
                output_shape[j] = int(in_prod * -1)
        output_shape = [output_shape]
    return output_shape


# 构建节点
def createShuffle(layer, nodename, inname, outname, input_shape):
    # 获取output_shape
    output_shape = getReshapeOutShape(layer, input_shape)

    # 构建node
    node = Node.c2oNode(layer, nodename, "Reshape", inname, outname, input_shape, output_shape)
    return node
