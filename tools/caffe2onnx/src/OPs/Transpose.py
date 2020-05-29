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
import typing


def getTransposeAttri(layer) -> typing.Dict:
    if layer.type == "ShuffleChannel":
        # 超参数字典
        perm_array = [0, 2, 1, 3, 4]
        attributes = {"perm": perm_array}
        return attributes
    else:
        orders = layer.permute_param.order
        attributes = {"perm": orders}
        return attributes


# 计算输出维度
def getTransposeOutShape(layer, input_shape, attributes):
    if layer.type == "ShuffleChannel":
        n, g, c, h, w = input_shape[0][0], input_shape[0][1], input_shape[0][2], input_shape[0][3], input_shape[0][4]

        output_shape = [[n, c, g, h, w]]
        return output_shape
    else:
        orders = attributes.get("perm")
        shape = []
        for order in orders:
            shape.append(input_shape[0][order])
        return [shape]

# 构建节点
def createTranspose(layer, node_name, input_name, output_name, input_shape) -> Node:
    attributes = getTransposeAttri(layer)

    output_shape = getTransposeOutShape(layer, input_shape, attributes)

    node = Node.c2oNode(layer, node_name, "Transpose", input_name, output_name, input_shape, output_shape, attributes)
    return node
