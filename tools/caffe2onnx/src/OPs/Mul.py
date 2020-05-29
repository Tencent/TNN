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
import numpy as np
from typing import *
from onnx import TensorProto
import copy


def need_add_reshape(input_shape: List[List]) -> bool:
    return len(input_shape[0]) != len(input_shape[1])


def get_param_shape(input_shape: List[List]) -> List:
    input = input_shape[0]
    scale = copy.deepcopy(input_shape[1])
    if len(input) > len(scale):
        for i in range(len(input) - len(scale)):
            scale.append(1)
    return scale

def broadcast_scale(input_shape: List[List]) -> List[List]:
    input = input_shape[0]
    scale = input_shape[1]
    if len(input) > len(scale):
        for i in range(len(input) - len(scale)):
            scale.append(1)
        broadcast_shape = [input, scale]
    elif len(input) < len(scale):
        print("the scale should be less than input")
        exit(-1)
    else:
        broadcast_shape = [input, scale]
    return broadcast_shape


def get_mul_output_shape(input_shape: List[List]) -> List[List]:
    output_shape = input_shape[0]
    return [output_shape]


def create_mul_node(layer, node_name, input_name, output_name, input_shape):

    output_shape = get_mul_output_shape(input_shape)

    node = Node.c2oNode(layer, node_name, 'Mul', input_name, output_name, input_shape, output_shape)

    return node
