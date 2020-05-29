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
from typing import *
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
    output_shape = input_shape[1]
    return [output_shape]


def create_axpy_mul_node(layer, node_name, input_name, output_name, input_shape):

    new_node_name = node_name + "_middle"
    output_shape = get_mul_output_shape(input_shape)
    new_input_name = [input_name[0], input_name[1]]
    new_output_name = [output_name[0] + "_mul"]
    new_input_shape = [input_shape[0], input_shape[1]]

    node = Node.c2oNode(layer, new_node_name, 'Mul', new_input_name, new_output_name, new_input_shape, output_shape)

    return node

def get_add_output_shape(input_shape):

    output_shape = input_shape[1]

    return [output_shape]

def create_axpy_add_node(layer, node_name, input_name, output_name, input_shape):

    output_shape = get_add_output_shape(input_shape)
    new_input_name = [node_name + "_mul", input_name[2]]
    new_input_shape = [input_shape[1], input_shape[2]]
    node = Node.c2oNode(layer, node_name, "Add", new_input_name, output_name, input_shape, output_shape)

    return node
