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

def get_attribute(layer):
    attributes = {}
    max_attribute = 0
    min_attribute = 0
    if layer.type == 'ReLU6':
        max_attribute = 6.0
        min_attribute = 0

    attribute = {
        'max': max_attribute,
        'min': min_attribute
    }
    return attributes


def get_clip_output_shape(input_shape):
    output_shape = input_shape
    return output_shape


def create_clip_node(layer, node_name, input_name, output_name, input_shape):
    # onnx 1.6.0 don't use
    # attributes = get_attribute(layer)
    output_shape = get_clip_output_shape(input_shape)
    node = Node.c2oNode(layer, node_name, 'Clip', input_name, output_name, input_shape, output_shape)
    return node
