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

def get_crop_param(layer, input_shape):
    axis: int = layer.crop_param.axis
    crop_offset = layer.crop_param.offset

    if not crop_offset:
        offset_0 = 0
    else:
        offset_0 = crop_offset[0]

    offset = []
    starts = []
    axes = []
    ends = []

    for i in range(len(input_shape[0])):
        if i < axis:
            start = 0
            end = input_shape[1][i]
        else:
            if (i - axis) >= len(crop_offset):
                offset.append(offset_0)
            else:
                offset.append(crop_offset[i - axis])

            start = offset[i - axis]
            end = start + input_shape[1][i]

        if input_shape[0][i] != input_shape[1][i]:
            axes.append(i)
            starts.append(start)
            ends.append(end)

    return starts, ends, axes

def get_crop_output_shape(layer, input_shape):  
    return [input_shape[1]]


def create_crop_node(layer, node_name, input_name, output_name, input_shape):
    output_shape = get_crop_output_shape(layer, input_shape)
    node = Node.c2oNode(layer, node_name, "Slice", input_name, output_name, input_shape, output_shape)
    return node
