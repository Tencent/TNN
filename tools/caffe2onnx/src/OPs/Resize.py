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

import onnx
from typing import *
from onnx import helper
from typing import *
import ctypes
import src.c2oObject as Node
import numpy as np


def create_attributes(layer) -> Dict:
    coordinate_transformation_mode = 'half_pixel'
    cubic_coeff_a = -0.75
    exclude_outside = 0
    extrapolation_value = 0.0
    mode = 'nearest'
    nearest_mode = 'round_prefer_floor'
    attributes = {
        "coordinate_transformation_mode": coordinate_transformation_mode,
        "cubic_coeff_a": cubic_coeff_a,
        "exclude_outside": exclude_outside,
        "extrapolation_value": extrapolation_value,
        "mode": mode,
        "nearest_mode" :nearest_mode
    }
    return attributes


def caculate_output_shape(layer, input_shape) -> List:
    scale = layer.upsample_param.scale
    scales = [1.0,1.0,scale,scale]
    output_shape = [np.multiply(np.array(scales,dtype=np.int),np.array(input_shape[0])).tolist()]
    return output_shape



def create_resize_node(layer,
                       node_name: str,
                       inputs_name: List[str],
                       outputs_name: List[str],
                       inputs_shape: List, ) -> onnx.NodeProto:
    attributes = create_attributes(layer)

    outputs_shape = caculate_output_shape(layer, inputs_shape)

    node = Node.c2oNode(layer, node_name, "Resize",
                        inputs_name, outputs_name,
                        inputs_shape, outputs_shape,
                        attributes)
    return node
