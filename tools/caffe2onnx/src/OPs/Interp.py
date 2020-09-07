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

def get_interp_attri(layer, input_shape):
    # scale = layer.upsample_param.scale
    # scales = [1.0,1.0,scale,scale]
    # dict = {"scales":scales,"mode":"nearest"}#Upsample将scales放入参数里面了
    # dict = {"width_scale": scale,"height_scale":scale, "mode": "nearest"}#在OpenVINO读onnx的时候要求用width_scale和height_scale
    height = layer.interp_param.height
    width = layer.interp_param.width
    zoom_factor = layer.interp_param.zoom_factor
    shrink_factor = layer.interp_param.shrink_factor
    pad_beg = layer.interp_param.pad_beg
    pad_end = layer.interp_param.pad_end
    H, W = input_shape[0][2], input_shape[0][3]

    sacles = [1.0, 1.0, 1.0, 1.0]
    if height > H and width > W:
        if height / H == width / W:
            scale = float(height / H)
            scales = [1.0, 1.0, scale, scale]
            attributes = {"mode": "linear",
                          'scales': scales}
            return attributes
    if height == 0 and width == 0:
        if zoom_factor > 1 and shrink_factor == 1:
            height_in_eff = height + pad_beg + pad_end
            width_in_eff = width + pad_beg + pad_end
            height_out = height_in_eff + (height_in_eff - 1) * (zoom_factor -1)
            width_out = width_in_eff + (width_in_eff - 1) * (zoom_factor -1)
            scale_height = float(height_out /height_in_eff)
            scale_width = float(width_out /width_in_eff)
            scales = [1.0, 1.0, scale_height, scale_width]
            attributes = {"mode": "linear",
                          'scales': scales}
            return attributes
        else:
            print("do not support interp type")
            exit(-1)


def get_interp_output_shape(layer, input_shape, attributes):
    scales = attributes.get("scales")
    output_shape = [np.multiply(np.array(scales, dtype=np.int), np.array(input_shape[0])).tolist()]
    return output_shape

# TODO interp 只支持放大的情况，后期会将 onnx 升级到 1.6.0 , 并使用 resize 替换
def create_interp_node(layer, node_name, input_name, output_name, input_shape):
    attributes = get_interp_attri(layer, input_shape)
    output_shape = get_interp_output_shape(layer, input_shape, attributes)

    # print(output_shape)
    node = Node.c2oNode(layer, node_name, "Upsample", input_name, output_name, input_shape, output_shape, attributes)
    return node
