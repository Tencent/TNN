// Tencent is pleased to support the open source community by making TNN available.
//
// Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include "ncnn_layer_type.h"
#include "tnn/core/layer_type.h"

#include <map>
#include <string>

namespace TNN_NS {

static std::map<std::string, LayerType> global_layer_type_map = {
    {"Convolution", LAYER_CONVOLUTION},
    {"ConvolutionDepthWise", LAYER_CONVOLUTION},
    {"Deconvolution", LAYER_DECONVOLUTION},
    {"DeconvolutionDepthWise", LAYER_DECONVOLUTION},
    {"BatchNorm", LAYER_BATCH_NORM},
    {"InnerProduct", LAYER_INNER_PRODUCT},
    {"Pooling", LAYER_POOLING},
    {"Softmax", LAYER_SOFTMAX},
    {"ReLU", LAYER_RELU},
    {"Sigmoid", LAYER_SIGMOID},
    {"Tanh", LAYER_TANH},
    {"HardSwish", LAYER_HARDSWISH},
    {"HardSigmoid", LAYER_HARDSIGMOID},
    {"LRN", LAYER_LRN},
    {"AbsVal", LAYER_ABS},
    {"Split", LAYER_SPLITING},
    {"Concat", LAYER_CONCAT},
    {"Reshape", LAYER_RESHAPE},
    {"Slice", LAYER_SLICE},
    {"Flatten", LAYER_FLATTEN},
    {"Dropout", LAYER_DROPOUT},
    {"ShuffleChannel", LAYER_SHUFFLE_CHANNEL},
    {"Crop", LAYER_STRIDED_SLICE},
    {"Permute", LAYER_PERMUTE},
    {"Interp", LAYER_UPSAMPLE},
    {"MemoryData", LAYER_CONST},
    {"ELU", LAYER_ELU},
    {"PReLU", LAYER_PRELU},
    {"Clip", LAYER_CLIP},
    {"Padding", LAYER_PAD},
    {"SELU", LAYER_SELU},
    {"DetectionOutput", LAYER_DETECTION_OUTPUT},
    {"InstanceNorm", LAYER_INST_BATCH_NORM},
    {"PriorBox", LAYER_PRIOR_BOX},
    {"DetectionOutput", LAYER_DETECTION_OUTPUT},
    {"Reorg", LAYER_REORG},
    {"Normalize", LAYER_NORMALIZE},
    {"RoiPooling", LAYER_ROIPOOLING},
    {"Scale", LAYER_SCALE}
};

LayerType ConvertNCNNLayerType(std::string layer_type_str) {
    if (global_layer_type_map.count(layer_type_str) > 0) {
        return global_layer_type_map[layer_type_str];
    } else {
        return LAYER_NOT_SUPPORT;
    }
}

}  // namespace TNN_NS
