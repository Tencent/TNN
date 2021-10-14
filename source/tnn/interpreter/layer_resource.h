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

#ifndef TNN_SOURCE_TNN_INTERPRETER_LAYER_RESOURCE_H_
#define TNN_SOURCE_TNN_INTERPRETER_LAYER_RESOURCE_H_

#include <map>
#include <memory>
#include <string>

#include "tnn/core/layer_type.h"
#include "tnn/interpreter/raw_buffer.h"

namespace TNN_NS {

typedef std::map<std::string, DimsVector> BlobShapesMap;
typedef std::map<std::string, std::shared_ptr<RawBuffer> > ConstantResource;
typedef std::map<std::string, int > ConstantResourceFlag;

struct LayerResource {
    std::string name = "";
    // default virtual destructor
    virtual ~LayerResource(){};
};

// @brief conv layer filter format
typedef enum { OIHW = 0, IHWO = 1, OIDHW = 2 } ConvLayerFilterFormat;

// @brief ConvLayerResource different device holds different handle
struct ConvLayerResource : public LayerResource {
    // conv layer filter format
    ConvLayerFilterFormat filter_format = OIHW;

    // conv layer handle
    // NOTE: for deconv, the weight's default format is [n][i][o][h][w]
    RawBuffer filter_handle;

    // bias handle
    RawBuffer bias_handle;

    // extra scale handle for different precision
    RawBuffer scale_handle;
};

struct BatchNormLayerResource : public LayerResource {
    // bn k buffer
    RawBuffer scale_handle;

    // bn b buffer
    RawBuffer bias_handle;
};

struct InstanceNormLayerResource : public BatchNormLayerResource {};

struct EltwiseLayerResource : public LayerResource {
    // elements
    RawBuffer element_handle;

    std::vector<int> element_shape;
};

struct InnerProductLayerResource : public LayerResource {
    // weight buffer
    RawBuffer weight_handle;

    // bias buffer
    RawBuffer bias_handle;

    // extra scale handle for different precision
    RawBuffer scale_handle;
};

struct PReluLayerResource : public LayerResource {
    // slope
    RawBuffer slope_handle;
};

struct IntScaleResource : public LayerResource {
    // scale buffer
    RawBuffer scale_handle;
    // bias buffer
    RawBuffer bias_handle;
};

// @brief HdrGuideLayerResource different device holds different handle
struct HdrGuideLayerResource : public LayerResource {
    // ccm weight
    RawBuffer ccm_weight_handle;
    // ccm bias
    RawBuffer ccm_bias_handle;
    // shifts
    RawBuffer shifts_handle;
    // slopes
    RawBuffer slopes_handle;
    // projection weights
    RawBuffer projection_weight_handle;
    // projection bias
    RawBuffer projection_bias_handle;
};

struct ConstLayerResource : public LayerResource {
    // const weights
    RawBuffer weight_handle;
};

struct DetectionPostProcessLayerResource : public LayerResource {
    RawBuffer anchors_handle;
};

struct ScatterNDLayerResource : public LayerResource {
    RawBuffer indices;
    // optional
    RawBuffer updates;
};

struct ScatterLayerResource : public LayerResource {
    RawBuffer indices;
    RawBuffer updates;
};

struct ScatterElementsLayerResource : public LayerResource {
    RawBuffer data;
};

struct GatherLayerResource : public LayerResource {
    //RawBuffer has dims
    RawBuffer data;
    RawBuffer indices;
};

struct ConstantOfShapeLayerResource : public LayerResource {
    //RawBuffer has dims
    RawBuffer value;
};

struct SqueezeLayerResource : public LayerResource {
    std::vector<int> data_dims;
    RawBuffer data;
};

struct UnsqueezeLayerResource : public SqueezeLayerResource {};

struct MatMulLayerResource : public LayerResource {
    RawBuffer weight;
};

struct BiasAddLayerResource : public LayerResource {
    RawBuffer bias_handle;
};


}  // namespace TNN_NS

#endif  // TNN_SOURCE_TNN_INTERPRETER_LAYER_RESOURCE_H_
