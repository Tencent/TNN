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

#ifndef TNN_SOURCE_TNN_INTERPRETER_LAYER_PARAM_H_
#define TNN_SOURCE_TNN_INTERPRETER_LAYER_PARAM_H_

#include <limits.h>

#include <cfloat>
#include <map>
#include <string>
#include <vector>

#include "tnn/core/common.h"

namespace TNN_NS {

struct LayerParam {
    virtual ~LayerParam() {}
    /**layer type*/
    std::string type;
    /**layer name*/
    std::string name;
    bool quantized = false;
    // weight data size for ncnn param
    size_t weight_data_size = 0;
};

enum ActivationType {
    ActivationType_None  = 0x0000,
    ActivationType_ReLU  = 0x0001,
    ActivationType_ReLU6 = 0x0002,
    ActivationType_SIGMOID_MUL = 0x0100,
};

enum FusionType {
    FusionType_None                = 0x0000,
    FusionType_Conv_Add_Activation = 0x0001,
    FusionType_Conv_Activation_Add = 0x0002,
};

struct BatchNormLayerParam : public LayerParam {
    int channels = 0;
    float eps    = 0.f;
};
struct InstanceNormLayerParam : public LayerParam {
    int channels = 0;
    float eps    = 0.01f;
};

struct ConvLayerParam : public LayerParam {
    int pad_type = -1;
    // input channels of blob, devide by group
    int input_channel = 0;
    // the total output channels of blob, not devide by group
    int output_channel = 0;
    //[w_begin w_end h_begin h_end d_begin d_end]
    std::vector<int> pads;
    // order [w h d]
    std::vector<int> kernels;
    // order [w h d]
    std::vector<int> strides;
    // order [w h d]
    std::vector<int> dialations;
    int group           = 1;
    int bias            = 0;
    int activation_type = ActivationType_None;
    int fusion_type     = FusionType_None;
};

struct PadLayerParam : public LayerParam {
    //[w_begin, w_end, h_begin, h_end, c_begin, c_end]
    std::vector<int> pads;
    // 0:const 1:reflect 2:edge
    int type = 0;
    float value = 0.0f;
};

struct PoolingLayerParam : public LayerParam {
    int pool_type = 0;
    //-1:caffe typy default 0:SAME 1:VALID
    int pad_type  = -1;
    int ceil_mode = 1;

    //[w_begin w_end h_begin h_end d_begin d_end]
    std::vector<int> pads;
    // order [w h d]
    std::vector<int> kernels;
    std::vector<int> kernels_params;
    // order [w h d]
    std::vector<int> strides;

    // order [w h d] for adaptive pool
    std::vector<int> kernel_indexs;
};

struct RoiPoolingLayerParam : public LayerParam {
    // pool type of roi pooling
    int pool_type = 0;

    // scale of the input image / roi
    float spatial_scale = 1.0f;

    // output spatial dimensions, [WHD]
    std::vector<int> pooled_dims;
};

struct UpsampleLayerParam : public LayerParam {
    //1: nereast 2:bilinear/linear
    int mode          = 0;
    int align_corners = 0;

    // order [w h d]
    std::vector<float> scales;
    // order [w h d]
    std::vector<int> dims;
};

struct SoftmaxLayerParam : public LayerParam {
    int axis = 1;
};

struct PowLayerParam : public LayerParam {
    float exponent = 1.0f;
    float scale    = 1.0f;
    float shift    = 0.0f;
};

struct NormalizeLayerParam : public LayerParam {
    float epsilon = 1e-12f;
    int axis      = 1;
    int p         = 2;

    int across_spatial = 0;
    int channel_shared = 1;
};

struct ReshapeLayerParam : public LayerParam {
    // reshape_type:
    // onnx caffe reshape(nchw): 0
    // Tensorflow TFLite reshape(nhwc): 1
    int reshape_type = 0;
    int axis         = 0;
    int num_axes     = 0;
    std::vector<int> shape;
};

struct PermuteLayerParam : public LayerParam {
    std::vector<int> orders;
};

struct ScaleLayerParam : public LayerParam {
    int axis      = 1;
    int num_axes  = 1;
    int bias_term = 0;
};

struct SplitVLayerParam : public LayerParam {
    int axis = 1;
    // size of each slice
    std::vector<int> slices;
};

struct ReduceLayerParam : public LayerParam {
    int keep_dims = 0;
    std::vector<int> axis;
    // ignore axis, reduce all to one
    int all_reduce = 0;
};

struct ReduceSumLayerParam : public ReduceLayerParam {};

struct ReduceMeanLayerParam : public ReduceLayerParam {};

struct ReduceMaxLayerParam : public ReduceLayerParam {};

struct InnerProductLayerParam : public LayerParam {
    int num_output = 0;
    int has_bias   = 0;
    int transpose  = 0;
    int axis       = 0;
};

struct ConcatLayerParam : public LayerParam {
    int axis = 1;
};

struct PReluLayerParam : public LayerParam {
    int channel_shared = 0;
    int has_filler;
};

struct EluLayerParam : public LayerParam {
    float alpha = 1.0;
};

struct ClipLayerParam : public LayerParam {
    float min = -FLT_MAX;
    float max = FLT_MAX;
};

struct SeluLayerParam : public LayerParam {
    float alpha;
    float gamma;
};

//前闭后开区间
struct StrideSliceLayerParam : public LayerParam {
    // order [w h d c n]
    std::vector<int> begins;
    // order [w h d c n]
    std::vector<int> ends;
    // order [w h d c n]
    std::vector<int> strides;
};

struct SliceLayerParam : public LayerParam {
    // size of each slice
    std::vector<int> slices;
    int axis;
};

struct ElementWiseLayerParam : public LayerParam {};

typedef enum {
    // unknown or decided by runtime
    BroadcastTypeUnknown = -1,
    // no broadcast
    BroadcastTypeNormal = 0,
    // broadcast single element
    BroadcastTypeSingle = 1,
    // broadcast channel
    BroadcastTypeChannel = 2,
    // broadcast channel x height x width
    BroadcastTypeElement = 3,
    // broadcast height x width
    BroadcastTypeHeightWidth = 4,
    // broadcast width
    BroadcastTypeWidth = 5
} BroadcastType;

struct MultidirBroadcastLayerParam : public ElementWiseLayerParam {
    int input0_broadcast_type = BroadcastTypeUnknown;
    int input1_broadcast_type = BroadcastTypeUnknown;
    int weight_input_index    = 1;
};

struct HardSwishLayerParam : public MultidirBroadcastLayerParam {
    float alpha = 1.0f;
    float beta  = 0.0f;
};

struct HardSigmoidLayerParam : public LayerParam {
    float alpha = 1.0f;
    float beta  = 0.0f;
};

typedef enum {
    // only data_type
    QUANT_ONLY   = 0,
    DEQUANT_ONLY = 1,
    // data_type + layout for arm
    QUANT_NCHW4_2_NHWC   = 2,
    DEQUANT_NHWC_2_NCHW4 = 3
    // to be continued
} ReformatType;

struct ReformatLayerParam : public LayerParam {
    DataType src_type;
    DataType dst_type;
    DataFormat src_format;
    DataFormat dst_format;
    ReformatType type;
};

struct ShuffleLayerParam : public LayerParam {
    int group;
};

struct PriorBoxLayerParam : public LayerParam {
    std::vector<float> min_sizes;
    std::vector<float> max_sizes;
    bool clip = false;
    bool flip = true;

    std::vector<float> variances;
    std::vector<float> aspect_ratios;
    // order [img_h, img_w]
    int img_w;
    int img_h;
    // order [step_h, step_w]
    float step_w;
    float step_h;

    float offset = 0.5;
};

struct DetectionOutputLayerParam : public LayerParam {
    int num_classes;
    bool share_location;
    int background_label_id;
    bool variance_encoded_in_target;
    int code_type;
    int keep_top_k;
    float confidence_threshold;

    struct nms_param {
        float nms_threshold;
        int top_k;
    } nms_param;
    float eta;
};

struct DetectionPostProcessLayerParam : public LayerParam {
    int max_detections;
    int max_classes_per_detection;
    int detections_per_class;
    bool use_regular_nms;
    float nms_score_threshold;
    float nms_iou_threshold;
    int num_classes;
    // y_scale, x_scale, h_scale, w_scale
    std::vector<float> center_size_encoding;
    bool has_anchors;
    int num_anchors;
    int anchors_coord_num;
};

struct LRNLayerParam : public LayerParam {
    float alpha;
    float beta;
    float bias;
    int size;
};

struct ReorgLayerParam : public LayerParam {
    int stride;
    bool forward;
    int mode; // DCR: 0  CRD: 1
};

struct ConstLayerParam : public LayerParam {
    std::vector<int> dims;
};

struct SignedMulLayerParam : public LayerParam {
    float alpha = 1.0f;
    float beta  = 1.0f;
    float gamma = 2.0f;
};

struct SqueezeLayerParam : public LayerParam {
    std::vector<int> axes;
};

struct ArgMaxOrMinLayerParam : public LayerParam {
    int mode;
    int axis;
    int keep_dims;
    int select_last_index;
};

struct PixelShuffleLayerParam : public LayerParam {
    int upscale_factor;
};

}  // namespace TNN_NS

#endif  // TNN_SOURCE_TNN_INTERPRETER_LAYER_PARAM_H
