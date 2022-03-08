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
#include <set>
#include <memory>
#include <string>
#include <vector>

#include "tnn/core/common.h"

namespace TNN_NS {

#define PARAM_COPY(param_type)                                                                                         \
public:                                                                                                                \
    virtual std::shared_ptr<LayerParam> Copy() {                                                                       \
        std::shared_ptr<LayerParam> param(new param_type());                                                           \
        param_type* param_ptr = dynamic_cast<param_type*>(param.get());                                                \
        if (nullptr == param_ptr) {                                                                                    \
            LOGE("dynamic cast to %s failed\n", #param_type);                                                          \
            return nullptr;                                                                                            \
        }                                                                                                              \
        *param_ptr = *this;                                                                                            \
        return param;                                                                                                  \
    }

struct LayerParam {
    virtual ~LayerParam() {}
    /**layer type*/
    std::string type;
    /**layer name*/
    std::string name;
    bool quantized = false;
    // use int8 save, float32 interpreting
    bool dynamic_range_quantized = false;
    // weight data size for ncnn param
    size_t weight_data_size = 0;
    // extra config set, such as arm conv algo (gemm or winograd)
    std::set<std::string> extra_config;

    PARAM_COPY(LayerParam)
};

enum ActivationType {
    ActivationType_None        = 0x0000,
    ActivationType_ReLU        = 0x0001,
    ActivationType_ReLU6       = 0x0002,
    ActivationType_SIGMOID_MUL = 0x0100,
};

enum FusionType {
    FusionType_None                = 0x0000,
    FusionType_Conv_Add_Activation = 0x0001,
    FusionType_Conv_Activation_Add = 0x0002,
};

struct BatchNormLayerParam : public LayerParam {
    int channels = 0;
    float eps    = 1e-5f;

    PARAM_COPY(BatchNormLayerParam)
};
struct InstanceNormLayerParam : public LayerParam {
    int channels = 0;
    float eps    = 1e-5f;

    PARAM_COPY(InstanceNormLayerParam)
};

struct GroupNormLayerParam : public LayerParam {
    int group = 0;
    float eps = 1e-5f;

    PARAM_COPY(GroupNormLayerParam)
};

struct LayerNormLayerParam : public LayerParam {
    int reduce_dims_size = 0;
    float eps            = 1e-5f;

    PARAM_COPY(LayerNormLayerParam)
};

struct GridSampleLayerParam : public LayerParam {
    // 1: nereast 2: bilinear/linear 3: cubic
    int mode = 2;
    // 0:const 1:reflect 2:edge
    int pad_type      = 0;
    int align_corners = 0;

    PARAM_COPY(GridSampleLayerParam)
};

struct TileLayerParam : public LayerParam {
    // nchw order
    std::vector<int> reps;

    PARAM_COPY(TileLayerParam)
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

    PARAM_COPY(ConvLayerParam)
};

struct PadLayerParam : public LayerParam {
    // for old Pad the order is  [w_begin, w_end, h_begin, h_end, c_begin, c_end]
    // for PadV2 the order correspand to input dims, same as ONNX, like [x1_begin, x2_begin,...,x1_end, x2_end,...]
    std::vector<int> pads;
    // 0:const 1:reflect 2:edge
    int type    = 0;
    float value = 0.0f;

    PARAM_COPY(PadLayerParam)
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

    int is_adaptive_pool = 0;
    int is_global_pool   = 0;
    // order [w h d]
    std::vector<int> output_shape;

    PARAM_COPY(PoolingLayerParam)
};

struct RoiPoolingLayerParam : public LayerParam {
    // pool type of roi pooling
    int pool_type = 0;

    // scale of the input image / roi
    float spatial_scale = 1.0f;

    // output spatial dimensions, [WHD]
    std::vector<int> pooled_dims;

    PARAM_COPY(RoiPoolingLayerParam)
};

struct UpsampleLayerParam : public LayerParam {
    // 1: nereast 2: bilinear/linear 3: cubic
    int mode          = 0;
    int align_corners = 0;

    // order [w h d]
    std::vector<float> scales;
    // order [w h d]
    std::vector<int> dims;

    PARAM_COPY(UpsampleLayerParam)
};

struct RangeLayerParam : public LayerParam {
    DataType data_type = DATA_TYPE_FLOAT;
    RangeData start    = {0};
    RangeData limit    = {0};

    // designated initializer may cause compile error in msvc
    RangeData delta = {1};
    // RangeData delta = { .i = 1};

    PARAM_COPY(RangeLayerParam)
};

struct SoftmaxLayerParam : public LayerParam {
    int axis = 1;

    PARAM_COPY(SoftmaxLayerParam)
};

struct PowLayerParam : public LayerParam {
    float exponent = 1.0f;
    float scale    = 1.0f;
    float shift    = 0.0f;

    PARAM_COPY(PowLayerParam)
};

struct NormalizeLayerParam : public LayerParam {
    float epsilon = 1e-12f;
    int axis      = 1;
    int p         = 2;

    int across_spatial = 0;
    int channel_shared = 1;

    PARAM_COPY(NormalizeLayerParam)
};

struct ReshapeLayerParam : public LayerParam {
    // reshape_type:
    // onnx caffe reshape(nchw): 0
    // Tensorflow TFLite reshape(nhwc): 1
    int reshape_type = 0;
    int axis         = 0;
    int num_axes     = 0;
    std::vector<int> shape;

    PARAM_COPY(ReshapeLayerParam)
};

struct PermuteLayerParam : public LayerParam {
    std::vector<int> orders;

    PARAM_COPY(PermuteLayerParam)
};

struct CastLayerParam : public LayerParam {
    int to   = 0;
    int from = 0;  // used for HUAWEI_NPU

    PARAM_COPY(CastLayerParam)
};

struct HistogramLayerParam : public LayerParam {
    int depth;
    PARAM_COPY(HistogramLayerParam)
};

struct OneHotLayerParam : public LayerParam {
    int axis        = -1;
    int depth       = -1;
    float value_off = 0;
    float value_on  = 1;

    PARAM_COPY(OneHotLayerParam)
};

struct BitShiftLayerParam : public LayerParam {
    // 0: rigth 1:left
    int direction = 0;
    int bits      = 0;
    PARAM_COPY(BitShiftLayerParam)
};

struct ScaleLayerParam : public LayerParam {
    int axis      = 1;
    int num_axes  = 1;
    int bias_term = 0;

    PARAM_COPY(ScaleLayerParam)
};

struct SplitVLayerParam : public LayerParam {
    int axis = 1;
    // size of each slice
    std::vector<int> slices;
    // judge whether slices is specified or calculated by equal sized parts
    bool is_split_specified = true;

    PARAM_COPY(SplitVLayerParam)
};

struct ReduceLayerParam : public LayerParam {
    int keep_dims = 0;
    std::vector<int> axis;
    // ignore axis, reduce all to one
    int all_reduce = 0;

    PARAM_COPY(ReduceLayerParam)
};

struct ReduceSumLayerParam : public ReduceLayerParam {
    PARAM_COPY(ReduceSumLayerParam)
};

struct ReduceMeanLayerParam : public ReduceLayerParam {
    PARAM_COPY(ReduceMeanLayerParam)
};

struct ReduceMaxLayerParam : public ReduceLayerParam {
    PARAM_COPY(ReduceMaxLayerParam)
};

struct InnerProductLayerParam : public LayerParam {
    int num_output = 0;
    int has_bias   = 0;
    int transpose  = 0;
    int axis       = 0;

    PARAM_COPY(InnerProductLayerParam)
};

struct ConcatLayerParam : public LayerParam {
    int axis = 1;

    PARAM_COPY(ConcatLayerParam)
};

struct PReluLayerParam : public LayerParam {
    int channel_shared = 0;
    int has_filler;

    PARAM_COPY(PReluLayerParam)
};

struct EluLayerParam : public LayerParam {
    float alpha = 1.0;

    PARAM_COPY(EluLayerParam)
};

struct ClipLayerParam : public LayerParam {
    float min = -FLT_MAX;
    float max = FLT_MAX;

    PARAM_COPY(ClipLayerParam)
};

struct SeluLayerParam : public LayerParam {
    float alpha;
    float gamma;

    PARAM_COPY(SeluLayerParam)
};

//前闭后开区间
struct StrideSliceLayerParam : public LayerParam {
    // order [w h d c n]
    std::vector<int> begins;
    // order [w h d c n]
    std::vector<int> ends;
    // order [w h d c n]
    std::vector<int> strides;

    PARAM_COPY(StrideSliceLayerParam)
};

struct StrideSliceV2LayerParam : public LayerParam {
    std::vector<int> begins;
    std::vector<int> ends;
    std::vector<int> axes;
    std::vector<int> strides;

    PARAM_COPY(StrideSliceV2LayerParam)
};

struct SliceLayerParam : public LayerParam {
    // size of each slice
    std::vector<int> slices;
    int axis;

    PARAM_COPY(SliceLayerParam)
};

struct ElementWiseLayerParam : public LayerParam {
    PARAM_COPY(ElementWiseLayerParam)
};

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
    BroadcastTypeWidth = 5,
    // broadcast for any dim
    BroadcastTypeGeneral = 6,
    // broadcast channel x height
    BroadcastTypeChannelHeight = 7,
    // broadcast channel x width
    BroadcastTypeChannelWidth = 8,
} BroadcastType;

struct MultidirBroadcastLayerParam : public ElementWiseLayerParam {
    int input0_broadcast_type = BroadcastTypeUnknown;
    int input1_broadcast_type = BroadcastTypeUnknown;
    int weight_input_index    = 1;

    PARAM_COPY(MultidirBroadcastLayerParam)
};

struct HardSwishLayerParam : public MultidirBroadcastLayerParam {
    float alpha = 1.0f;
    float beta  = 0.0f;

    PARAM_COPY(HardSwishLayerParam)
};

struct HardSigmoidLayerParam : public LayerParam {
    float alpha = 1.0f;
    float beta  = 0.0f;

    PARAM_COPY(HardSigmoidLayerParam)
};

typedef enum {
    // only data_type
    QUANT_ONLY   = 0,
    DEQUANT_ONLY = 1,
    // data_type + layout for arm
    QUANT_NCHW4_2_NHWC   = 2,
    DEQUANT_NHWC_2_NCHW4 = 3,
    // data_type + layout for half data type in armv8.2
    NC4HW4FP32_2_NC8HW8FP16 = 4,
    NC8HW8FP16_2_NC4HW4FP32 = 5,
    // nchw <-> nc4hw4 fp32
    NC4HW4FP32_2_NCHWFP32 = 6,
    NCHWFP32_2_NC4HW4FP32 = 7,
    // nchw <-> nc8hw8 fp16
    NC8HW8FP16_2_NCHWFP16 = 8,
    NCHWFP16_2_NC8HW8FP16 = 9,
    // nchw <-> nc4hw4 int32
    NC4HW4INT32_2_NCHWINT32 = 10,
    NCHWINT32_2_NC4HW4INT32 = 11,
    // to be continued
} ReformatType;

struct ReformatLayerParam : public LayerParam {
    DataType src_type     = DATA_TYPE_AUTO;
    DataType dst_type     = DATA_TYPE_AUTO;
    DataFormat src_format = DATA_FORMAT_AUTO;
    DataFormat dst_format = DATA_FORMAT_AUTO;
    ReformatType type;

    PARAM_COPY(ReformatLayerParam)
};

struct ShuffleLayerParam : public LayerParam {
    int group;

    PARAM_COPY(ShuffleLayerParam)
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

    PARAM_COPY(PriorBoxLayerParam)
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

    PARAM_COPY(DetectionOutputLayerParam)
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

    PARAM_COPY(DetectionPostProcessLayerParam)
};

struct LRNLayerParam : public LayerParam {
    float alpha;
    float beta;
    float bias;
    int size;

    PARAM_COPY(LRNLayerParam)
};

struct ReorgLayerParam : public LayerParam {
    int stride;
    bool forward;
    int mode;  // DCR: 0  CRD: 1

    PARAM_COPY(ReorgLayerParam)
};

struct ConstLayerParam : public LayerParam {
    std::vector<int> dims;

    PARAM_COPY(ConstLayerParam)
};

struct SignedMulLayerParam : public LayerParam {
    float alpha = 1.0f;
    float beta  = 1.0f;
    float gamma = 2.0f;

    PARAM_COPY(SignedMulLayerParam)
};

struct SqueezeLayerParam : public LayerParam {
    // Note the axes is ascending order,  see SqueezeLayer::InferOutputShape and UnsqueezeLayer::InferOutputShape
    std::vector<int> axes;
    bool data_in_resource = false;

    PARAM_COPY(SqueezeLayerParam)
};

struct UnsqueezeLayerParam : public SqueezeLayerParam {
    PARAM_COPY(UnsqueezeLayerParam)
};

struct ArgMaxOrMinLayerParam : public LayerParam {
    int mode;
    int axis;
    int keep_dims = 1;
    int select_last_index;

    PARAM_COPY(ArgMaxOrMinLayerParam)
};

struct PixelShuffleLayerParam : public LayerParam {
    int upscale_factor;
    int axis;

    PARAM_COPY(PixelShuffleLayerParam)
};

struct GatherLayerParam : public LayerParam {
    int axis                 = 0;
    bool data_in_resource    = false;
    bool indices_in_resource = true;

    PARAM_COPY(GatherLayerParam)
};

struct GatherNDLayerParam : public LayerParam {
    int batch_dims = 0;
    PARAM_COPY(GatherNDLayerParam)
};

struct LSTMONNXLayerParam : public LayerParam {
    float clip_threshold = 0;
    int hidden_size      = 0;
    // 0: forward 1:reverse 2:bidirection
    int direction = 0;

    PARAM_COPY(LSTMONNXLayerParam)
};

struct ExpandLayerParam : public LayerParam {
    std::vector<int> shape;

    PARAM_COPY(ExpandLayerParam)
};

struct MatMulLayerParam : public LayerParam {
    int weight_position = -1;
    DimsVector matrix_a_dims;
    DimsVector matrix_b_dims;
    int axis = 0;

    PARAM_COPY(MatMulLayerParam)
};

struct RoiAlignLayerParam : public LayerParam {
    // 0: max, 1: avg
    int mode = 1;
    int output_height;
    int output_width;
    int sampling_ratio;
    float spatial_scale;

    PARAM_COPY(RoiAlignLayerParam)
};

struct FlattenLayerParam : public LayerParam {
    int axis = 1;

    PARAM_COPY(FlattenLayerParam)
};

struct EinsumLayerParam : public LayerParam {
    std::string equation;
    int out_size;
    bool has_zero_size_dim = false;
    std::vector<std::vector<int>> perm_shapes;
    std::vector<std::size_t> dim_last_op;
    std::vector<DimsVector> operand_dims;

    PARAM_COPY(EinsumLayerParam)
};

struct TopKLayerParam : public LayerParam {
    int axis    = -1;
    int largest = 1;
    int sorted  = 1;
    int k;

    PARAM_COPY(TopKLayerParam)
};

struct NonMaxSuppressionLayerParam : public LayerParam {
    int center_point_box               = 0;
    int64_t max_output_boxes_per_class = 0;
    float iou_threshold                = 0.0f;
    float score_threshold              = 0.0f;

    PARAM_COPY(NonMaxSuppressionLayerParam)
};

struct ScatterLayerParam : public LayerParam {
    int axis = 0;

    PARAM_COPY(ScatterLayerParam)
};

struct ScatterElementsLayerParam : public LayerParam {
    int axis = 0;
    // 0: eq, 1: add
    int op = 0;

    PARAM_COPY(ScatterElementsLayerParam);
};

struct LogSoftmaxLayerParam : public LayerParam {
    int axis = 1;

    PARAM_COPY(LogSoftmaxLayerParam)
};

};  // namespace TNN_NS

#endif  // TNN_SOURCE_TNN_INTERPRETER_LAYER_PARAM_H
