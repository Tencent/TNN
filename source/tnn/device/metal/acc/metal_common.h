
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

#ifndef TNN_METAL_COMMON_H_
#define TNN_METAL_COMMON_H_

#ifndef FLT_MAX
#define FLT_MIN 1.175494351e-38F
#define FLT_MAX 3.402823466e+38F
#endif

#ifndef TNN_METAL_FULL_PRECISION
#define TNN_METAL_FULL_PRECISION 0
#endif

#ifndef UP_DIV
#define UP_DIV(x, y) (((x) + (y)-1) / (y))
#endif
#ifndef ROUND_UP
#define ROUND_UP(x, y) (((x) + (y)-1) / (y) * (y))
#endif

/**Base Param Struct **/
struct MetalParams {
    int input_width;
    int input_height;
    int input_size;
    int input_slice;

    int output_width;
    int output_height;
    int output_size;
    int output_slice;
    int share_channel = 0;
    int batch;
};

// keep as same as BroadcastType in layer_param.h
#define kBroadcastTypeNormal 0x0000
#define kBroadcastTypeSingle 0x0001
#define kBroadcastTypeChannel 0x0002
#define kBroadcastTypeElement 0x0003
#define kBroadcastTypeHeightWidth 0x0004

/** Broadcast Param Struct **/
struct MetalBroadcastParams {
    int input_width;
    int input_height;
    int input_size;
    int input_slice;
    int output_width;
    int output_height;
    int output_size;
    int output_slice;
    int broadcast_input0;
    int broadcast_input1;
    int batch;
};

/**Pow Param Struct **/
struct MetalPowParams {
    int input_width;
    int input_height;
    int input_size;
    int input_slice;
    int output_width;
    int output_height;
    int output_size;
    int output_slice;
    float exponent = 1.0;
    float scale    = 1.0;
    float shift    = 0.0;
    int batch;
};

/** Hard Sigmoid Param Struct **/
struct MetalHardSigmoidParams {
    int input_width;
    int input_height;
    int input_size;
    int input_slice;
    int output_width;
    int output_height;
    int output_size;
    int output_slice;
    float alpha = 1.0;
    float beta  = 0.0;
    float min   = 0.0;
    float max   = 0.0;
    int batch;
    int broadcast_input0;
    int broadcast_input1;
};

/** Elu Param Struct **/
struct MetalEluParams {
    int input_width;
    int input_height;
    int input_size;
    int input_slice;
    int output_width;
    int output_height;
    int output_size;
    int output_slice;
    float alpha = 1.0;
    int batch;
};

/** Clip Param Struct **/
struct MetalClipParams {
    int input_width;
    int input_height;
    int input_size;
    int input_slice;
    int output_width;
    int output_height;
    int output_size;
    int output_slice;
    float min = -FLT_MAX;
    float max = FLT_MAX;
    int batch;
};

/** Selu Param Struct **/
struct MetalSeluParams {
    int input_width;
    int input_height;
    int input_size;
    int input_slice;
    int output_width;
    int output_height;
    int output_size;
    int output_slice;
    float alpha = 1.67326;
    float gamma = 1.0507;
    int batch;
};

/** LRN Param Struct **/
struct MetalLRNParams {
    int input_width;
    int input_height;
    int input_channel;
    int input_size;
    int input_slice;
    int output_width;
    int output_height;
    int output_size;
    int output_slice;
    float alpha;
    float beta;
    float bias;
    int size;

    int batch;
};

/** Stride Slice Param Struct **/
struct MetalStrideSliceParams {
    int input_width;
    int input_height;
    int input_size;
    int input_slice;
    int output_width;
    int output_height;
    int output_size;
    int output_slice;
    int batch;
    int begin_n;
    int begin_c;
    int begin_h;
    int begin_w;
    int stride_n;
    int stride_c;
    int stride_h;
    int stride_w;
};

/** Shuffle Param Struct **/
struct MetalShuffleParams {
    int input_width;
    int input_height;
    int input_channel;
    int input_size;
    int input_slice;
    int output_width;
    int output_height;
    int output_size;
    int output_slice;
    int group             = 0;
    int channel_per_group = 0;
    int batch;
};

/** Inner Product Param Struct **/
struct MetalInnerProductParams {
    int input_width;
    int input_height;
    int input_size;
    int input_slice;
    int output_width;
    int output_height;
    int output_size;
    int output_slice;
    int batch;
    int has_bias;
    int activation = -1;
};

/** Conv Param Struct **/
struct MetalConvParams {
    int input_width;
    int input_height;
    int input_size;
    int input_slice;
    int input_slice_per_group;
    int output_width;
    int output_height;
    int output_size;
    int output_slice;
    int output_slice_per_group;
    int batch;
    int threadgroup_input_slice;
    int kernel_x;
    int kernel_y;
    int kernel_size;
    int stride_x;
    int stride_y;
    int pad_x;
    int pad_y;
    int dilation_x;
    int dilation_y;
    int kernel_delta_x;
    int kernel_delta_y;
    int input_delta_x;
    int input_delta_y;
    int has_bias;
    int activation = -1;
};

/** Winograd Param Struct **/
struct MetalWinogradParams {
    int input_width;
    int input_height;
    int input_size;
    int input_slice;
    int output_width;
    int output_height;
    int output_size;
    int output_slice;
    int batch;
    int pad_x;
    int pad_y;
    int unit_width;
    int unit_height;
    int unit;
    int has_bias;
    int activation = -1;
};

/** Mat Mul Param Struct **/
struct MetalMatMul4x4Params {
    int output_width;
    int output_height;
    int multi_length;
    int group;
};

/** Pool Param Struct **/
struct MetalPoolParams {
    int input_width;
    int input_height;
    int input_size;
    int input_slice;
    int output_width;
    int output_height;
    int output_size;
    int output_slice;
    int batch;
    int kernel_x;
    int kernel_y;
    int stride_x;
    int stride_y;
    int pad_x;
    int pad_y;
};

/** Upsample Param Struct **/
struct MetalUpsampleParams {
    int input_width;
    int input_height;
    int input_size;
    int input_slice;
    int output_width;
    int output_height;
    int output_size;
    int output_slice;
    int batch;
    // input_x/output_x
    float scale_x;
    // input_y/output_y
    float scale_y;
};

/** Concat Param Struct **/
struct MetalConcatParams {
    int input_width;
    int input_height;
    int input_size;
    int input_channel_0;
    int input_slice_0;
    int input_channel_1;
    int input_slice_1;
    int input_channel_offset;
    int output_width;
    int output_size;
    int output_channel;
    int output_slice;
    int batch;
};

/** Normalize Param Struct **/
struct MetalNormalizeParams {
    int output_width;
    int output_height;
    int output_size;
    int output_slice;
    int batch;
    float epsilon = 1e-12;
};

/** Reduce Param Struct
 *  ReduceL1
 *  ReduceL2
 *  ReduceLogSum
 *  ReduceLogSumExp
 *  ReduceMax
 *  ReduceMean
 *  ReduceMin
 *  ReduceProd
 *  ReduceSum
 *  ReduceSumSquare
 * **/
struct MetalReduceParams {
    int input_width;
    int input_height;
    int input_size;
    int input_slice;
    int input_batch;
    int output_width;
    int output_height;
    int output_size;
    int output_slice;
    int output_batch;
    int batch;
    int axis;
    int input_channel;
    int input_channel_mode_4;
};

/** Multi-axis Reduce Param Struct **/
struct MetalMultiAxisReduceParams {
    int input_width;
    int input_height;
    int input_size;
    int input_slice;
    int input_batch;
    int output_width;
    int output_height;
    int output_size;
    int output_slice;
    int output_batch;
    int batch;
    int input_channel;
    int input_channel_mode_4;
    int reduce_length;
    int reduce_flag[4] = {0};
};

/** Softmax Param Struct **/
struct MetalSoftmaxParams {
    int output_width;
    int output_height;
    int output_size;
    int output_slice;
    int channel_remain = 0;
    int batch;
};

/** Pad Param Struct **/
struct MetalPadParams {
    int input_width;
    int input_height;
    int input_size;
    int input_slice;
    int output_width;
    int output_height;
    int output_size;
    int output_slice;
    int batch;
    float value = 0;
    int pad_l;
    int pad_r;
    int pad_t;
    int pad_b;
    int pad_c_b;
    int pad_c_e;
    int input_channel;
};

/** Image Converter Param Struct **/
struct MetalImageConverterParams {
    int width;
    int height;
    int size;
    int channel;
    int slice;
    int batch = 1;
    float scale_x;
    float scale_y;
    float scale_z;
    float scale_w;
    float bias_x;
    float bias_y;
    float bias_z;
    float bias_w;
    int bgra_to_rgba;
};

/** Signed Mul Param Struct **/
struct MetalSignedMulParams {
    int input_width;
    int input_height;
    int input_size;
    int input_slice;
    int output_width;
    int output_height;
    int output_size;
    int output_slice;
    int batch;
    float alpha;
    float beta;
    float gamma_inv;
};

struct MetalResizeParams {
    int width;
    int height;
    int size;
    int channel;
    int slice;
    int batch;
    
    float scale_w;
    float scale_h;
    
    int resized_width;
    int resized_height;
    int type;
};

struct MetalCropParams {
    int width;
    int height;
    int size;
    int channel;
    int slice;
    int batch;
    
    int crop_width;
    int crop_height;
    int top_left_x;
    int top_left_y;
};

struct MetalWarpAffineParams {
    int width;
    int height;
    int size;
    int channel;
    int slice;
    int batch;
    
    int resized_width;
    int resized_height;
    // double is not supported in Metal
    float transform_inv[2][3];
    int interp_type;
    int border_type;
    float border_val;
};

struct MetalCopyParams {
    int width;
    int height;
    int size;
    int channel;
    int slice;
    int batch;
};

struct MetalCopyMakeBorderParam {
    int width;
    int height;
    int channel;
    int batch;
    int top;
    int bottom ;
    int left;
    int right;
    int border_type;
    float border_val;
};


struct MetalBGR2GrayParams {
    int width;
    int height;
    int size;
    int channel;
    int slice;
    int batch;
};

/** Reshape Param Struct **/
struct MetalReshapeParams {
    int input_width;
    int input_height;
    int input_size;
    int input_slice;
    int input_channel;

    int output_width;
    int output_height;
    int output_size;
    int output_slice;
    int output_channel;
    int batch;
};

/** ArgMaxOrMin Param Struct **/
struct MetalArgMaxOrMinParams {
    int input_channel;
    int outer_size;
    int inner_size;
    int reduce_size;
    int mode;
};

/** PixelShuffle Param Struct **/
struct MetalPixelShuffleParams {
    int input_width;
    int input_height;
    int input_slice;
    int input_channel;

    int output_width;
    int output_height;
    int output_slice;
    int batch;

    int upscale_factor;
};

/** Reorg Param Struct **/
struct MetalReorgParams {
    int input_width;
    int input_height;
    int input_slice;
    int input_channel;

    int output_width;
    int output_height;
    int output_slice;
    int output_channel;
    int batch;

    int stride;
    int mode; // DCR: 0  CRD: 1
};

#define SetDefaultMetalParams(metal_params, dims_input, dims_output)                                                   \
    do {                                                                                                               \
        metal_params.input_width   = dims_input[3];                                                                    \
        metal_params.input_height  = dims_input[2];                                                                    \
        metal_params.input_size    = metal_params.input_height * metal_params.input_width;                             \
        metal_params.input_slice   = UP_DIV(dims_input[1], 4);                                                         \
        metal_params.output_width  = dims_output[3];                                                                   \
        metal_params.output_height = dims_output[2];                                                                   \
        metal_params.output_size   = metal_params.output_height * metal_params.output_width;                           \
        metal_params.output_slice  = UP_DIV(dims_output[1], 4);                                                        \
        metal_params.batch         = dims_output[0];                                                                   \
    } while (0)

#define SetDefaultMetalConvParams(metal_params, conv_param)                                                            \
    do {                                                                                                               \
        metal_params.activation  = conv_param->activation_type;                                                        \
        metal_params.has_bias    = conv_param->bias;                                                                   \
        metal_params.kernel_x    = conv_param->kernels[0];                                                             \
        metal_params.kernel_y    = conv_param->kernels[1];                                                             \
        metal_params.kernel_size = metal_params.kernel_x * metal_params.kernel_y;                                      \
        metal_params.stride_x    = conv_param->strides[0];                                                             \
        metal_params.stride_y    = conv_param->strides[1];                                                             \
        metal_params.pad_x       = conv_param->pads[0];                                                                \
        metal_params.pad_y       = conv_param->pads[2];                                                                \
        metal_params.dilation_x  = conv_param->dialations[0];                                                          \
        metal_params.dilation_y  = conv_param->dialations[1];                                                          \
    } while (0)

#endif  // TNN_METAL_COMMON_H_
