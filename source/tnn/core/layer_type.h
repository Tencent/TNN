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

#ifndef TNN_SOURCE_TNN_CORE_LAYER_TYPE_H_
#define TNN_SOURCE_TNN_CORE_LAYER_TYPE_H_

#include <string>
#include "tnn/core/macro.h"

namespace TNN_NS {

/*
 * Here, we list all the layer types in TNN.
 * Not all of following types are used now.
 */
enum LayerType {
    // pls do not change the existing numbers
    LAYER_NOT_SUPPORT      = 0,
    LAYER_CONVOLUTION      = 1,
    LAYER_BATCH_NORM       = 2,
    LAYER_POOLING          = 4,
    LAYER_RELU             = 5,
    LAYER_FC               = 6,
    LAYER_SPLITING         = 7,
    LAYER_CONCAT           = 8,
    LAYER_RESHAPE          = 9,
    LAYER_FLATTEN          = 10,
    LAYER_DROPOUT          = 11,
    LAYER_LRN              = 12,
    LAYER_PROPOSAL         = 13,
    LAYER_ROIPOOLING       = 14,
    LAYER_ELTWISE          = 15,
    LAYER_SCALE            = 16,
    LAYER_ARB_DIM_SPP      = 17,
    LAYER_BATCH_NORM_EX    = 18,
    LAYER_INNER_PRODUCT    = 19,
    LAYER_RESHAPEC         = 20,
    LAYER_SOFTMAX          = 21,
    LAYER_DECONVOLUTION    = 22,
    LAYER_SIGMOID          = 23,
    LAYER_PERMUTE          = 32,
    LAYER_PRIOR_BOX        = 33,
    LAYER_DETECTION_OUTPUT = 34,
    LAYER_PRELU            = 35,
    LAYER_ADD              = 38,
    LAYER_TANH             = 39,
    LAYER_LEAKY_RELU       = 40,
    LAYER_ABS              = 41,
    LAYER_MUL              = 42,
    LAYER_INST_BATCH_NORM  = 43,
    LAYER_PAD              = 44,
    LAYER_NORMALIZE        = 45,
    LAYER_QUANTIZEV2       = 46,
    LAYER_LSTM             = 47,
    // Quantization related layers
    LAYER_QUANTIZEDPOOLING                                  = 49,
    LAYER_DEQUANTIZE                                        = 50,
    LAYER_QUANTIZEDRESHAPE                                  = 51,
    LAYER_CONVOLUTION_DEPTHWISE                             = 52,
    LAYER_QUANTIZEDBIASADD                                  = 53,
    LAYER_BIASADD                                           = 54,
    LAYER_CONTINUATION_INDICATOR                            = 55,
    LAYER_QUANTIZEDRELU                                     = 56,
    LAYER_STRIDED_SLICE                                     = 57,
    LAYER_RESHAPE_TENSORFLOW                                = 58,
    LAYER_QUANTIZEDINNERPRODUCT                             = 60,
    LAYER_LSTM_CTC                                          = 61,
    LAYER_LABEL_SEQUENCE_ACCURACY                           = 62,
    LAYER_SHUFFLE_CHANNEL                                   = 63,
    LAYER_IM2COL_TRANSPOSE                                  = 64,
    LAYER_IM2COL                                            = 65,
    LAYER_TRANSPOSE                                         = 66,
    LAYER_FILEINPUT                                         = 67,
    LAYER_REVERSE                                           = 68,
    LAYER_POWER                                             = 69,
    LAYER_NEG                                               = 70,
    LAYER_TENSORDOT                                         = 71,
    LAYER_SHAPE                                             = 72,
    LAYER_PROD                                              = 73,
    LAYER_CONST                                             = 100,
    LAYER_IDENTITY                                          = 101,
    LAYER_SLICE                                             = 102,
    LAYER_CAST                                              = 103,
    LAYER_GATHER                                            = 104,
    LAYER_MATMUL                                            = 105,
    LAYER_PACK                                              = 106,
    LAYER_PLACEHOLDER                                       = 107,
    LAYER_SUB                                               = 108,
    LAYER_ADD_TF                                            = 109,
    LAYER_MUL_TF                                            = 110,
    LAYER_SLICE_TF                                          = 111,
    LAYER_SPLIT_TF                                          = 113,
    LAYER_NEGRELUMUL                                        = 114,
    LAYER_NCHW2NHWC                                         = 115,
    LAYER_NHWC2NCHW                                         = 116,
    LAYER_QUANTIZEDCONVOLUTION                              = 117,
    LAYER_SQUEEZE                                           = 118,
    LAYER_PRELUX                                            = 121,
    LAYER_REQUANTIZE                                        = 122,
    LAYER_BATCH_NORM_QUANTIZE                               = 123,
    LAYER_BATCH_NORM_CXX_QUANTIZE                           = 124,
    LAYER_RELU6                                             = 125,
    LAYER_QUANTIZED_CONCAT                                  = 126,
    LAYER_QUANTIZED_NCHW_TO_NCHW4                           = 127,
    LAYER_DEQUANTIZED_NCHW4_TO_NCHW                         = 128,
    LAYER_SQUARE                                            = 129,
    LAYER_SQRT                                              = 130,
    LAYER_REORG                                             = 131,
    LAYER_ELU                                               = 132,
    LAYER_REDUCE_SUM                                        = 133,
    LAYER_REALDIV                                           = 134,
    LAYER_BN                                                = 135,
    LAYER_INTERP                                            = 136,
    LAYER_MAXIMUM                                           = 137,
    LAYER_RSQRT                                             = 138,
    LAYER_DETECTION_OUTPUT_REF                              = 139,
    LAYER_MINIMUM                                           = 140,
    LAYER_EXP                                               = 141,
    LAYER_DEQUANTIZE_NCHW4_TO_NCHW                          = 142,
    LAYER_QUANTIZED_BATCH_NORM_CXX_SIGNED_INPUT             = 143,
    LAYER_QUANTIZED_ADD_SIGNED_INPUT                        = 144,
    LAYER_QUANTIZED_CONVOLUTION_SIGNED_INPUT                = 145,
    LAYER_QUANTIZED_RELU_SIGNED_INPUT                       = 146,
    LAYER_LOGSIGMOID                                        = 147,
    LAYER_REPEAT                                            = 148,
    LAYER_UPSAMPLE                                          = 149,
    LAYER_POOLING_NCHWC4                                    = 150,
    LAYER_QUANTIZED_CONVOLUTION_DEQUANTIZE_BN_QUANTIZE_RELU = 151,
    LAYER_DEQUANTIZE_BN_ADD_BN_QUANTIZE                     = 152,
    LAYER_SPLITV                                            = 153,
    LAYER_BATCH_NORM_QUANTIZE_V2                            = 154,
    LAYER_QUANTIZE_BY_CHANNEL                               = 155,
    LAYER_DEQUANTIZE_NCHW4_TO_NCHW_BY_CHANNEL               = 156,
    LAYER_QUANTIZED_CONVOLUTION_BY_CHANNEL                  = 157,
    LAYER_QUANTIZED_FUSED_CBR_BY_CHANNEL                    = 158,
    LAYER_UNPACK                                            = 159,
    LAYER_FILL                                              = 160,
    LAYER_RESIZE_BICUBIC                                    = 161,
    LAYER_FUSED_BATCH_NORM                                  = 162,
    LAYER_UNSQUEEZE                                         = 164,
    LAYER_GRU                                               = 165,
    LAYER_HARDTANH                                          = 166,
    LAYER_ADAPTIVE_AVG_POOL                                 = 167,
    LAYER_ADAPTIVE_MAX_POOL                                 = 168,
    LAYER_REDUCE_MEAN                                       = 169,
    LAYER_REFORMAT                                          = 170,
    LAYER_CLIP                                              = 171,
    LAYER_HARDSIGMOID                                       = 172,
    LAYER_HARDSWISH                                         = 173,
    LAYER_SOFTPLUS                                          = 174,
    LAYER_DIV                                               = 175,
    LAYER_SIGN                                              = 176,
    LAYER_REDUCE_MAX                                        = 177,
    LAYER_COS                                               = 178,
    LAYER_ACOS                                              = 179,
    LAYER_SIN                                               = 180,
    LAYER_ASIN                                              = 181,
    LAYER_TAN                                               = 182,
    LAYER_ATAN                                              = 183,
    LAYER_LOG                                               = 184,
    LAYER_RECIPROCAL                                        = 185,
    LAYER_FLOOR                                             = 186,
    LAYER_SELU                                              = 187,
    LAYER_REDUCE_L1                                         = 188,
    LAYER_REDUCE_L2                                         = 189,
    LAYER_REDUCE_LOG_SUM                                    = 190,
    LAYER_REDUCE_LOG_SUM_EXP                                = 191,
    LAYER_REDUCE_MIN                                        = 192,
    LAYER_REDUCE_PROD                                       = 193,
    LAYER_REDUCE_SUM_SQUARE                                 = 194,
    LAYER_CEIL                                              = 195,
    LAYER_SIGNED_MUL                                        = 196,
    LAYER_DETECTION_POST_PROCESS                            = 197,
    LAYER_SQUARED_DIFFERENCE                                = 198,
    LAYER_ARG_MAX_OR_MIN                                    = 199,

    LAYER_CONVOLUTION_3D                                    = 201,
    LAYER_POOLING_3D                                        = 202,

    LAYER_HDRGUIDE                                          = 302,
    LAYER_PIXEL_SHUFFLE                                     = 303,
    LAYER_SOFTSIGN                                          = 304,

    LAYER_BLOB_SCALE                                        = 600,

    LAYER_INT8_RANGE                                        = 700,
    LAYER_TRT_ENGINE                                        = 701,

    LAYER_CBAM_FUSED_REDUCE                                 = 800,
    LAYER_CBAM_FUSED_POOLING                                = 801
};

LayerType GlobalConvertLayerType(std::string layer_type_str);

}  // namespace TNN_NS

#endif  // TNN_SOURCE_TNN_CORE_LAYER_TYPE_H_
