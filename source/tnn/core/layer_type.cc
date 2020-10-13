#// Tencent is pleased to support the open source community by making TNN available.
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

#include "tnn/core/layer_type.h"

#include <map>
#include <string>

namespace TNN_NS {

static const std::string int8_prefix = "Int8";

static std::map<std::string, LayerType> global_layer_type_map = {
    // LAYER_Convolution, including depthwise convolution
    {"Convolution", LAYER_CONVOLUTION},
    {"Convolution3D", LAYER_CONVOLUTION_3D},
    {"BatchNormalization", LAYER_BATCH_NORM},
    {"BatchNormCxx", LAYER_BATCH_NORM},
    {"Softmax", LAYER_SOFTMAX},
    {"Pooling", LAYER_POOLING},
    {"Pooling3D", LAYER_POOLING_3D},
    {"Pooling_split_CC", LAYER_POOLING},
    {"ReLU", LAYER_RELU},
    {"Relu", LAYER_RELU},
    {"Split", LAYER_SPLITING},
    {"Concat", LAYER_CONCAT},
    {"Reshape", LAYER_RESHAPE},
    {"Flatten", LAYER_FLATTEN},
    {"Dropout", LAYER_DROPOUT},
    // LAYER_LRN
    {"LRN", LAYER_LRN},
    {"Proposal", LAYER_PROPOSAL},
    {"ROIPooling", LAYER_ROIPOOLING},
    {"Eltwise", LAYER_ELTWISE},
    {"Scale", LAYER_SCALE},
    {"ArbitraryDimensionSpp", LAYER_ARB_DIM_SPP},
    {"BatchNorm", LAYER_BATCH_NORM_EX},
    // LAYER_FC
    {"InnerProduct", LAYER_INNER_PRODUCT},
    {"ReshapeC", LAYER_RESHAPEC},
    {"SoftmaxCaffe", LAYER_SOFTMAX},
    {"Deconvolution", LAYER_DECONVOLUTION},
    {"Sigmoid", LAYER_SIGMOID},
    {"Convolution_nhwc", LAYER_CONVOLUTION},
    {"BatchNormCxx_nhwc", LAYER_BATCH_NORM},
    {"Pooling_nhwc", LAYER_POOLING},
    {"Softmax_nhwc", LAYER_SOFTMAX},
    {"Concat_nhwc", LAYER_CONCAT},
    {"Flatten_nhwc", LAYER_FLATTEN},
    {"Permute", LAYER_PERMUTE},
    // SSD related layer type
    {"PriorBox", LAYER_PRIOR_BOX},
    {"DetectionOutput", LAYER_DETECTION_OUTPUT},
    {"PReLU", LAYER_PRELU},
    {"InnerProduct_nhwc", LAYER_INNER_PRODUCT},
    {"PReLU_nhwc", LAYER_PRELU},
    {"Add", LAYER_ADD},
    {"Tanh", LAYER_TANH},
    {"LeakyRelu", LAYER_LEAKY_RELU},
    {"Abs", LAYER_ABS},
    {"Mul", LAYER_MUL},
    {"InstBatchNormCxx", LAYER_INST_BATCH_NORM},
    {"Pad", LAYER_PAD},
    {"Normalize", LAYER_NORMALIZE},
    {"QuantizeV2", LAYER_QUANTIZEV2},
    {"Lstm", LAYER_LSTM},
    {"QuantizedConvolution_nhwc", LAYER_CONVOLUTION},
    {"QuantizedPooling", LAYER_POOLING},
    // 50
    {"Dequantize", LAYER_DEQUANTIZE},
    {"QuantizedReshapeTensorflow", LAYER_RESHAPE},
    {"ConvolutionDepthwise", LAYER_CONVOLUTION_DEPTHWISE},
    {"QuantizedBiasAdd", LAYER_BIASADD},
    {"QuantizedSum", LAYER_BIASADD},
    {"BiasAdd", LAYER_BIASADD},
    {"ContinuationIndicator", LAYER_CONTINUATION_INDICATOR},
    {"QuantizedReLU", LAYER_RELU},
    {"QuantizedAdd", LAYER_ADD},
    {"StridedSlice", LAYER_STRIDED_SLICE},
    {"ReshapeTensorflow", LAYER_RESHAPE_TENSORFLOW},
    {"QuantizedInnerProduct", LAYER_INNER_PRODUCT},
    {"lstm_ctc", LAYER_LSTM_CTC},
    {"LabelsequenceAccuracy", LAYER_LABEL_SEQUENCE_ACCURACY},
    {"ShuffleChannel", LAYER_SHUFFLE_CHANNEL},
    {"Im2colTranspose", LAYER_IM2COL_TRANSPOSE},
    {"Im2col", LAYER_IM2COL},
    {"Transpose", LAYER_TRANSPOSE},
    {"FileInput", LAYER_FILEINPUT},
    {"Reverse", LAYER_REVERSE},
    {"Power", LAYER_POWER},
    {"Neg", LAYER_NEG},
    {"Tensordot", LAYER_TENSORDOT},
    {"Shape", LAYER_SHAPE},
    {"Prod", LAYER_PROD},
    // 100
    {"Const", LAYER_CONST},
    {"Identity", LAYER_IDENTITY},
    {"Slice", LAYER_SLICE},
    {"SliceCaffe", LAYER_SLICE},
    {"Cast", LAYER_CAST},
    {"Gather", LAYER_GATHER},
    {"MatMul", LAYER_MATMUL},
    {"Pack", LAYER_PACK},
    {"Placeholder", LAYER_PLACEHOLDER},
    {"Sub", LAYER_SUB},
    {"Add_tf", LAYER_ADD_TF},
    {"Mul_tf", LAYER_MUL_TF},
    {"Slice_tf", LAYER_SLICE_TF},
    {"StridedSlice_nhwc", LAYER_STRIDED_SLICE},
    {"Split_tf", LAYER_SPLIT_TF},
    {"NegReLUMul", LAYER_NEGRELUMUL},
    {"NCHW2NHWC", LAYER_NCHW2NHWC},
    {"NHWC2NCHW", LAYER_NHWC2NCHW},
    {"QuantizedConvolution", LAYER_CONVOLUTION},
    {"Squeeze", LAYER_SQUEEZE},
    {"PReLU_X", LAYER_PRELUX},
    {"Requantize", LAYER_REQUANTIZE},
    {"QuantizedBNGlobal", LAYER_BATCH_NORM_QUANTIZE},
    {"QuantizedMul", LAYER_BATCH_NORM_CXX_QUANTIZE},
    {"QuantizedBatchNormCxx", LAYER_BATCH_NORM_CXX_QUANTIZE},
    {"ReLU6", LAYER_RELU6},
    {"Relu6", LAYER_RELU6},
    {"QuantizedConcat", LAYER_CONCAT},
    {"QuantizeNCHWTONCHW4", LAYER_QUANTIZED_NCHW_TO_NCHW4},
    {"DequantizeNCHW4TONCHW", LAYER_DEQUANTIZED_NCHW4_TO_NCHW},
    {"Square", LAYER_SQUARE},
    {"Sqrt", LAYER_SQRT},
    {"Reorg", LAYER_REORG},
    {"Elu", LAYER_ELU},
    {"Reduce_Sum", LAYER_REDUCE_SUM},
    {"ReduceMean", LAYER_REDUCE_MEAN},
    {"ReduceMax", LAYER_REDUCE_MAX},
    {"RealDiv", LAYER_REALDIV},
    {"BN", LAYER_BN},
    {"Interp", LAYER_INTERP},
    {"Maximum", LAYER_MAXIMUM},
    {"Rsqrt", LAYER_RSQRT},
    {"DetectionOutputREF", LAYER_DETECTION_OUTPUT_REF},
    {"Minimum", LAYER_MINIMUM},
    {"Exp", LAYER_EXP},
    {"DequantizeNCHW4TONCHWByChannel", LAYER_DEQUANTIZE_NCHW4_TO_NCHW},
    {"QuantizedBatchNormCxxSignedInput", LAYER_QUANTIZED_BATCH_NORM_CXX_SIGNED_INPUT},
    {"QuantizedAddSignedInput", LAYER_QUANTIZED_ADD_SIGNED_INPUT},
    {"QuantizedConvolutionSignedInput", LAYER_QUANTIZED_CONVOLUTION_SIGNED_INPUT},
    {"QuantizedReluSignedInput", LAYER_QUANTIZED_RELU_SIGNED_INPUT},
    {"LogSigmoid", LAYER_LOGSIGMOID},
    {"Repeat", LAYER_REPEAT},
    {"Upsample", LAYER_UPSAMPLE},
    // 150
    {"Pooling_nchwc4", LAYER_POOLING_NCHWC4},
    {"QConv2DDequantizeMulAddQuantizeQRelu", LAYER_QUANTIZED_CONVOLUTION_DEQUANTIZE_BN_QUANTIZE_RELU},
    {"DequantizeBnAddBnQuantize", LAYER_DEQUANTIZE_BN_ADD_BN_QUANTIZE},
    {"SplitV", LAYER_SPLITV},
    {"BatchNormQuantizeV2", LAYER_BATCH_NORM_QUANTIZE_V2},
    {"QuantizeV2ByChannel", LAYER_QUANTIZE_BY_CHANNEL},
    {"DequantizeNCHW4TONCHWByChannel", LAYER_DEQUANTIZE_NCHW4_TO_NCHW_BY_CHANNEL},
    {"QuantizedConvolutionByChannel", LAYER_QUANTIZED_CONVOLUTION_BY_CHANNEL},
    {"QFusedCBRByChannel", LAYER_QUANTIZED_FUSED_CBR_BY_CHANNEL},
    {"Unpack", LAYER_UNPACK},
    {"Fill", LAYER_FILL},
    {"ResizeBicubic", LAYER_RESIZE_BICUBIC},
    {"FusedBatchNorm", LAYER_FUSED_BATCH_NORM},
    {"Unsqueeze", LAYER_UNSQUEEZE},
    {"Gru", LAYER_GRU},
    {"HardTanH", LAYER_HARDTANH},
    {"AdaptiveAvgPool2d", LAYER_ADAPTIVE_AVG_POOL},
    {"AdaptiveMaxPool2d", LAYER_ADAPTIVE_MAX_POOL},
    {"HDRGuide", LAYER_HDRGUIDE},
    {"BlobScale", LAYER_BLOB_SCALE},
    {"Reformat", LAYER_REFORMAT},
    {"Clip", LAYER_CLIP},
    {"HardSigmoid", LAYER_HARDSIGMOID},
    {"HardSwish", LAYER_HARDSWISH},
    {"Softplus", LAYER_SOFTPLUS},
    {"Div", LAYER_DIV},
    {"Sign", LAYER_SIGN},
    {"Cos", LAYER_COS},
    {"Acos", LAYER_ACOS},
    {"Sin", LAYER_SIN},
    {"Asin", LAYER_ASIN},
    {"Tan", LAYER_TAN},
    {"Atan", LAYER_ATAN},
    {"Log", LAYER_LOG},
    {"Reciprocal", LAYER_RECIPROCAL},
    {"Selu", LAYER_SELU},
    {"Floor", LAYER_FLOOR},
    {"Ceil", LAYER_CEIL},
    {"ReduceL1", LAYER_REDUCE_L1},
    {"ReduceL2", LAYER_REDUCE_L2},
    {"ReduceLogSum", LAYER_REDUCE_LOG_SUM},
    {"ReduceLogSumExp", LAYER_REDUCE_LOG_SUM_EXP},
    {"ReduceMin", LAYER_REDUCE_MIN},
    {"ReduceProd", LAYER_REDUCE_PROD},
    {"ReduceSum", LAYER_REDUCE_SUM},
    {"ReduceSumSquare", LAYER_REDUCE_SUM_SQUARE},
    // LAYER_INT8_RANGE
    // LAYER_TRT_ENGINE

    {"SignedMul", LAYER_SIGNED_MUL},
    {"DetectionPostProcess", LAYER_DETECTION_POST_PROCESS},
    {"SquaredDifference", LAYER_SQUARED_DIFFERENCE},
    {"ArgMaxOrMin", LAYER_ARG_MAX_OR_MIN},
    {"PixelShuffle", LAYER_PIXEL_SHUFFLE},
    {"Expand", LAYER_EXPAND},
    {"ScatterND", LAYER_SCATTER_ND},
    {"QuantizedSigmoid", LAYER_SIGMOID},
};

LayerType GlobalConvertLayerType(std::string layer_type_str) {
    if (global_layer_type_map.count(layer_type_str) > 0) {
        return global_layer_type_map[layer_type_str];
    } else {
        return LAYER_NOT_SUPPORT;
    }
}

}  // namespace TNN_NS
