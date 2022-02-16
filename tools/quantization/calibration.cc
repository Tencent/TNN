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

#include "calibration.h"
#include <algorithm>
#include <cmath>
#include <random>
#include "file_reader.h"
#include "tnn/core/macro.h"
#include "tnn/core/tnn.h"
#include "tnn/interpreter/tnn/model_packer.h"
#include "tnn/interpreter/tnn/objseri.h"
#include "tnn/utils/dims_vector_utils.h"

namespace TNN_NS {

static const std::set<LayerType> kQuantizedLayerTypeStr = {LAYER_CONVOLUTION, LAYER_ADD, LAYER_CONCAT,
                                                           LAYER_INNER_PRODUCT};

static const std::set<LayerType> kBlobScaleMergeLayerTypeStr = {LAYER_RELU, LAYER_POOLING};

static void InitWeightScaleADMM(const float* weights, const int size, const int output_channel, bool merge_channel,
                                float* weight_scale, const int quantize_bits) {
    int weight_scale_count = merge_channel ? 1 : output_channel;
    const int s_size       = size / weight_scale_count;
    const int bound        = std::pow(2, quantize_bits - 1) - 1;

    for (int i = 0; i < weight_scale_count; i++) {
        float avg = 0;
        float max = 0;
        float val_abs;

        for (int j = 0; j < s_size; j++) {
            val_abs = std::fabs(weights[i * s_size + j]);
            avg += val_abs;
            if (val_abs > max) {
                max = val_abs;
            }
        }
        avg = avg / float(s_size);

        if (quantize_bits > 2) {
            weight_scale[i] = max / (bound * 1.25);
        } else {
            weight_scale[i] = avg;
        }
    }
}

static void UpdateQuantizedWeightsADMM(const float* weights, const int size, const int output_channel,
                                       bool merge_channel, float* weight_scale, const int quantize_bits,
                                       int8_t* quantized_weights) {
    int weight_scale_count = merge_channel ? 1 : output_channel;
    const int s_size       = size / weight_scale_count;
    const float bound      = std::pow(2, quantize_bits - 1) - 1;
    const float eps        = 1e-9f;
    float weight_quan;
    ASSERT(quantize_bits > 4);

    for (int i = 0; i < size; i++) {
        weight_quan          = weights[i] / (weight_scale[i / s_size] + eps);
        quantized_weights[i] = std::min(bound, std::max(-bound, std::roundf(weight_quan)));
    }
}

static void UpdateAlphaADMM(const float* weights, const int size, const int output_channel, bool merge_channel,
                            float* weight_scale, int8_t* quantized_weights) {
    int weight_scale_count = merge_channel ? 1 : output_channel;
    const int s_size       = size / weight_scale_count;
    const float eps        = 1e-9f;

    for (int i = 0; i < weight_scale_count; i++) {
        const int offset = i * s_size;
        float sum1       = 0;
        float sum2       = 0;

        for (int j = 0; j < s_size; j++) {
            sum1 += weights[offset + j] * quantized_weights[offset + j];
            sum2 += quantized_weights[offset + j] * quantized_weights[offset + j];
        }
        weight_scale[i] = sum1 / (sum2 + eps);
    }
}

Calibration::Calibration() {}

Calibration::~Calibration() {}

Status Calibration::Init(NetworkConfig& net_config, ModelConfig& model_config, InputShapesMap inputs_shape) {
    TNN tnn;
    Status status = tnn.Init(model_config);
    if (status != TNN_OK) {
        LOGE("tnn init failed!\n");
        return TNNERR_INVALID_MODEL;
    }
    instance_ = tnn.CreateInst(net_config, status);
    if (status != TNN_OK) {
        LOGE("tnn create instance failed!\n");
        return TNNERR_INST_ERR;
    }

    interpreter_ = std::dynamic_pointer_cast<DefaultModelInterpreter>(instance_->GetInterpreter());
    if (interpreter_ == nullptr) {
        LOGE("instance init failed!\n");
        return TNNERR_INST_ERR;
    }

    return TNN_OK;
}

int Calibration::SetCalibrationParams(CalibrationParam params) {
    cali_params_ = params;

    if (cali_params_.blob_quantize_method == ADMM) {
        LOGE("Not support ADMM in quantizing blobs!\n");
        cali_params_.blob_quantize_method = MIN_MAX;
        return -1;
    }

    if (cali_params_.weights_quantize_method == KL_DIVERGENCE || cali_params_.weights_quantize_method == ACIQ_GAUS ||
        cali_params_.weights_quantize_method == ACIQ_LAPLACE) {
        LOGE("Not support KL_DIVERGENCE or ACIQ methond in quantizing weights!\n");
        cali_params_.weights_quantize_method = MIN_MAX;
        return -1;
    }

    return 0;
}

Status Calibration::RunCalibration(DataSet& dataset) {
    // Compute Feature Scale
    int ret = CalBlobScale(dataset);
    if (ret != 0) {
        LOGE("calcluate blob scale failed!\n");
        return TNNERR_QUANTIZE_ERROR;
    }

    // Quantize params
    ret = QuantizeParams();
    if (ret != 0) {
        LOGE("quantize params failed!\n");
        return TNNERR_QUANTIZE_ERROR;
    }

    // Merge Blob Scale of some layers
    ret = MergeBlobScale();
    if (ret != 0) {
        LOGE("merge blob scale failed!\n");
        return TNNERR_QUANTIZE_ERROR;
    }

    return TNN_OK;
}

Status Calibration::Serialize(std::string proto_path, std::string model_path) {
    NetStructure* net_struct  = interpreter_->GetNetStructure();
    NetResource* net_resource = interpreter_->GetNetResource();
    if (net_struct == nullptr || net_resource == nullptr) {
        LOGE("net struct or net resource is null\n");
        return TNNERR_INVALID_MODEL;
    }

    TNN_NS::ModelPacker packer(net_struct, net_resource);

    Status status = packer.Pack(proto_path, model_path);
    if (status != TNN_OK) {
        LOGE("pack the model failed!\n");
        return status;
    }

    return TNN_OK;
}

int Calibration::CalBlobScale(DataSet& dataset) {
    printf("Start to calculate blob scale ...\n");
    NetResource* net_resource = interpreter_->GetNetResource();

    Status status = instance_->Reshape(dataset.input_shape);
    if (status != TNN_OK) {
        LOGE("instance reshape failed!\n");
        return -1;
    }

    // Init Feature map
    int ret = InitFeatureMap();
    if (ret != 0) {
        LOGE("init feautre map for quantize failed!\n");
        return ret;
    }
    printf("\tInit Feature Map done!\n");

    // Collect the Range of Feature map
    ret = UpdateBlobRange(dataset);
    if (ret != 0) {
        LOGE("collect feautre map range failed!\n");
        return ret;
    }
    printf("\tCollect Blob Range done!\n");

    // Calculate Distribute of Feature map
    ret = UpdateBlobDistribute(dataset);
    if (ret != 0) {
        LOGE("update feautre map distribute failed!\n");
        return ret;
    }
    printf("\tCollect Blob Distribution done!\n");

    // Compute Scale of Feature map and save to resource map
    for (auto& item : feature_map_) {
        std::vector<float> scale_vec;
        std::vector<int8_t> zero_point_vec;

        std::string input_scale_name = item.first->GetBlobDesc().name + BLOB_SCALE_SUFFIX;
        int ret = item.second->CalculateScale(scale_vec, zero_point_vec);
        if (ret != 0) {
            LOGE("CalculateScale (%s) failed\n", input_scale_name.c_str());
            return ret;
        }
        LayerResource* blob_scale_res;
        blob_scale_res = CreateIntScale(scale_vec, zero_point_vec);
        net_resource->resource_map[input_scale_name] = std::shared_ptr<LayerResource>(blob_scale_res);
        printf("\t====> Calculate (%s) done!\n", input_scale_name.c_str());
    }

    return 0;
}

int Calibration::InitFeatureMap() {
    feature_map_.clear();

    BlobStatisticCallback func = [&](std::vector<Blob*>& blobs, LayerInfo* info) {
        LayerType layer_type = info->type;
        if (kQuantizedLayerTypeStr.find(layer_type) != kQuantizedLayerTypeStr.end() ||
            kBlobScaleMergeLayerTypeStr.find(layer_type) != kBlobScaleMergeLayerTypeStr.end()) {
            for (auto blob : blobs) {
                if (feature_map_.find(blob) == feature_map_.end()) {
                    std::shared_ptr<ScaleCalculator> scale_cal(new ScaleCalculator());
                    if (scale_cal->Init(blob, cali_params_.merge_blob_channel, cali_params_.blob_quantize_method) ==
                        0) {
                        feature_map_[blob] = scale_cal;
                    }
                }

                // set FC layer input and ouput blob to merge channel
                if (layer_type == LAYER_INNER_PRODUCT) {
                    if (feature_map_.find(blob) != feature_map_.end()) {
                        feature_map_[blob]->SetMergeChannel(true);
                    }
                }
            }
        }
    };

    instance_->ForwardWithCallback(func, func);

    // set input blob quantize method to MIN_MAX
    BlobMap input_blobs;
    Status status = instance_->GetAllInputBlobs(input_blobs);
    if (status != TNN_OK) {
        LOGE("instance get input blobs failed!\n");
        return -1;
    }
    for (auto item : input_blobs) {
        if (feature_map_.find(item.second) != feature_map_.end()) {
            feature_map_[item.second]->SetQuantizeMethod(MIN_MAX);
        }
    }

    return 0;
}

int Calibration::UpdateBlobRange(DataSet& dataset) {
    BlobMap input_blobs;
    Status status = instance_->GetAllInputBlobs(input_blobs);
    if (status != TNN_OK) {
        LOGE("instance get input blobs failed!\n");
        return -1;
    }
    Blob* input_blob = input_blobs.begin()->second;

    BlobStatisticCallback func = [&](std::vector<Blob*>& blobs, LayerInfo* info) {
        for (auto blob : blobs) {
            if (feature_map_.find(blob) != feature_map_.end()) {
                feature_map_[blob]->UpdateRange();
            }
        }
    };

    FileReader file_reader;
    file_reader.SetBiasValue(cali_params_.input_bias);
    file_reader.SetScaleValue(cali_params_.input_scale);
    file_reader.SetReverseChannel(cali_params_.reverse_channel);
    for (auto file_pack : dataset.file_list) {
        for (auto item : feature_map_) {
            item.second->ClearRangeFlag();
        }

        status = file_reader.Read(input_blob, file_pack.first, file_pack.second);
        if (status != TNN_OK) {
            LOGE("read input file (%s) failed!\n", file_pack.first.c_str());
            continue;
        }
        instance_->ForwardWithCallback(func, func);
    }

    return 0;
}

int Calibration::UpdateBlobDistribute(DataSet& dataset) {
    for (auto& item : feature_map_) {
        item.second->ResetDistribute();
    }

    BlobMap input_blobs;
    Status status = instance_->GetAllInputBlobs(input_blobs);
    if (status != TNN_OK) {
        LOGE("instance get input blobs failed!\n");
        return -1;
    }
    Blob* input_blob = input_blobs.begin()->second;

    BlobStatisticCallback func = [&](std::vector<Blob*>& blobs, LayerInfo* info) {
        for (auto blob : blobs) {
            if (feature_map_.find(blob) != feature_map_.end()) {
                feature_map_[blob]->UpdateDistribute();
            }
        }
    };

    FileReader file_reader;
    file_reader.SetBiasValue(cali_params_.input_bias);
    file_reader.SetScaleValue(cali_params_.input_scale);
    file_reader.SetReverseChannel(cali_params_.reverse_channel);
    for (auto file_pack : dataset.file_list) {
        for (auto& item : feature_map_) {
            item.second->ClearDistributeFlag();
        }

        status = file_reader.Read(input_blob, file_pack.first, file_pack.second);
        if (status != TNN_OK) {
            LOGE("read input file (%s) failed!\n", file_pack.first.c_str());
            continue;
        }
        instance_->ForwardWithCallback(func, func);
    }

    return 0;
}

IntScaleResource* Calibration::CreateIntScale(std::vector<float> scale_vec, std::vector<int8_t> zero_point_vec) {
    IntScaleResource* int8scale = new IntScaleResource();
    // scale
    RawBuffer scale(scale_vec.size() * sizeof(float));
    float* k_data = scale.force_to<float*>();
    memcpy(k_data, scale_vec.data(), scale_vec.size() * sizeof(float));
    int8scale->scale_handle = scale;

    // zero_point
    RawBuffer zero_point(zero_point_vec.size() * sizeof(char));
    zero_point.SetDataType(DATA_TYPE_INT8);
    int8_t* sb_data = zero_point.force_to<int8_t*>();
    memcpy(sb_data, zero_point_vec.data(), zero_point_vec.size() * sizeof(char));
    int8scale->zero_point_handle = zero_point;

    // bias
    RawBuffer bias(scale_vec.size() * sizeof(int32_t));
    bias.SetDataType(DATA_TYPE_INT32);
    int32_t* b_data = bias.force_to<int32_t*>();
    memset(b_data, 0, scale_vec.size() * sizeof(int32_t));
    int8scale->bias_handle = bias;
    return int8scale;
}

IntScaleResource* Calibration::CreateIntScale(std::vector<float> scale_vec) {
    IntScaleResource* int8scale = new IntScaleResource();
    // scale
    RawBuffer scale(scale_vec.size() * sizeof(float));
    float* k_data = scale.force_to<float*>();
    memcpy(k_data, scale_vec.data(), scale_vec.size() * sizeof(float));
    int8scale->scale_handle = scale;

    // bias
    RawBuffer bias(scale_vec.size() * sizeof(int32_t));
    bias.SetDataType(DATA_TYPE_INT32);
    int32_t* b_data = bias.force_to<int32_t*>();
    memset(b_data, 0, scale_vec.size() * sizeof(int32_t));
    int8scale->bias_handle = bias;
    return int8scale;
}

int Calibration::QuantizeParams() {
    printf("Start to Quantize Parameters ...\n");
    NetStructure* net_struct  = interpreter_->GetNetStructure();
    NetResource* net_resource = interpreter_->GetNetResource();

    for (auto& item : net_struct->layers) {
        LayerType layer_type = item->type;

        // skip constant layer
        auto const_layers = net_resource->constant_layers;
        if (const_layers.find(item->name) != const_layers.end()) {
            continue;
        }

        if (kQuantizedLayerTypeStr.find(layer_type) != kQuantizedLayerTypeStr.end()) {
            // assign NetStructure
            item->param->quantized = true;

            // assign NetResource
            if (layer_type == LAYER_CONVOLUTION) {
                printf("\tQuantize Convolution parameters...\n");
                if (net_resource->resource_map.find(item->name) == net_resource->resource_map.end()) {
                    LOGE("Convolution resource not found (name: %s)", item->name.c_str());
                    return -1;
                }

                ConvLayerResource* conv_res =
                    dynamic_cast<ConvLayerResource*>(net_resource->resource_map[item->name].get());
                ConvLayerParam* conv_param        = dynamic_cast<ConvLayerParam*>(item->param.get());
                std::string input_blob_scale_name = item->inputs[0] + BLOB_SCALE_SUFFIX;
                if (net_resource->resource_map.find(input_blob_scale_name) == net_resource->resource_map.end()) {
                    LOGE("Blob Scale resource not found (name: %s)", input_blob_scale_name.c_str());
                    return -1;
                }
                IntScaleResource* blob_scale =
                    dynamic_cast<IntScaleResource*>(net_resource->resource_map[input_blob_scale_name].get());
                int ret = QuantizeConvParams(conv_res, conv_param, blob_scale);
                if (ret != 0) {
                    LOGE(
                        "Quantize convolution weights failed! (layer name: "
                        "%s)\n",
                        item->name.c_str());
                    return -1;
                }
                printf("\t====> done!\n");

            } else if (layer_type == LAYER_INNER_PRODUCT) {
                printf("\tQuantize InnerProduct parameters...\n");
                if (net_resource->resource_map.find(item->name) == net_resource->resource_map.end()) {
                    LOGE("InnerProduct resource not found (name: %s)", item->name.c_str());
                    return -1;
                }

                InnerProductLayerResource* fc_res =
                    dynamic_cast<InnerProductLayerResource*>(net_resource->resource_map[item->name].get());
                InnerProductLayerParam* fc_param  = dynamic_cast<InnerProductLayerParam*>(item->param.get());
                std::string input_blob_scale_name = item->inputs[0] + BLOB_SCALE_SUFFIX;
                if (net_resource->resource_map.find(input_blob_scale_name) == net_resource->resource_map.end()) {
                    LOGE("Blob Scale resource not found (name: %s)", input_blob_scale_name.c_str());
                    return -1;
                }
                IntScaleResource* blob_scale =
                    dynamic_cast<IntScaleResource*>(net_resource->resource_map[input_blob_scale_name].get());
                int ret = QuantizeFcParams(fc_res, fc_param, blob_scale);
                if (ret != 0) {
                    LOGE(
                        "Quantize InnerProduct weights failed! (layer name: "
                        "%s)\n",
                        item->name.c_str());
                    return -1;
                }
                printf("\t====> done!\n");
            } else if (layer_type == LAYER_ADD) {
                // if one of the input of add layer is in layer resource, then this layer will not be quantized
                if (net_resource->resource_map.find(item->name) != net_resource->resource_map.end()) {
                    auto layer_resource = net_resource->resource_map[item->name].get();
                    auto layer_res      = dynamic_cast<EltwiseLayerResource*>(layer_resource);
                    if (layer_res != nullptr) {
                        item->param->quantized = false;
                    }
                }
            }
        }
    }

    return 0;
}

int Calibration::QuantizeConvParams(ConvLayerResource* resource, ConvLayerParam* param, IntScaleResource* input_scale) {
    int group          = param->group;
    int output_channel = param->output_channel;
    int kernel_size    = DimsVectorUtils::Count(param->kernels);
    int size           = resource->filter_handle.GetDataCount();
    if (size % (kernel_size * output_channel) != 0) {
        LOGE("invalid weight size!\n");
        return -1;
    }
    if (output_channel % group != 0) {
        LOGE("invalid conv param (output channel, group)!\n");
        return -1;
    }
    int input_channel_per_group  = size / output_channel / kernel_size;
    int output_channel_per_group = output_channel / group;
    int oc_stride                = input_channel_per_group * kernel_size;

    std::vector<float> weight_multiby_inputscale(size);
    bool merge_channel = false;
    if (input_scale->scale_handle.GetDataCount() == 1)
        merge_channel = true;

    bool is_depthwise = (output_channel_per_group == 1 && input_channel_per_group == 1);

    // multi weights by input_scale
    float* input_scale_data = input_scale->scale_handle.force_to<float*>();
    auto filter_handle      = resource->filter_handle;
    if (resource->filter_handle.GetDataType() == DATA_TYPE_HALF) {
        LOGI("Fp16 model is used to quantize, precision may be lower than fp32 model!");
        filter_handle = ConvertHalfHandle(filter_handle);
    }
    float* weight_data = filter_handle.force_to<float*>();
    for (int group_idx = 0; group_idx < group; group_idx++) {
        for (int oc = 0; oc < output_channel_per_group; ++oc) {
            for (int ic = 0; ic < input_channel_per_group; ++ic) {
                int s_idx = ic + group_idx * input_channel_per_group;
                for (int i = 0; i < kernel_size; ++i) {
                    int idx = (group_idx * output_channel_per_group + oc) * oc_stride + ic * kernel_size + i;
                    if (merge_channel)
                        s_idx = 0;
                    if (is_depthwise) {
                        weight_multiby_inputscale[idx] = weight_data[idx];
                    } else {
                        weight_multiby_inputscale[idx] = weight_data[idx] * input_scale_data[s_idx];
                    }
                }
            }
        }
    }

    // quantize weights
    RawBuffer weight_quantized(size * sizeof(char));
    weight_quantized.SetDataType(DATA_TYPE_INT8);
    int weight_scale_size = output_channel;
    if (cali_params_.merge_weights_channel)
        weight_scale_size = 1;
    RawBuffer weight_scale(weight_scale_size * sizeof(float));
    RawBuffer weight_zero_point(weight_scale_size * sizeof(char));
    weight_zero_point.SetDataType(DATA_TYPE_INT8);

    float* weight_scale_data      = weight_scale.force_to<float*>();
    int8_t* weight_zero_point_data = weight_zero_point.force_to<int8_t*>();
    int8_t* weight_quantized_data = weight_quantized.force_to<int8_t*>();

    int ret                       = CalQuantizedWeights(weight_multiby_inputscale.data(), size, output_channel,
                                  cali_params_.merge_weights_channel, weight_quantized_data, weight_scale_data, weight_zero_point_data);
    if (ret != 0) {
        LOGE("Calculate quantized weights failed!\n");
        return ret;
    }

    // for depthwise conv, need to mul weight_scale by input_scale
    if (is_depthwise) {
        for (int i = 0; i < weight_scale_size; ++i) {
            int s_idx = i;
            if (merge_channel)
                s_idx = 0;
            weight_scale_data[i] = weight_scale_data[i] * input_scale_data[s_idx];
        }
    }

    resource->filter_handle = weight_quantized;
    resource->scale_handle  = weight_scale;
    resource->zero_point_handle  = weight_zero_point;

    // quantize bias
    if (param->bias) {
        auto fp32_bias_handle = ConvertHalfHandle(resource->bias_handle);
        float* bias_data      = fp32_bias_handle.force_to<float*>();
        RawBuffer bias_quantized(output_channel * sizeof(int32_t));
        bias_quantized.SetDataType(DATA_TYPE_INT32);
        int32_t* bias_quantized_data = bias_quantized.force_to<int32_t*>();

        for (int oc = 0; oc < output_channel; ++oc) {
            if (weight_scale_data[oc] == 0) {
                bias_quantized_data[oc] = 0;
            } else {
                int weight_scale_idx = oc;
                if (cali_params_.merge_weights_channel)
                    weight_scale_idx = 0;
                bias_quantized_data[oc] = static_cast<int32_t>(bias_data[oc] / weight_scale_data[weight_scale_idx]);
            }
        }

        resource->bias_handle = bias_quantized;
    }

    return 0;
}

int Calibration::QuantizeFcParams(InnerProductLayerResource* resource, InnerProductLayerParam* param,
                                  IntScaleResource* input_scale) {
    int output_channel = param->num_output;
    int size           = resource->weight_handle.GetDataCount();
    if (size % output_channel != 0) {
        LOGE("invalid weight size!\n");
        return -1;
    }
    int oc_stride = size / output_channel;

    std::vector<float> weight_multiby_inputscale(size);
    if (input_scale->scale_handle.GetDataCount() != 1) {
        LOGE("invalid scale size!\n");
        return -1;
    }

    // multi weights by input_scale
    float* input_scale_data = input_scale->scale_handle.force_to<float*>();
    auto weight_handle      = resource->weight_handle;
    if (resource->weight_handle.GetDataType() == DATA_TYPE_HALF) {
        LOGI("Fp16 model is used to quantize, precision may be lower than fp32 model!");
        weight_handle = ConvertHalfHandle(weight_handle);
    }
    float* weight_data = weight_handle.force_to<float*>();
    for (int i = 0; i < size; ++i) {
        weight_multiby_inputscale[i] = weight_data[i] * input_scale_data[0];
    }

    // quantize weights
    RawBuffer weight_quantized(size * sizeof(char));
    weight_quantized.SetDataType(DATA_TYPE_INT8);
    int weight_scale_size = output_channel;
    if (cali_params_.merge_weights_channel)
        weight_scale_size = 1;
    RawBuffer weight_scale(weight_scale_size * sizeof(float));
    RawBuffer weight_zero_point(weight_scale_size * sizeof(char));
    weight_zero_point.SetDataType(DATA_TYPE_INT8);

    float* weight_scale_data      = weight_scale.force_to<float*>();
    int8_t* weight_zero_point_data = weight_zero_point.force_to<int8_t*>();
    int8_t* weight_quantized_data = weight_quantized.force_to<int8_t*>();
    int ret                       = CalQuantizedWeights(weight_multiby_inputscale.data(), size, output_channel,
                                  cali_params_.merge_weights_channel, weight_quantized_data, weight_scale_data, weight_zero_point_data);
    if (ret != 0) {
        LOGE("Calculate quantized weights failed!\n");
        return ret;
    }

    resource->weight_handle = weight_quantized;
    resource->scale_handle  = weight_scale;
    resource->zero_point_handle  = weight_zero_point;

    // quantize bias
    if (param->has_bias) {
        auto fp32_bias_handle = ConvertHalfHandle(resource->bias_handle);
        float* bias_data      = fp32_bias_handle.force_to<float*>();
        RawBuffer bias_quantized(output_channel * sizeof(int32_t));
        bias_quantized.SetDataType(DATA_TYPE_INT32);
        int32_t* bias_quantized_data = bias_quantized.force_to<int32_t*>();

        for (int oc = 0; oc < output_channel; ++oc) {
            if (weight_scale_data[oc] == 0) {
                bias_quantized_data[oc] = 0;
            } else {
                int weight_scale_idx = oc;
                if (cali_params_.merge_weights_channel)
                    weight_scale_idx = 0;
                bias_quantized_data[oc] = static_cast<int32_t>(bias_data[oc] / weight_scale_data[weight_scale_idx]);
            }
        }

        resource->bias_handle = bias_quantized;
    }

    return 0;
}

int Calibration::CalQuantizedWeights(const float* weights, const int size, const int output_channel, bool merge_channel,
                                     int8_t* quantized_weights, float* weight_scale, int8_t* weight_zero_point) {
    ASSERT(size % output_channel == 0);

    if (cali_params_.weights_quantize_method == MIN_MAX) {
        // MIN_MAX
        int weight_scale_count = merge_channel ? 1 : output_channel;
        int s_size             = size / weight_scale_count;
        for (int s_idx = 0; s_idx < weight_scale_count; ++s_idx) {
            const float* weight_start = weights + s_idx * s_size;
            int8_t* weight_q_start    = quantized_weights + s_idx * s_size;
            auto minmax               = std::minmax_element(weight_start, weight_start + s_size);
            float max_val_abs         = std::max(std::abs(*minmax.first), std::abs(*minmax.second));

            weight_scale[s_idx]    = max_val_abs / 127.0f;
            float scale_float2int8 = 1.0f;
            if (max_val_abs != 0)
                scale_float2int8 = 1 / weight_scale[s_idx];

            // quantize weights
            for (int i = 0; i < s_size; ++i) {
                int value         = static_cast<int>(std::round(weight_start[i] * scale_float2int8));
                weight_q_start[i] = std::min(127, std::max(-127, value));
            }
        }
    } else if (cali_params_.weights_quantize_method == ADMM) {
        // ADMM
        int weight_scale_count  = merge_channel ? 1 : output_channel;
        int s_size              = size / weight_scale_count;
        const int quantize_bits = 8;

        InitWeightScaleADMM(weights, size, output_channel, merge_channel, weight_scale, quantize_bits);

        int iter           = 0;
        float pre_sum      = 0;
        float cur_sum      = 0;
        const int max_iter = 1000;

        for (int i = 0; i < size; i++) {
            pre_sum += std::fabs(weights[i]);
        }
        // update weights quan
        while (iter < max_iter) {
            UpdateQuantizedWeightsADMM(weights, size, output_channel, merge_channel, weight_scale, quantize_bits,
                                       quantized_weights);
            UpdateAlphaADMM(weights, size, output_channel, merge_channel, weight_scale, quantized_weights);
            iter++;
        }

        for (int i = 0; i < size; i++) {
            cur_sum += std::fabs(quantized_weights[i] * weight_scale[i / s_size]);
        }
        // LOGD("iter: %d  with diff %f\n", iter, pre_sum - cur_sum);
    } else if (cali_params_.weights_quantize_method == ASY_MIN_MAX) {
        // ASY_MIN_MAX
        int weight_scale_count  = merge_channel ? 1 : output_channel;
        int s_size              = size / weight_scale_count;
        for (int s_idx = 0; s_idx < weight_scale_count; ++s_idx) {
            const float* weight_start = weights + s_idx * s_size;
            int8_t* weight_q_start    = quantized_weights + s_idx * s_size;
            auto minmax               = std::minmax_element(weight_start, weight_start + s_size);
            float weight_min = std::min(.0f, *minmax.first);
            float weight_max = std::max(.0f, *minmax.second);

            weight_scale[s_idx] = (weight_max - weight_min) / 254.0f;
            float scale_float2int8 = 1.0f;
            if (weight_max != weight_min){
                scale_float2int8 = 1 / weight_scale[s_idx];
            }else{
                LOGE("Single constant input is not supported\n");
                return -1;
            }
            int8_t bias = 127 - static_cast<int>(std::round(weight_max * scale_float2int8));
            weight_zero_point[s_idx] = bias;
            // quantize weights
            for (int i = 0; i < s_size; ++i) {
                int value        = static_cast<int>(std::round(weight_start[i] * scale_float2int8)) + bias;
                weight_q_start[i] = std::min(127, std::max(-127, value));
            }
        }
    }
    else {
        LOGE("Not support yet (method: %d) for quantize weights", cali_params_.weights_quantize_method);
        return -1;
    }

    return 0;
}

int Calibration::MergeBlobScale() {
    printf("Start to Merge Blob Scale ...\n");
    NetStructure* net_struct  = interpreter_->GetNetStructure();
    NetResource* net_resource = interpreter_->GetNetResource();

    for (auto& item : net_struct->layers) {
        MergeBlobScaleRecursion(item.get(), net_struct, net_resource);
    }

    return 0;
}

void Calibration::MergeBlobScaleRecursion(LayerInfo* layer_info, NetStructure* net_struct, NetResource* net_resource) {
    LayerType layer_type = layer_info->type;
    // Skip average pooling
    if (layer_type == LAYER_POOLING) {
        auto param = dynamic_cast<PoolingLayerParam*>(layer_info->param.get());
        if (param->pool_type == 1) {
            return;
        }
    }
    if (kBlobScaleMergeLayerTypeStr.find(layer_type) != kBlobScaleMergeLayerTypeStr.end()) {
        ASSERT(layer_info->inputs.size() == 1 && layer_info->outputs.size() == 1)
        LayerInfo* pre_layer_info = GetLayerInfoFromOutpubBlobName(layer_info->inputs[0], net_struct);
        if (pre_layer_info != nullptr && pre_layer_info->param->quantized) {
            // merge blob scale
            std::string input_scale_name  = layer_info->inputs[0] + BLOB_SCALE_SUFFIX;
            std::string output_scale_name = layer_info->outputs[0] + +BLOB_SCALE_SUFFIX;
            if (net_resource->resource_map.find(input_scale_name) != net_resource->resource_map.end() &&
                net_resource->resource_map.find(output_scale_name) != net_resource->resource_map.end()) {
                net_resource->resource_map[input_scale_name] = net_resource->resource_map[output_scale_name];
                layer_info->param->quantized                 = true;
            }

            MergeBlobScaleRecursion(pre_layer_info, net_struct, net_resource);
        }
    }
}

LayerInfo* Calibration::GetLayerInfoFromOutpubBlobName(std::string blob_name, NetStructure* net_struct) {
    LayerInfo* layer_info = nullptr;
    for (auto item : net_struct->layers) {
        for (auto name : item->outputs) {
            if (name == blob_name) {
                layer_info = item.get();
                return layer_info;
            }
        }
    }

    return layer_info;
}

}  // namespace TNN_NS
