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

#include "test/unit_test/unit_test_common.h"

#include <iostream>
#include <sstream>

#include "test/flags.h"
#include "test/test_utils.h"
#include "tnn/core/macro.h"
#include "tnn/interpreter/default_model_interpreter.h"
#include "tnn/utils/bfp16.h"

namespace TNN_NS {

IntScaleResource* CreateIntScale(int channel) {
    IntScaleResource* int8scale = new IntScaleResource();
    // scale
    RawBuffer scale(channel * sizeof(float));
    float* k_data = scale.force_to<float*>();
    InitRandom(k_data, channel, 0.f, 1.0f);
    for (int k = 0; k < channel; k++) {
        k_data[k] = std::fabs(k_data[k] - 0.f) < FLT_EPSILON ? 1.f : k_data[k];
    }
    int8scale->scale_handle = scale;
    // scale zero point
    RawBuffer zero_point(channel * sizeof(int8_t));
    zero_point.SetDataType(DATA_TYPE_INT8);
    int8scale->zero_point_handle = zero_point;

    // bias
    RawBuffer bias(channel * sizeof(int32_t));
    int32_t* b_data = bias.force_to<int32_t*>();
    InitRandom(b_data, channel, 32);
    int8scale->bias_handle = bias;
    return int8scale;
}

void SetUpEnvironment(AbstractDevice** cpu, AbstractDevice** device,
                       Context** cpu_context, Context** device_context) {
    NetworkConfig config;
    config.device_type = ConvertDeviceType(FLAGS_dt);
    config.enable_tune_kernel = FLAGS_et;
    if (FLAGS_lp.length() > 0) {
        config.library_path = {FLAGS_lp};
    }
    TNN_NS::Status ret = TNN_NS::TNN_OK;

    // cpu
    *cpu = GetDevice(DEVICE_NAIVE);
    ASSERT(*cpu != NULL);

    *cpu_context = (*cpu)->CreateContext(0);
    ASSERT(*cpu_context != NULL);

    // device
    *device = GetDevice(config.device_type);
    ASSERT(*device != NULL);

    *device_context = (*device)->CreateContext(config.device_id);
    ASSERT(*device_context != NULL);


    if (!FLAGS_ub) {
        (*device_context)->SetPrecision(PRECISION_HIGH);
    } else {
        (*device_context)->SetEnableTuneKernel(config.enable_tune_kernel);
    }

    ret = (*device_context)->LoadLibrary(config.library_path);
    ASSERT(ret == TNN_OK);
}

InputShapesMap GenerateInputShapeMap(std::vector<std::vector<int>>& input_vec) {
    InputShapesMap shape_map;
    for (int i = 0; i < input_vec.size(); ++i) {
        std::ostringstream ostr;
        ostr << "input" << i;
        shape_map[ostr.str()] = input_vec[i];
    }
    return shape_map;
}

InputDataTypeMap GenerateInputDataTypeMap(const std::vector<DataType>& input_dtype) {
    InputDataTypeMap dtype_map;
    for (int i = 0; i < input_dtype.size(); ++i) {
        std::ostringstream ostr;
        ostr << "input" << i;
        dtype_map[ostr.str()] = input_dtype[i];
    }
    return dtype_map;
}

std::shared_ptr<AbstractModelInterpreter> GenerateInterpreter(std::string layer_type_str,
                                                              std::vector<std::vector<int>> input_vec,
                                                              std::shared_ptr<LayerParam> param,
                                                              std::shared_ptr<LayerResource> resource,
                                                              int output_count,
                                                              std::vector<DataType> input_dtype) {
    auto interpreter = CreateModelInterpreter(MODEL_TYPE_TNN);
    if (!interpreter) {
        return nullptr;
    }
    DefaultModelInterpreter* default_interpreter = dynamic_cast<DefaultModelInterpreter*>(interpreter);
    if (!default_interpreter) {
        return nullptr;
    }

    NetStructure* net_structure = default_interpreter->GetNetStructure();
    NetResource* net_resource   = default_interpreter->GetNetResource();

    // generate net structure
    net_structure->inputs_shape_map    = GenerateInputShapeMap(input_vec);
    net_structure->input_data_type_map = GenerateInputDataTypeMap(input_dtype);

    std::shared_ptr<LayerInfo> layer_info = std::make_shared<LayerInfo>();
    layer_info->type                      = GlobalConvertLayerType(layer_type_str);
    layer_info->type_str                  = layer_type_str;
    layer_info->name                      = "layer_name";
    for (auto item : net_structure->inputs_shape_map) {
        layer_info->inputs.push_back(item.first);
        net_structure->blobs.insert(item.first);
    }
    for (int i = 0; i < output_count; ++i) {
        std::ostringstream ostr;
        ostr << "output" << i;
        layer_info->outputs.push_back(ostr.str());
        net_structure->outputs.insert(ostr.str());
        net_structure->blobs.insert(ostr.str());
    }
    layer_info->param = param;
    net_structure->layers.push_back(layer_info);

    // generate net resource
    if (nullptr != resource) {
        net_resource->resource_map["layer_name"] = resource;
    }

    return std::shared_ptr<AbstractModelInterpreter>(interpreter);
}

}  // namespace TNN_NS
