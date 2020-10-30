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

    // bias
    RawBuffer bias(channel * sizeof(int32_t));
    int32_t* b_data = bias.force_to<int32_t*>();
    InitRandom(b_data, channel, 32);
    int8scale->bias_handle = bias;
    return int8scale;
}

void SetUpEnvironment(AbstractDevice** cpu, AbstractDevice** device, Context** cpu_context, Context** device_context) {
    NetworkConfig config;
    config.device_type = ConvertDeviceType(FLAGS_dt);
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

    ret = (*device_context)->LoadLibrary(config.library_path);
    ASSERT(ret == TNN_OK);
}

std::string GenerateHeadProto(std::vector<int> input_dims, int output_count) {
    std::ostringstream ostr;
    ostr << "\"1 124 1 4206624770,\"\n";
    ostr << "\"input";
    for (auto i : input_dims)
        ostr << " " << i;
    ostr << " ,\"\n";
    ostr << "\" input ";
    for (int i = 0; i < output_count; ++i) {
        ostr << "output" << i << " ";
    }
    ostr << ",\"\n";
    ostr << "\"";
    if (1 == output_count) {
        ostr << "output ";
    } else {
        for (int i = 0; i < output_count; ++i) {
            ostr << "output" << i << " ";
        }
    }
    ostr << ",\"\n";
    ostr << "\"1 ,\"\n";
    return ostr.str();
}

}  // namespace TNN_NS
