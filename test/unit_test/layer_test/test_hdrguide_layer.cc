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

#include "test/unit_test/layer_test/layer_test.h"
#include "test/unit_test/unit_test_common.h"
#include "test/unit_test/utils/network_helpers.h"
#include "tnn/utils/dims_vector_utils.h"

namespace TNN_NS {

class HdrGuideLayerTest : public LayerTest, public ::testing::WithParamInterface<std::tuple<int, int, int>> {};

INSTANTIATE_TEST_SUITE_P(LayerTest, HdrGuideLayerTest, ::testing::Combine(BASIC_BATCH_CHANNEL_SIZE));

TEST_P(HdrGuideLayerTest, HdrGuideLayer) {
    // get param
    int batch      = std::get<0>(GetParam());
    int channel    = std::get<1>(GetParam());
    int input_size = std::get<2>(GetParam());
    DeviceType dev = ConvertDeviceType(FLAGS_dt);

    if (channel != 3) {
        GTEST_SKIP();
    }

    if (DEVICE_METAL != dev && DEVICE_OPENCL != dev) {
        GTEST_SKIP();
    }

    // blob desc
    auto inputs_desc  = CreateInputBlobsDesc(batch, channel, input_size, 1, DATA_TYPE_FLOAT);
    auto outputs_desc = CreateOutputBlobsDesc(1, DATA_TYPE_FLOAT);

    // param
    LayerParam param;
    param.name = "HDRGuide";

    // resource
    HdrGuideLayerResource resource;

    int size = 9;
    RawBuffer ccm_weight(size * sizeof(float));
    float* ccm_weight_data = ccm_weight.force_to<float*>();
    InitRandom(ccm_weight_data, size, 1.0f);
    resource.ccm_weight_handle = ccm_weight;

    size = 3;
    RawBuffer ccm_bias(size * sizeof(float));
    float* ccm_bias_data = ccm_bias.force_to<float*>();
    InitRandom(ccm_bias_data, size, 1.0f);
    resource.ccm_bias_handle = ccm_bias;

    size = 12;
    RawBuffer shifts(size * sizeof(float));
    float* shifts_data = shifts.force_to<float*>();
    InitRandom(shifts_data, size, 1.0f);
    resource.shifts_handle = shifts;

    size = 12;
    RawBuffer slopes(size * sizeof(float));
    float* slopes_data = slopes.force_to<float*>();
    InitRandom(slopes_data, size, 1.0f);
    resource.slopes_handle = slopes;

    size = 3;
    RawBuffer p_weight(size * sizeof(float));
    float* p_weight_data = p_weight.force_to<float*>();
    InitRandom(p_weight_data, size, 1.0f);
    resource.projection_weight_handle = p_weight;

    size = 1;
    RawBuffer p_bias(size * sizeof(float));
    float* p_bias_data = p_bias.force_to<float*>();
    InitRandom(p_bias_data, size, 1.0f);
    resource.projection_bias_handle = p_bias;

    Run(LAYER_HDRGUIDE, &param, &resource, inputs_desc, outputs_desc);
}

}  // namespace TNN_NS
