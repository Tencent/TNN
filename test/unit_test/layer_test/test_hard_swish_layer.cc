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

class HardSwishLayerTest
    : public LayerTest,
      public ::testing::WithParamInterface<std::tuple<int, int, int, float, float, int, DataType>> {};

INSTANTIATE_TEST_SUITE_P(LayerTest, HardSwishLayerTest,
                         ::testing::Combine(
                            // batch
                            testing::Values(1),
                            // channel Values(1, 6, 8, 13),
                            testing::Values(1, 6, 8, 13),
                            // size Values(1, 6, 8, 13),
                            testing::Values(6),
                            // alpha Values(2, 1, 0.5),
                            testing::Values(2, 1, 0.5),
                            // beta Values(0, 2, 1.5, 3),
                            testing::Values(0, 2, 1.5, 3),
                            // input count
                            testing::Values(1, 2),
                            // data_type
                            testing::Values(DATA_TYPE_FLOAT)));

TEST_P(HardSwishLayerTest, HardSwishLayer) {
    // get param
    int batch       = std::get<0>(GetParam());
    int channel     = std::get<1>(GetParam());
    int input_size  = std::get<2>(GetParam());
    float alpha     = std::get<3>(GetParam());
    float beta      = std::get<4>(GetParam());
    int input_count = std::get<5>(GetParam());

    DataType data_type = std::get<6>(GetParam());
    DeviceType dev     = ConvertDeviceType(FLAGS_dt);

    // blob desc
    std::vector<BlobDesc> inputs_desc;
    BlobDesc input_desc;
    input_desc.dims.push_back(batch);
    input_desc.dims.push_back(channel);
    input_desc.dims.push_back(input_size);
    input_desc.dims.push_back(input_size);
    input_desc.device_type = DEVICE_NAIVE;
    input_desc.data_type   = data_type;
    for (int i = 0; i < input_count; ++i)
        inputs_desc.push_back(input_desc);

    std::vector<BlobDesc> outputs_desc;
    BlobDesc output_desc;
    output_desc.data_type   = data_type;
    output_desc.device_type = DEVICE_NAIVE;
    outputs_desc.push_back(output_desc);

    // param
    HardSwishLayerParam param;
    param.name  = "HardSwish";
    param.alpha = alpha;
    param.beta  = beta;

    Run(LAYER_HARDSWISH, &param, nullptr, inputs_desc, outputs_desc);
}

}  // namespace TNN_NS
