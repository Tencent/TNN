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

class StrideSliceLayerTest : public LayerTest,
                             public ::testing::WithParamInterface<std::tuple<int, int, int, int, int, int, int>> {};

INSTANTIATE_TEST_SUITE_P(LayerTest, StrideSliceLayerTest,
                         ::testing::Combine(BASIC_BATCH_CHANNEL_SIZE,
                                            // begins offset
                                            testing::Values(0, 1),
                                            // channel stride Values(1, 2),
                                            testing::Values(1, 2),
                                            // w,h stride Values(1, 2),
                                            testing::Values(1, 2),
                                            // ends offset Values(0, -1)
                                            testing::Values(0, -1)));

TEST_P(StrideSliceLayerTest, StrideSliceLayer) {
    // get param
    int batch          = std::get<0>(GetParam());
    int channel        = std::get<1>(GetParam());
    int input_size     = std::get<2>(GetParam());
    int begin_offset   = std::get<3>(GetParam());
    int channel_stride = std::get<4>(GetParam());
    int wh_stride      = std::get<5>(GetParam());
    int end_offset     = std::get<6>(GetParam());

    DeviceType dev = ConvertDeviceType(FLAGS_dt);

    // param
    std::shared_ptr<StrideSliceLayerParam> param(new StrideSliceLayerParam());
    param->name    = "StrideSlice";
    param->begins  = {begin_offset, begin_offset, begin_offset, 0};
    param->strides = {wh_stride, wh_stride, channel_stride, 1};
    param->ends    = {input_size + end_offset, input_size + end_offset, channel + end_offset, batch};

    for (int i = 0; i < param->begins.size(); ++i) {
        if (param->begins[i] >= param->ends[i]) {
            GTEST_SKIP();
        }
    }

    // generate interpreter
    std::vector<int> input_dims = {batch, channel, input_size, input_size};
    auto interpreter            = GenerateInterpreter("StridedSlice", {input_dims}, param);
    Run(interpreter);
}

}  // namespace TNN_NS
