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

class ReorgLayerTest : public LayerTest,
                       public ::testing::WithParamInterface<std::tuple<int, int, int, int, bool, int>> {};

INSTANTIATE_TEST_SUITE_P(LayerTest, ReorgLayerTest,
                         ::testing::Combine(testing::Values(1, 2), testing::Values(36, 72),
                                            testing::Values(6, 12, 18, 36),
                                            // stride
                                            testing::Values(2, 3),
                                            // reverse
                                            testing::Values(true, false), testing::Values(0, 1)));

TEST_P(ReorgLayerTest, ReorgLayer) {
    // get param
    int batch      = std::get<0>(GetParam());
    int channel    = std::get<1>(GetParam());
    int input_size = std::get<2>(GetParam());
    int stride     = std::get<3>(GetParam());
    bool forward   = std::get<4>(GetParam());
    int mode       = std::get<5>(GetParam());  // 0 : DCR, 1: CRD
    DeviceType dev = ConvertDeviceType(FLAGS_dt);

    if (DEVICE_HUAWEI_NPU == dev) {
        GTEST_SKIP();
    }

    if (mode == 1 && forward == 0) {
        // illegal case
        GTEST_SKIP();
    }

    // param
    std::shared_ptr<ReorgLayerParam> param(new ReorgLayerParam());
    param->name    = "Reorg";
    param->stride  = stride;
    param->forward = forward;
    param->mode    = mode;

    // generate interpreter
    std::vector<int> input_dims = {batch, channel, input_size, input_size};
    auto interpreter            = GenerateInterpreter("Reorg", {input_dims}, param);
    Run(interpreter);
}

}  // namespace TNN_NS
