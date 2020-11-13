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

class ConcatLayerTest : public LayerTest,
                        public ::testing::WithParamInterface<std::tuple<int, int, int, int, int, DataType>> {};

INSTANTIATE_TEST_SUITE_P(LayerTest, ConcatLayerTest,
                         ::testing::Combine(BASIC_BATCH_CHANNEL_SIZE,
                                            // axis
                                            testing::Values(1, 2, 3),
                                            // input cnt
                                            testing::Values(2, 3),
                                            // dtype
                                            testing::Values(DATA_TYPE_INT8, DATA_TYPE_FLOAT)));

TEST_P(ConcatLayerTest, ConcatLayer) {
    // get param
    int batch          = std::get<0>(GetParam());
    int channel        = std::get<1>(GetParam());
    int input_size     = std::get<2>(GetParam());
    int axis           = std::get<3>(GetParam());
    int input_count    = std::get<4>(GetParam());
    DataType data_type = std::get<5>(GetParam());
    DeviceType dev     = ConvertDeviceType(FLAGS_dt);

    if (data_type == DATA_TYPE_INT8 && DEVICE_ARM != dev) {
        GTEST_SKIP();
    }

    // param
    std::shared_ptr<ConcatLayerParam> param(new ConcatLayerParam());
    param->name = "Concat";
    param->axis = axis;

    // generate interpreter
    std::vector<std::vector<int>> input_dims_vec;
    for (int i = 0; i < input_count; ++i)
        input_dims_vec.push_back({batch, channel, input_size, input_size});
    auto interpreter = GenerateInterpreter("Concat", input_dims_vec, param);
    Run(interpreter);
}

}  // namespace TNN_NS
