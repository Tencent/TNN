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

#include "test/unit_test/layer_test/test_binary_layer.h"

namespace TNN_NS {

class AddLayerTest : public BinaryLayerTest {
public:
    AddLayerTest() : BinaryLayerTest(LAYER_ADD) {}
};

INSTANTIATE_TEST_SUITE_P(LayerTest, AddLayerTest,
                         ::testing::Combine(BASIC_BATCH_CHANNEL_SIZE,
                                            // input cnt
                                            testing::Values(1, 2),
                                            // param size type (1, channel, chw, hw)
                                            testing::Values(0, 1, 2, 3),
                                            // weight index
                                            testing::Values(-1, 0, 1),
                                            // data_type
                                            testing::Values(DATA_TYPE_FLOAT, DATA_TYPE_INT8)));

TEST_P(AddLayerTest, BinaryLayerTest) {
    int batch               = std::get<0>(GetParam());
    int channel             = std::get<1>(GetParam());
    int input_size          = std::get<2>(GetParam());
    int input_cnt           = std::get<3>(GetParam());
    int param_size_type     = std::get<4>(GetParam());
    int weight_index        = std::get<5>(GetParam());
    DataType blob_data_type = std::get<6>(GetParam());

    if (blob_data_type == DATA_TYPE_INT8) {
        // currently only single batch and non-broadcasting add is implemented
        if (batch != 1 || input_cnt != 2 || param_size_type != 2) {
            GTEST_SKIP();
        }
    }

    RunBinaryTest("Add");
}
}  // namespace TNN_NS
