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

class DivLayerTest : public BinaryLayerTest {
public:
    DivLayerTest() : BinaryLayerTest(LAYER_DIV) {}
};

INSTANTIATE_TEST_SUITE_P(LayerTest, DivLayerTest,
                         ::testing::Combine(BASIC_BATCH_CHANNEL_SIZE,
                                            // input cnt
                                            testing::Values(1, 2),
                                            // param size type (1, channel, chw, hw)
                                            testing::Values(0, 1, 2, 3),
                                            // weight index
                                            testing::Values(-1, 0, 1),
                                            // data_type
                                            testing::Values(DATA_TYPE_FLOAT)));

TEST_P(DivLayerTest, BinaryLayerTest) {
    RunBinaryTest("Div", true);
}

}  // namespace TNN_NS
