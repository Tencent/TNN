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

static bool TestFilter(DeviceType device_type, DataType data_type) {
    if (device_type == DEVICE_NAIVE)
        return true;
    
    if (device_type == DEVICE_METAL && data_type == DATA_TYPE_FLOAT)
        return true;
    
    return false;
}

class LSTMLayerTest : public LayerTest,
                              public ::testing::WithParamInterface<std::tuple<int, int, int, int, int, DataType>> {};
// seq_len, batch, input, output
// direction: 0, 1, 2
INSTANTIATE_TEST_SUITE_P(LayerTest, LSTMLayerTest,
                         ::testing::Combine(testing::Values(1, 4, 16),  // seq_len
                                            testing::Values(1, 2, 4),   // batch_size
                                            testing::Values(1, 8, 32),  // input_size
                                            testing::Values(1, 16, 32), // hidden_size
                                            testing::Values(0, 1, 2),   // direction, 0:forward, 1:backward, 2:bi-direction
                                            testing::Values(DATA_TYPE_FLOAT, DATA_TYPE_HALF)));

TEST_P(LSTMLayerTest, LSTMONNXLayer) {
    // get param
    int seq_len        = std::get<0>(GetParam());
    int batch          = std::get<1>(GetParam());
    int input_size     = std::get<2>(GetParam());
    int output_size    = std::get<3>(GetParam());
    int direction      = std::get<4>(GetParam());
    DataType dtype     = std::get<5>(GetParam());
    DeviceType dev     = ConvertDeviceType(FLAGS_dt);
    if (!TestFilter(dev, dtype)) {
        GTEST_SKIP();
    }

    // param
    std::shared_ptr<LSTMONNXLayerParam> param(new LSTMONNXLayerParam());
    param->name        = "LSTMONNX";
    param->hidden_size = output_size;
    param->direction   = direction;

    // generate interpreter
    const int num_directions = param->direction==2? 2: 1;
    std::vector<int> input_dims = {seq_len, batch, input_size};
    std::vector<int> wi_dims    = {num_directions, 4, output_size, input_size};
    std::vector<int> wh_dims    = {num_directions, 4, output_size, output_size};
    std::vector<int> bias_dims  = {num_directions, 8, output_size};
    auto interpreter            = GenerateInterpreter("LSTMONNX", {input_dims, wi_dims, wh_dims, bias_dims}, param, nullptr, 3);

    Precision precision = PRECISION_AUTO;
    if (DATA_TYPE_BFP16 == dtype) {
        precision = PRECISION_LOW;
    }
    Run(interpreter);
}

}  // namespace TNN_NS
