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

class ArgMaxOrMinLayerTest
    : public LayerTest,
      public ::testing::WithParamInterface<std::tuple<int, int, int, int, int, int, int, DataType>> {};

INSTANTIATE_TEST_SUITE_P(LayerTest, ArgMaxOrMinLayerTest,
                         ::testing::Combine(BASIC_BATCH_CHANNEL_SIZE,
                                            // mode: ArgMax or ArgMin
                                            testing::Values(0, 1),
                                            // axis
                                            testing::Values(0, 1, 2, 3),
                                            // keep dims
                                            testing::Values(1),
                                            // select_last_index: we will support this feature in future;
                                            testing::Values(0), testing::Values(DATA_TYPE_FLOAT)));

TEST_P(ArgMaxOrMinLayerTest, ArgMaxOrMinLayer) {
    // get param
    int batch             = std::get<0>(GetParam());
    int channel           = std::get<1>(GetParam());
    int input_size        = std::get<2>(GetParam());
    int mode              = std::get<3>(GetParam());
    int axis              = std::get<4>(GetParam());
    int keep_dims         = std::get<5>(GetParam());
    int select_last_index = std::get<6>(GetParam());
    DataType dtype        = std::get<7>(GetParam());
    DeviceType dev        = ConvertDeviceType(FLAGS_dt);

    if (DEVICE_HUAWEI_NPU == dev) {
        GTEST_SKIP();
    }

    if (dtype != DATA_TYPE_FLOAT) {
        GTEST_SKIP();
    }

    // param
    std::shared_ptr<ArgMaxOrMinLayerParam> param(new ArgMaxOrMinLayerParam());
    param->name              = "ArgMaxOrMin";
    param->mode              = mode;
    param->axis              = axis;
    param->keep_dims         = keep_dims;
    param->select_last_index = select_last_index;

    // generate proto string
    std::vector<int> input_dims = {batch, channel, input_size, input_size};
    auto interpreter            = GenerateInterpreter("ArgMaxOrMin", {input_dims}, param);
    Run(interpreter);
}

}  // namespace TNN_NS
