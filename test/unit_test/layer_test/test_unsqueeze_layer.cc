
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

class UnsqueezeLayerTest : public LayerTest,
                              public ::testing::WithParamInterface<std::tuple<int, int, int, int, DataType>> {};
// seq_len, batch, input, output
// direction: 0, 1, 2
INSTANTIATE_TEST_SUITE_P(LayerTest, UnsqueezeLayerTest,
                         ::testing::Combine(testing::Values(1, 7, 13),    // dim0
                                            testing::Values(1, 3, 11), //dim1
                                            testing::Values(1, 2, 8),    // dim2
                                            testing::Values(0, 1, 2, 3),  // axis
                                            testing::Values(DATA_TYPE_FLOAT, DATA_TYPE_HALF)));

TEST_P(UnsqueezeLayerTest, UnsqueezeLayer) {
    // get param
    int dim0         = std::get<0>(GetParam());
    int dim1         = std::get<1>(GetParam());
    int dim2         = std::get<2>(GetParam());
    int axis         = std::get<3>(GetParam());
    DataType dtype   = std::get<4>(GetParam());
    DeviceType dev   = ConvertDeviceType(FLAGS_dt);
    if (!TestFilter(dev, dtype)) {
        GTEST_SKIP();
    }

    // param
    /*
    struct SqueezeLayerParam : public LayerParam {
    std::vector<int> axes;
    bool data_in_resource = false;

    PARAM_COPY(SqueezeLayerParam)
};
*/
    std::shared_ptr<UnsqueezeLayerParam> param(new UnsqueezeLayerParam());
    param->name        = "Unsqueeze";
    std::vector<int> axes = {axis};
    param->axes = axes;

    // generate interpreter
    std::vector<int> input_dims = {dim0, dim1, dim2};
    input_dims.insert(input_dims.begin()+axis, 1);
    auto interpreter            = GenerateInterpreter("Squeeze", {input_dims}, param);

    Precision precision = PRECISION_AUTO;
    if (DATA_TYPE_BFP16 == dtype) {
        precision = PRECISION_LOW;
    }
    Run(interpreter);
}

}  // namespace TNN_NS
