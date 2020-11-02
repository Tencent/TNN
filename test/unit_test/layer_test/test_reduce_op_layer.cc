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

#include "layer_test.h"
#include "tnn/utils/dims_vector_utils.h"
#include "unit_test_common.h"
#include "utils/network_helpers.h"

namespace TNN_NS {

static std::string GenerateReduceProto(std::string op_type, ReduceLayerParam param) {
    std::ostringstream ostr;
    ostr << "\"" << op_type << " layer_name 1 1 input output " << param.keep_dims << " ";
    for (auto axis : param.axis) {
        ostr << axis << " ";
    }
    ostr << ",\"";
    return ostr.str();
}

class ReduceOpLayerTest : public LayerTest,
                          public ::testing::WithParamInterface<std::tuple<int, int, int, int, int, DataType>> {};

INSTANTIATE_TEST_SUITE_P(LayerTest, ReduceOpLayerTest,
                         ::testing::Combine(testing::Values(1), testing::Values(2, 3, 4, 10, 32, 512),
                                            testing::Values(9, 10, 16, 19, 512), testing::Values(9, 10, 16, 19, 512),
                                            // axis
                                            testing::Values(0, 1, 2, 3),
                                            // dtype
                                            testing::Values(DATA_TYPE_FLOAT)));

TEST_P(ReduceOpLayerTest, ReduceOpLayer) {
    // get param
    int batch          = std::get<0>(GetParam());
    int channel        = std::get<1>(GetParam());
    int input_height   = std::get<2>(GetParam());
    int input_width    = std::get<3>(GetParam());
    int axis           = std::get<4>(GetParam());
    DataType data_type = std::get<5>(GetParam());
    DeviceType dev     = ConvertDeviceType(FLAGS_dt);

    if ((channel == 512 && input_height == 512) || (input_width == 512 && input_height == 512) ||
        (channel == 512 && input_width == 512)) {
        GTEST_SKIP();
    }

    // param
    ReduceLayerParam* param = new ReduceLayerParam();
    param->name             = "ReduceOp";
    param->axis             = {axis};

    auto param_share = std::shared_ptr<LayerParam>(param);
    // generate interpreter
    std::vector<int> input_dims = {batch, channel, input_height, input_width};
    auto interpreter1           = GenerateInterpreter("ReduceMax", {input_dims}, param_share);
    Run(interpreter1);
    auto interpreter2 = GenerateInterpreter("ReduceMin", {input_dims}, param_share);
    Run(interpreter2);
    auto interpreter3 = GenerateInterpreter("ReduceMean", {input_dims}, param_share);
    Run(interpreter3);
    auto interpreter4 = GenerateInterpreter("ReduceSum", {input_dims}, param_share);
    Run(interpreter4);
    auto interpreter5 = GenerateInterpreter("ReduceL1", {input_dims}, param_share);
    Run(interpreter5);
    auto interpreter6 = GenerateInterpreter("ReduceL2", {input_dims}, param_share);
    Run(interpreter6);
    auto interpreter7 = GenerateInterpreter("ReduceLogSum", {input_dims}, param_share);
    Run(interpreter7);
    auto interpreter8 = GenerateInterpreter("ReduceLogSumExp", {input_dims}, param_share);
    Run(interpreter8);
    auto interpreter9 = GenerateInterpreter("ReduceProd", {input_dims}, param_share);
    Run(interpreter9);
    auto interpreter10 = GenerateInterpreter("ReduceSumSquare", {input_dims}, param_share);
    Run(interpreter10);
}

}  // namespace TNN_NS
