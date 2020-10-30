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

    // blob desc
    std::vector<BlobDesc> inputs_desc;
    BlobDesc input_desc;
    input_desc.dims.push_back(batch);
    input_desc.dims.push_back(channel);
    input_desc.dims.push_back(input_height);
    input_desc.dims.push_back(input_width);
    input_desc.device_type = DEVICE_NAIVE;
    input_desc.data_type   = data_type;
    inputs_desc.push_back(input_desc);
    auto outputs_desc = CreateOutputBlobsDesc(1, data_type);

    // param
    ReduceMaxLayerParam param;
    param.name = "ReduceOp";
    param.axis = {axis};

    // all reduce different op layer run
    Run(LAYER_REDUCE_MAX, &param, nullptr, inputs_desc, outputs_desc);
    Run(LAYER_REDUCE_MIN, &param, nullptr, inputs_desc, outputs_desc);
    Run(LAYER_REDUCE_MEAN, &param, nullptr, inputs_desc, outputs_desc);
    Run(LAYER_REDUCE_L2, &param, nullptr, inputs_desc, outputs_desc);
    Run(LAYER_REDUCE_LOG_SUM, &param, nullptr, inputs_desc, outputs_desc);
    Run(LAYER_REDUCE_LOG_SUM_EXP, &param, nullptr, inputs_desc, outputs_desc);
    Run(LAYER_REDUCE_PROD, &param, nullptr, inputs_desc, outputs_desc);
    Run(LAYER_REDUCE_SUM_SQUARE, &param, nullptr, inputs_desc, outputs_desc);
    Run(LAYER_REDUCE_SUM, &param, nullptr, inputs_desc, outputs_desc);
}

TEST_P(ReduceOpLayerTest, ReduceOpLayerWithProto) {
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
    ReduceLayerParam param;
    param.name = "ReduceOp";
    param.axis = {axis};

    // generate proto string
    std::vector<int> input_dims = {batch, channel, input_height, input_width};
    std::string head            = GenerateHeadProto({input_dims});

    RunWithProto(head + GenerateReduceProto("ReduceMax", param));
    RunWithProto(head + GenerateReduceProto("ReduceMin", param));
    RunWithProto(head + GenerateReduceProto("ReduceMean", param));
    RunWithProto(head + GenerateReduceProto("ReduceSum", param));
    RunWithProto(head + GenerateReduceProto("ReduceL1", param));
    RunWithProto(head + GenerateReduceProto("ReduceL2", param));
    RunWithProto(head + GenerateReduceProto("ReduceLogSum", param));
    RunWithProto(head + GenerateReduceProto("ReduceLogSumExp", param));
    RunWithProto(head + GenerateReduceProto("ReduceProd", param));
    RunWithProto(head + GenerateReduceProto("ReduceSumSquare", param));
}

}  // namespace TNN_NS
