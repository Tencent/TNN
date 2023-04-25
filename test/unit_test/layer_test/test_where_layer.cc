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
#include "tnn/utils/dims_utils.h"

namespace TNN_NS {

class WhereLayerTest : public LayerTest,
                    public ::testing::WithParamInterface<std::tuple<int, int, int, int, int, int, DataType>> {};

INSTANTIATE_TEST_SUITE_P(LayerTest, WhereLayerTest,
                         ::testing::Combine(testing::Values(1, 2), testing::Values(1, 3, 4, 10), testing::Values(3, 9),
                                            // param size type (1, channel, chw, hw, batch)
                                            testing::Values(0, 1, 2, 3, 4),
                                            // weight index
                                            testing::Values(-1, 0, 1),
                                            // dims
                                            testing::Values(2, 3, 4, 5),
                                            // data_type
                                            testing::Values(DATA_TYPE_FLOAT, DATA_TYPE_HALF)));

TEST_P(WhereLayerTest, WhereLayer) {
    // get param
    int batch           = std::get<0>(GetParam());
    int channel         = std::get<1>(GetParam());
    int input_size      = std::get<2>(GetParam());
    int param_size_type = std::get<3>(GetParam());
    int weight_idx      = std::get<4>(GetParam());
    int dims_size       = std::get<5>(GetParam());
    DataType data_type  = std::get<6>(GetParam());
    DeviceType dev      = ConvertDeviceType(FLAGS_dt);

    if(CheckDataTypeSkip(data_type)) {
        GTEST_SKIP();
    }

    if (DEVICE_ARM != dev && DEVICE_OPENCL != dev && DEVICE_METAL != dev) {
        GTEST_SKIP();
    }

    std::vector<int> param_dims;
    std::vector<int> input_dims = {batch, channel, input_size, input_size, input_size};
    if (0 == param_size_type) {
        param_dims  = {1, 1, 1, 1, 1};
    } else if (1 == param_size_type) {
        param_dims  = {1, channel, 1, 1, 1};
    } else if (2 == param_size_type) {
        param_dims  = {1, channel, input_size, input_size, input_size};
    } else if (3 == param_size_type) {
        param_dims  = {1, 1, input_size, input_size, input_size};
    } else if (4 == param_size_type) {
        param_dims  = {batch, 1, 1, 1, input_size};
    }

    for (int i = dims_size; i < 5; ++i) {
        param_dims.pop_back();
        input_dims.pop_back();
    }

    std::vector<int> input0_dims;
    std::vector<int> input1_dims;
    if (-1 == weight_idx) {
        // the size of input are same
        input0_dims = input_dims;
        input1_dims = input_dims;
    } else {
        if (0 == weight_idx) {
            input0_dims = {1};
            input1_dims = input_dims;
        } else if (1 == weight_idx) {
            input0_dims = input_dims;
            input1_dims = {1};
        }
    }

    std::shared_ptr<LayerParam> param(new LayerParam());
    param->name = "Where";

    Precision precision = SetPrecision(dev, data_type);
    if (DATA_TYPE_INT8 == data_type) {
        param->quantized = true;
    }
    ensure_input_positive_ = true;
    int8_input_max_ = 2;

    auto interpreter = GenerateInterpreter("Where", {input0_dims, input1_dims, param_dims}, param, nullptr, 1, {data_type, data_type, DATA_TYPE_INT8});
    DataFormat data_format = dims_size != 5 ? DATA_FORMAT_NCHW : DATA_FORMAT_NCDHW;
    Run(interpreter, precision, data_format, data_format);
}

}  // namespace TNN_NS
