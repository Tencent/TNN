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

class MatMulLayerTest : public LayerTest,
                        public ::testing::WithParamInterface<std::tuple<std::vector<int>, std::vector<int>, int>> {};

bool IsCrossBroadCast(std::vector<int> dim0, std::vector<int> dim1) {
    auto dim0_extend = dim0;
    auto dim1_extend = dim1;
    if (dim0.size() > dim1.size()) {
        for (int i = 0; i < dim0.size() - dim1.size(); ++i) {
            dim1_extend.insert(dim1_extend.begin(), 1);
        }
    } else if (dim0.size() < dim1.size()) {
        for (int i = 0; i < dim1.size() - dim0.size(); ++i) {
            dim0_extend.insert(dim0_extend.begin(), 1);
        }
    }

    bool dim0_broadcast = false;
    bool dim1_broadcast = false;
    for (int i = 0; i < dim0_extend.size() - 2; ++i) {
        if (dim0_extend[i] != dim1_extend[i]) {
            if (dim0_extend[i] > dim1_extend[i]) {
                dim1_broadcast = true;
            } else if (dim0_extend[i] < dim1_extend[i]) {
                dim0_broadcast = true;
            }
        }
    }

    return dim0_broadcast & dim1_broadcast;
}

bool IsBothDiffBatch(std::vector<int> dim0, std::vector<int> dim1) {
    int dim0_batch = DimsVectorUtils::Count(dim0) / DimsVectorUtils::Count(dim0, dim0.size() - 2);
    int dim1_batch = DimsVectorUtils::Count(dim1) / DimsVectorUtils::Count(dim1, dim1.size() - 2);
    return dim0_batch > 1 && dim1_batch > 1 && dim0_batch != dim1_batch;
}

INSTANTIATE_TEST_SUITE_P(
    LayerTest, MatMulLayerTest,
    ::testing::Combine(::testing::Values(std::vector<int>({3, 4, 8, 16}), std::vector<int>({3, 1, 8, 16}),
                                         std::vector<int>({1, 4, 8, 16}), std::vector<int>({4, 8, 16}),
                                         std::vector<int>({1, 8, 16}), std::vector<int>({8, 16})),
                       ::testing::Values(std::vector<int>({16, 9}), std::vector<int>({1, 16, 9}),
                                         std::vector<int>({4, 16, 9}), std::vector<int>({3, 4, 16, 9}),
                                         std::vector<int>({1, 4, 16, 9}), std::vector<int>({3, 1, 16, 9})),
                       ::testing::Values(-1, 0, 1)));

TEST_P(MatMulLayerTest, MatMulLayer) {
    // get param
    std::vector<int> input0_dim = std::get<0>(GetParam());
    std::vector<int> input1_dim = std::get<1>(GetParam());
    int weight_pos              = std::get<2>(GetParam());

    DeviceType dev = ConvertDeviceType(FLAGS_dt);

    if (DEVICE_HUAWEI_NPU == dev) {
        GTEST_SKIP();
    }

    if (DEVICE_APPLE_NPU == dev && (weight_pos == 1 || weight_pos == 0)) {
        GTEST_SKIP();
    }

    if (IsCrossBroadCast(input0_dim, input1_dim)) {
        GTEST_SKIP();
    }
    if (IsBothDiffBatch(input0_dim, input1_dim)) {
        GTEST_SKIP();
    }

    std::shared_ptr<MatMulLayerParam> param(new MatMulLayerParam());
    param->name            = "MatMul";
    param->weight_position = weight_pos;

    std::vector<int> weight_dim;
    if (weight_pos == 0) {
        weight_dim = input0_dim;
    } else if (weight_pos == 1) {
        weight_dim = input1_dim;
    }

    std::shared_ptr<MatMulLayerResource> resource = nullptr;
    if (weight_pos != -1) {
        int param_count = DimsVectorUtils::Count(weight_dim);
        resource        = std::shared_ptr<MatMulLayerResource>(new MatMulLayerResource());
        RawBuffer buffer(param_count * sizeof(float), weight_dim);
        float* buffer_data = buffer.force_to<float*>();
        InitRandom(buffer_data, param_count, 1.0f);
        resource->weight = buffer;
    }

    // generate interpreter
    std::shared_ptr<AbstractModelInterpreter> interpreter;
    if (weight_pos == 0) {
        interpreter = GenerateInterpreter("MatMul", {input1_dim}, param, resource);
    } else if (weight_pos == 1) {
        interpreter = GenerateInterpreter("MatMul", {input0_dim}, param, resource);
    } else {
        interpreter = GenerateInterpreter("MatMul", {input0_dim, input1_dim}, param);
    }
    Run(interpreter);
}

}  // namespace TNN_NS
