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
#include "tnn/utils/dims_utils.h"
#include "unit_test_common.h"
#include "utils/network_helpers.h"

namespace TNN_NS {

static bool TestFilter(DeviceType device_type, int input_dim_size, int axis_size) {
    if (device_type == DEVICE_NAIVE || device_type == DEVICE_ARM)
        return true;

    if (device_type == DEVICE_OPENCL && input_dim_size <= 4)
        return true;
    if (device_type == DEVICE_OPENCL && input_dim_size > 4 && axis_size == 1)
        return true;
    if (device_type == DEVICE_APPLE_NPU && input_dim_size < 5)
        return true;
    // in order to skip when axis = {3, -2} 
    // [ n  d  k  h  w]
    // ->         *
    // [ 0  1  2  3  4]
    // [-5 -4 -3 -2 -1]
    //            *  <-
    // axis=3 and axis=-2 are point to the same axis, when dims=5
    // this cause coreml model error
    if (device_type == DEVICE_APPLE_NPU && input_dim_size > 4 && axis_size == 1)  
        return true;
        
    return false;
}

static std::string GenerateReduceProto(std::string op_type, ReduceLayerParam param) {
    std::ostringstream ostr;
    ostr << "\"" << op_type << " layer_name 1 1 input output " << param.keep_dims << " ";
    for (auto axis : param.axis) {
        ostr << axis << " ";
    }
    ostr << ",\"";
    return ostr.str();
}

static void UpdateReduceAxis(std::vector<int>& axes, const int dim_count) {
    const auto f = [=](int& v){ v = v < 0? v+dim_count : v;};
    std::for_each(axes.begin(), axes.end(), f);
}

static bool HasAxis(std::vector<int> axes, const int axis, const int dim_count) {
    UpdateReduceAxis(axes, dim_count);
    auto it = std::find(axes.begin(), axes.end(), axis);
    return (it != axes.end());
}

static bool IsDiscontinuous(std::vector<int> axes, const int dim_count) {
    UpdateReduceAxis(axes, dim_count);
    auto min_val = *std::min_element(axes.begin(), axes.end());
    auto max_val = *std::max_element(axes.begin(), axes.end());
    for(auto v=min_val; v<=max_val; ++v) {
        if (std::find(axes.begin(), axes.end(), v) == axes.end())
            return true;
    }
    return false;
}

class ReduceOpLayerTest
    : public LayerTest,
      public ::testing::WithParamInterface<std::tuple<int, int, int, int, int, int, std::vector<int>, DataType>> {};

INSTANTIATE_TEST_SUITE_P(LayerTest, ReduceOpLayerTest,
                         ::testing::Combine(testing::Values(1, 2),
                                            testing::Values(2, 3, 9, 128),
                                            testing::Values(9, 10, 19, 128),
                                            testing::Values(9, 10, 19, 128),
                                            // keep_dim
                                            testing::Values(0, 1),
                                            // dim count
                                            testing::Values(2, 3, 4, 5),
                                            // axis
                                            testing::Values(std::vector<int>({0}), std::vector<int>({1}), std::vector<int>({2}),
                                                            std::vector<int>({3}), std::vector<int>({1, 2}),
                                                            std::vector<int>({1, -1}), std::vector<int>({3, -2}),
                                                            std::vector<int>({1, -2, -1})),
                                            // dtype
                                            testing::Values(DATA_TYPE_FLOAT, DATA_TYPE_INT32)));

TEST_P(ReduceOpLayerTest, ReduceOpLayer) {
    // get param
    int batch          = std::get<0>(GetParam());
    int channel        = std::get<1>(GetParam());
    int input_height   = std::get<2>(GetParam());
    int input_width    = std::get<3>(GetParam());
    int keep_dims      = std::get<4>(GetParam());
    int dim_count      = std::get<5>(GetParam());
    auto& axis         = std::get<6>(GetParam());
    DataType data_type = std::get<7>(GetParam());
    DeviceType dev     = ConvertDeviceType(FLAGS_dt);

    // input is 1-dimensional
    if (dim_count == 1 || axis[0] == 0) {
        if (dev == DEVICE_ARM) {
            if (dim_count == 1 && axis[0] == 0) {
                // reduce on the first dimensional
            } else {
                GTEST_SKIP();
            }
        } else {
            GTEST_SKIP();
        }
    }

    // only test one case for large inputs
    if ((channel == 128 && (input_height > 9 || input_width > 9)) ||
        (input_width == 128 && (channel > 2 || input_height > 9)) ||
        (input_height == 128 && (channel > 2 || input_width > 9))) {
        GTEST_SKIP();
    }

    if (!TestFilter(dev, dim_count, axis.size())) {
        GTEST_SKIP();
    }
    // skip output dims size is 1;
    if (keep_dims== 0 && axis.size() == dim_count - 1) {
        GTEST_SKIP();
    }
    // blobconverter cannot handle 1-dimensional blob, skip it for now
    if (dim_count <= axis.size()+1 && keep_dims == 0) {
        if (dev == DEVICE_ARM && (dim_count == axis.size()+1 && keep_dims == 0)) {
            // arm can support reduce to 1-dimensional blob
        } else {
            GTEST_SKIP();
        }
    }

    for(const auto& d: axis) {
        if (d >= dim_count || d + dim_count < 0) {
            GTEST_SKIP();
        }
    }

    if ((HasAxis(axis, 0, dim_count) || IsDiscontinuous(axis, dim_count)) && DEVICE_CUDA == dev) {
        GTEST_SKIP();
    }

    // param
    std::shared_ptr<ReduceLayerParam> param(new ReduceLayerParam());
    param->name = "ReduceOp";
    param->axis = axis;
    param->keep_dims = keep_dims;

    // generate interpreter
    //std::vector<int> input_dims = {batch, channel, input_height, input_width};
    std::vector<int> input_dims = {batch, channel};
    for(int i=input_dims.size(); i<dim_count; ++i) {
        if (i % 2 == 0) input_dims.push_back(input_height);
        else input_dims.push_back(input_width);
    }

    if (dim_count > 4) {
        for (int i = 4; i < dim_count; i++) {
            input_dims[i] = std::min(input_height, input_width);
        }
    }

    // input is 1-dimensional
    if (dim_count == 1) {
        input_dims = {channel};
    }
    
    // APPLE_NPU can not support DATA_TYPE_INT32
    if (dev == DEVICE_APPLE_NPU && data_type == DATA_TYPE_INT32) {
        GTEST_SKIP();
    }

    if (DEVICE_HUAWEI_NPU != dev) {
        auto interpreter1 = GenerateInterpreter("ReduceMax", {input_dims}, param);
        Run(interpreter1);
        auto interpreter2 = GenerateInterpreter("ReduceMin", {input_dims}, param);
        Run(interpreter2);
        auto interpreter3 = GenerateInterpreter("ReduceMean", {input_dims}, param);
        Run(interpreter3);
        if(DEVICE_APPLE_NPU != dev){
            auto interpreter8 = GenerateInterpreter("ReduceLogSumExp", {input_dims}, param);
            Run(interpreter8);
            if (DEVICE_CUDA != dev) {
                auto interpreter5 = GenerateInterpreter("ReduceL1", {input_dims}, param);
                Run(interpreter5);
                auto interpreter6 = GenerateInterpreter("ReduceL2", {input_dims}, param);
                Run(interpreter6);
                auto interpreter7 = GenerateInterpreter("ReduceLogSum", {input_dims}, param);
                Run(interpreter7);
                auto interpreter10 = GenerateInterpreter("ReduceSumSquare", {input_dims}, param);
                Run(interpreter10);
            }
        }
    }
    auto interpreter4 = GenerateInterpreter("ReduceSum", {input_dims}, param);
    Run(interpreter4);
    if (DEVICE_CUDA != dev && DEVICE_APPLE_NPU != dev) {
        auto interpreter9 = GenerateInterpreter("ReduceProd", {input_dims}, param);
        Run(interpreter9);
    }
}

}  // namespace TNN_NS
