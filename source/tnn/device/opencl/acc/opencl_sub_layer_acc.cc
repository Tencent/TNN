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

#include "tnn/device/opencl/acc/opencl_binary_layer_acc.h"
#include "tnn/device/opencl/imagebuffer_convertor.h"

namespace TNN_NS {

DECLARE_OPENCL_BINARY_ACC(Sub);

Status OpenCLSubLayerAcc::Init(Context *context, LayerParam *param, LayerResource *resource,
                               const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    LOGD("Init Sub Acc\n");
    Status ret = OpenCLBinaryLayerAcc::Init(context, param, resource, inputs, outputs);
    CHECK_TNN_OK(ret)

    op_name_ = "Sub";

    // create kernel
    std::set<std::string> build_options;
    std::string compute;
    compute = "in0-in1";
    if (inputs.size() == 2) {
        if (broadcast_param_.input0_broadcast_type == BroadcastTypeNormal) {
            compute = "in0-in1";
        } else {
            compute = "in1-in0";
        }
    } else if (inputs.size() == 1) {
        if (kernel_name_ != "BinaryElementWise" && kernel_name_ != "BinaryBroadcast5D" &&
            kernel_name_ != "BinaryBroadcast") {
            if (broadcast_param_.input0_broadcast_type == BroadcastTypeNormal) {
                if (broadcast_param_.weight_input_index == 0) {
                    compute = "in1-in0";
                }
            } else if (broadcast_param_.input1_broadcast_type == BroadcastTypeNormal) {
                if (broadcast_param_.weight_input_index == 0) {
                    compute = "in1-in0";
                }
            }
        }
    }
    build_options.emplace(" -DOPERATOR=" + compute);
    build_options.insert(build_options_.begin(), build_options_.end());
    ret = CreateExecuteUnit(execute_units_[0], "binary", kernel_name_, build_options);
    if (ret != TNN_OK) {
        LOGE("create execute unit failed!\n");
        return ret;
    }

    return TNN_OK;
}

OpenCLSubLayerAcc::~OpenCLSubLayerAcc() {}

REGISTER_OPENCL_ACC(Sub, LAYER_SUB)
REGISTER_OPENCL_LAYOUT(LAYER_SUB, DATA_FORMAT_NHC4W4);

}  // namespace TNN_NS
