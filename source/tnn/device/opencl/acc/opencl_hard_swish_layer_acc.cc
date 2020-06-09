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

class OpenCLHardSwishLayerAcc : public OpenCLBinaryLayerAcc {
public:
    virtual Status Init(Context *context, LayerParam *param, LayerResource *resource, const std::vector<Blob *> &inputs,
                        const std::vector<Blob *> &outputs) override;
    virtual Status Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) override;
    virtual ~OpenCLHardSwishLayerAcc() override;

private:
    void ExtendInputs(const std::vector<Blob *> &inputs);

    bool need_extend_input_ = false;
    std::vector<Blob *> inputs_extend_ = {};
};

OpenCLHardSwishLayerAcc::~OpenCLHardSwishLayerAcc() {}

Status OpenCLHardSwishLayerAcc::Init(Context *context, LayerParam *param, LayerResource *resource,
                                     const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    LOGD("Init HardSwish Acc\n");

    if (nullptr == resource && 1 == inputs.size()) {
        need_extend_input_ = true;
    }

    ExtendInputs(inputs);

    Status ret = OpenCLBinaryLayerAcc::Init(context, param, resource, inputs_extend_, outputs);
    CHECK_TNN_OK(ret)

    op_name_ = "HardSwish";

    // create kernel
    std::string kernel_name = kernel_name_ + "_HardSwish";
    std::set<std::string> build_options;
    std::string compute;
    if (broadcast_param_.input0_broadcast_type == BroadcastTypeNormal) {
        compute =
            "in0*clamp(in1*(FLOAT)(alpha)+(FLOAT)(beta),(FLOAT)0.0f,(FLOAT)1."
            "0f)";
    } else {
        compute =
            "in1*clamp(in0*(FLOAT)(alpha)+(FLOAT)(beta),(FLOAT)0.0f,(FLOAT)1."
            "0f)";
    }
    build_options.emplace(" -DOPERATOR=" + compute);
    ret = CreateExecuteUnit(execute_units_[0], "hard_swish", kernel_name, build_options);
    if (ret != TNN_OK) {
        LOGE("create execute unit failed!\n");
        return ret;
    }

    return TNN_OK;
}

Status OpenCLHardSwishLayerAcc::Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    LOGD("HardSwish Acc Reshape\n");
    HardSwishLayerParam *hs_param = dynamic_cast<HardSwishLayerParam *>(param_);
    if (!hs_param) {
        LOGE("Error: layer param is null\n");
        return Status(TNNERR_MODEL_ERR, "Error: layer param is null");
    }

    ExtendInputs(inputs);

    Status ret = OpenCLBinaryLayerAcc::Reshape(inputs_extend_, outputs);
    CHECK_TNN_OK(ret)

    execute_units_[0].ocl_kernel.setArg(kernel_arg_idx_++, hs_param->alpha);
    execute_units_[0].ocl_kernel.setArg(kernel_arg_idx_++, hs_param->beta);

    return TNN_OK;
}

void OpenCLHardSwishLayerAcc::ExtendInputs(const std::vector<Blob *> &inputs) {
    inputs_extend_ = inputs;
    if (need_extend_input_) {
        inputs_extend_.clear();
        inputs_extend_.resize(2);
        inputs_extend_[0] = inputs[0];
        inputs_extend_[1] = inputs[0];
    }
}

REGISTER_OPENCL_ACC(HardSwish, LAYER_HARDSWISH)

}  // namespace TNN_NS
