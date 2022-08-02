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

#include <vector>

#include "tnn/device/opencl/acc/opencl_layer_acc.h"
#include "tnn/device/opencl/imagebuffer_convertor.h"
#include "tnn/device/opencl/opencl_memory.h"
#include "tnn/utils/data_type_utils.h"

namespace TNN_NS {

class OpenCLGroupNormLayerAcc : public OpenCLLayerAcc {
public:
    virtual Status Init(Context *context, LayerParam *param, LayerResource *resource, const std::vector<Blob *> &inputs,
                        const std::vector<Blob *> &outputs) override;

    virtual ~OpenCLGroupNormLayerAcc() override;

    virtual Status Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) override;
};

Status OpenCLGroupNormLayerAcc::Init(Context *context, LayerParam *param, LayerResource *resource,
                                     const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    LOGD("Init GroupNorm Acc\n");
    Status ret = OpenCLLayerAcc::Init(context, param, resource, inputs, outputs);
    CHECK_TNN_OK(ret)

    op_name_ = "GroupNorm";

    GroupNormLayerParam *group_norm_layer_param = dynamic_cast<GroupNormLayerParam *>(param_);

    const int channels_per_group = outputs[0]->GetBlobDesc().dims[1] / group_norm_layer_param->group;
    if (channels_per_group % 4 != 0 || (outputs[0]->GetBlobDesc().dims[1] % 4) != 0) {
        LOGE(
            "channels_per_group = %d, output_channels = %d, both channels_per_group and output_channels must be "
            "divisible by 4\n",
            channels_per_group, outputs[0]->GetBlobDesc().dims[1]);
        return Status(TNNERR_OPENCL_ACC_INIT_ERROR,
                      "both channels_per_group and output_channels must be divisible by 4");
    }

    std::string kernel_name  = "GroupNorm";
    std::string program_name = "group_norm";
    ret                      = CreateExecuteUnit(execute_units_[0], program_name, kernel_name, build_options_);
    if (ret != TNN_OK) {
        LOGE("create execute unit failed!\n");
        return ret;
    }

    return TNN_OK;
}

OpenCLGroupNormLayerAcc::~OpenCLGroupNormLayerAcc() {}

Status OpenCLGroupNormLayerAcc::Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    LOGD("GroupNorm Layer Reshape\n");
    ASSERT(inputs.size() == 3);
    Status ret = OpenCLLayerAcc::Reshape(inputs, outputs);
    CHECK_TNN_OK(ret)

    GroupNormLayerParam *group_norm_layer_param = dynamic_cast<GroupNormLayerParam *>(param_);
    auto input_dims                             = inputs[0]->GetBlobDesc().dims;
    auto output_dims                            = outputs[0]->GetBlobDesc().dims;

    uint32_t idx = SetExecuteUnit2DSizeInfoDefault(execute_units_[0], output_dims);
    execute_units_[0].ocl_kernel.setArg(idx++, *((cl::Image *)inputs[0]->GetHandle().base));
    execute_units_[0].ocl_kernel.setArg(idx++, *((cl::Image *)inputs[1]->GetHandle().base));
    execute_units_[0].ocl_kernel.setArg(idx++, *((cl::Image *)inputs[2]->GetHandle().base));
    execute_units_[0].ocl_kernel.setArg(idx++, group_norm_layer_param->group);
    execute_units_[0].ocl_kernel.setArg(idx++, group_norm_layer_param->eps);
    execute_units_[0].ocl_kernel.setArg(idx++, output_dims.size() * sizeof(int), output_dims.data());
    execute_units_[0].ocl_kernel.setArg(idx++, *((cl::Image *)outputs[0]->GetHandle().base));
    return TNN_OK;
}

REGISTER_OPENCL_ACC(GroupNorm, LAYER_GROUP_NORM)
REGISTER_OPENCL_LAYOUT(LAYER_GROUP_NORM, DATA_FORMAT_NHC4W4);

}  // namespace TNN_NS
