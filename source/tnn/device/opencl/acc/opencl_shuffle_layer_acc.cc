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

#include "tnn/device/opencl/acc/opencl_layer_acc.h"
#include "tnn/device/opencl/opencl_utils.h"

namespace TNN_NS {

DECLARE_OPENCL_ACC(Shuffle);

Status OpenCLShuffleLayerAcc::Init(Context *context, LayerParam *param, LayerResource *resource,
                                   const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    LOGD("Init ShuffleChannel Acc\n");
    Status ret = OpenCLLayerAcc::Init(context, param, resource, inputs, outputs);
    CHECK_TNN_OK(ret)

    run_3d_ndrange_ = true;
    op_name_        = "ShuffleChannel";

    // create kernel
    std::string kernel_name = "ShuffleChannel";
    ret                     = CreateExecuteUnit(execute_units_[0], "shuffle", kernel_name, build_options_);
    if (ret != TNN_OK) {
        LOGE("create execute unit failed!\n");
        return ret;
    }

    return TNN_OK;
}

Status OpenCLShuffleLayerAcc::Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    LOGD("ShuffleChannel Acc Reshape\n");
    Status ret = OpenCLLayerAcc::Reshape(inputs, outputs);
    CHECK_TNN_OK(ret)

    auto shuffle_param = dynamic_cast<ShuffleLayerParam *>(param_);
    if (shuffle_param == nullptr) {
        LOGE("ShuffleChannelLayerParam is null!\n");
        return Status(TNNERR_MODEL_ERR, "ShuffleChannelLayerParam is null!");
    }

    ASSERT(inputs.size() == 1);

    auto input_dims  = inputs[0]->GetBlobDesc().dims;
    auto output_dims = outputs[0]->GetBlobDesc().dims;
    if (shuffle_param->group <= 0 || DimsFunctionUtils::GetDim(input_dims, 1) % shuffle_param->group != 0) {
        LOGE("invalid group size in Shuffle layer!\n");
        return Status(TNNERR_LAYER_ERR, "invalid group size in Shuffle layer!");
    }

    uint32_t idx = SetExecuteUnit3DSizeInfoDefault(execute_units_[0], outputs[0]->GetBlobDesc().dims);
    int group_size = DimsFunctionUtils::GetDim(output_dims, 1) / shuffle_param->group;
    execute_units_[0].ocl_kernel.setArg(idx++, *((cl::Image *)inputs[0]->GetHandle().base));
    execute_units_[0].ocl_kernel.setArg(idx++, *((cl::Image *)outputs[0]->GetHandle().base));
    execute_units_[0].ocl_kernel.setArg(idx++, shuffle_param->group);
    execute_units_[0].ocl_kernel.setArg(idx++, group_size);
    execute_units_[0].ocl_kernel.setArg(idx++, DimsFunctionUtils::GetDim(output_dims, 1)); //output channel

    return TNN_OK;
}

REGISTER_OPENCL_ACC(Shuffle, LAYER_SHUFFLE_CHANNEL)
REGISTER_OPENCL_LAYOUT(LAYER_SHUFFLE_CHANNEL, DATA_FORMAT_NHC4W4);

}  // namespace TNN_NS
