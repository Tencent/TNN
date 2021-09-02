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

#include "tnn/device/opencl/acc/opencl_reformat_layer_acc.h"
#include "tnn/utils/dims_utils.h"

namespace TNN_NS {

Status OpenCLReformatLayerAcc::Init(Context *context, LayerParam *param, LayerResource *resource,
                                 const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    LOGD("Init Reformat Acc\n");
    Status ret = OpenCLLayerAcc::Init(context, param, resource, inputs, outputs);
    CHECK_TNN_OK(ret)

    ASSERT(inputs.size() == outputs.size());

    run_3d_ndrange_ = false;
    op_name_        = "Reformat";

    auto reformat_param = dynamic_cast<ReformatLayerParam *>(param);
    CHECK_PARAM_NULL(reformat_param);

    if (reformat_param->src_format == DATA_FORMAT_NHC4W4 && reformat_param->dst_format == DATA_FORMAT_CNH4) {
        kernel_name_ = "NHC4W4ImageToCNH4Image";
    } else if (reformat_param->src_format == DATA_FORMAT_CNH4 && reformat_param->dst_format == DATA_FORMAT_NHC4W4) {
        kernel_name_ = "CNH4ImageToNHC4W4Image";
    } else {
        LOGE("OpenCLReformatLayerAcc::Init Error: src_fmt: %d, dst_fmt: %d, src_type: %d, dst_type: %d\n",
             reformat_param->src_format, reformat_param->dst_format, reformat_param->src_type,
             reformat_param->dst_type);
        return Status(TNNERR_MODEL_ERR, "OpenCLReformatLayerAcc::Init unsupport reformat type");
    }

    // create kernel
    const int blob_size = inputs.size();
    execute_units_.resize(blob_size);
    for (int i = 0; i < blob_size; i++) {
        ret = CreateExecuteUnit(execute_units_[i], "image_to_image", kernel_name_);
    }

    if (ret != TNN_OK) {
        LOGE("create execute unit failed!\n");
        return ret;
    }

    return TNN_OK;
}

OpenCLReformatLayerAcc::~OpenCLReformatLayerAcc() {}

Status OpenCLReformatLayerAcc::Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    LOGD("Reformat Layer Reshape\n");
    Status ret = OpenCLLayerAcc::Reshape(inputs, outputs);
    CHECK_TNN_OK(ret)

    const int blob_size = inputs.size();
    for (int i = 0; i < blob_size; i++) {
        auto input_dims  = inputs[i]->GetBlobDesc().dims;
        auto output_dims = outputs[i]->GetBlobDesc().dims;

        if (input_dims != output_dims || input_dims.size() != 3) {
            LOGE("Reformat Layer input dims invalid\n");
            return Status(TNNERR_OPENCL_ACC_RESHAPE_ERROR, "Reformat Layer input dims invalid");
        }

        int n = DimsFunctionUtils::GetDim(input_dims, 0);
        int c = DimsFunctionUtils::GetDim(input_dims, 1);
        int h = DimsFunctionUtils::GetDim(input_dims, 2);
        if (kernel_name_ == "NHC4W4ImageToCNH4Image") {
            execute_units_[i].global_work_size = {static_cast<uint32_t>(UP_DIV(h, 4)), static_cast<uint32_t>(c * n)};
            execute_units_[i].local_work_size  = {64, 1};
        } else if (kernel_name_ == "CNH4ImageToNHC4W4Image") {
            execute_units_[i].global_work_size = {static_cast<uint32_t>(UP_DIV(c, 4)), static_cast<uint32_t>(n * h)};
            execute_units_[i].local_work_size  = {1, 64};
        }

        uint32_t idx = 0;
        execute_units_[i].ocl_kernel.setArg(idx++, execute_units_[i].global_work_size[0]);
        execute_units_[i].ocl_kernel.setArg(idx++, execute_units_[i].global_work_size[1]);
        execute_units_[i].ocl_kernel.setArg(idx++, *((cl::Image *)inputs[i]->GetHandle().base));
        execute_units_[i].ocl_kernel.setArg(idx++, n);
        execute_units_[i].ocl_kernel.setArg(idx++, h);
        execute_units_[i].ocl_kernel.setArg(idx++, *((cl::Image *)outputs[i]->GetHandle().base));
    }

    return TNN_OK;
}

std::vector<DataFormat> OpenCLReformatLayerAcc::SupportDataFormat(DataType data_type, int dims_size, BlobType blob_type) {
    std::vector<DataFormat> support_list{DATA_FORMAT_NHC4W4, DATA_FORMAT_CNH4};
    return support_list;
}

REGISTER_OPENCL_ACC(Reformat, LAYER_REFORMAT)
REGISTER_OPENCL_LAYOUT(LAYER_REFORMAT, DATA_FORMAT_NHC4W4)
REGISTER_OPENCL_LAYOUT(LAYER_REFORMAT, DATA_FORMAT_CNH4)

}  // namespace TNN_NS
