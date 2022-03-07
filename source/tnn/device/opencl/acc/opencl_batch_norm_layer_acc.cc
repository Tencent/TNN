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

class OpenCLBatchNormLayerAcc : public OpenCLLayerAcc {
public:
    virtual Status Init(Context *context, LayerParam *param, LayerResource *resource, const std::vector<Blob *> &inputs,
                        const std::vector<Blob *> &outputs) override;

    virtual ~OpenCLBatchNormLayerAcc() override;

    virtual Status Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) override;

private:
    Status ConvertWeights(float *weights_data_ptr, int output_channel);

private:
    bool share_channel_ = false;
    std::shared_ptr<OpenCLMemory> ocl_k_ = nullptr;
    std::shared_ptr<OpenCLMemory> ocl_b_ = nullptr;
};

Status OpenCLBatchNormLayerAcc::Init(Context *context, LayerParam *param, LayerResource *resource,
                                     const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    LOGD("Init BatchNorm Acc\n");
    Status ret = OpenCLLayerAcc::Init(context, param, resource, inputs, outputs);
    CHECK_TNN_OK(ret)

    run_3d_ndrange_ = true;
    op_name_        = "BatchNorm";

    auto input_dims = inputs[0]->GetBlobDesc().dims;
    int channels    = DimsFunctionUtils::GetDim(input_dims, 1);

    BatchNormLayerResource *batchnorm_resource = dynamic_cast<BatchNormLayerResource *>(resource);
    if (batchnorm_resource == nullptr) {
        LOGE("BatchNormLayerResource is null!\n");
        return Status(TNNERR_MODEL_ERR, "BatchNormLayerResource is null");
    }

    RawBuffer &scale_handle = batchnorm_resource->scale_handle;
    RawBuffer &bias_handle  = batchnorm_resource->bias_handle;
    DataType data_type      = scale_handle.GetDataType();

    share_channel_ = scale_handle.GetBytesSize() == DataTypeUtils::GetBytesSize(data_type);
    bool has_bias  = bias_handle.GetBytesSize() != 0;

    ret = ConvertChannelWeights(scale_handle, ocl_k_, channels, true, share_channel_);
    CHECK_TNN_OK(ret)

    // get bias
    ret = ConvertChannelWeights(bias_handle, ocl_b_, channels, has_bias, share_channel_);
    CHECK_TNN_OK(ret)

    // create kernel
    std::string kernel_name = "BatchNormGS3D";
    ret                     = CreateExecuteUnit(execute_units_[0], "batch_norm", kernel_name, build_options_);
    if (ret != TNN_OK) {
        LOGE("create execute unit failed!\n");
        return ret;
    }

    return TNN_OK;
}

OpenCLBatchNormLayerAcc::~OpenCLBatchNormLayerAcc() {}

Status OpenCLBatchNormLayerAcc::Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    LOGD("BatchNorm Layer Reshape\n");
    ASSERT(inputs.size() == 1);
    Status ret = OpenCLLayerAcc::Reshape(inputs, outputs);
    CHECK_TNN_OK(ret)

    auto output_dims = outputs[0]->GetBlobDesc().dims;
    uint32_t idx = SetExecuteUnit3DSizeInfoDefault(execute_units_[0], output_dims);
    execute_units_[0].ocl_kernel.setArg(idx++, *((cl::Image *)inputs[0]->GetHandle().base));
    execute_units_[0].ocl_kernel.setArg(idx++, *((cl::Image *)ocl_k_->GetData()));
    execute_units_[0].ocl_kernel.setArg(idx++, *((cl::Image *)ocl_b_->GetData()));
    execute_units_[0].ocl_kernel.setArg(idx++, *((cl::Image *)outputs[0]->GetHandle().base));
    return TNN_OK;
}

REGISTER_OPENCL_ACC(BatchNorm, LAYER_BATCH_NORM)
REGISTER_OPENCL_ACC(BatchNorm, LAYER_SCALE)
REGISTER_OPENCL_LAYOUT(LAYER_BATCH_NORM, DATA_FORMAT_NHC4W4);
REGISTER_OPENCL_LAYOUT(LAYER_SCALE, DATA_FORMAT_NHC4W4);

}  // namespace TNN_NS
