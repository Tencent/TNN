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
#include "tnn/device/opencl/imagebuffer_convertor.h"
#include "tnn/device/opencl/opencl_memory.h"
#include "tnn/utils/dims_vector_utils.h"

namespace TNN_NS {

class OpenCLPReluLayerAcc : public OpenCLLayerAcc {
public:
    virtual Status Init(Context *context, LayerParam *param, LayerResource *resource, const std::vector<Blob *> &inputs,
                        const std::vector<Blob *> &outputs) override;

    virtual ~OpenCLPReluLayerAcc() override;

    virtual Status Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) override;

#if TNN_PROFILE
    virtual double GetFlops() override;
    virtual double GetBandwidth() override;
#endif

private:
    Status ConvertWeights(float *weights_data_ptr, int output_channel);

private:
    bool share_channel_ = false;
    shared_ptr<OpenCLMemory> ocl_scope_ = nullptr;
};

Status OpenCLPReluLayerAcc::Init(Context *context, LayerParam *param, LayerResource *resource,
                                 const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    LOGD("Init PRelu Acc\n");
    Status ret = OpenCLLayerAcc::Init(context, param, resource, inputs, outputs);
    CHECK_TNN_OK(ret)

    run_3d_ndrange_ = false;
    op_name_        = "PRelu";

    auto input_dims = inputs[0]->GetBlobDesc().dims;
    int channels    = input_dims[1];

    auto layer_param = dynamic_cast<PReluLayerParam *>(param);
    if (layer_param == nullptr) {
        LOGE("PReluLayerParam is null!\n");
        return Status(TNNERR_MODEL_ERR, "PReluLayerParam is null");
    }
    share_channel_ = layer_param->channel_shared;

    auto layer_res = dynamic_cast<PReluLayerResource *>(resource);
    if (layer_res == nullptr) {
        LOGE("PReluLayerResource is null!\n");
        return Status(TNNERR_MODEL_ERR, "PReluLayerResource is null");
    }
    RawBuffer &scope_handle = layer_res->slope_handle;
    DataType data_type      = scope_handle.GetDataType();

    ConvertChannelWeights(scope_handle, ocl_scope_, channels, true, share_channel_);

    // create kernel
    std::string kernel_name = "PRelu";
    if (run_3d_ndrange_)
        kernel_name = "PReluGS3D";
    ret = CreateExecuteUnit(execute_units_[0], "prelu", kernel_name);
    if (ret != TNN_OK) {
        LOGE("create execute unit failed!\n");
        return ret;
    }

    return TNN_OK;
}

OpenCLPReluLayerAcc::~OpenCLPReluLayerAcc() {}

Status OpenCLPReluLayerAcc::Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    LOGD("PRelu Acc Reshape\n");
    ASSERT(inputs.size() == 1);
    auto output_dims = outputs[0]->GetBlobDesc().dims;
    uint32_t idx = 0;
    if (run_3d_ndrange_) {
        idx = SetExecuteUnit3DSizeInfoDefault(execute_units_[0], output_dims);
    } else {
        idx = SetExecuteUnit2DSizeInfoDefault(execute_units_[0], output_dims);
    }
    execute_units_[0].ocl_kernel.setArg(idx++, *((cl::Image *)inputs[0]->GetHandle().base));
    if (!run_3d_ndrange_) {
        //set output width
        execute_units_[0].ocl_kernel.setArg(idx++, output_dims[3]);
    }
    execute_units_[0].ocl_kernel.setArg(idx++, *((cl::Image *)ocl_scope_->GetData()));
    execute_units_[0].ocl_kernel.setArg(idx++, *((cl::Image *)outputs[0]->GetHandle().base));
    return TNN_OK;
}

#if TNN_PROFILE
double OpenCLPReluLayerAcc::GetFlops() {
    return 2.0 * DimsVectorUtils::Count(output_dims_) / 1000.0 / 1000.0;
}

double OpenCLPReluLayerAcc::GetBandwidth() {
    OpenCLRuntime *opencl_runtime = OpenCLRuntime::GetInstance();
    int data_type_size            = opencl_runtime->GetPrecision() != PRECISION_HIGH ? 2 : 4;
    return 2.0 * DimsVectorUtils::Count(output_dims_) * data_type_size / 1000.0 / 1000.0;
}
#endif

REGISTER_OPENCL_ACC(PRelu, LAYER_PRELU)

}  // namespace TNN_NS
