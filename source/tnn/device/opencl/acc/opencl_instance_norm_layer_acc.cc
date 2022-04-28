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

#include <stdio.h>
#include <vector>

#include "tnn/device/opencl/acc/opencl_layer_acc.h"
#include "tnn/device/opencl/imagebuffer_convertor.h"
#include "tnn/device/opencl/opencl_memory.h"
#include "tnn/utils/data_type_utils.h"
#include "tnn/utils/dims_utils.h"

namespace TNN_NS {

class OpenCLInstanceNormLayerAcc : public OpenCLLayerAcc {
public:
    virtual Status Init(Context *context, LayerParam *param, LayerResource *resource, const std::vector<Blob *> &inputs,
                        const std::vector<Blob *> &outputs) override;

    virtual ~OpenCLInstanceNormLayerAcc() override;

    virtual Status Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) override;

#if TNN_PROFILE
    virtual double GetFlops() override;
    virtual double GetBandwidth() override;
#endif

private:
    Status ConvertWeights(float *weights_data_ptr, int output_channel);
    Status AllocateImage(int batch, int output_channel);
    Status BuildVarBiasKernel(int width);
    std::vector<uint32_t> GetLocalWS();

private:
    bool share_channel_ = false;
    shared_ptr<OpenCLMemory> ocl_k_ = nullptr;
    shared_ptr<OpenCLMemory> ocl_b_ = nullptr;
    shared_ptr<OpenCLMemory> ocl_var_ = nullptr;
    shared_ptr<OpenCLMemory> ocl_bias_ = nullptr;
    int thread_block_w_ = 8;
    int bench_idx_      = 0;
};

Status OpenCLInstanceNormLayerAcc::Init(Context *context, LayerParam *param, LayerResource *resource,
                                        const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    LOGD("Init InstanceNorm Acc\n");
    Status ret = OpenCLLayerAcc::Init(context, param, resource, inputs, outputs);
    CHECK_TNN_OK(ret)

    run_3d_ndrange_ = false;
    op_name_        = "InstanceNorm";

    InstanceNormLayerResource *instnorm_resource = dynamic_cast<InstanceNormLayerResource *>(resource);
    if (instnorm_resource == nullptr) {
        LOGE("InstanceNormLayerResource is null!\n");
        return Status(TNNERR_MODEL_ERR, "InstanceNormLayerResource is null");
    }

    RawBuffer &scale_handle = instnorm_resource->scale_handle;
    RawBuffer &bias_handle  = instnorm_resource->bias_handle;
    DataType data_type      = scale_handle.GetDataType();

    auto input_dims = inputs[0]->GetBlobDesc().dims;
    int batch       = DimsFunctionUtils::GetDim(input_dims, 0);
    int channels    = DimsFunctionUtils::GetDim(input_dims, 1);
    int width       = DimsFunctionUtils::GetDim(input_dims, 3);

    share_channel_ = scale_handle.GetBytesSize() == DataTypeUtils::GetBytesSize(data_type);
    bool has_bias  = bias_handle.GetBytesSize() != 0;
    //convert scale
    ret = ConvertChannelWeights(scale_handle, ocl_k_, channels, true, share_channel_);
    CHECK_TNN_OK(ret)

    //convert bias
    ret = ConvertChannelWeights(bias_handle, ocl_b_, channels, has_bias, share_channel_);
    CHECK_TNN_OK(ret)

    // allocate var and bias
    ret = AllocateImage(batch, channels);
    CHECK_TNN_OK(ret)

    // create kernel
    execute_units_.resize(2);
    ret = BuildVarBiasKernel(width);
    if (ret != TNN_OK) {
        LOGE("create execute unit failed!\n");
        return ret;
    }
    //create execute unit
    ret = CreateExecuteUnit(execute_units_[1], "batch_norm", "BatchNormBatch", build_options_);
    if (ret != TNN_OK) {
        LOGE("create execute unit failed!\n");
        return ret;
    }

    return TNN_OK;
}

OpenCLInstanceNormLayerAcc::~OpenCLInstanceNormLayerAcc() {}

Status OpenCLInstanceNormLayerAcc::Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    LOGD("Instance Norm Layer Reshape\n");
    ASSERT(inputs.size() == 1);
    Status ret = OpenCLLayerAcc::Reshape(inputs, outputs);
    CHECK_TNN_OK(ret)

    auto input_dims  = inputs[0]->GetBlobDesc().dims;
    auto output_dims = outputs[0]->GetBlobDesc().dims;

    const int channel_blocks = UP_DIV(DimsFunctionUtils::GetDim(input_dims, 1), 4);

    // unit0
    execute_units_[0].global_work_size = {static_cast<uint32_t>(thread_block_w_ * thread_block_w_),
                                        static_cast<uint32_t>(channel_blocks) * static_cast<uint32_t>(DimsFunctionUtils::GetDim(input_dims, 0))};
    execute_units_[0].local_work_size = GetLocalWS();

    uint32_t idx = 0;
    execute_units_[0].ocl_kernel.setArg(idx++, execute_units_[0].global_work_size[0]);
    execute_units_[0].ocl_kernel.setArg(idx++, execute_units_[0].global_work_size[1]);
    execute_units_[0].ocl_kernel.setArg(idx++, *((cl::Image *)inputs[0]->GetHandle().base));
    execute_units_[0].ocl_kernel.setArg(idx++, *((cl::Image *)ocl_k_->GetData()));
    execute_units_[0].ocl_kernel.setArg(idx++, *((cl::Image *)ocl_b_->GetData()));
    execute_units_[0].ocl_kernel.setArg(idx++, channel_blocks);
    //input_height
    execute_units_[0].ocl_kernel.setArg(idx++, DimsFunctionUtils::GetDim(input_dims, 2));
    //input_width
    execute_units_[0].ocl_kernel.setArg(idx++, DimsFunctionUtils::GetDim(input_dims, 3));
    //input_height * input_width
    execute_units_[0].ocl_kernel.setArg(idx++, DimsFunctionUtils::GetDim(input_dims, 3) *
            DimsFunctionUtils::GetDim(input_dims, 2));
    execute_units_[0].ocl_kernel.setArg(idx++, *((cl::Image *)ocl_var_->GetData()));
    execute_units_[0].ocl_kernel.setArg(idx++, *((cl::Image *)ocl_bias_->GetData()));

    // unit1
    idx = SetExecuteUnit2DSizeInfoDefault(execute_units_[1], input_dims);
    execute_units_[1].ocl_kernel.setArg(idx++, *((cl::Image *)inputs[0]->GetHandle().base));
    execute_units_[1].ocl_kernel.setArg(idx++, *((cl::Image *)ocl_var_->GetData()));
    execute_units_[1].ocl_kernel.setArg(idx++, *((cl::Image *)ocl_bias_->GetData()));
    //input_width
    execute_units_[1].ocl_kernel.setArg(idx++, DimsFunctionUtils::GetDim(input_dims, 3));
    //input_height
    execute_units_[1].ocl_kernel.setArg(idx++, DimsFunctionUtils::GetDim(input_dims, 2));
    execute_units_[1].ocl_kernel.setArg(idx++, *((cl::Image *)outputs[0]->GetHandle().base));
    return TNN_OK;
}

Status OpenCLInstanceNormLayerAcc::AllocateImage(int batch, int output_channel) {
    OpenCLRuntime *opencl_runtime = OpenCLRuntime::GetInstance();

    int image_size            = UP_DIV(output_channel, 4);
    cl_channel_type data_type = CL_FLOAT;
    if (opencl_runtime->GetPrecision() != PRECISION_HIGH)
        data_type = CL_HALF_FLOAT;

    cl_int ret             = CL_SUCCESS;
    cl::Image2D *image_var = new cl::Image2D(*opencl_runtime->Context(), CL_MEM_READ_WRITE,
                                             cl::ImageFormat(CL_RGBA, data_type), image_size, batch, 0, nullptr, &ret);
    if (ret != CL_SUCCESS) {
        CHECK_CL_SUCCESS(ret)
        if (nullptr != image_var)
            delete image_var;
        return Status(TNNERR_OPENCL_MEMALLOC_ERROR, "OpenCL malloc memory failed");
    }
    ocl_var_.reset(new OpenCLMemory(TNN_CL_IMAGE));
    ocl_var_->SetData(image_var, true);

    cl::Image2D *image_bias = new cl::Image2D(*opencl_runtime->Context(), CL_MEM_READ_WRITE,
                                              cl::ImageFormat(CL_RGBA, data_type), image_size, batch, 0, nullptr, &ret);
    if (ret != CL_SUCCESS) {
        CHECK_CL_SUCCESS(ret)
        if (nullptr != image_bias)
            delete image_bias;
        return Status(TNNERR_OPENCL_MEMALLOC_ERROR, "OpenCL malloc memory failed");
    }
    ocl_bias_.reset(new OpenCLMemory(TNN_CL_IMAGE));
    ocl_bias_->SetData(image_bias, true);

    return TNN_OK;
}

Status OpenCLInstanceNormLayerAcc::BuildVarBiasKernel(int width) {
    Status ret = TNN_OK;
    while (1) {
        std::set<std::string> build_options;
        char temp_str[32];
        memset(temp_str, 0, 32);
        snprintf(temp_str, 31, "-DTHREAD_BLOCK_W=%d", thread_block_w_);
        build_options.emplace(temp_str);
        build_options.insert(build_options_.begin(), build_options_.end());
        std::string kernel_name = "InstanceNormVarBias_LocalMem";
        ret                     = CreateExecuteUnit(execute_units_[0], "instance_norm", kernel_name, build_options);

        if (execute_units_[0].workgroupsize_max >= thread_block_w_ * thread_block_w_) {
            break;
        } else {
            while (execute_units_[0].workgroupsize_max < thread_block_w_ * thread_block_w_) {
                thread_block_w_--;
            }
        }
    }
    return ret;
}

std::vector<uint32_t> OpenCLInstanceNormLayerAcc::GetLocalWS() {
    std::vector<uint32_t> lws(2, 0);

    lws[0] = thread_block_w_ * thread_block_w_;
    lws[1] = 1;

    return lws;
}

#if TNN_PROFILE
double OpenCLInstanceNormLayerAcc::GetFlops() {
    if (0 == bench_idx_) {
        return 1.0 * DimsVectorUtils::Count(output_dims_) * (1 + 3) / 1000.0 / 1000.0;
    } else if (1 == bench_idx_) {
        return 1.0 * DimsVectorUtils::Count(output_dims_) / 1000.0 / 1000.0;
    } else {
        return 0;
    }
}

double OpenCLInstanceNormLayerAcc::GetBandwidth() {
    if (0 == bench_idx_) {
        bench_idx_ = (bench_idx_ + 1) % execute_units_.size();
        return (2.0 * DimsVectorUtils::Count(output_dims_) + 2.0 * output_dims_[1]) / 1000.0 / 1000.0;
    } else if (1 == bench_idx_) {
        bench_idx_ = (bench_idx_ + 1) % execute_units_.size();
        return (DimsVectorUtils::Count(output_dims_) + 2.0 * output_dims_[0] * output_dims_[3]) / 1000.0 / 1000.0;
    } else {
        return 0;
    }
}
#endif

REGISTER_OPENCL_ACC(InstanceNorm, LAYER_INST_BATCH_NORM)
REGISTER_OPENCL_LAYOUT(LAYER_INST_BATCH_NORM, DATA_FORMAT_NHC4W4);

}  // namespace TNN_NS
