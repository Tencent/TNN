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
// specific language governing permissions and limitations under the License./

#include "tnn/device/opencl/acc/opencl_layer_acc.h"
#include "tnn/device/opencl/imagebuffer_convertor.h"
#include "tnn/device/opencl/opencl_memory.h"
#include "tnn/device/opencl/opencl_utils.h"
#include "tnn/utils/pribox_generator_utils.h"

namespace TNN_NS {

class OpenCLPriorBoxLayerAcc : public OpenCLLayerAcc {
public:
    virtual ~OpenCLPriorBoxLayerAcc();

    Status Init(Context *context, LayerParam *param, LayerResource *resource, const std::vector<Blob *> &inputs,
                const std::vector<Blob *> &outputs);

    virtual Status Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs);

    virtual Status Forward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs);

private:
    Status ConvertPriorBox(std::vector<float> &priorbox, DimsVector dims);

private:
    shared_ptr<OpenCLMemory> ocl_priorbox_ = nullptr;
    PriorBoxLayerParam *param_ = nullptr;
    const int PRIORBOX_CHANNEL = 2;
};

OpenCLPriorBoxLayerAcc::~OpenCLPriorBoxLayerAcc(){};

Status OpenCLPriorBoxLayerAcc::Init(Context *context, LayerParam *param, LayerResource *resource,
                                    const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    Status ret = OpenCLLayerAcc::Init(context, param, resource, inputs, outputs);
    CHECK_TNN_OK(ret)

    run_3d_ndrange_ = false;
    op_name_        = "PriorBox";

    param_ = dynamic_cast<PriorBoxLayerParam *>(param);
    if (!param_) {
        return Status(TNNERR_MODEL_ERR, "Error: PriorBoxLayerParam is empyt");
    }

    return TNN_OK;
}

Status OpenCLPriorBoxLayerAcc::Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    std::vector<float> priorbox = GeneratePriorBox(inputs, outputs, param_);
    auto dims = outputs[0]->GetBlobDesc().dims;
    Status ret                  = ConvertPriorBox(priorbox, dims);
    return ret;
}

Status OpenCLPriorBoxLayerAcc::Forward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    OpenCLRuntime *opencl_runtime = OpenCLRuntime::GetInstance();
    DimsVector dims               = GetImageShape(ocl_priorbox_.get());

    Status ret = TNN_OK;
#if TNN_PROFILE
    std::shared_ptr<OpenCLProfilingData> pdata(new OpenCLProfilingData());
    UpdateProfilingData(pdata.get(), {}, {});

    ret = CopyImageToImage(opencl_runtime, ocl_context_, *((cl::Image *)ocl_priorbox_->GetData()),
                           *((cl::Image *)outputs[0]->GetHandle().base), dims[0], dims[1], false, pdata.get());
    ocl_context_->AddProfilingData(pdata);
#else
    ret = CopyImageToImage(opencl_runtime, ocl_context_, *((cl::Image *)ocl_priorbox_->GetData()),
                           *((cl::Image *)outputs[0]->GetHandle().base), dims[0], dims[1], false);
#endif
    return ret;
}

Status OpenCLPriorBoxLayerAcc::ConvertPriorBox(std::vector<float> &priorbox, DimsVector dims) {
    OpenCLRuntime *opencl_runtime = OpenCLRuntime::GetInstance();
    cl_int ret                    = CL_SUCCESS;

    shared_ptr<OpenCLMemory> priorbox_buffer(new OpenCLMemory(TNN_CL_BUFFER));

    cl::Buffer priorbox_clbuffer(*opencl_runtime->Context(), CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR,
                                 priorbox.size() * sizeof(float), nullptr, &ret);

    priorbox_buffer->SetData(&priorbox_clbuffer);

    if (ret != CL_SUCCESS) {
        CHECK_CL_SUCCESS(ret)
        return Status(TNNERR_OPENCL_MEMALLOC_ERROR, "OpenCL malloc memory falied");
    }
    auto priorbox_clbuffer_ptr = ocl_context_->CommandQueue()->enqueueMapBuffer(
        priorbox_clbuffer, true, CL_MAP_WRITE, 0, priorbox.size() * sizeof(float), nullptr, nullptr, &ret);
    if (ret != CL_SUCCESS) {
        CHECK_CL_SUCCESS(ret)
        return Status(TNNERR_OPENCL_MEMMAP_ERROR, "OpenCL MemMap failed");
    }

    memcpy(priorbox_clbuffer_ptr, priorbox.data(), priorbox.size() * sizeof(float));
    ret = ocl_context_->CommandQueue()->enqueueUnmapMemObject(priorbox_clbuffer, priorbox_clbuffer_ptr);
    if (ret != CL_SUCCESS) {
        CHECK_CL_SUCCESS(ret)
        return Status(TNNERR_OPENCL_MEMUNMAP_ERROR, "OpenCL MemUnMap falied");
    }

    // create ocl_pribox_
    int ocl_priorbox_w        = UP_DIV(PRIORBOX_CHANNEL, 4);
    int ocl_priorbox_h        = priorbox.size() / PRIORBOX_CHANNEL;
    cl_channel_type data_type = CL_FLOAT;
    if (opencl_runtime->GetPrecision() != PRECISION_HIGH)
        data_type = CL_HALF_FLOAT;
    cl::Image2D *image =
        new cl::Image2D(*opencl_runtime->Context(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_RGBA, data_type),
                        ocl_priorbox_w, ocl_priorbox_h, 0, nullptr, &ret);
    if (ret != CL_SUCCESS) {
        CHECK_CL_SUCCESS(ret)
        if (nullptr != image)
            delete image;
        return Status(TNNERR_OPENCL_MEMALLOC_ERROR, "OpenCL malloc memory falied");
    }
    ocl_priorbox_.reset(new OpenCLMemory(TNN_CL_IMAGE));
    ocl_priorbox_->SetData(image, true);

    ImageBufferConvertor convertor(opencl_runtime, ocl_context_->CommandQueue());
    return convertor.ConvertBufferToImage(priorbox_buffer.get(), NCHW_BUFFER, dims, ocl_priorbox_.get(), true);
}

REGISTER_OPENCL_ACC(PriorBox, LAYER_PRIOR_BOX)

}  // namespace TNN_NS
