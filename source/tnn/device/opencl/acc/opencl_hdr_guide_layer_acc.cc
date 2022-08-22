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
#include "tnn/utils/data_type_utils.h"

namespace TNN_NS {

class OpenCLHdrGuideLayerAcc : public OpenCLLayerAcc {
public:
    virtual Status Init(Context *context, LayerParam *param, LayerResource *resource, const std::vector<Blob *> &inputs,
                        const std::vector<Blob *> &outputs) override;

    virtual ~OpenCLHdrGuideLayerAcc() override;

    virtual Status Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) override;

private:
    Status ConvertWeights(shared_ptr<OpenCLMemory> &ocl_memory, float *weight, float *bias, int pixel_count);
    Status ConvertTrans(shared_ptr<OpenCLMemory> &ocl_blob, float *data_ptr, float default_val);
    bool InputParamCheck(RawBuffer &ccm_weight_handle, RawBuffer &ccm_bias_handle,
                         RawBuffer &shifts_handle, RawBuffer &slopes_handle,
                         RawBuffer &p_weight_handle, RawBuffer &p_bias_handle);

private:
    shared_ptr<OpenCLMemory> ocl_ccm_ = nullptr;
    shared_ptr<OpenCLMemory> ocl_shifts_ = nullptr;
    shared_ptr<OpenCLMemory> ocl_slopes_ = nullptr;
    shared_ptr<OpenCLMemory> ocl_projection_ = nullptr;
};

bool OpenCLHdrGuideLayerAcc::InputParamCheck(
        RawBuffer &ccm_weight_handle, RawBuffer &ccm_bias_handle,
        RawBuffer &shifts_handle, RawBuffer &slopes_handle,
        RawBuffer &p_weight_handle, RawBuffer &p_bias_handle) {
    return (ccm_weight_handle.GetDataCount() != 9 || ccm_bias_handle.GetDataCount() != 3 ||
            shifts_handle.GetDataCount() != 12 || slopes_handle.GetDataCount() != 12 ||
            p_weight_handle.GetDataCount() != 3 || p_bias_handle.GetDataCount() != 1);
}

Status OpenCLHdrGuideLayerAcc::Init(Context *context, LayerParam *param, LayerResource *resource,
                                    const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    LOGD("Init HDRGuide Acc\n");
    Status ret = OpenCLLayerAcc::Init(context, param, resource, inputs, outputs);
    CHECK_TNN_OK(ret)

    run_3d_ndrange_ = false;
    op_name_        = "HDRGuide";

    HdrGuideLayerResource *hdr_guide_resource = dynamic_cast<HdrGuideLayerResource *>(resource);
    RawBuffer &ccm_weight_handle              = hdr_guide_resource->ccm_weight_handle;
    RawBuffer &ccm_bias_handle                = hdr_guide_resource->ccm_bias_handle;
    RawBuffer &shifts_handle                  = hdr_guide_resource->shifts_handle;
    RawBuffer &slopes_handle                  = hdr_guide_resource->slopes_handle;
    RawBuffer &p_weight_handle                = hdr_guide_resource->projection_weight_handle;
    RawBuffer &p_bias_handle                  = hdr_guide_resource->projection_bias_handle;
    if (InputParamCheck(ccm_weight_handle, ccm_bias_handle, shifts_handle,
                        slopes_handle, p_weight_handle, p_bias_handle)) {
        LOGE("Invalid data size of HDRGuide Param!\n");
        return Status(TNNERR_OPENCL_ACC_INIT_ERROR, "Invalid data size of HDRGuide Param!");
    }

    // get ccm weight
    if (ccm_weight_handle.GetDataType() == DATA_TYPE_FLOAT) {
        float *weight_ptr = ccm_weight_handle.force_to<float *>();
        float *bias_ptr   = ccm_bias_handle.force_to<float *>();
        if (weight_ptr == nullptr || bias_ptr == nullptr) {
            return Status(TNNERR_OPENCL_ACC_INIT_ERROR, "pointer is null");
        }
        ret = ConvertWeights(ocl_ccm_, weight_ptr, bias_ptr, 3);
        CHECK_TNN_OK(ret)
    } else {
        auto weight_ptr = GetFloatFromRawBuffer(ccm_weight_handle);  // handle the memory
        auto bias_ptr   = GetFloatFromRawBuffer(ccm_bias_handle);    // handle the memory
        if (weight_ptr == nullptr || bias_ptr == nullptr) {
            return Status(TNNERR_OPENCL_ACC_INIT_ERROR, "pointer is null");
        }
        ret = ConvertWeights(ocl_ccm_, weight_ptr.get(), bias_ptr.get(), 3);
        CHECK_TNN_OK(ret)
    }

    // get ocl_shifts_
    if (shifts_handle.GetDataType() == DATA_TYPE_FLOAT) {
        float *data_ptr = shifts_handle.force_to<float *>();
        if (data_ptr == nullptr) {
            return Status(TNNERR_OPENCL_ACC_INIT_ERROR, "pointer is null");
        }
        ret = ConvertTrans(ocl_shifts_, data_ptr, 0);
        CHECK_TNN_OK(ret)
    } else {
        auto data_ptr = GetFloatFromRawBuffer(shifts_handle);  // handle the memory
        if (data_ptr == nullptr) {
            return Status(TNNERR_OPENCL_ACC_INIT_ERROR, "pointer is null");
        }
        ret = ConvertTrans(ocl_shifts_, data_ptr.get(), 0);
        CHECK_TNN_OK(ret)
    }

    // get ocl_slopes_
    if (slopes_handle.GetDataType() == DATA_TYPE_FLOAT) {
        float *data_ptr = slopes_handle.force_to<float *>();
        if (data_ptr == nullptr) {
            return Status(TNNERR_OPENCL_ACC_INIT_ERROR, "pointer is null");
        }
        ret = ConvertTrans(ocl_slopes_, data_ptr, 1);
        CHECK_TNN_OK(ret)
    } else {
        auto data_ptr = GetFloatFromRawBuffer(slopes_handle);  // handle the memory
        if (data_ptr == nullptr) {
            return Status(TNNERR_OPENCL_ACC_INIT_ERROR, "pointer is null");
        }
        ret = ConvertTrans(ocl_slopes_, data_ptr.get(), 1);
        CHECK_TNN_OK(ret)
    }

    // get projection weight
    if (p_weight_handle.GetDataType() == DATA_TYPE_FLOAT) {
        float *weight_ptr = p_weight_handle.force_to<float *>();
        float *bias_ptr   = p_bias_handle.force_to<float *>();
        if (weight_ptr == nullptr || bias_ptr == nullptr) {
            return Status(TNNERR_OPENCL_ACC_INIT_ERROR, "pointer is null");
        }
        ret = ConvertWeights(ocl_projection_, weight_ptr, bias_ptr, 1);
        CHECK_TNN_OK(ret)
    } else {
        auto weight_ptr = GetFloatFromRawBuffer(p_weight_handle);  // handle the memory
        auto bias_ptr   = GetFloatFromRawBuffer(p_bias_handle);    // handle the memory
        if (weight_ptr == nullptr || bias_ptr == nullptr) {
            return Status(TNNERR_OPENCL_ACC_INIT_ERROR, "pointer is null");
        }
        ret = ConvertWeights(ocl_projection_, weight_ptr.get(), bias_ptr.get(), 1);
        CHECK_TNN_OK(ret)
    }

    // create kernel
    std::string kernel_name = "HdrGuide";
    ret                     = CreateExecuteUnit(execute_units_[0], "hdr_guide", kernel_name, build_options_);
    if (ret != TNN_OK) {
        LOGE("create execute unit failed!\n");
        return ret;
    }

    return TNN_OK;
}

OpenCLHdrGuideLayerAcc::~OpenCLHdrGuideLayerAcc() {}

Status OpenCLHdrGuideLayerAcc::Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    LOGD("HDRGuide Layer Reshape\n");
    ASSERT(inputs.size() == 1);
    Status ret = OpenCLLayerAcc::Reshape(inputs, outputs);
    CHECK_TNN_OK(ret)

    auto input_dims  = inputs[0]->GetBlobDesc().dims;
    auto output_dims = outputs[0]->GetBlobDesc().dims;
    uint32_t idx     = SetExecuteUnit2DSizeInfoDefault(execute_units_[0], output_dims);
    execute_units_[0].ocl_kernel.setArg(idx++, *((cl::Image *)inputs[0]->GetHandle().base));
    execute_units_[0].ocl_kernel.setArg(idx++, *((cl::Image *)outputs[0]->GetHandle().base));
    execute_units_[0].ocl_kernel.setArg(idx++, *((cl::Image *)ocl_ccm_->GetData()));
    execute_units_[0].ocl_kernel.setArg(idx++, *((cl::Image *)ocl_shifts_->GetData()));
    execute_units_[0].ocl_kernel.setArg(idx++, *((cl::Image *)ocl_slopes_->GetData()));
    execute_units_[0].ocl_kernel.setArg(idx++, *((cl::Image *)ocl_projection_->GetData()));

    return TNN_OK;
}

Status OpenCLHdrGuideLayerAcc::ConvertWeights(shared_ptr<OpenCLMemory> &ocl_memory, float *weight, float *bias,
                                              int pixel_count) {
    OpenCLRuntime *opencl_runtime = OpenCLRuntime::GetInstance();

    int buffer_size = pixel_count * 4;
    cl_int ret      = CL_SUCCESS;
    cl::Buffer clbuffer(*opencl_runtime->Context(), CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR,
                        buffer_size * sizeof(float), nullptr, &ret);
    if (ret != CL_SUCCESS) {
        CHECK_CL_SUCCESS(ret)
        return Status(TNNERR_OPENCL_MEMALLOC_ERROR, "OpenCL malloc memory failed");
    }
    float *clbuffer_ptr = static_cast<float *>(ocl_context_->CommandQueue()->enqueueMapBuffer(
        clbuffer, true, CL_MAP_WRITE, 0, buffer_size * sizeof(float), nullptr, nullptr, &ret));
    if (ret != CL_SUCCESS) {
        CHECK_CL_SUCCESS(ret)
        return Status(TNNERR_OPENCL_MEMMAP_ERROR, "OpenCL MemMap failed");
    }

    for (int y = 0; y < pixel_count; ++y) {
        float *data_ptr = clbuffer_ptr + y * 4;
        for (int x = 0; x < 3; ++x) {
            data_ptr[x] = weight[y * 3 + x];
        }
        data_ptr[3] = bias[y];
    }

    ret = ocl_context_->CommandQueue()->enqueueUnmapMemObject(clbuffer, clbuffer_ptr);
    if (ret != CL_SUCCESS) {
        CHECK_CL_SUCCESS(ret)
        return Status(TNNERR_OPENCL_MEMUNMAP_ERROR, "OpenCL MemUnMap failed");
    }

    // create ocl_blob
    int image_w               = pixel_count;
    int image_h               = 1;
    cl_channel_type data_type = CL_FLOAT;
    if (opencl_runtime->GetPrecision() != PRECISION_HIGH)
        data_type = CL_HALF_FLOAT;
    cl::Image2D *image = new cl::Image2D(*opencl_runtime->Context(), CL_MEM_READ_WRITE,
                                         cl::ImageFormat(CL_RGBA, data_type), image_w, image_h, 0, nullptr, &ret);
    if (ret != CL_SUCCESS) {
        CHECK_CL_SUCCESS(ret)
        if (nullptr != image)
            delete image;
        return Status(TNNERR_OPENCL_MEMALLOC_ERROR, "OpenCL malloc memory failed");
    }
    ocl_memory.reset(new OpenCLMemory(TNN_CL_IMAGE));
    ocl_memory->SetData(image, true);

    // convert
    shared_ptr<OpenCLMemory> input_memory(new OpenCLMemory(TNN_CL_BUFFER));
    input_memory->SetData(&clbuffer);
    ImageBufferConvertor convertor(opencl_runtime, ocl_context_->CommandQueue());
    return convertor.ConvertBufferToImage(input_memory.get(), ARGUMENT, {buffer_size}, ocl_memory.get(), true);
}

Status OpenCLHdrGuideLayerAcc::ConvertTrans(shared_ptr<OpenCLMemory> &ocl_blob, float *weight, float default_val) {
    OpenCLRuntime *ocl_runtime = OpenCLRuntime::GetInstance();

    int buf_size   = 4 * 4;
    cl_int ocl_ret = CL_SUCCESS;
    cl::Buffer clbuf(*ocl_runtime->Context(), CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, buf_size * sizeof(float),
                     nullptr, &ocl_ret);
    if (ocl_ret != CL_SUCCESS) {
        CHECK_CL_SUCCESS(ocl_ret)
        return Status(TNNERR_OPENCL_MEMALLOC_ERROR, "OpenCL malloc memory failed");
    }
    float *clbuf_ptr = static_cast<float *>(ocl_context_->CommandQueue()->enqueueMapBuffer(
        clbuf, true, CL_MAP_WRITE, 0, buf_size * sizeof(float), nullptr, nullptr, &ocl_ret));
    if (ocl_ret != CL_SUCCESS) {
        CHECK_CL_SUCCESS(ocl_ret)
        return Status(TNNERR_OPENCL_MEMMAP_ERROR, "OpenCL MemMap failed");
    }

    for (int y = 0; y < 4; ++y) {
        float *data_ptr = clbuf_ptr + y * 4;
        for (int x = 0; x < 3; ++x) {
            data_ptr[x] = weight[y + x * 4];
        }
        data_ptr[3] = default_val;
    }

    ocl_ret = ocl_context_->CommandQueue()->enqueueUnmapMemObject(clbuf, clbuf_ptr);
    if (ocl_ret != CL_SUCCESS) {
        CHECK_CL_SUCCESS(ocl_ret)
        return Status(TNNERR_OPENCL_MEMUNMAP_ERROR, "OpenCL MemUnMap failed");
    }

    // create ocl_blob
    int img_width          = 4;
    int img_height         = 1;
    cl_channel_type data_t = CL_FLOAT;
    if (ocl_runtime->GetPrecision() != PRECISION_HIGH)
        data_t = CL_HALF_FLOAT;
    cl::Image2D *img = new cl::Image2D(*ocl_runtime->Context(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_RGBA, data_t),
                                       img_width, img_height, 0, nullptr, &ocl_ret);
    if (ocl_ret != CL_SUCCESS) {
        CHECK_CL_SUCCESS(ocl_ret)
        if (nullptr != img)
            delete img;
        return Status(TNNERR_OPENCL_MEMALLOC_ERROR, "OpenCL malloc memory failed");
    }
    ocl_blob.reset(new OpenCLMemory(TNN_CL_IMAGE));
    ocl_blob->SetData(img, true);

    // convert
    shared_ptr<OpenCLMemory> input(new OpenCLMemory(TNN_CL_BUFFER));
    input->SetData(&clbuf);
    ImageBufferConvertor convertor(ocl_runtime, ocl_context_->CommandQueue());
    return convertor.ConvertBufferToImage(input.get(), ARGUMENT, {buf_size}, ocl_blob.get(), true);
}

REGISTER_OPENCL_ACC(HdrGuide, LAYER_HDRGUIDE)
REGISTER_OPENCL_LAYOUT(LAYER_HDRGUIDE, DATA_FORMAT_NHC4W4);

}  // namespace TNN_NS
