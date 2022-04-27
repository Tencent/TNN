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
#include "tnn/device/opencl/acc/opencl_reshape_layer_acc.h"
#include "tnn/device/opencl/imagebuffer_convertor.h"
#include "tnn/device/opencl/opencl_memory.h"
#include "tnn/utils/data_type_utils.h"
#include "tnn/utils/dims_utils.h"

namespace TNN_NS {

class OpenCLInnerProductLayerAcc : public OpenCLLayerAcc {
public:
    virtual Status Init(Context *context, LayerParam *param, LayerResource *resource, const std::vector<Blob *> &inputs,
                        const std::vector<Blob *> &outputs) override;

    virtual ~OpenCLInnerProductLayerAcc() override;

    virtual Status Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) override;

    virtual Status Forward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) override;

private:
    Status InitReshapeLayer(const std::vector<Blob *> &inputs);
    Status ConvertWeights(float *weights_data_ptr, int weight_w, int weight_h);

private:
    int num_output_ = 0;
    int transpose_ = 0;
    int axis_ = 0;
    shared_ptr<OpenCLMemory> ocl_weights_ = nullptr;
    shared_ptr<OpenCLMemory> ocl_bias_ = nullptr;

    bool need_reshape_ = false;
    ReshapeLayerParam reshape_param_;
    shared_ptr<OpenCLReshapeLayerAcc> reshape_layer_acc_ = nullptr;
    std::vector<Blob *> reshape_outputs_ = {};
    shared_ptr<Blob> reshape_output_blob_ = nullptr;
    shared_ptr<cl::Image2D> reshape_output_image_ = nullptr;
};

Status OpenCLInnerProductLayerAcc::Init(Context *context, LayerParam *param, LayerResource *resource,
                                        const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    LOGD("Init InnerProduct Acc\n");
    Status ret = OpenCLLayerAcc::Init(context, param, resource, inputs, outputs);
    CHECK_TNN_OK(ret)

    run_3d_ndrange_ = false;
    op_name_        = "InnerProduct";

    InnerProductLayerParam *innerproduct_param = dynamic_cast<InnerProductLayerParam *>(param);
    CHECK_PARAM_NULL(innerproduct_param);

    num_output_  = innerproduct_param->num_output;
    int has_bias = innerproduct_param->has_bias;
    transpose_   = innerproduct_param->transpose;
    axis_        = innerproduct_param->axis;

    InnerProductLayerResource *innerproduct_resource = dynamic_cast<InnerProductLayerResource *>(resource);
    CHECK_PARAM_NULL(innerproduct_resource);
    RawBuffer &weight_handle = innerproduct_resource->weight_handle;
    RawBuffer &bias_handle   = innerproduct_resource->bias_handle;
    DataType data_type       = weight_handle.GetDataType();

    // get weights
    int weights_height = weight_handle.GetBytesSize() / DataTypeUtils::GetBytesSize(data_type) / num_output_;
    int weights_width  = num_output_;
    if (weight_handle.GetDataType() == DATA_TYPE_FLOAT) {
        // get float pointer from raw buffer.
        float *weights_data_ptr = weight_handle.force_to<float *>();
        if (weights_data_ptr == nullptr) {
            return Status(TNNERR_OPENCL_ACC_INIT_ERROR, "pointer is null");
        }
        ret = ConvertWeights(weights_data_ptr, weights_width, weights_height);
        CHECK_TNN_OK(ret)
    } else {
        // if handle is half, need convert to float first.
        auto float_data_ptr = GetFloatFromRawBuffer(weight_handle);
        if (float_data_ptr == nullptr) {
            return Status(TNNERR_OPENCL_ACC_INIT_ERROR, "pointer is null");
        }
        ret = ConvertWeights(float_data_ptr.get(), weights_width, weights_height);
        CHECK_TNN_OK(ret)
    }

    // get bias
    ret = ConvertChannelWeights(innerproduct_resource->bias_handle, ocl_bias_, num_output_, has_bias);
    CHECK_TNN_OK(ret)

    // create kernel
    std::string kernel_name = "Innerproduct";
    ret                     = CreateExecuteUnit(execute_units_[0], "innerproduct", kernel_name, build_options_);
    if (ret != TNN_OK) {
        LOGE("create execute unit failed!\n");
        return ret;
    }

    return TNN_OK;
}

OpenCLInnerProductLayerAcc::~OpenCLInnerProductLayerAcc() {}

Status OpenCLInnerProductLayerAcc::Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    LOGD("InnerProduct Layer Reshape\n");
    ASSERT(inputs.size() == 1);
    Status ret = OpenCLLayerAcc::Reshape(inputs, outputs);
    CHECK_TNN_OK(ret)

    auto input_dims     = inputs[0]->GetBlobDesc().dims;
    auto output_dims    = outputs[0]->GetBlobDesc().dims;
    auto output_height  = DimsFunctionUtils::GetDim(output_dims, 2);
    auto output_width   = DimsFunctionUtils::GetDim(output_dims, 3);
    auto input_height   = DimsFunctionUtils::GetDim(input_dims, 2);
    auto input_width    = DimsFunctionUtils::GetDim(input_dims, 3);
    // now only support axis is channel, output width and output height is 1.
    if (axis_ != 1 || output_height != 1 || output_width != 1) {
        LOGE("Invalid InnerParameter param or input/output size!\n");
        return Status(TNNERR_OPENCL_ACC_RESHAPE_ERROR, "Invalid InnerParameter param or input/output size!");
    }

    // if input width and input height is not 1, need reshape first.
    if (input_height != 1 || input_width != 1) {
        need_reshape_ = true;
    }

    // init
    if (need_reshape_) {
        ret = InitReshapeLayer(inputs);
        CHECK_TNN_OK(ret)
    }

    // reshape
    if (need_reshape_) {
        if (reshape_layer_acc_ == nullptr) {
            return Status(TNNERR_OPENCL_ACC_RESHAPE_ERROR, "reshape layer acc in InnerProduct is null");
        }
        ret = reshape_layer_acc_->Reshape(inputs, reshape_outputs_);
        CHECK_TNN_OK(ret)
    }

    // calcuate M,K,N
    int N = num_output_;
    int M = DimsVectorUtils::Count(inputs[0]->GetBlobDesc().dims, 0, axis_);
    int K = DimsVectorUtils::Count(inputs[0]->GetBlobDesc().dims, axis_);

    const int K_blocks = UP_DIV(K, 4);
    const int remain   = K % 4;

    execute_units_[0].global_work_size = {static_cast<uint32_t>(UP_DIV(N, 4)), static_cast<uint32_t>(M)};
    execute_units_[0].local_work_size  = {64, 1};

    uint32_t idx = 0;
    execute_units_[0].ocl_kernel.setArg(idx++, execute_units_[0].global_work_size[0]);
    execute_units_[0].ocl_kernel.setArg(idx++, execute_units_[0].global_work_size[1]);
    if (need_reshape_) {
        execute_units_[0].ocl_kernel.setArg(idx++, *((cl::Image *)reshape_outputs_[0]->GetHandle().base));
    } else {
        execute_units_[0].ocl_kernel.setArg(idx++, *((cl::Image *)inputs[0]->GetHandle().base));
    }
    execute_units_[0].ocl_kernel.setArg(idx++, *((cl::Image *)ocl_weights_->GetData()));
    execute_units_[0].ocl_kernel.setArg(idx++, *((cl::Image *)ocl_bias_->GetData()));
    execute_units_[0].ocl_kernel.setArg(idx++, K_blocks);
    execute_units_[0].ocl_kernel.setArg(idx++, remain);
    execute_units_[0].ocl_kernel.setArg(idx++, *((cl::Image *)outputs[0]->GetHandle().base));

    return TNN_OK;
}

Status OpenCLInnerProductLayerAcc::Forward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    Status ret = TNN_OK;
    if (need_reshape_) {
        // reshape first
        if (reshape_layer_acc_ == nullptr) {
            return Status(TNNERR_OPENCL_ACC_FORWARD_ERROR, "reshape layer acc in InnerProduct is null");
        }
        ret = reshape_layer_acc_->Forward(inputs, reshape_outputs_);
        CHECK_TNN_OK(ret)
    }

    return OpenCLLayerAcc::Forward(inputs, outputs);
}

Status OpenCLInnerProductLayerAcc::InitReshapeLayer(const std::vector<Blob *> &inputs) {
    Status ret = TNN_OK;

    reshape_layer_acc_ = std::make_shared<OpenCLReshapeLayerAcc>();
    if (reshape_layer_acc_ == nullptr) {
        LOGE("Create Reshape Layer Acc in InnerProduct failed!\n");
        return Status(TNNERR_CREATE_LAYER, "Create Reshape Layer Acc in InnerProduct failed!");
    }

    // create output_blob
    BlobDesc output_desc    = inputs[0]->GetBlobDesc();
    output_desc.data_format = DATA_FORMAT_NHC4W4;
    auto dims               = inputs[0]->GetBlobDesc().dims;
    output_desc.dims[0]     = DimsFunctionUtils::GetDim(dims, 0);
    output_desc.dims[1]     = DimsFunctionUtils::GetDim(dims, 1) * DimsFunctionUtils::GetDim(dims, 2) * DimsFunctionUtils::GetDim(dims, 3);
    output_desc.dims[2]     = 1;
    output_desc.dims[3]     = 1;
    reshape_output_blob_    = std::make_shared<Blob>(output_desc);
    if (reshape_output_blob_ == nullptr) {
        LOGE("Create reshape output blob in InnerProduct failed!\n");
        return Status(TNNERR_CREATE_LAYER, "Create reshape output blob in InnerProduct failed!");
    }
    reshape_outputs_.clear();
    reshape_outputs_.push_back(reshape_output_blob_.get());

    // create output_image
    OpenCLRuntime *opencl_runtime = OpenCLRuntime::GetInstance();
    DimsVector imageshape{(int)(UP_DIV(DimsFunctionUtils::GetDim(output_desc.dims, 1), 4)),
        DimsFunctionUtils::GetDim(output_desc.dims, 0)};
    cl_channel_type data_type = CL_FLOAT;
    if (opencl_runtime->GetPrecision() != PRECISION_HIGH)
        data_type = CL_HALF_FLOAT;
    cl_int err            = CL_SUCCESS;
    reshape_output_image_ = std::make_shared<cl::Image2D>(*opencl_runtime->Context(), CL_MEM_READ_WRITE,
                                                          cl::ImageFormat(CL_RGBA, data_type), imageshape[0],
                                                          imageshape[1], 0, nullptr, &err);
    if (err != CL_SUCCESS) {
        CHECK_CL_SUCCESS(err)
        return Status(TNNERR_OPENCL_MEMALLOC_ERROR, "OpenCL malloc memory failed");
    }
    BlobHandle blob_handle;
    blob_handle.base = reshape_output_image_.get();
    reshape_output_blob_->SetHandle(blob_handle);

    // Init LayerAcc
    reshape_param_.name         = layer_name_ + "_Reshape";
    reshape_param_.reshape_type = 0;
    reshape_param_.axis         = 0;
    reshape_param_.num_axes     = 4;
    reshape_param_.shape        = {0, -1, 1, 1};
    reshape_layer_acc_->Init(ocl_context_, &reshape_param_, nullptr, inputs, reshape_outputs_);

    return ret;
}

Status OpenCLInnerProductLayerAcc::ConvertWeights(float *weights_data_ptr, int weight_w, int weight_h) {
    OpenCLRuntime *opencl_runtime = OpenCLRuntime::GetInstance();

    // traspose
    shared_ptr<float> weights_data_ptr_trans(new float[weight_w * weight_h]);
    for (size_t i = 0; i < weight_h; i++) {
        for (size_t j = 0; j < weight_w; j++) {
            weights_data_ptr_trans.get()[j + i * weight_w] = weights_data_ptr[i + j * weight_h];
        }
    }

    // copy weights data into clBuffer
    DimsVector weight_shape{weight_h, weight_w, 1, 1};
    shared_ptr<OpenCLMemory> weight_buffer(new OpenCLMemory(TNN_CL_BUFFER));
    cl_int ret = CL_SUCCESS;
    cl::Buffer buffer(*opencl_runtime->Context(), CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR,
                      DimsVectorUtils::Count(weight_shape) * sizeof(float), nullptr, &ret);
    if (ret != CL_SUCCESS) {
        CHECK_CL_SUCCESS(ret)
        return Status(TNNERR_OPENCL_MEMALLOC_ERROR, "OpenCL malloc memory failed");
    }
    weight_buffer->SetData(&buffer);
    auto weight_clbuffer_ptr = ocl_context_->CommandQueue()->enqueueMapBuffer(
        buffer, true, CL_MAP_WRITE, 0, DimsVectorUtils::Count(weight_shape) * sizeof(float), nullptr, nullptr, &ret);
    if (ret != CL_SUCCESS) {
        CHECK_CL_SUCCESS(ret)
        return Status(TNNERR_OPENCL_MEMMAP_ERROR, "OpenCL MemMap failed");
    }
    memcpy(weight_clbuffer_ptr, weights_data_ptr_trans.get(), DimsVectorUtils::Count(weight_shape) * sizeof(float));
    ret = ocl_context_->CommandQueue()->enqueueUnmapMemObject(buffer, weight_clbuffer_ptr);
    if (ret != CL_SUCCESS) {
        CHECK_CL_SUCCESS(ret)
        return Status(TNNERR_OPENCL_MEMUNMAP_ERROR, "OpenCL MemUnMap failed");
    }

    // create ocl_weights_
    DimsVector weight_imageshape{(int)(UP_DIV(weight_w, 4)), weight_h};
    cl_channel_type data_type = CL_FLOAT;
    if (opencl_runtime->GetPrecision() != PRECISION_HIGH)
        data_type = CL_HALF_FLOAT;
    cl::Image2D *image =
        new cl::Image2D(*opencl_runtime->Context(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_RGBA, data_type),
                        weight_imageshape[0], weight_imageshape[1], 0, nullptr, &ret);
    if (ret != CL_SUCCESS) {
        CHECK_CL_SUCCESS(ret)
        if (nullptr != image)
            delete image;
        return Status(TNNERR_OPENCL_MEMALLOC_ERROR, "OpenCL malloc memory failed");
    }
    ocl_weights_.reset(new OpenCLMemory(TNN_CL_IMAGE));
    ocl_weights_->SetData(image, true);

    // transfer from clBuffer to clImage
    ImageBufferConvertor convertor(opencl_runtime, ocl_context_->CommandQueue());
    return convertor.ConvertBufferToImage(weight_buffer.get(), NHWC_BUFFER, weight_shape, ocl_weights_.get(), true);
}

REGISTER_OPENCL_ACC(InnerProduct, LAYER_INNER_PRODUCT)
REGISTER_OPENCL_LAYOUT(LAYER_INNER_PRODUCT, DATA_FORMAT_NHC4W4);

}  // namespace TNN_NS
