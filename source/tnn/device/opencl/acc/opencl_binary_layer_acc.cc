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
#include "tnn/utils/dims_utils.h"
#include "tnn/utils/data_type_utils.h"

namespace TNN_NS {

Status OpenCLBinaryLayerAcc::Init(Context *context, LayerParam *param, LayerResource *resource,
                                  const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    LOGD("Init Binary Acc\n");

    output_dims_size_ = outputs[0]->GetBlobDesc().dims.size();
    Status ret = OpenCLLayerAcc::Init(context, param, resource, inputs, outputs);
    CHECK_TNN_OK(ret)

    run_3d_ndrange_ = false;

    auto broadcast_param = dynamic_cast<MultidirBroadcastLayerParam *>(param);
    CHECK_PARAM_NULL(broadcast_param);
    broadcast_param_ = *broadcast_param;

    EltwiseLayerResource *layer_res = dynamic_cast<EltwiseLayerResource *>(resource);
    if (layer_res == nullptr) {
        if (inputs.size() == 2) {
            if (broadcast_param_.input0_broadcast_type == BroadcastTypeNormal &&
                broadcast_param_.input1_broadcast_type == BroadcastTypeNormal) {
                // inputs[0] and inputs[1] are equal
                input_idx_ = 0;
                param_idx_ = 1;
            } else if (broadcast_param_.input0_broadcast_type != BroadcastTypeNormal &&
                       broadcast_param_.input1_broadcast_type == BroadcastTypeNormal) {
                // inputs[0] is the param
                input_idx_ = 1;
                param_idx_ = 0;

            } else if (broadcast_param_.input0_broadcast_type == BroadcastTypeNormal &&
                       broadcast_param_.input1_broadcast_type != BroadcastTypeNormal) {
                // inputs[1] is the param
                input_idx_ = 0;
                param_idx_ = 1;

            } else {
                return Status(TNNERR_PARAM_ERR, "input dims is illegal");
            }

        } else {
            return Status(TNNERR_PARAM_ERR, "inputs size shound be 2 without binary resource");
        }
    } else {
        auto param_dims = layer_res->element_shape;
        input_idx_      = 0;
        if (inputs.size() != 1) {
            return Status(TNNERR_PARAM_ERR, "input size should be 1");
        }

        int diff = output_dims_size_ - param_dims.size();
        for (int i = 0; i < diff; i++) {
            param_dims.insert(param_dims.begin(), 1);
        }

        if (layer_res->element_handle.GetDataType() == DATA_TYPE_FLOAT) {
            float *data_ptr = layer_res->element_handle.force_to<float *>();
            ret             = ConvertParam(data_ptr, param_dims);
            CHECK_TNN_OK(ret)
        } else {
            auto float_data_ptr = GetFloatFromRawBuffer(layer_res->element_handle);  // handle the memory
            if (float_data_ptr == nullptr) {
                return Status(TNNERR_OPENCL_ACC_INIT_ERROR, "convert res to float failed");
            }
            ret = ConvertParam(float_data_ptr.get(), param_dims);
            CHECK_TNN_OK(ret)
        }
    }

    kernel_name_ = GetKernelName(broadcast_param_);

    return TNN_OK;
}

OpenCLBinaryLayerAcc::~OpenCLBinaryLayerAcc() {}

Status OpenCLBinaryLayerAcc::Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    LOGD("Binary Acc Reshape\n");
    Status ret = OpenCLLayerAcc::Reshape(inputs, outputs);
    CHECK_TNN_OK(ret)

    auto output_dims = outputs[0]->GetBlobDesc().dims;

    kernel_arg_idx_ = SetExecuteUnit2DSizeInfoDefault(execute_units_[0], output_dims);
    // set input0 and input1
    if (inputs.size() == 2) {
        execute_units_[0].ocl_kernel.setArg(kernel_arg_idx_++, *((cl::Image *)inputs[input_idx_]->GetHandle().base));
        execute_units_[0].ocl_kernel.setArg(kernel_arg_idx_++, *((cl::Image *)inputs[param_idx_]->GetHandle().base));
    } else {
        if (broadcast_param_.input0_broadcast_type == BroadcastTypeNormal && broadcast_param_.weight_input_index == 0) {
            execute_units_[0].ocl_kernel.setArg(kernel_arg_idx_++, *((cl::Image *)binary_params_->GetData()));
            execute_units_[0].ocl_kernel.setArg(kernel_arg_idx_++, *((cl::Image *)inputs[0]->GetHandle().base));
        } else {
            execute_units_[0].ocl_kernel.setArg(kernel_arg_idx_++, *((cl::Image *)inputs[0]->GetHandle().base));
            execute_units_[0].ocl_kernel.setArg(kernel_arg_idx_++, *((cl::Image *)binary_params_->GetData()));
        }
    }
    // set optional param
    if (kernel_name_ == "BinaryChannel" || kernel_name_ == "BinaryCHW" ||
        kernel_name_ == "BinaryHW" || kernel_name_ == "BinaryWidth") {
        execute_units_[0].ocl_kernel.setArg(kernel_arg_idx_++, DimsFunctionUtils::GetDim(output_dims, 2));
        execute_units_[0].ocl_kernel.setArg(kernel_arg_idx_++, DimsFunctionUtils::GetDim(output_dims, 3));
        int param_batch = 1;
        if (inputs.size() == 2) {
            auto param_dims = inputs[param_idx_]->GetBlobDesc().dims;
            param_batch = DimsFunctionUtils::GetDim(param_dims, 0);
        }
        execute_units_[0].ocl_kernel.setArg(kernel_arg_idx_++, param_batch);
    }

    // set output
    execute_units_[0].ocl_kernel.setArg(kernel_arg_idx_++, *((cl::Image *)outputs[0]->GetHandle().base));

    return TNN_OK;
}

std::string OpenCLBinaryLayerAcc::GetKernelName(const MultidirBroadcastLayerParam &param) {
    if (param.input0_broadcast_type == BroadcastTypeSingle || param.input1_broadcast_type == BroadcastTypeSingle) {
        return "BinarySingle";
    } else if (param.input0_broadcast_type == BroadcastTypeChannel ||
               param.input1_broadcast_type == BroadcastTypeChannel) {
        return "BinaryChannel";
    } else if (param.input0_broadcast_type == BroadcastTypeElement ||
               param.input1_broadcast_type == BroadcastTypeElement) {
        return "BinaryCHW";
    } else if (param.input0_broadcast_type == BroadcastTypeHeightWidth ||
               param.input1_broadcast_type == BroadcastTypeHeightWidth) {
        return "BinaryHW";
    } else if (param.input0_broadcast_type == BroadcastTypeWidth || param.input1_broadcast_type == BroadcastTypeWidth) {
        return "BinaryWidth";
    } else {
        return "BinaryElementWise";
    }
}

Status OpenCLBinaryLayerAcc::ConvertParam(float *param_data_ptr, std::vector<int> param_dims) {
    OpenCLRuntime *opencl_runtime = OpenCLRuntime::GetInstance();

    // copy param data into clBuffer
    shared_ptr<OpenCLMemory> param_buffer(new OpenCLMemory(TNN_CL_BUFFER));
    int param_size  = DimsVectorUtils::Count(param_dims);
    int buffer_size = DimsFunctionUtils::GetDim(param_dims, 0) *
                      ROUND_UP(DimsFunctionUtils::GetDim(param_dims, 1), 4) *
                      DimsFunctionUtils::GetDim(param_dims, 2) * DimsFunctionUtils::GetDim(param_dims, 3);
    cl_int ret      = CL_SUCCESS;
    cl::Buffer param_clbuffer(*opencl_runtime->Context(), CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR,
                              buffer_size * sizeof(float), nullptr, &ret);
    if (ret != CL_SUCCESS) {
        CHECK_CL_SUCCESS(ret)
        return Status(TNNERR_OPENCL_MEMALLOC_ERROR, "OpenCL malloc memory failed");
    }
    param_buffer->SetData(&param_clbuffer);
    auto param_clbuffer_ptr = ocl_context_->CommandQueue()->enqueueMapBuffer(
        param_clbuffer, true, CL_MAP_WRITE, 0, buffer_size * sizeof(float), nullptr, nullptr, &ret);
    if (ret != CL_SUCCESS) {
        CHECK_CL_SUCCESS(ret)
        return Status(TNNERR_OPENCL_MEMMAP_ERROR, "OpenCL MemMap failed");
    }
    memset(param_clbuffer_ptr, 0, buffer_size * sizeof(float));
    memcpy(param_clbuffer_ptr, param_data_ptr, param_size * sizeof(float));
    ret = ocl_context_->CommandQueue()->enqueueUnmapMemObject(param_clbuffer, param_clbuffer_ptr);
    if (ret != CL_SUCCESS) {
        CHECK_CL_SUCCESS(ret)
        return Status(TNNERR_OPENCL_MEMUNMAP_ERROR, "OpenCL MemUnMap failed");
    }

    // create binary_param_
    int climage_w             = UP_DIV(DimsFunctionUtils::GetDim(param_dims, 1), 4) * DimsFunctionUtils::GetDim(param_dims, 3);
    int climage_h             = DimsFunctionUtils::GetDim(param_dims, 0) * DimsFunctionUtils::GetDim(param_dims, 2);
    cl_channel_type data_type = CL_FLOAT;
    if (opencl_runtime->GetPrecision() != PRECISION_HIGH)
        data_type = CL_HALF_FLOAT;
    cl::Image2D *image = new cl::Image2D(*opencl_runtime->Context(), CL_MEM_READ_WRITE,
                                         cl::ImageFormat(CL_RGBA, data_type), climage_w, climage_h, 0, nullptr, &ret);
    if (ret != CL_SUCCESS) {
        CHECK_CL_SUCCESS(ret)
        if (nullptr != image)
            delete image;
        return Status(TNNERR_OPENCL_MEMALLOC_ERROR, "OpenCL malloc memory failed");
    }
    binary_params_.reset(new OpenCLMemory(TNN_CL_IMAGE));
    binary_params_->SetData(image, true);

    // convert nchw buffer to Image
    ImageBufferConvertor convertor(opencl_runtime, ocl_context_->CommandQueue());
    return convertor.ConvertBufferToImage(param_buffer.get(), NCHW_BUFFER, param_dims, binary_params_.get(), true);
}

Status OpenCLBinaryLayerAcc::ReloadConstantBlobs(const std::vector<Blob *> &inputs, bool only_reload_shape_differ_blob) {
    auto const_resource = const_resource_;
    auto const_resource_flag = const_resource_flag_;
    auto const_blob_map = const_blob_map_;
    for (auto iter : inputs) {
        auto name = iter->GetBlobDesc().name;
        if (const_resource == nullptr || const_resource->find(name) == const_resource->end()) {
            continue;
        }

        if (only_reload_shape_differ_blob && const_resource_flag &&
            const_resource_flag->find(name) == const_resource_flag->end()) {
            continue;
        }

        auto buffer = (*const_resource)[name];
        std::shared_ptr<Blob> blob = nullptr;
        if (const_blob_map.find(name) != const_blob_map.end()) {
            blob = const_blob_map[name];
        }
        auto buffer_dims = buffer->GetBufferDims();
        if (output_dims_size_ != buffer_dims.size()) {
            std::shared_ptr<RawBuffer> new_buffer(new RawBuffer(*buffer));
            int diff = output_dims_size_ - buffer_dims.size();
            for (int i = 0; i < diff; i++) {
                buffer_dims.insert(buffer_dims.begin(), 1);
            }
            new_buffer->SetBufferDims(buffer_dims);
            buffer = new_buffer;
        }
        auto status = RawBuffer2OpenCLBlob(buffer.get(), blob);
        RETURN_ON_NEQ(status, TNN_OK);

        blob->SetFlag(DATA_FLAG_CHANGE_NEVER);
        auto dims = iter->GetBlobDesc().dims;
        auto data_type_size = DataTypeUtils::GetBytesSize(iter->GetBlobDesc().data_type);
        const_blob_map[name] = blob;
        iter->SetHandle(blob->GetHandle());
        iter->GetBlobDesc() = blob->GetBlobDesc();
        LOGD("Reload constant blob: %s\n", name.c_str());
    }
    const_blob_map_ = const_blob_map;
    return TNN_OK;
}

}  // namespace TNN_NS
