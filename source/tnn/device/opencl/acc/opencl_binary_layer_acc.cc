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
#include "tnn/utils/data_type_utils.h"
#include "tnn/utils/dims_utils.h"

namespace TNN_NS {

Status OpenCLBinaryLayerAcc::Init(Context *context, LayerParam *param, LayerResource *resource,
                                  const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    LOGD("Init Binary Acc\n");

    output_dims_size_ = outputs[0]->GetBlobDesc().dims.size();
    Status ret        = OpenCLLayerAcc::Init(context, param, resource, inputs, outputs);
    CHECK_TNN_OK(ret)

    run_3d_ndrange_ = false;

    auto broadcast_param = dynamic_cast<MultidirBroadcastLayerParam *>(param);
    CHECK_PARAM_NULL(broadcast_param);
    broadcast_param_ = *broadcast_param;

    EltwiseLayerResource *layer_res = dynamic_cast<EltwiseLayerResource *>(resource);
    if (layer_res == nullptr) {
        if (inputs.size() == 2) {
            param_dims_ = inputs[0]->GetBlobDesc().dims;
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

            } else if (output_dims_size_ == 5) {
                input_idx_ = 0;
                param_idx_ = 1;
                if (broadcast_param_.input0_broadcast_type != BroadcastTypeNormal) {
                    input_idx_ = 1;
                    param_idx_ = 0;
                }
            } else {
                return Status(TNNERR_PARAM_ERR, "input dims is illegal");
            }

        } else {
            return Status(TNNERR_PARAM_ERR, "inputs size shound be 2 without binary resource");
        }
    } else {
        param_dims_ = layer_res->element_shape;
        input_idx_  = 0;
        if (inputs.size() != 1) {
            return Status(TNNERR_PARAM_ERR, "input size should be 1");
        }

        int diff = output_dims_size_ - param_dims_.size();
        for (int i = 0; i < diff; i++) {
            param_dims_.insert(param_dims_.begin(), 1);
        }

        if (layer_res->element_handle.GetDataType() == DATA_TYPE_FLOAT) {
            float *data_ptr = layer_res->element_handle.force_to<float *>();
            ret             = ConvertParam(data_ptr, param_dims_);
            CHECK_TNN_OK(ret)
        } else {
            auto float_data_ptr = GetFloatFromRawBuffer(layer_res->element_handle);  // handle the memory
            if (float_data_ptr == nullptr) {
                return Status(TNNERR_OPENCL_ACC_INIT_ERROR, "convert res to float failed");
            }
            ret = ConvertParam(float_data_ptr.get(), param_dims_);
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
        if (kernel_name_ == "BinaryBroadcast" || kernel_name_ == "BinaryBroadcast5D" ||
            kernel_name_ == "BinaryElementWise") {  // only in0 - in1
            if (broadcast_param_.weight_input_index == 0) {
                execute_units_[0].ocl_kernel.setArg(kernel_arg_idx_++, *((cl::Image *)binary_params_->GetData()));
                execute_units_[0].ocl_kernel.setArg(kernel_arg_idx_++, *((cl::Image *)inputs[0]->GetHandle().base));
            } else {
                execute_units_[0].ocl_kernel.setArg(kernel_arg_idx_++, *((cl::Image *)inputs[0]->GetHandle().base));
                execute_units_[0].ocl_kernel.setArg(kernel_arg_idx_++, *((cl::Image *)binary_params_->GetData()));
            }
        } else if (kernel_name_ == "BinaryChannel" || kernel_name_ == "BinaryCHW" || kernel_name_ == "BinaryHW" ||
                   kernel_name_ == "BinaryWidth" || kernel_name_ == "BinarySingle") {  // maybe in0-in1 or in1-in0
            if (broadcast_param_.input0_broadcast_type == BroadcastTypeNormal) {
                if (broadcast_param_.weight_input_index == 0) {  // weight is input0, input is input1
                    execute_units_[0].ocl_kernel.setArg(kernel_arg_idx_++, *((cl::Image *)binary_params_->GetData()));
                    execute_units_[0].ocl_kernel.setArg(kernel_arg_idx_++, *((cl::Image *)inputs[0]->GetHandle().base));
                } else {  // in1 - in0
                    execute_units_[0].ocl_kernel.setArg(kernel_arg_idx_++, *((cl::Image *)inputs[0]->GetHandle().base));
                    execute_units_[0].ocl_kernel.setArg(kernel_arg_idx_++, *((cl::Image *)binary_params_->GetData()));
                }
            } else if (broadcast_param_.input1_broadcast_type == BroadcastTypeNormal) {  // input1 is normal
                if (broadcast_param_.weight_input_index == 0) {  // weight is input0, input is input1, in1 - in0
                    execute_units_[0].ocl_kernel.setArg(kernel_arg_idx_++, *((cl::Image *)inputs[0]->GetHandle().base));
                    execute_units_[0].ocl_kernel.setArg(kernel_arg_idx_++, *((cl::Image *)binary_params_->GetData()));
                } else {  // in0 - in1
                    execute_units_[0].ocl_kernel.setArg(kernel_arg_idx_++, *((cl::Image *)binary_params_->GetData()));
                    execute_units_[0].ocl_kernel.setArg(kernel_arg_idx_++, *((cl::Image *)inputs[0]->GetHandle().base));
                }
            }
        }
    }

    // set optional param
    if (kernel_name_ == "BinaryChannel" || kernel_name_ == "BinaryCHW" || kernel_name_ == "BinaryHW" ||
        kernel_name_ == "BinaryWidth") {
        execute_units_[0].ocl_kernel.setArg(kernel_arg_idx_++, DimsFunctionUtils::GetDim(output_dims, 2));
        execute_units_[0].ocl_kernel.setArg(kernel_arg_idx_++, DimsFunctionUtils::GetDim(output_dims, 3));
        int param_batch = 1;
        if (inputs.size() == 2) {
            auto param_dims = inputs[param_idx_]->GetBlobDesc().dims;
            param_batch     = DimsFunctionUtils::GetDim(param_dims, 0);
        }
        execute_units_[0].ocl_kernel.setArg(kernel_arg_idx_++, param_batch);
    } else if (kernel_name_ == "BinaryBroadcast") {
        std::vector<int> output_shape(4), input0_shape(4), input1_shape(4);
        if (inputs.size() == 2) {
            if (inputs[input_idx_]->GetBlobDesc().dims.size() > 4 ||
                inputs[param_idx_]->GetBlobDesc().dims.size() > 4) {
                return Status(TNNERR_OPENCL_ACC_RESHAPE_ERROR, "opencl binary layer inputs not support dims > 4");
            }
            for (int i = 0; i < 4; ++i) {
                input0_shape[i] = DimsFunctionUtils::GetDim(inputs[input_idx_]->GetBlobDesc().dims, i);
                input1_shape[i] = DimsFunctionUtils::GetDim(inputs[param_idx_]->GetBlobDesc().dims, i);
            }
        } else {
            if (inputs[input_idx_]->GetBlobDesc().dims.size() > 4 || param_dims_.size() > 4) {
                return Status(TNNERR_OPENCL_ACC_RESHAPE_ERROR, "opencl binary layer inputs not support dims > 4");
            }
            if (broadcast_param_.weight_input_index == 0) {
                for (int i = 0; i < 4; ++i) {
                    input0_shape[i] = DimsFunctionUtils::GetDim(param_dims_, i);
                    input1_shape[i] = DimsFunctionUtils::GetDim(inputs[input_idx_]->GetBlobDesc().dims, i);
                }
            } else {
                for (int i = 0; i < 4; ++i) {
                    input0_shape[i] = DimsFunctionUtils::GetDim(inputs[input_idx_]->GetBlobDesc().dims, i);
                    input1_shape[i] = DimsFunctionUtils::GetDim(param_dims_, i);
                }
            }
        }
        for (int i = 0; i < 4; ++i) {
            output_shape[i] = DimsFunctionUtils::GetDim(output_dims, i);
        }

        execute_units_[0].ocl_kernel.setArg(kernel_arg_idx_++, 4 * sizeof(int), output_shape.data());
        execute_units_[0].ocl_kernel.setArg(kernel_arg_idx_++, 4 * sizeof(int), input0_shape.data());
        execute_units_[0].ocl_kernel.setArg(kernel_arg_idx_++, 4 * sizeof(int), input1_shape.data());
        execute_units_[0].ocl_kernel.setArg(kernel_arg_idx_++, UP_DIV(input0_shape[1], 4));
        execute_units_[0].ocl_kernel.setArg(kernel_arg_idx_++, UP_DIV(input1_shape[1], 4));
    } else if (kernel_name_ == "BinaryBroadcast5D") {
        const int n_dims = 5;
        std::vector<int> output_shape(n_dims), input0_shape(n_dims), input1_shape(n_dims);
        if (inputs.size() == 2) {
            for (int i = 0; i < n_dims; ++i) {
                input0_shape[i] = DimsFunctionUtils::GetDim(inputs[input_idx_]->GetBlobDesc().dims, i);
                input1_shape[i] = DimsFunctionUtils::GetDim(inputs[param_idx_]->GetBlobDesc().dims, i);
            }
        } else {
            if (broadcast_param_.weight_input_index == 0) {
                for (int i = 0; i < n_dims; ++i) {
                    input0_shape[i] = DimsFunctionUtils::GetDim(param_dims_, i);
                    input1_shape[i] = DimsFunctionUtils::GetDim(inputs[input_idx_]->GetBlobDesc().dims, i);
                }
            } else {
                for (int i = 0; i < n_dims; ++i) {
                    input0_shape[i] = DimsFunctionUtils::GetDim(inputs[input_idx_]->GetBlobDesc().dims, i);
                    input1_shape[i] = DimsFunctionUtils::GetDim(param_dims_, i);
                }
            }
        }

        for (int i = 0; i < n_dims; ++i) {
            output_shape[i] = DimsFunctionUtils::GetDim(output_dims, i);
        }

        execute_units_[0].ocl_kernel.setArg(kernel_arg_idx_++, n_dims * sizeof(int), output_shape.data());
        execute_units_[0].ocl_kernel.setArg(kernel_arg_idx_++, n_dims * sizeof(int), input0_shape.data());
        execute_units_[0].ocl_kernel.setArg(kernel_arg_idx_++, n_dims * sizeof(int), input1_shape.data());
        execute_units_[0].ocl_kernel.setArg(kernel_arg_idx_++, UP_DIV(input0_shape[1], 4));
        execute_units_[0].ocl_kernel.setArg(kernel_arg_idx_++, UP_DIV(input1_shape[1], 4));
    }

    // set output
    execute_units_[0].ocl_kernel.setArg(kernel_arg_idx_++, *((cl::Image *)outputs[0]->GetHandle().base));

    return TNN_OK;
}

std::string OpenCLBinaryLayerAcc::GetKernelName(const MultidirBroadcastLayerParam &param) {
    if (output_dims_size_ == 5) {
        return "BinaryBroadcast5D";
    }

    if (param.input0_broadcast_type == BroadcastTypeNormal && param.input1_broadcast_type == BroadcastTypeNormal) {
        return "BinaryElementWise";
    } else if (param.input0_broadcast_type == BroadcastTypeSingle ||
               param.input1_broadcast_type == BroadcastTypeSingle) {
        return "BinarySingle";
    } else if ((param.input0_broadcast_type == BroadcastTypeChannel &&
                param.input1_broadcast_type == BroadcastTypeNormal) ||
               (param.input1_broadcast_type == BroadcastTypeChannel &&
                param.input0_broadcast_type == BroadcastTypeNormal)) {
        return "BinaryChannel";
    } else if ((param.input0_broadcast_type == BroadcastTypeElement &&
                param.input1_broadcast_type == BroadcastTypeNormal) ||
               (param.input1_broadcast_type == BroadcastTypeElement &&
                param.input0_broadcast_type == BroadcastTypeNormal)) {
        return "BinaryCHW";
    } else if ((param.input0_broadcast_type == BroadcastTypeHeightWidth &&
                param.input1_broadcast_type == BroadcastTypeNormal) ||
               (param.input1_broadcast_type == BroadcastTypeHeightWidth &&
                param.input0_broadcast_type == BroadcastTypeNormal)) {
        return "BinaryHW";
    } else if ((param.input0_broadcast_type == BroadcastTypeWidth &&
                param.input1_broadcast_type == BroadcastTypeNormal) ||
               (param.input1_broadcast_type == BroadcastTypeWidth &&
                param.input0_broadcast_type == BroadcastTypeNormal)) {
        return "BinaryWidth";
    } else {
        return "BinaryBroadcast";
    }
}

Status OpenCLBinaryLayerAcc::ConvertParam(float *param_data_ptr, std::vector<int> param_dims) {
    OpenCLRuntime *opencl_runtime = OpenCLRuntime::GetInstance();

    // copy param data into clBuffer
    shared_ptr<OpenCLMemory> param_buffer(new OpenCLMemory(TNN_CL_BUFFER));
    int param_size  = DimsVectorUtils::Count(param_dims);
    int buffer_size = DimsFunctionUtils::GetDim(param_dims, 0) * ROUND_UP(DimsFunctionUtils::GetDim(param_dims, 1), 4) *
                      DimsFunctionUtils::GetDim(param_dims, 2) * DimsFunctionUtils::GetDim(param_dims, 3);
    if (param_dims.size() > 4) {
        for (int i = 4; i < param_dims.size(); i++) {
            buffer_size *= DimsFunctionUtils::GetDim(param_dims, i);
        }
    }
    cl_int ret = CL_SUCCESS;
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
    int climage_w = UP_DIV(DimsFunctionUtils::GetDim(param_dims, 1), 4) * DimsFunctionUtils::GetDim(param_dims, 3);
    int climage_h = DimsFunctionUtils::GetDim(param_dims, 0) * DimsFunctionUtils::GetDim(param_dims, 2);
    if (param_dims.size() == 5) {
        climage_w = UP_DIV(DimsFunctionUtils::GetDim(param_dims, 1), 4) * DimsFunctionUtils::GetDim(param_dims, 4);
        climage_h = DimsFunctionUtils::GetDim(param_dims, 0) * DimsFunctionUtils::GetDim(param_dims, 2) *
                    DimsFunctionUtils::GetDim(param_dims, 3);
    }
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

Status OpenCLBinaryLayerAcc::ReloadConstantBlobs(const std::vector<Blob *> &inputs,
                                                 bool only_reload_shape_differ_blob) {
    auto const_resource      = const_resource_;
    auto const_resource_flag = const_resource_flag_;
    auto const_blob_map      = const_blob_map_;
    for (auto iter : inputs) {
        auto name = iter->GetBlobDesc().name;
        if (const_resource == nullptr || const_resource->find(name) == const_resource->end()) {
            continue;
        }

        if (only_reload_shape_differ_blob && const_resource_flag &&
            const_resource_flag->find(name) == const_resource_flag->end()) {
            continue;
        }

        auto buffer                = (*const_resource)[name];
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
        auto dims            = iter->GetBlobDesc().dims;
        auto data_type_size  = DataTypeUtils::GetBytesSize(iter->GetBlobDesc().data_type);
        const_blob_map[name] = blob;
        iter->SetHandle(blob->GetHandle());
        iter->GetBlobDesc() = blob->GetBlobDesc();
        LOGD("Reload constant blob: %s\n", name.c_str());
    }
    const_blob_map_ = const_blob_map;
    return TNN_OK;
}

}  // namespace TNN_NS
