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

#include "tnn/device/directx/acc/directx_binary_layer_acc.h"

#include "tnn/core/macro.h"
#include "tnn/utils/data_type_utils.h"
#include "tnn/utils/dims_utils.h"
#include "tnn/device/directx/directx_memory.h"
#include "tnn/device/directx/directx_util.h"
// #include "tnn/device/opencl/imagebuffer_convertor.h"

namespace TNN_NS {

namespace directx {

Status DirectXBinaryLayerAcc::Init(Context *context, LayerParam *param, LayerResource *resource,
                                  const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    LOGD("Init Binary Acc\n");

    output_dims_size_ = outputs[0]->GetBlobDesc().dims.size();
    Status ret        = DirectXLayerAcc::Init(context, param, resource, inputs, outputs);
    RETURN_ON_NEQ(ret, TNN_OK);

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

        float *data_ptr = layer_res->element_handle.force_to<float *>();
        std::shared_ptr<float> data = std::shared_ptr<float>(data_ptr, [](float *){});
        if (layer_res->element_handle.GetDataType() != DATA_TYPE_FLOAT) {
            data = GetFloatFromRawBuffer(layer_res->element_handle);  
            if (data == nullptr) {
                return Status(TNNERR_DX_ACC_INIT_ERR, "convert res to float failed");
            }
        } 
        RETURN_ON_NEQ(ConvertParam(data.get(), param_dims_), TNN_OK);
    }

    kernel_name_ = GetKernelName(broadcast_param_);

    return TNN_OK;
}

DirectXBinaryLayerAcc::~DirectXBinaryLayerAcc() {}

Status DirectXBinaryLayerAcc::Forward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {

    Status status = DirectXLayerAcc::Forward(inputs, outputs);
    RETURN_ON_NEQ(status, TNN_OK);

    auto d3d_context = GetID3DContext();

    auto in_memory = DirectXMemory::CreateRefMemoryFromBlob(inputs[0]); 
    auto out_memory = DirectXMemory::CreateRefMemoryFromBlob(outputs[0]); 

    auto in_srv = in_memory->GetSRV();
    auto out_uav = out_memory->GetUAV();

    auto in_buffer = (ID3D11Buffer *) in_memory->GetData();
    auto out_buffer = (ID3D11Buffer *) out_memory->GetData();
    std::shared_ptr<ID3D11ComputeShader> cs;
    Status ret = GetShaderByName("buffer_add", cs);
    RETURN_ON_NEQ(ret, TNN_OK);

    typedef struct launch_param {
        UINT n;
        UINT c;
        UINT h;
        UINT w;
    } launch_param_t;

    launch_param_t args;
    args.n = inputs[0]->GetBlobDesc().dims[0];
    args.c = inputs[0]->GetBlobDesc().dims[1];
    args.h = inputs[0]->GetBlobDesc().dims[2];
    args.w = inputs[0]->GetBlobDesc().dims[3];

    std::shared_ptr<ID3D11Buffer> const_buffer;
    ret = CreateConstBuffer<launch_param_t>(args, GetID3DDevice(), const_buffer);
    RETURN_ON_NEQ(ret, TNN_OK);

    const int THREADS_PER_BLOCK = 128;
    const int ELE_PER_THREAD    = 4;

    const int ele_count = DimsVectorUtils::Count(inputs[0]->GetBlobDesc().dims);

    ret = DispatchShader(cs, {in_srv}, {out_uav}, {const_buffer.get()}, {UP_DIV(ele_count, THREADS_PER_BLOCK * ELE_PER_THREAD)});

    return ret;
}

Status DirectXBinaryLayerAcc::Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    LOGD("Binary Acc Reshape\n");
    Status ret = DirectXLayerAcc::Reshape(inputs, outputs);
    RETURN_ON_NEQ(ret, TNN_OK);

    auto output_dims = outputs[0]->GetBlobDesc().dims;

    /*
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
    */

    return TNN_OK;
}

std::string DirectXBinaryLayerAcc::GetKernelName(const MultidirBroadcastLayerParam &param) {
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

Status DirectXBinaryLayerAcc::ConvertParam(float *param_data_ptr, std::vector<int> param_dims) {

    // copy param data into DirectX Buffer
    // TODO: to DirectX Texture2D 
    shared_ptr<DirectXMemory> param_buffer = DirectXMemory::CreateBufferMemoryFromHost(
                                                param_data_ptr, param_dims, DATA_TYPE_FLOAT, DATA_FORMAT_NCHW);
    if (!param_buffer) {
        LOGE("param transfer to GPU failed.");
        return Status(TNNERR_DX_BUFFER_ALOCATE_ERR, "param transfer to GPU failed.");
    }
    binary_params_ = std::move(param_buffer);
    return TNN_OK;
}

Status DirectXBinaryLayerAcc::ReloadConstantBlobs(const std::vector<Blob *> &inputs,
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
        auto status = RawBuffer2DirectXBlob(buffer.get(), blob);
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

} // namespace directx

}  // namespace TNN_NS
