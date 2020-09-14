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

#include "tnn/device/metal/acc//metal_cpu_adapter_acc.h"

#include "tnn/core/abstract_device.h"
#include "tnn/utils/blob_converter.h"
#include "tnn/utils/data_format_converter.h"

namespace TNN_NS {

MetalCpuAdapterAcc::MetalCpuAdapterAcc(LayerType impl_layer_type) {
    impl_layer_type_ = impl_layer_type;
    DeviceType device_list[2] = {DEVICE_ARM, DEVICE_X86};
    for(auto device_type : device_list) {
        auto device = GetDevice(device_type);
        if(device != NULL) {
            auto acc = device->CreateLayerAcc(impl_layer_type_);
            if(acc != NULL) {
                cpu_adapter_acc_ = acc;
                impl_device_type_ = device_type;
                impl_device_context_ = device->CreateContext(0);
                break;
            }
        }
    }
}

Status MetalCpuAdapterAcc::Init(Context *context, LayerParam *param, LayerResource *resource,
                                const std::vector<Blob *> &inputs,
                                const std::vector<Blob *> &outputs) {
    if(cpu_adapter_acc_ == NULL) {
        return Status(TNNERR_MODEL_ERR, "cpu adapter acc is null");
    }
    auto status = AbstractLayerAcc::Init(context, param, resource, inputs, outputs);
    RETURN_ON_NEQ(status, TNN_OK);
    
    metal_context_ = dynamic_cast<MetalContext *>(context);
    if (metal_context_ == nullptr) {
        return Status(TNNERR_NULL_PARAM, "Metal Context Convert failed");
    }
    
    //check input and output data type
    for(auto input : inputs) {
        auto desc = input->GetBlobDesc();
        if (desc.data_type != DATA_TYPE_FLOAT && desc.data_type != DATA_TYPE_HALF) {
            LOGE("layer acc with tyoe (%d) is nil\n", (int)impl_layer_type_);
            return Status(TNNERR_NULL_PARAM, "layer acc is nil");
        }
    }

    for(auto output : outputs) {
        auto desc = output->GetBlobDesc();
        if (desc.data_type != DATA_TYPE_FLOAT && desc.data_type != DATA_TYPE_HALF) {
            LOGE("layer acc with tyoe (%d) is nil\n", (int)impl_layer_type_);
            return Status(TNNERR_NULL_PARAM, "layer acc is nil");
        }
    }
    
    //TODO: test with bfp16 mode
    
    for(auto input : inputs) {
        auto desc = input->GetBlobDesc();
        desc.device_type = impl_device_type_;
        desc.data_format = DATA_FORMAT_AUTO;
        desc.data_type = DATA_TYPE_FLOAT;
        cpu_blob_in_.push_back(new Blob(desc, true));
    }

    for(auto output : outputs) {
        auto desc = output->GetBlobDesc();
        desc.device_type = impl_device_type_;
        desc.data_format = DATA_FORMAT_AUTO;
        desc.data_type = DATA_TYPE_FLOAT;
        cpu_blob_out_.push_back(new Blob(desc, true));
    }
    
    //cpu acc init
    status = cpu_adapter_acc_->Init(impl_device_context_, param, resource, cpu_blob_in_, cpu_blob_out_);
    
    return status;
}

MetalCpuAdapterAcc::~MetalCpuAdapterAcc() {
    for(auto input : cpu_blob_in_) {
        delete input;
    }
    cpu_blob_in_.clear();
    
    for(auto output : cpu_blob_out_) {
        delete output;
    }
    cpu_blob_out_.clear();
    
    if (cpu_adapter_acc_) {
        delete cpu_adapter_acc_;
    }
    cpu_adapter_acc_ = nullptr;
    
    if (impl_device_context_) {
        delete impl_device_context_;
    }
    impl_device_context_ = nullptr;
    
}

Status MetalCpuAdapterAcc::Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    for(int i = 0; i < inputs.size(); ++i) {
        auto device_input = inputs[i];
        auto cpu_input = cpu_blob_in_[i];
        cpu_input->GetBlobDesc().dims = device_input->GetBlobDesc().dims;
    }
    for(int i = 0; i < outputs.size(); ++i) {
        auto device_output = outputs[i];
        auto cpu_output = cpu_blob_out_[i];
        cpu_output->GetBlobDesc().dims = device_output->GetBlobDesc().dims;
    }
    return cpu_adapter_acc_->Reshape(cpu_blob_in_, cpu_blob_out_);
}

Status MetalCpuAdapterAcc::Forward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    void* command_queue = nullptr;
    metal_context_->GetCommandQueue(&command_queue);

    Status status = TNN_OK;
    //convert data from metal to cpu
    for(int i = 0; i < inputs.size(); ++i) {
        auto device_input = inputs[i];
        auto cpu_input = cpu_blob_in_[i];
        auto dims = cpu_input->GetBlobDesc().dims;
        //To optimize
        BlobConverter blob_converter(device_input);
        MatConvertParam param;
        if(DATA_FORMAT_NCHW == cpu_input->GetBlobDesc().data_format) {
            Mat mat(impl_device_type_, NCHW_FLOAT, cpu_input->GetBlobDesc().dims, cpu_input->GetHandle().base);
            status = blob_converter.ConvertToMat(mat, param, command_queue);
            if (status != TNN_OK) {
                return status;
            }
        } else {
            //To optimize, use convert to change format
            Mat mat(impl_device_type_, NCHW_FLOAT, cpu_input->GetBlobDesc().dims);
            status = blob_converter.ConvertToMat(mat, param, command_queue);
            if (status != TNN_OK) {
                return status;
            }
            float* src_data = reinterpret_cast<float*>(mat.GetData());
            float* dst_data = reinterpret_cast<float*>(cpu_input->GetHandle().base);
            DataFormatConverter::ConvertFromNCHWToNCHW4Float(src_data, dst_data, dims[0], dims[1], dims[2], dims[3]);
        }
    }

    //cpu acc forword
    status = cpu_adapter_acc_->Forward(cpu_blob_in_, cpu_blob_out_);
    if (status != TNN_OK) {
        return status;
    }

    //convert data from cpu to metal
    for(int i = 0; i < outputs.size(); ++i) {
        auto device_output = outputs[i];
        auto cpu_output = cpu_blob_out_[i];
        auto dims = cpu_output->GetBlobDesc().dims;
        // use the shape of cpu_blob
        device_output->GetBlobDesc().dims = dims;

        BlobConverter blob_converter(device_output);
        MatConvertParam param;
        if(DATA_FORMAT_NCHW == cpu_output->GetBlobDesc().data_format) {
            Mat mat(impl_device_type_, NCHW_FLOAT, cpu_output->GetBlobDesc().dims, cpu_output->GetHandle().base);
            status = blob_converter.ConvertFromMat(mat, param, command_queue);
            if (status != TNN_OK) {
                return status;
            }
        } else {
            //To optimize, use convert to change format
            Mat mat(impl_device_type_, NCHW_FLOAT, dims);
            float* src_data = reinterpret_cast<float*>(cpu_output->GetHandle().base);
            float* dst_data = reinterpret_cast<float*>(mat.GetData());
            DataFormatConverter::ConvertFromNCHW4ToNCHWFloat(src_data, dst_data, dims[0], dims[1], dims[2], dims[3]);
            status = blob_converter.ConvertFromMat(mat, param, command_queue);
            if (status != TNN_OK) {
                return status;
            }
        }
    }

    return status;
}

std::vector<DataFormat> MetalCpuAdapterAcc::SupportDataFormat(DataType data_type, int dims_size) {
    std::vector<DataFormat> support_list;
    if (dims_size == 4) {
        support_list.push_back(DATA_FORMAT_NC4HW4);
    }
    return support_list;
}

}
