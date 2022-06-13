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

#include "tnn/device/x86/acc/x86_cpu_adapter_acc.h"

#include "tnn/core/abstract_device.h"
#include "tnn/utils/blob_converter.h"
#include "tnn/utils/cpu_utils.h"
#include "tnn/utils/data_format_converter.h"
#include "tnn/utils/dims_utils.h"

namespace TNN_NS {

X86CpuAdapterAcc::X86CpuAdapterAcc(LayerType impl_layer_type) {
    impl_layer_type_         = impl_layer_type;
    cpu_adapter_acc_         = nullptr;
    impl_device_type_        = DEVICE_NAIVE;
    impl_device_context_     = nullptr;
    DeviceType device_list[] = {DEVICE_NAIVE};
    for (auto device_type : device_list) {
        auto device = GetDevice(device_type);
        if (device != nullptr) {
            auto acc = device->CreateLayerAcc(impl_layer_type_);
            if (acc != nullptr) {
                cpu_adapter_acc_     = acc;
                impl_device_type_    = device_type;
                impl_device_         = device;
                impl_device_context_ = device->CreateContext(0);
                break;
            }
        }
    }
}

Status X86CpuAdapterAcc::Init(Context *context, LayerParam *param, LayerResource *resource,
                              const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    if (cpu_adapter_acc_ == nullptr) {
        return Status(TNNERR_LAYER_ERR, "cpu adapter acc is nil");
    }
    auto status = AbstractLayerAcc::Init(context, param, resource, inputs, outputs);
    RETURN_ON_NEQ(status, TNN_OK);

    context_ = dynamic_cast<X86Context *>(context);
    if (context_ == nullptr) {
        return Status(TNNERR_NULL_PARAM, "X86 Context Convert failed");
    }

    std::set<DataType> support_data_types = {DATA_TYPE_FLOAT, DATA_TYPE_INT32, DATA_TYPE_INT8};

    // check input and output data type
    for (auto input : inputs) {
        auto desc = input->GetBlobDesc();
        if (support_data_types.count(desc.data_type) == 0) {
            LOGE("layer acc with type (%d) is nil\n", (int)impl_layer_type_);
            return Status(TNNERR_NULL_PARAM, "layer acc is nil");
        }
    }

    for (auto output : outputs) {
        auto desc = output->GetBlobDesc();
        if (support_data_types.count(desc.data_type) == 0) {
            LOGE("layer acc with type (%d) is nil\n", (int)impl_layer_type_);
            return Status(TNNERR_NULL_PARAM, "layer acc is nil");
        }
    }

    for (auto input : inputs) {
        auto desc        = input->GetBlobDesc();
        desc.device_type = impl_device_type_;
        desc.data_format = GetCpuLayerAccDataFormat();
        desc.data_type   = GetCpuLayerAccPrecision(desc.data_type);
        cpu_blob_in_.push_back(new Blob(desc, false));
    }

    for (auto output : outputs) {
        auto desc        = output->GetBlobDesc();
        desc.device_type = impl_device_type_;
        desc.data_format = GetCpuLayerAccDataFormat();
        desc.data_type   = GetCpuLayerAccPrecision(desc.data_type);
        cpu_blob_out_.push_back(new Blob(desc, false));
    }

    // cpu acc init
    status = cpu_adapter_acc_->Init(impl_device_context_, param, resource, cpu_blob_in_, cpu_blob_out_);
    RETURN_ON_NEQ(status, TNN_OK);

    cpu_adapter_acc_->SetRuntimeMode(runtime_model_);
    cpu_adapter_acc_->SetConstantResource(const_resource_);

    return status;
}

X86CpuAdapterAcc::~X86CpuAdapterAcc() {
    for (auto input : cpu_blob_in_) {
        delete input;
    }
    cpu_blob_in_.clear();

    for (auto output : cpu_blob_out_) {
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

Status X86CpuAdapterAcc::Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    for (int i = 0; i < inputs.size(); ++i) {
        auto device_input             = inputs[i];
        auto cpu_input                = cpu_blob_in_[i];
        auto dims                     = device_input->GetBlobDesc().dims;
        cpu_input->GetBlobDesc().dims = dims;
    }

    for (int i = 0; i < outputs.size(); ++i) {
        auto device_output             = outputs[i];
        auto cpu_output                = cpu_blob_out_[i];
        auto dims                      = device_output->GetBlobDesc().dims;
        cpu_output->GetBlobDesc().dims = dims;
    }

    return cpu_adapter_acc_->Reshape(cpu_blob_in_, cpu_blob_out_);
}

Status X86CpuAdapterAcc::Forward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    Status status = TNN_OK;
    // convert data from x86 to cpu
    status = ConvertBlobForAdaptorAcc(inputs, cpu_blob_in_, true);
    RETURN_ON_NEQ(status, TNN_OK);
    status = ConvertBlobForAdaptorAcc(outputs, cpu_blob_out_, false);
    RETURN_ON_NEQ(status, TNN_OK);

    // cpu acc forward
    status = cpu_adapter_acc_->Forward(cpu_blob_in_, cpu_blob_out_);

    return status;
}

Status X86CpuAdapterAcc::ConvertBlobForAdaptorAcc(const std::vector<Blob *> &device_blobs,
                                                  const std::vector<Blob *> &cpu_blobs, bool device_to_cpu) {
    Status status = TNN_OK;

    for (int i = 0; i < device_blobs.size(); ++i) {
        auto device_blob = device_blobs[i];
        auto cpu_blob    = cpu_blobs[i];

        auto device_handle = device_blob->GetHandle();
        cpu_blob->SetHandle(device_handle);
    }
    return status;
}

std::vector<DataFormat> X86CpuAdapterAcc::SupportDataFormat(DataType data_type, int dims_size, BlobType blob_type) {
    std::vector<DataFormat> support_list;
    support_list.push_back(DATA_FORMAT_NCHW);
    return support_list;
}

DataType X86CpuAdapterAcc::GetCpuLayerAccPrecision(DataType data_type) {
    return data_type;
}

DataFormat X86CpuAdapterAcc::GetCpuLayerAccDataFormat() {
    // cpu only support nchw
    return DATA_FORMAT_NCHW;
}

}  // namespace TNN_NS
