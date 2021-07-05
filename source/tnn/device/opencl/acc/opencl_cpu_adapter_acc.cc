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

#include "tnn/device/opencl/acc/opencl_cpu_adapter_acc.h"

#include "tnn/core/abstract_device.h"
#include "tnn/utils/blob_converter.h"
#include "tnn/utils/data_format_converter.h"
#include "tnn/utils/dims_utils.h"
#include "tnn/utils/cpu_utils.h"

namespace TNN_NS {

inline MatType MatTypeByBlob(const BlobDesc& desc) {
    // TODO: opencl blob converter support fp16 mat
    return NCHW_FLOAT;
}

static void PackOrUnpackData(void *src, void *dst, DataType data_type, DimsVector& dims, bool pack) {
    // TODO: PackOrUnpackData support fp16 mat
    if (DATA_TYPE_FLOAT == data_type) {
        float *src_data = reinterpret_cast<float*>(src);
        float *dst_data = reinterpret_cast<float*>(dst);
        if (pack) {
            DataFormatConverter::ConvertFromNCHWToNCHW4Float(src_data, dst_data, dims[0], dims[1],
                DimsFunctionUtils::GetDim(dims, 2), DimsFunctionUtils::GetDim(dims, 3));
        } else {
            DataFormatConverter::ConvertFromNCHW4ToNCHWFloat(src_data, dst_data, dims[0], dims[1],
                DimsFunctionUtils::GetDim(dims, 2), DimsFunctionUtils::GetDim(dims, 3));
        }
    }
}

OpenCLCpuAdapterAcc::OpenCLCpuAdapterAcc(LayerType impl_layer_type) {
    impl_layer_type_ = impl_layer_type;
    cpu_adapter_acc_ = NULL;
    impl_device_type_ = DEVICE_ARM;
    impl_device_context_ = NULL;
    DeviceType device_list[2] = {DEVICE_ARM, DEVICE_X86};
    for(auto device_type : device_list) {
        auto device = GetDevice(device_type);
        if(device != NULL) {
            auto acc = device->CreateLayerAcc(impl_layer_type_);
            if(acc != NULL) {
                cpu_adapter_acc_     = acc;
                impl_device_type_    = device_type;
                impl_device_         = device;
                impl_device_context_ = device->CreateContext(0);
                break;
            }
        }
    }
}

Status OpenCLCpuAdapterAcc::Init(Context *context, LayerParam *param, LayerResource *resource, const std::vector<Blob *> &inputs,
                        const std::vector<Blob *> &outputs) {
    if(cpu_adapter_acc_ == NULL) {
       return Status(TNNERR_OPENCL_ACC_INIT_ERROR, "cpu adapter acc is null");
    }
    auto status = AbstractLayerAcc::Init(context, param, resource, inputs, outputs);
    RETURN_ON_NEQ(status, TNN_OK);

    ocl_context_ = dynamic_cast<OpenCLContext *>(context);
    if (ocl_context_ == nullptr) {
        return Status(TNNERR_NULL_PARAM, "OpenCL Context Convert failed");
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
        desc.data_format = GetCpuLayerAccDataFormat();
        desc.data_type = GetCpuLayerAccPrecision();
        cpu_blob_in_.push_back(new Blob(desc, true));
    }

    for(auto output : outputs) {
        auto desc = output->GetBlobDesc();
        desc.device_type = impl_device_type_;
        desc.data_format = GetCpuLayerAccDataFormat();
        desc.data_type = GetCpuLayerAccPrecision();
        cpu_blob_out_.push_back(new Blob(desc, true));
    }

    //cpu acc init
    status = cpu_adapter_acc_->Init(impl_device_context_, param, resource, cpu_blob_in_, cpu_blob_out_);
    RETURN_ON_NEQ(status, TNN_OK);

    cpu_adapter_acc_->SetRuntimeMode(runtime_model_);
    cpu_adapter_acc_->SetConstantResource(const_resource_);

    return status;
}

OpenCLCpuAdapterAcc::~OpenCLCpuAdapterAcc() {
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

Status OpenCLCpuAdapterAcc::Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    for(int i = 0; i < inputs.size(); ++i) {
        auto device_input = inputs[i];
        auto cpu_input = cpu_blob_in_[i];
        auto dims = device_input->GetBlobDesc().dims;
        cpu_input->GetBlobDesc().dims = dims;
    }

    for(int i = 0; i < outputs.size(); ++i) {
        auto device_output = outputs[i];
        auto cpu_output = cpu_blob_out_[i];
        auto dims = device_output->GetBlobDesc().dims;
        cpu_output->GetBlobDesc().dims = dims;
    }

    return cpu_adapter_acc_->Reshape(cpu_blob_in_, cpu_blob_out_);
}

Status OpenCLCpuAdapterAcc::Forward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    void* command_queue = nullptr;
    ocl_context_->GetCommandQueue(&command_queue);

    Status status = TNN_OK;
    //convert data from opencl to cpu
    status = ConvertBlobForAdaptorAcc(inputs, cpu_blob_in_, true);
    RETURN_ON_NEQ(status, TNN_OK);

    //cpu acc forward
    status = cpu_adapter_acc_->Forward(cpu_blob_in_, cpu_blob_out_);
    RETURN_ON_NEQ(status, TNN_OK);

    //convert data from cpu to opencl
    status = ConvertBlobForAdaptorAcc(outputs, cpu_blob_out_, false);

    return status;
}

Status OpenCLCpuAdapterAcc::ConvertBlobForAdaptorAcc(const std::vector<Blob *> & device_blobs,
                                                     const std::vector<Blob *> & cpu_blobs,
                                                     bool device_to_cpu) {
    Status status = TNN_OK;
    void* command_queue = nullptr;
    ocl_context_->GetCommandQueue(&command_queue);
    for (int i = 0; i < device_blobs.size(); ++i) {
        auto device_blob = device_blobs[i];
        auto cpu_blob    = cpu_blobs[i];

        if (const_resource_ != nullptr &&
            const_resource_->find(device_blob->GetBlobDesc().name) != const_resource_->end()) {
                continue;
        }

        auto dims = cpu_blob->GetBlobDesc().dims;
        if (!device_to_cpu) {
            device_blob->GetBlobDesc().dims = dims;
        }

        BlobConverter blob_converter(device_blob);
        MatConvertParam param;
        const auto& cpu_blob_desc = cpu_blob->GetBlobDesc();

        if (DATA_FORMAT_NCHW == cpu_blob_desc.data_format) {
            if (device_to_cpu) {
                Mat mat(impl_device_type_, MatTypeByBlob(cpu_blob_desc), cpu_blob_desc.dims, cpu_blob->GetHandle().base);
                status = blob_converter.ConvertToMat(mat, param, command_queue);
            } else {
                Mat mat(impl_device_type_, MatTypeByBlob(cpu_blob_desc), cpu_blob_desc.dims, cpu_blob->GetHandle().base);
                status = blob_converter.ConvertFromMat(mat, param, command_queue);
            }
            RETURN_ON_NEQ(status, TNN_OK);
        } else {
            //To optimize, use convert to change format
            Mat mat(impl_device_type_, MatTypeByBlob(cpu_blob_desc), cpu_blob_desc.dims);
            if (device_to_cpu) {
                status = blob_converter.ConvertToMat(mat, param, command_queue);
                RETURN_ON_NEQ(status, TNN_OK);
                PackOrUnpackData(mat.GetData(), cpu_blob->GetHandle().base, cpu_blob_desc.data_type, dims, true);
            } else {
                PackOrUnpackData(cpu_blob->GetHandle().base, mat.GetData(), cpu_blob_desc.data_type, dims, false);
                status = blob_converter.ConvertFromMat(mat, param, command_queue);
                RETURN_ON_NEQ(status, TNN_OK);
            }
        }
    }
    return status;
}

std::vector<DataFormat> OpenCLCpuAdapterAcc::SupportDataFormat(DataType data_type, int dims_size, BlobType blob_type) {
    std::vector<DataFormat> support_list;
    if (dims_size >= 2) {
        support_list.push_back(DATA_FORMAT_NHC4W4);
    }
    return support_list;
}

DataType OpenCLCpuAdapterAcc::GetCpuLayerAccPrecision() {
    // TODO: opencl blob converter support fp16 mat
    return DATA_TYPE_FLOAT;
}

 DataFormat OpenCLCpuAdapterAcc::GetCpuLayerAccDataFormat() {
     auto cpu_layouts = impl_device_->GetImplementedLayout(impl_layer_type_);
     return cpu_layouts->layouts[0];
 }

}
