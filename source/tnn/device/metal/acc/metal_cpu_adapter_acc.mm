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
#include "tnn/utils/dims_utils.h"
#include "tnn/device/arm/arm_util.h"
#include "tnn/utils/cpu_utils.h"

namespace TNN_NS {
inline MatType MatTypeByBlob(const BlobDesc& desc) {    
    if (desc.data_type == DATA_TYPE_FLOAT)
        return NCHW_FLOAT;
    
    if (desc.data_type == DATA_TYPE_HALF)
        return RESERVED_BFP16_TEST;
    
    return INVALID;
}

static void PackOrUnpackData(void *src, void *dst, DataType data_type, DimsVector& dims, bool pack) {
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
#if TNN_ARM82
    else if (DATA_TYPE_HALF == data_type) {
        // TODO: how to packc8 in a device-independent way?
        const int batch   = dims[0];
        const int channel = dims[1];
        const int hw      = DimsFunctionUtils::GetDimProduct(dims, 2);
        fp16_t *src_data = reinterpret_cast<fp16_t*>(src);
        fp16_t *dst_data = reinterpret_cast<fp16_t*>(dst);
        if (pack) {
            for(int n=0; n<batch; ++n) {
                PackC8(dst_data+n*channel*hw, src_data+n*channel*hw , hw, channel);
            }
        } else {
            for(int n=0; n<batch; ++n) {
                UnpackC8(dst_data+n*channel*hw, src_data+n*channel*hw , hw, channel);
            }
        }
    }
#endif
}

MetalCpuAdapterAcc::MetalCpuAdapterAcc(LayerType impl_layer_type) {
    impl_layer_type_ = impl_layer_type;
    DeviceType device_list[2] = {DEVICE_ARM, DEVICE_X86};
    for(auto device_type : device_list) {
        auto device = GetDevice(device_type);
        if(device != NULL) {
            auto acc = device->CreateLayerAcc(impl_layer_type_);
            if(acc != NULL) {
                cpu_adapter_acc_  = acc;
                impl_device_type_ = device_type;
                impl_device_      = GetDevice(impl_device_type_);
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
            LOGE("layer acc with type (%d) is nil\n", (int)impl_layer_type_);
            return Status(TNNERR_NULL_PARAM, "layer acc is nil");
        }
    }

    for(auto output : outputs) {
        auto desc = output->GetBlobDesc();
        if (desc.data_type != DATA_TYPE_FLOAT && desc.data_type != DATA_TYPE_HALF) {
            LOGE("layer acc with type (%d) is nil\n", (int)impl_layer_type_);
            return Status(TNNERR_NULL_PARAM, "layer acc is nil");
        }
    }
    
    //TODO: test with bfp16 mode
    
    for(auto input : inputs) {
        auto desc = input->GetBlobDesc();
        desc.device_type = impl_device_type_;
        desc.data_format = GetCpuLayerAccDataFormat();
        desc.data_type   = GetCpuLayerAccPrecision(input->GetBlobDesc().data_type);
        cpu_blob_in_.push_back(new Blob(desc, true));
    }

    for(auto output : outputs) {
        auto desc = output->GetBlobDesc();
        desc.device_type = impl_device_type_;
        desc.data_format = GetCpuLayerAccDataFormat();
        desc.data_type   = GetCpuLayerAccPrecision(output->GetBlobDesc().data_type);
        cpu_blob_out_.push_back(new Blob(desc, true));
    }
    
    //cpu acc init
    status = cpu_adapter_acc_->Init(impl_device_context_, param, resource, cpu_blob_in_, cpu_blob_out_);
    RETURN_ON_NEQ(status, TNN_OK);

    cpu_adapter_acc_->SetRuntimeMode(runtime_model_);
    cpu_adapter_acc_->SetConstantResource(const_resource_);
    
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

DataType MetalCpuAdapterAcc::GetCpuLayerAccPrecision(DataType metal_blob_data_type) {
    if (metal_blob_data_type == DATA_TYPE_HALF) {
        static bool cpu_support_fp16 = CpuUtils::CpuSupportFp16();
        bool layer_implemented_fp16  = impl_device_->GetImplementedPrecision(impl_layer_type_)->fp16_implemented;
        return (cpu_support_fp16 && layer_implemented_fp16) ? DATA_TYPE_HALF : DATA_TYPE_FLOAT;
    } else if (metal_blob_data_type == DATA_TYPE_FLOAT) {
        return DATA_TYPE_FLOAT;
    }
    return DATA_TYPE_FLOAT;
}

 DataFormat MetalCpuAdapterAcc::GetCpuLayerAccDataFormat() {
    auto cpu_layouts = impl_device_->GetImplementedLayout(impl_layer_type_);
    return cpu_layouts->layouts[0];
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

Status MetalCpuAdapterAcc::ConvertBlobForAdaptorAcc(const std::vector<Blob *> & metal_blobs,
                                                const std::vector<Blob *> & cpu_blobs, bool metal_to_cpu) {
    Status status = TNN_OK;
    void* command_queue = nullptr;
    metal_context_->GetCommandQueue(&command_queue);
    for(int i = 0; i < metal_blobs.size(); ++i) {
        auto device_blob = metal_blobs[i];
        auto cpu_blob    = cpu_blobs[i];

        // leave constant blobs to device layer acc
        if (const_resource_ != nullptr &&
            const_resource_->find(device_blob->GetBlobDesc().name) != const_resource_->end()) {
                continue;
        }

        auto dims = cpu_blob->GetBlobDesc().dims;
        if (!metal_to_cpu) {
            device_blob->GetBlobDesc().dims = dims;
        }

        BlobConverter blob_converter(device_blob);
        MatConvertParam param;
        const auto& cpu_blob_desc = cpu_blob->GetBlobDesc();

        if(DATA_FORMAT_NCHW == cpu_blob_desc.data_format) {
            if (metal_to_cpu) {
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
            if (metal_to_cpu) {
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

Status MetalCpuAdapterAcc::Forward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    Status status = TNN_OK;

    status = ConvertBlobForAdaptorAcc(inputs, cpu_blob_in_, true);
    RETURN_ON_NEQ(status, TNN_OK);
    //cpu acc forward
    status = cpu_adapter_acc_->Forward(cpu_blob_in_, cpu_blob_out_);
    RETURN_ON_NEQ(status, TNN_OK);

    status = ConvertBlobForAdaptorAcc(outputs, cpu_blob_out_, false);

    return status;
}

std::vector<DataFormat> MetalCpuAdapterAcc::SupportDataFormat(DataType data_type, int dims_size, BlobType blob_type) {
    return {DATA_FORMAT_NC4HW4};
}

}
