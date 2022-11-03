//
//  arm_layer_acc.cpp
//  tnn
//
//  Created by seanxcwang on 2019/9/17.
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
//

#include "tnn/device/arm/acc/arm_layer_acc.h"

#include "tnn/core/profile.h"
#include "tnn/device/arm/arm_context.h"
#include "tnn/utils/blob_transfer_utils.h"
#include "tnn/utils/data_type_utils.h"
#include "tnn/utils/dims_utils.h"

namespace TNN_NS {

Status ArmLayerAcc::Init(Context *context, LayerParam *param, LayerResource *resource,
                         const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    AbstractLayerAcc::Init(context, param, resource, inputs, outputs);
    context_ = reinterpret_cast<ArmContext *>(context);

    param_    = param;
    resource_ = resource;
    k_param_  = std::make_shared<ArmKernelParam>();

    // init base k_param_
    auto input_dim  = inputs[0]->GetBlobDesc().dims;
    auto output_dim = outputs[0]->GetBlobDesc().dims;
    int ic          = DimsFunctionUtils::GetDim(input_dim, 1);
    int ih          = DimsFunctionUtils::GetDim(input_dim, 2);
    int iw          = DimsFunctionUtils::GetDim(input_dim, 3);
    int oc          = DimsFunctionUtils::GetDim(output_dim, 1);
    int oh          = DimsFunctionUtils::GetDim(output_dim, 2);
    int ow          = DimsFunctionUtils::GetDim(output_dim, 3);
    // k_param_ only used in conv, pooling, deconv
    k_param_->set_dims(ROUND_UP(ic, 4), ROUND_UP(ic, 8), ih, iw, ROUND_UP(oc, 4), ROUND_UP(oc, 8), oh, ow);

    RETURN_ON_NEQ(ReloadConstantBlobs(inputs, false), TNN_OK);

    return TNN_OK;
}

std::vector<DataFormat> ArmLayerAcc::SupportDataFormat(DataType data_type, int dims_size, BlobType blob_type) {
    std::vector<DataFormat> support_list;
    if (dims_size == 4) {
        if (data_type == DATA_TYPE_FLOAT || data_type == DATA_TYPE_BFP16)
            support_list.push_back(DATA_FORMAT_NC4HW4);
        else if (data_type == DATA_TYPE_INT8)
            support_list.push_back(DATA_FORMAT_NHWC4);
        else if (data_type == DATA_TYPE_HALF) {
            support_list.push_back(DATA_FORMAT_NC8HW8);
        }
    }
    return support_list;
}

bool ArmLayerAcc::UseNaiveConstantBlobs() {
    return false;
}

ArmLayerAcc::~ArmLayerAcc() {}

Status ArmLayerAcc::Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    // reinit k_param_ h,w
    auto input_dim  = inputs[0]->GetBlobDesc().dims;
    auto output_dim = outputs[0]->GetBlobDesc().dims;
    int ic          = DimsFunctionUtils::GetDim(input_dim, 1);
    int ih          = DimsFunctionUtils::GetDim(input_dim, 2);
    int iw          = DimsFunctionUtils::GetDim(input_dim, 3);
    int oc          = DimsFunctionUtils::GetDim(output_dim, 1);
    int oh          = DimsFunctionUtils::GetDim(output_dim, 2);
    int ow          = DimsFunctionUtils::GetDim(output_dim, 3);
    k_param_->set_dims(ROUND_UP(ic, 4), ROUND_UP(ic, 8), ih, iw, ROUND_UP(oc, 4), ROUND_UP(oc, 8), oh, ow);
    return TNN_OK;
}

Status ArmLayerAcc::ConfigBuffer2ArmBlobDesc(BlobDesc &desc) {
    return TNN_OK;
}

Status ArmLayerAcc::RawBuffer2ArmBlob(RawBuffer *buffer, std::shared_ptr<Blob> &blob, BlobDesc &desc) {
    if (!buffer) {
        LOGE("RawBuffer2ArmBlob:: buffer is null \n");
        return Status(TNNERR_PARAM_ERR, "RawBuffer2ArmBlob:: buffer is null");
    }

    const int count = blob ? DimsVectorUtils::Count(blob->GetBlobDesc().dims) : 0;

    if (!blob || buffer->GetDataCount() != count) {
        {
            desc.device_type = DEVICE_ARM;
            desc.dims        = buffer->GetBufferDims();
            ConfigBuffer2ArmBlobDesc(desc);
        }
        if (buffer->GetBytesSize() > 0) {
            blob = std::make_shared<Blob>(desc, true);
        } else {
            blob = std::make_shared<Blob>(desc, false);
        }
    }

    if (blob->GetHandle().base && buffer->GetBytesSize() > 0) {
        auto buff_dtype = buffer->GetDataType();
        auto blob_dtype = blob->GetBlobDesc().data_type;
        auto blob_fmt   = blob->GetBlobDesc().data_format;
        auto dims       = desc.dims;

        if (dims.size() < 2) {
            if (buff_dtype == blob_dtype) {
                memcpy(GetBlobHandlePtr(blob->GetHandle()), buffer->force_to<void *>(), buffer->GetBytesSize());
            } else {
                if (buff_dtype == DATA_TYPE_FLOAT && blob_dtype == DATA_TYPE_HALF) {
                    ConvertFromFloatToHalf(buffer->force_to<float *>(),
                                           GetBlobHandlePtr(blob->GetHandle()),
                                           buffer->GetBytesSize() / sizeof(float));
                } else if (buff_dtype == DATA_TYPE_HALF && blob_dtype == DATA_TYPE_FLOAT) {
                    ConvertFromHalfToFloat(buffer->force_to<void *>(),
                                           reinterpret_cast<float *>(GetBlobHandlePtr(blob->GetHandle())),
                                           buffer->GetBytesSize() / sizeof(fp16_t));
                } else {
                    LOGE("RawBuffer2ArmBlob:: unsupported buffer and blob data type: %d, %d\n", buff_dtype, blob_dtype);
                    return Status(TNNERR_PARAM_ERR, "RawBuffer2ArmBlob:: unsupported buffer and blob data type");
                }
            }
            return TNN_OK;
        }

        int batch       = DimsFunctionUtils::GetDim(dims, 0);
        int channel     = DimsFunctionUtils::GetDim(dims, 1);
        int hw          = DimsVectorUtils::Count(dims, 2);
        auto buff_count = batch * channel * hw;

        if (buff_dtype == DATA_TYPE_FLOAT) {
            auto src_ptr = buffer->force_to<float *>();
            if (blob_dtype == DATA_TYPE_FLOAT) {
                if (blob_fmt == DATA_FORMAT_NCHW) {
                    memcpy(reinterpret_cast<float *>(GetBlobHandlePtr(blob->GetHandle())), src_ptr,
                           buff_count * sizeof(float));
                } else {
                    PackFloatBlob(reinterpret_cast<float *>(GetBlobHandlePtr(blob->GetHandle())), src_ptr, batch,
                                  channel, hw);
                }
            } else if (blob_dtype == DATA_TYPE_HALF) {
                RawBuffer tmp_fp16_buff = RawBuffer(buff_count * sizeof(fp16_t));
                auto tmp_buff_ptr       = tmp_fp16_buff.force_to<fp16_t *>();
                ConvertFromFloatToHalf(src_ptr, tmp_buff_ptr, buff_count);
                if (blob_fmt == DATA_FORMAT_NCHW) {
                    memcpy(reinterpret_cast<fp16_t *>(GetBlobHandlePtr(blob->GetHandle())), tmp_buff_ptr,
                           buff_count * sizeof(fp16_t));
                } else {
                    PackHalfBlob(reinterpret_cast<fp16_t *>(GetBlobHandlePtr(blob->GetHandle())), tmp_buff_ptr, batch,
                                 channel, hw);
                }
            } else {
                LOGE("RawBuffer2ArmBlob:: unsupported blob data type: %d\n", blob_dtype);
                return Status(TNNERR_PARAM_ERR, "RawBuffer2ArmBlob:: unsupported blob data type");
            }
        } else {
            LOGE("RawBuffer2ArmBlob:: unsupported buffer data type: %d\n", buff_dtype);
            return Status(TNNERR_PARAM_ERR, "RawBuffer2ArmBlob:: unsupported buffer data type");
        }
    }

    return TNN_OK;
}

Status ArmLayerAcc::ReloadConstantBlobs(const std::vector<Blob *> &inputs, bool only_reload_shape_differ_blob) {
    auto const_resource = const_resource_;
    if (const_resource == nullptr) {
        return TNN_OK;
    }
    auto const_resource_flag = const_resource_flag_;
    auto const_blob_map      = const_blob_map_;

    // The default blob desc has the same data type and data format with non-constant input blob
    BlobDesc arm_default_desc;
    for (auto iter : inputs) {
        auto name = iter->GetBlobDesc().name;
        // skip const blobs
        if (const_resource->find(name) != const_resource->end()) {
            continue;
        }
        if (only_reload_shape_differ_blob &&
            !(const_resource_flag && const_resource_flag->find(name) == const_resource_flag->end())) {
            continue;
        }

        arm_default_desc.device_type = DEVICE_ARM;
        arm_default_desc.data_type   = iter->GetBlobDesc().data_type;
        arm_default_desc.data_format = iter->GetBlobDesc().data_format;
    }

    for (auto iter : inputs) {
        auto name = iter->GetBlobDesc().name;
        // deal with const blobs
        if (const_resource->find(name) == const_resource->end()) {
            continue;
        }
        if (only_reload_shape_differ_blob && const_resource_flag &&
            const_resource_flag->find(name) == const_resource_flag->end()) {
            continue;
        }

        LOGD("Reloading constant blob: %s, default data_type = %d, data_format = %d\n", name.c_str(),
             arm_default_desc.data_type, arm_default_desc.data_format);
        auto buffer                = (*const_resource)[name];
        std::shared_ptr<Blob> blob = nullptr;
        if (const_blob_map.find(name) != const_blob_map.end()) {
            blob = const_blob_map[name];
        }
        Status status;
        // convert fp16 to fp32 resource before converting from buffer to blob
        auto buffer_cvt = RawBuffer();
        if (buffer->GetDataType() == DATA_TYPE_HALF) {
            buffer_cvt = ConvertHalfHandle(*(buffer.get()));
            buffer_cvt.SetBufferDims(buffer->GetBufferDims());
        } else {
            buffer_cvt = *(buffer.get());
        }
        if (UseNaiveConstantBlobs()) {
            // the const blob has fp32 or integer dtype and nchw format
            status = RawBuffer2Blob(&buffer_cvt, blob);
        } else {
            // the const blob has the same dtype and format as other input blob of the layer
            arm_default_desc.dims = buffer_cvt.GetBufferDims();
            status                = RawBuffer2ArmBlob(&buffer_cvt, blob, arm_default_desc);
        }
        RETURN_ON_NEQ(status, TNN_OK);

        BlobDesc blob_desc = blob->GetBlobDesc();
        if (blob_desc.name.empty()) {
            blob_desc.name = iter->GetBlobDesc().name;
            blob->SetBlobDesc(blob_desc);
        }

        blob->SetFlag(DATA_FLAG_CHANGE_NEVER);
        const_blob_map[name] = blob;
        iter->SetHandle(blob->GetHandle());
        iter->GetBlobDesc() = blob->GetBlobDesc();
        LOGD("Reload constant blob: %s done\n", name.c_str());
    }
    const_blob_map_ = const_blob_map;
    return TNN_OK;
}

bool ArmLayerAcc::DataTypeSupported(DataType data_type) {
    if (data_type == DATA_TYPE_FLOAT || data_type == DATA_TYPE_BFP16 || data_type == DATA_TYPE_INT8 ||
        data_type == DATA_TYPE_HALF || data_type == DATA_TYPE_INT32) {
        return true;
    } else {
        return false;
    }
}

Status ArmLayerAcc::Forward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    Status status;
#if TNN_PROFILE
    auto pdata = std::make_shared<ProfilingData>();
    UpdateProfilingData(pdata.get(), param_, inputs[0]->GetBlobDesc().dims, outputs[0]->GetBlobDesc().dims);
    timer.Start();
#endif

    auto in_data_type = inputs[0]->GetBlobDesc().data_type;
    if (DataTypeSupported(in_data_type)) {
        status = this->DoForward(inputs, outputs);
    } else {
        LOGE("Error : arm layer acc got unsupported data type %d\n", in_data_type);
        return Status(TNNERR_LAYER_ERR, "Error: arm layer acc got unsupported data type.");
    }

#if TNN_PROFILE
    pdata->kernel_time = timer.TimeEclapsed();
    context_->AddProfilingData(pdata);
#endif

    RETURN_ON_NEQ(status, TNN_OK);

    return TNN_OK;
}

Status ArmLayerAcc::DoForward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    return Status(TNNERR_LAYER_ERR, "DoForward not implement");
}

}  // namespace TNN_NS
