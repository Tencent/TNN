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

#include "tnn/device/arm/acc/arm_nchw_layer_acc.h"

#include "tnn/device/arm/arm_common.h"
#include "tnn/utils/data_type_utils.h"
#include "tnn/utils/dims_utils.h"

namespace TNN_NS {

ArmNchwLayerAcc::~ArmNchwLayerAcc(){};

template <typename T>
Status ArmNchwLayerAcc::UnPackInputs(const std::vector<Blob *> &inputs) {
    int ic_round_up = 4;
    if (std::is_same<T, float>::value) {
        ic_round_up = 4;
    } else if (std::is_same<T, fp16_t>::value) {
        ic_round_up = 8;
    }

    for (int i = 0; i < inputs.size(); i++) {
        auto input_dims = inputs[i]->GetBlobDesc().dims;
        for (int n = 0; n < input_dims[0]; ++n) {
            auto in_count  = DimsVectorUtils::Count(input_dims, 2) * ROUND_UP(input_dims[1], ic_round_up);
            auto out_count = DimsVectorUtils::Count(input_dims, 2) * input_dims[1];
            T *src         = reinterpret_cast<T *>(GetBlobHandlePtr(inputs[i]->GetHandle())) + n * in_count;
            T *dst         = reinterpret_cast<T *>(GetBlobHandlePtr(nchw_blob_in[i]->GetHandle())) + n * out_count;
            UnpackCX(dst, src, DimsVectorUtils::Count(input_dims, 2), input_dims[1]);
        }
    }
    return TNN_OK;
}

template Status ArmNchwLayerAcc::UnPackInputs<float>(const std::vector<Blob *> &inputs);
template Status ArmNchwLayerAcc::UnPackInputs<fp16_t>(const std::vector<Blob *> &inputs);

template <typename T>
Status ArmNchwLayerAcc::PackOutputs(const std::vector<Blob *> &outputs) {
    int oc_round_up = 4;
    if (std::is_same<T, float>::value) {
        oc_round_up = 4;
    } else if (std::is_same<T, fp16_t>::value) {
        oc_round_up = 8;
    }

    for (int i = 0; i < outputs.size(); i++) {
        auto out_dims                  = nchw_blob_out[i]->GetBlobDesc().dims;
        outputs[i]->GetBlobDesc().dims = out_dims;
        for (int n = 0; n < out_dims[0]; ++n) {
            auto in_count  = DimsVectorUtils::Count(out_dims, 2) * out_dims[1];
            auto out_count = DimsVectorUtils::Count(out_dims, 2) * ROUND_UP(out_dims[1], oc_round_up);
            T *src         = reinterpret_cast<T *>(GetBlobHandlePtr(nchw_blob_out[i]->GetHandle())) + n * in_count;
            T *dst         = reinterpret_cast<T *>(GetBlobHandlePtr(outputs[i]->GetHandle())) + n * out_count;
            PackCX(dst, src, DimsVectorUtils::Count(out_dims, 2), out_dims[1]);
        }
    }
    return TNN_OK;
}

template Status ArmNchwLayerAcc::PackOutputs<float>(const std::vector<Blob *> &outputs);
template Status ArmNchwLayerAcc::PackOutputs<fp16_t>(const std::vector<Blob *> &outputs);

Status ArmNchwLayerAcc::AllocConvertBuffer(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    int space_id = 0;
    nchw_blob_in.clear();
    nchw_blob_out.clear();
    for (auto blob : inputs) {
        auto desc = blob->GetBlobDesc();
        BlobHandle handle;
        handle.base = context_->GetSharedWorkSpace(
            DimsVectorUtils::Count(desc.dims) * DataTypeUtils::GetBytesSize(desc.data_type), space_id++);
        nchw_blob_in.push_back(std::make_shared<Blob>(desc, handle));
    }

    for (auto blob : outputs) {
        auto desc = blob->GetBlobDesc();
        BlobHandle handle;
        handle.base = context_->GetSharedWorkSpace(
            DimsVectorUtils::Count(desc.dims) * DataTypeUtils::GetBytesSize(desc.data_type), space_id++);
        nchw_blob_out.push_back(std::make_shared<Blob>(desc, handle));
    }

    return TNN_OK;
}

Status ArmNchwLayerAcc::DoForward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    return Status(TNNERR_LAYER_ERR, "CALL ERROR: NCHW BASE TYPE, NOT IMPLEMENT");
}

std::vector<Blob *> ArmNchwLayerAcc::GetNchwBlobVector(const std::vector<std::shared_ptr<Blob>> &blobs) {
    std::vector<Blob *> ret;
    for (auto v : blobs) {
        ret.push_back(v.get());
    }
    return ret;
}

}  // namespace TNN_NS
