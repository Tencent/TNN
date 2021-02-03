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

#include "tnn/device/cuda/acc/cuda_layer_acc.h"

#include "tnn/core/abstract_device.h"
#include "tnn/utils/blob_transfer_utils.h"

namespace TNN_NS {

CudaLayerAcc::~CudaLayerAcc() {
    for (int i = 0; i < tempbufs_.size(); i++) {
        Status ret = device_->Free(tempbufs_[i].ptr);
        if (ret != TNN_OK) {
            LOGE("Error cuda free acc temp buf failed\n");
        }
    }
}

Status CudaLayerAcc::Init(Context *context, LayerParam *param, LayerResource *resource,
            const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    AbstractLayerAcc::Init(context, param, resource, inputs, outputs);

    device_   = dynamic_cast<CudaDevice*>(GetDevice(DEVICE_CUDA));
    param_    = param;
    resource_ = resource;
    context_ = dynamic_cast<CudaContext*>(context);

    auto status = ReloadConstantBlobs(inputs);
    RETURN_ON_NEQ(status, TNN_OK);

    return CudaLayerAcc::Reshape(inputs, outputs);
}


Status CudaLayerAcc::ReloadConstantBlobs(const std::vector<Blob *> &inputs) {
    void *command_queue = nullptr;
    context_->GetCommandQueue(&command_queue);

    auto const_resource = const_resource_;
    auto const_blob_map = const_blob_map_;
    for (auto iter : inputs) {
        auto name = iter->GetBlobDesc().name;
        if (const_resource == nullptr || const_resource->find(name) == const_resource->end()) {
            continue;
        }
        
        auto buffer = (*const_resource)[name];
        if (buffer->GetBytesSize() == 0 ) {
            continue;
        }

        std::shared_ptr<Blob> blob = nullptr;
        auto status = RawBuffer2Blob(buffer.get(), blob);
        RETURN_ON_NEQ(status, TNN_OK);

        BlobDesc desc = blob->GetBlobDesc();
        desc.device_type = DEVICE_CUDA;
        auto cuda_blob = std::make_shared<Blob>(desc, true);
        CopyToDevice(cuda_blob.get(), blob.get(), command_queue);
        cuda_blob->flag = DATA_FLAG_CHANGE_NEVER;
        const_blob_map[name] = cuda_blob;
        iter->SetHandle(cuda_blob->GetHandle());
        LOGD("CUDA Reload constant blob: %s\n", name.c_str());
    }
    const_blob_map_ = const_blob_map;
    return TNN_OK;
}


Status CudaLayerAcc::Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    return TNN_OK;
}

Status CudaLayerAcc::Forward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    return TNN_OK;
}

std::vector<DataFormat> CudaLayerAcc::SupportDataFormat(DataType data_type, int dims_size) {
    std::vector<DataFormat> support_list;
    if (dims_size == 4) {
        support_list.push_back(DATA_FORMAT_NCHW);
    }
    return support_list;
}

void CudaLayerAcc::CreateTempBuf(size_t size) {
    CudaTempBufUnit buf;
    device_->Allocate(&(buf.ptr), size);
    tempbufs_.push_back(buf);
}

}  //  namespace TNN_NS