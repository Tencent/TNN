// Tencent is pleased to support the open source community by making TNN available.
//
// Copyright (C) 2024 THL A29 Limited, a Tencent company. All rights reserved.
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

#include "tnn/device/atlas/torchaie/torchaie_blob_converter.h"
#include "tnn/device/atlas/torchaie/torchaie_network.h"
#include "tnn/utils/dims_utils.h"
#include <torch/torch.h>

namespace TNN_NS {

TorchAieBlobConverterAcc::TorchAieBlobConverterAcc(Blob* blob)
    : BlobConverterAcc(blob) {}

Status TorchAieBlobConverterAcc::ConvertToMat(Mat& mat, MatConvertParam param, void* command_queue) {
    std::string name = blob_->GetBlobDesc().name;
    TorchAieNetwork* network = reinterpret_cast<TorchAieNetwork*>(command_queue);
    at::Tensor tensor = network->GetOutputTensor(name);
    size_t byte_size = 0;
    auto dims = mat.GetDims();

    if (mat.GetMatType() == NCHW_FLOAT) {
        byte_size = 4 * DimsVectorUtils::Count(dims);
        tensor = tensor.toType(torch::kFloat).to("cpu").contiguous();
    } else if (mat.GetMatType() == NCHW_HALF) {
        byte_size = 2 * DimsVectorUtils::Count(dims);
        tensor = tensor.toType(torch::kFloat16).to("cpu").contiguous();
    } else {
        LOGE("torchaie blob converter only supports mat type NCHW_FLOAT or NCHW_HALF\n");
        return TNNERR_NET_ERR;
    }

    if (byte_size > 0) {
        memcpy(mat.GetData(), tensor.data_ptr(), byte_size);
    }

    return TNN_OK;
}

Status TorchAieBlobConverterAcc::ConvertToMatAsync(Mat& mat, MatConvertParam param, void* command_queue) {
    return ConvertToMat(mat, param, command_queue);
}

Status TorchAieBlobConverterAcc::ConvertFromMat(Mat& mat, MatConvertParam param, void* command_queue) {
    std::string name = blob_->GetBlobDesc().name;
    TorchAieNetwork* network = reinterpret_cast<TorchAieNetwork*>(command_queue);
    auto dims = mat.GetDims();
    std::vector<int64_t> dims_vector;

    for (auto i : dims) {
        dims_vector.push_back(static_cast<int64_t>(i));
    }

    c10::IntArrayRef tensor_shape(dims_vector.data(), dims_vector.size());
    at::Tensor tensor = torch::zeros(tensor_shape).to("cpu");
    size_t byte_size = 0;

    if (mat.GetMatType() == NCHW_FLOAT) {
        byte_size = 4 * DimsVectorUtils::Count(dims);
        tensor = tensor.toType(torch::kFloat).contiguous();
    } else if (mat.GetMatType() == NCHW_HALF) {
        byte_size = 2 * DimsVectorUtils::Count(dims);
        tensor = tensor.toType(torch::kFloat16).contiguous();
    } else {
        LOGE("torchaie blob converter only supports mat type NCHW_FLOAT or NCHW_HALF\n");
        return TNNERR_NET_ERR;
    }

    if (byte_size > 0) {
        memcpy(tensor.data_ptr(), mat.GetData(), byte_size);
    }

    return network->SetInputTensor(tensor, name);
}

Status TorchAieBlobConverterAcc::ConvertFromMatAsync(Mat& mat, MatConvertParam param, void* command_queue) {
    return ConvertFromMat(mat, param, command_queue);
}

DECLARE_BLOB_CONVERTER_CREATER(TorchAie);
REGISTER_BLOB_CONVERTER(TorchAie, DEVICE_ATLAS);

}  // namespace TNN_NS
