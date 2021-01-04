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

#ifndef TNN_SOURCE_TNN_DEVICE_CUDA_UTILS_CUDA_BLOB_CONVERTER_KERNEL_CUH_
#define TNN_SOURCE_TNN_DEVICE_CUDA_UTILS_CUDA_BLOB_CONVERTER_KERNEL_CUH_

#include "tnn/core/macro.h"

namespace TNN_NS {

void BlobToBGR(int batch, int CHW, int HW, const float *src, unsigned char *dst, cudaStream_t stream,
        int channels, float *scale, float *bias, bool reverse_channel);

void BlobToGray(int count, const float *src, unsigned char *dst, cudaStream_t stream, float scale, float bias);

void BGRToBlob(int batch, int CHW, int HW, const unsigned char *src, float *dst, cudaStream_t stream,
        int channels, float *scale, float *bias, bool reverse_channel);

void GrayToBlob(int count, const unsigned char *src, float *dst, cudaStream_t stream, float scale, float bias);

void ScaleBias(const float* src, float* dst, cudaStream_t stream, float* scale, float* bias, int batch, int channels, int hw);

}  //  namespace TNN_NS;

#endif  //  TNN_SOURCE_TNN_DEVICE_CUDA_UTILS_CUDA_BLOB_CONVERTER_KERNEL_CUH_
