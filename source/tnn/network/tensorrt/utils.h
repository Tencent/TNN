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

#ifndef TNN_SOURCE_TNN_NETWORK_TENSORRT_UTILS_H_
#define TNN_SOURCE_TNN_NETWORK_TENSORRT_UTILS_H_

#include <cuda_runtime_api.h>
#include <NvInfer.h>

#include "tnn/core/macro.h"
#include "tnn/core/common.h"
#include "tnn/interpreter/raw_buffer.h"

namespace TNN_NS {

std::string GetGpuType(int gpu_id);

std::string GetGpuArch(int gpu_id);

std::string GetCudaVersion();

std::string GetTrtVersion();

DataType ConvertTRTDataType(nvinfer1::DataType type);

DataFormat ConvertTRTDataFormat(nvinfer1::TensorFormat format);

nvinfer1::Dims ConvertPaddingToTRTDims(DimsVector dims);

nvinfer1::Dims ConvertToTRTDims(DimsVector dims);

nvinfer1::Dims ConvertToTRTDynamicDims(DimsVector dims);

nvinfer1::Dims ConvertToTRTDynamicDims(nvinfer1::Dims max_dims, nvinfer1::Dims min_dims);

nvinfer1::Dims ConvertToTRTDimsReverse(DimsVector dims);

nvinfer1::DataType ConvertToTRTDataType(DataType type);

nvinfer1::ILayer* AddReshapeToNetwork(nvinfer1::INetworkDefinition* network, nvinfer1::ITensor* input_tensor,
    DimsVector reshape_dims, const char* layer_name);

nvinfer1::Weights ConvertToWeights(RawBuffer *buf, bool zero_weight = false, DataType recommend_type = DATA_TYPE_FLOAT);

nvinfer1::ILayer* ConvertWeightToConstLayer(nvinfer1::INetworkDefinition* network, RawBuffer *buf,
    DimsVector recommend_dims=DimsVector(), int expand_dims = 0);

void BroadcastTensors(nvinfer1::INetworkDefinition* network, nvinfer1::ITensor*& t1, nvinfer1::ITensor*& t2);

void BroadcastTensors(nvinfer1::INetworkDefinition* network, nvinfer1::ITensor*& t1,
    nvinfer1::ITensor*& t2, nvinfer1::ITensor*& t3);

}  //  namespace TNN_NS

#endif  //  TNN_SOURCE_TNN_NETWORK_TENSORRT_UTILS_H_
