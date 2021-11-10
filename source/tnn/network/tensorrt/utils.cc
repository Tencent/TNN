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

#include <string.h>
#include <string>
#include <stdio.h>

#include "tnn/network/tensorrt/utils.h"
#include "tnn/network/tensorrt/shape_tensor.h"
#include "tnn/core/macro.h"
#include "tnn/utils/data_type_utils.h"

namespace TNN_NS {

std::string GetGpuType(int gpu_id) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, gpu_id);
    int length = strlen(prop.name);
    for(int i=0;i<length;i++) {
        char c = prop.name[i];
        if (((c >= 'a') && (c<='z')) ||
            ((c >= 'A') && (c<='Z')) ||
            ((c >= '0') && (c<='9'))) {
            continue;
        }
        prop.name[i] = '_';
    }
    return std::string(prop.name);
}

std::string GetGpuArch(int gpu_id) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, gpu_id);
    char ss[50];
    sprintf(ss, "sm%1d%1d", prop.major, prop.minor);
    return std::string(ss);
}

std::string GetCudaVersion() {
    int version_num;

#ifndef CUDART_VERSION
#error CUDART_VERSION Undefined!
#else
    version_num = CUDART_VERSION;
#endif 

    char ss[50];
    sprintf(ss, "%02d", version_num / 1000);

    return std::string(ss);
}

std::string GetTrtVersion() {
    int version_num;

#ifndef NV_TENSORRT_MAJOR
#error NV_TENSORRT_MAJOR Undefined!
#else
    version_num = NV_TENSORRT_MAJOR * 100 + NV_TENSORRT_MINOR * 10 + NV_TENSORRT_PATCH;
#endif 

    char ss[50];
    sprintf(ss, "%3d", version_num);

    return std::string(ss);
}

DataType ConvertTRTDataType(nvinfer1::DataType type) {
    switch (type) {
        case nvinfer1::DataType::kFLOAT :
            return DATA_TYPE_FLOAT;
        case nvinfer1::DataType::kHALF :
            return DATA_TYPE_HALF;
        case nvinfer1::DataType::kINT32 :
            return DATA_TYPE_INT32;
        default:
            return DATA_TYPE_FLOAT;
    }
}

DataFormat ConvertTRTDataFormat(nvinfer1::TensorFormat format) {
    switch (format) {
        case nvinfer1::TensorFormat::kLINEAR :
            return DATA_FORMAT_NCHW;
        case nvinfer1::TensorFormat::kCHW2 :
            return DATA_FORMAT_NC2HW2;
        case nvinfer1::TensorFormat::kCHW4 :
            return DATA_FORMAT_NC4HW4;
        case nvinfer1::TensorFormat::kCHW16 :
            return DATA_FORMAT_NC16HW16;
        default:
            return DATA_FORMAT_NCHW;
    }
}

nvinfer1::Dims ConvertToTRTDims(DimsVector dims) {
    int dims_size = dims.size();
    nvinfer1::Dims trt_dims;
    trt_dims.nbDims = dims_size;
    for(int i = 0; i < dims_size; ++i) {
        trt_dims.d[i] = dims[i];
    }
    return trt_dims;
}

nvinfer1::Dims ConvertToTRTDimsReverse(DimsVector dims) {
    int dims_size = dims.size();
    nvinfer1::Dims trt_dims;
    trt_dims.nbDims = dims_size;
    int offset = 0;
    for(int i = dims_size-1; i >=0; i--) {
        trt_dims.d[offset++] = dims[i];
    }
    return trt_dims;
}

nvinfer1::Dims ConvertToTRTDynamicDims(DimsVector dims) {
    int dims_size = dims.size();
    nvinfer1::Dims trt_dims;
    trt_dims.nbDims = dims_size;
    trt_dims.d[0] = -1;
    for(int i = 1; i < dims_size; ++i) {
        trt_dims.d[i] = dims[i];
    }
    return trt_dims;
}

nvinfer1::Dims ConvertToTRTDynamicDims(nvinfer1::Dims max_dims, nvinfer1::Dims min_dims) {
    int dims_size = max_dims.nbDims;
    nvinfer1::Dims trt_dims;
    trt_dims.nbDims = dims_size;
    for(int i = 0; i < dims_size; ++i) {
        if (i == 1)
            trt_dims.d[i] = (max_dims.d[i] != min_dims.d[i]) ? -1 : max_dims.d[i];
        else
            trt_dims.d[i] = -1;
    }
    return trt_dims;
}

nvinfer1::DataType ConvertToTRTDataType(DataType type) {
    switch (type) {
        case DATA_TYPE_FLOAT:
            return nvinfer1::DataType::kFLOAT;
        case DATA_TYPE_HALF: 
            return nvinfer1::DataType::kHALF;
        case DATA_TYPE_INT32: 
            return nvinfer1::DataType::kINT32;
        default:
            return nvinfer1::DataType::kFLOAT;
    } 
}

nvinfer1::ILayer* ConvertWeightToConstLayer(nvinfer1::INetworkDefinition* network, RawBuffer *buf,
                                                            DimsVector recommend_dims, int expand_dims) {

    size_t buf_size_in_bytes = buf->GetDataCount() * DataTypeUtils::GetBytesSize(buf->GetDataType());

    nvinfer1::Weights const_weight;
    const_weight.type = ConvertToTRTDataType(buf->GetDataType());
    const_weight.values = buf->force_to<void*>();
    const_weight.count = buf->GetDataCount();

    DimsVector buf_dims = buf->GetBufferDims();

    if (recommend_dims.size() > 0 ) {
        buf_dims = recommend_dims;
    }

    if (buf_dims.size() == 0) {
        if (buf->GetDataCount() == 0) {
            LOGE("Warning: ConvertWeightToConstLayer got empty buffer\n");
            return nullptr;
        } else if (buf->GetDataCount() > 1) {
            LOGE("ConvertWeightToConstLayer got empty shapes\n");
            return nullptr;
        }
    }

    auto weightDims = ConvertToTRTDims(buf_dims);
    int origin_dims = weightDims.nbDims;
    if(expand_dims > origin_dims) {
        weightDims.nbDims = expand_dims;
        int diff = expand_dims - origin_dims;
        for(int i = expand_dims - 1; i >= diff; --i) {
            weightDims.d[i] = weightDims.d[i-diff];
        }
        for(int i = 0; i < diff; ++i) {
            weightDims.d[i] = 1;
        }
    }

    nvinfer1::ILayer* constant_layer = network->addConstant(weightDims, const_weight); 
    return constant_layer;
}

nvinfer1::ILayer* AddReshapeToNetwork(nvinfer1::INetworkDefinition* network, nvinfer1::ITensor* input_tensor,
        DimsVector reshape_dims, const char* layer_name) {
    nvinfer1::IShuffleLayer* shuffle_layer = network->addShuffle(*input_tensor);
    if (shuffle_layer != nullptr) {
        shuffle_layer->setName(layer_name);
        shuffle_layer->setReshapeDimensions(ConvertToTRTDims(reshape_dims));
    }
    return shuffle_layer;
}

nvinfer1::Weights ConvertToWeights(RawBuffer *buf, bool zero_weight, DataType recommend_type) {
    Weights weights;
    if (!zero_weight) {
        weights.type = ConvertToTRTDataType(buf->GetDataType());
        weights.values = buf->force_to<void*>();
        weights.count = buf->GetDataCount();
    } else {
        weights.type = ConvertToTRTDataType(recommend_type);
        weights.values = nullptr;
        weights.count = 0;
    }

    return weights;
}

nvinfer1::Dims ConvertPaddingToTRTDims(DimsVector dims) {
    nvinfer1::Dims trt_dims;
    if (dims.size() == 6) {
        trt_dims.nbDims = 3;
        trt_dims.d[0] = dims[4];
        trt_dims.d[1] = dims[2];
        trt_dims.d[2] = dims[0];
    } else if (dims.size() == 4) {
        trt_dims.nbDims = 2;
        trt_dims.d[0] = dims[2];
        trt_dims.d[1] = dims[0];
    }
    return trt_dims;
}

static void BroadcastTensor(nvinfer1::INetworkDefinition* network, nvinfer1::ITensor*& t, const int nbDims) {
    const auto inputDims = shapeOf(*t);
    const int nbInputDims = inputDims.size();
    assert(nbInputDims <= nbDims);
    if (nbInputDims < nbDims) {
        nvinfer1::IShuffleLayer* reshape = addShuffle(network, *t, concat(network, fillShapeVector(
            network, 1, shapeVector(nbDims - nbInputDims)), shapeOf(*t)));
        t = reshape->getOutput(0);
    }
}

void BroadcastTensors(nvinfer1::INetworkDefinition* network, nvinfer1::ITensor*& t1, nvinfer1::ITensor*& t2) {
    const int t1Dims = t1->getDimensions().nbDims;
    const int t2Dims = t2->getDimensions().nbDims;

    if (t1Dims == t2Dims) {
        return;
    }

    if (t1Dims > t2Dims) {
        BroadcastTensor(network, t2, t1Dims);
        return;
    }

    BroadcastTensor(network, t1, t2Dims);
}

void BroadcastTensors(nvinfer1::INetworkDefinition* network, nvinfer1::ITensor*& t1,
        nvinfer1::ITensor*& t2, nvinfer1::ITensor*& t3) {
    const int maxDims = std::max({t1->getDimensions().nbDims, t2->getDimensions().nbDims, t3->getDimensions().nbDims});
    BroadcastTensor(network, t1, maxDims);
    BroadcastTensor(network, t2, maxDims);
    BroadcastTensor(network, t3, maxDims);
}

}  //  namespace TNN_NS
