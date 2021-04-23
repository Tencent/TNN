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

#ifndef TNN_SOURCE_TNN_NETWORK_TENSORRT_TENSORRT_TENSOR_H_
#define TNN_SOURCE_TNN_NETWORK_TENSORRT_TENSORRT_TENSOR_H_

#include "NvInfer.h"

#include "tnn/core/status.h"
#include "tnn/interpreter/layer_resource.h"
#include "tnn/extern_wrapper/foreign_tensor.h"

namespace TNN_NS {

// @brief Base Type of a TensorRT Tensor
class TensorRTTensor : public ForeignTensor {
public:
    explicit TensorRTTensor() {};

    // @brief virtual destructor
    virtual ~TensorRTTensor() {};

    // @brief get the ITensor
    nvinfer1::ITensor* GetTensor() {
        return m_trt_tensor;
    }

    Status SetTensor(nvinfer1::ITensor* tensor) {
        m_trt_tensor = tensor;
        return TNN_OK;
    }

    IntScaleResource * GetIntResource() {
        return resource_;
    }

    void SetIntResource(IntScaleResource * resource) {
        resource_ = resource;
    }

    void SetInt8Mode(bool flag) {
        int8_mode = flag;
    }

    bool GetInt8Mode() {
        return int8_mode;
    }

    bool IsQuantized() {
        return quantized;
    }

    void SetQuantized() {
        quantized = true;
    }

    bool IsShapeTensor() {
        return shape_tensor;
    }

    void SetShapeTensor() {
        shape_tensor = true;
    }

    void SetShapeBlobName(std::string name) {
        shape_blob_name = name;
    }

    std::string GetShapeBlobName() {
        return shape_blob_name;
    }

private:
    bool int8_mode = false;
    bool quantized = false;
    bool shape_tensor = false;
    std::string shape_blob_name;
    IntScaleResource *resource_ = nullptr;
    nvinfer1::ITensor* m_trt_tensor = nullptr;
};

}  //  namespace TNN_NS

#endif  //  TNN_SOURCE_TNN_NETWORK_TENSORRT_TENSORRT_TENSOR_H_
