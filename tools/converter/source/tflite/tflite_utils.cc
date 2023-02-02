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

#include "tflite_utils.h"

#include <cstring>

#include "tnn/core/common.h"
#include "tnn/core/macro.h"
#include "tnn/core/status.h"
#include "tnn/interpreter/layer_resource.h"
#include "tnn/interpreter/raw_buffer.h"
#include "tnn/interpreter/tnn/objseri.h"
#include "tnn/utils/dims_vector_utils.h"

namespace TNN_CONVERTER {

bool TFLiteConvertOHWI2OIHW(const float* src, float* dst, int CO, int KH, int KW, int CI) {
    ASSERT(CO > 0);
    ASSERT(KH > 0);
    ASSERT(KW > 0);
    ASSERT(CI > 0);
    ASSERT(src != nullptr);
    for (int co = 0; co < CO; ++co) {
        for (int ci = 0; ci < CI; ++ci) {
            for (int h = 0; h < KH; ++h) {
                for (int w = 0; w < KW; ++w) {
                    dst[(co * CI + ci) * KH * KW + h * KW + w] = src[(co * KH + h) * KW * CI + w * CI + ci];
                }
            }
        }
    }
    return true;
}

bool TFLiteConvertOHWI2IOHW(const float* src, float* dst, int CO, int KH, int KW, int CI) {
    ASSERT(CI > 0);
    ASSERT(KH > 0);
    ASSERT(KW > 0);
    ASSERT(CO > 0);
    ASSERT(src != nullptr);
    for (int ci = 0; ci < CI; ++ci) {
        for (int co = 0; co < CO; ++co) {
            for (int h = 0; h < KH; ++h) {
                for (int w = 0; w < KW; ++w) {
                    dst[(ci * CO + co) * KH * KW + h * KW + w] = src[(co * KH + h) * KW * CI + w * CI + ci];
                }
            }
        }
    }
    return true;
}

bool ConvertShapeFormatTFLite(std::vector<int32_t>& shape) {
    if (shape.empty()) {
        LOGE("TNN Converter do not support wrong shape!\n");
        return false;
    }
    if (shape.size() < 3) {
        return true;
    } else if (shape.size() == 3) {
        auto h = shape[1];
        auto c = shape[2];
        shape[1] = c;
        shape[2] = h;
    } else if (shape.size() == 4) {
        // shape [n, h , w, c] -> shape [n, c, h, w]
        auto h   = shape[1];
        auto w   = shape[2];
        auto c   = shape[3];
        shape[1] = c;
        shape[2] = h;
        shape[3] = w;
    } else {
        LOGE("TNN Converter do not support wrong shape!\n");
        return false;
    }
    return true;
}

bool ConvertPermFormatTFLite(std::vector<int32_t>& perm) {
    if (perm.empty()) {
        LOGE("TNN Converter do not support wrong perm!\n");
        return false;
    }

    int perm_size = perm.size();
    if (perm_size > 4) {
        LOGE("TNN Transpose do not support perm's size larger than 4!\n");
        return false;
    }

    for (int i = perm_size; i < 4; i++) {
        perm.emplace_back(i);
    }

    std::map<int, int> nhwc_to_nchw;
    nhwc_to_nchw[0] = 0;
    nhwc_to_nchw[1] = 2;
    nhwc_to_nchw[2] = 3;
    nhwc_to_nchw[3] = 1;

    for (auto& v : perm) {
        v = nhwc_to_nchw[v];
    }
    ConvertShapeFormatTFLite(perm);

    return true;
}

// template <typename T>
bool ConvertConstFormatTFLite(int32_t const* dst, int32_t const* src, std::vector<int32_t> shape) {
    ASSERT(shape.size() == 2);
    ASSERT(shape[0] == 4);
    int data_size = shape[1];
    // std::memcpy((void*)(dst + 0 * data_size), src + 0 * data_size, data_size*sizeof(int32_t));
    std::memcpy((void*)(dst + 0 * data_size), src + 2 * data_size, data_size * sizeof(int32_t));
    std::memcpy((void*)(dst + 1 * data_size), src + 1 * data_size, data_size * sizeof(int32_t));
    std::memcpy((void*)(dst + 2 * data_size), src + 3 * data_size, data_size * sizeof(int32_t));
    return true;
}

int ConvertAxisFormatTFLite(int axis, int input_shape_size) {
    assert(axis > -4 && axis < 4);
    if (axis < 0) {
        axis += input_shape_size;
    }

    if (input_shape_size == 2) {
        return axis;
    } else if (input_shape_size == 3) {
        // [n,h,c] -> [n,c,h]
        switch (axis) {
            case 1:
                return 2;
            case 2:
                return 1;
            default:
                return 0;
        }
    } else if (input_shape_size == 4) {
        switch (axis) {
            case 0:
                return 0;
            case 1:
                return 2;
            case 2:
                return 3;
            default:
                return 1;
        }
    }

    return axis;
}

int Count(std::vector<int> shape) {
    if (shape.empty()) {
        return 0;
    }
    int count = 1;
    for (auto i : shape) {
        count *= i;
    }
    return count;
}

int SizeofTFLiteTensorData(tflite::TensorType type) {
    switch (type) {
        case tflite::TensorType_FLOAT32:
            return sizeof(float);
        case tflite::TensorType_INT32:
            return sizeof(int32_t);
        case tflite::TensorType_INT16:
            return sizeof(int16_t);
        case tflite::TensorType_INT64:
            return sizeof(int64_t);
        default:
            return 0;
    }
    return 0;
}

void Mask(std::vector<int> shape, int mask, int upper, std::vector<int>& v) {
    int window = 0x1;
    for (int i = 0; i < shape.size(); ++i) {
        if (mask & window) {
            // upper == 0: 处理的是 begin，取的是 0
            // upper != 0: 处理的是 ends， 取最大值
            v[i] = upper == 0? 0: shape[i];
        }
        window = window << 1;
    }
}

TNN_NS::Status CreateBlobScaleResource(TNN_NS::NetResource& net_resource,
                                      const std::vector<std::unique_ptr<tflite::TensorT>>& tf_lite_tensors, int tensor_index) {
    const auto& tensor    = tf_lite_tensors[tensor_index];
    const auto scale_name = tensor->name + BLOB_SCALE_SUFFIX;
    if (net_resource.resource_map.find(scale_name) == net_resource.resource_map.end()) {
        const auto scale_resource             = std::make_shared<TNN_NS::IntScaleResource>();
        scale_resource->name                  = scale_name;
        net_resource.resource_map[scale_name] = scale_resource;
        auto& quantization                    = tensor->quantization;
        std::vector<float> scales             = quantization->scale;
        if (scales.empty()) {
            return TNN_NS::Status(TNN_NS::TNNERR_CONVERT_INVALID_MODEL, "The scale size is empty\n");
        }
        auto scale_handle = TNN_NS::RawBuffer(scales.size() * sizeof(float));
        scale_handle.SetDataType(TNN_NS::DATA_TYPE_FLOAT);
        TNN_NS::DimsVector scale_dims;
        scale_dims.push_back(scales.size());
        scale_handle.SetBufferDims(scale_dims);
        auto scale_handle_data = scale_handle.force_to<float*>();
        for (int i = 0; i < scales.size(); i++) {
            scale_handle_data[i] = scales[i];
        }
        scale_resource->scale_handle = scale_handle;
        // for symmetric quantization zero point always is 0
        auto zero_point_handle = TNN_NS::RawBuffer(scales.size() * sizeof(int8_t));
        zero_point_handle.SetDataType(TNN_NS::DATA_TYPE_INT8);
        zero_point_handle.SetBufferDims(scale_dims);
        scale_resource->zero_point_handle = zero_point_handle;

        auto bias_handle = TNN_NS::RawBuffer(scales.size() * sizeof(int32_t));
        bias_handle.SetDataType(TNN_NS::DATA_TYPE_INT32);
        bias_handle.SetBufferDims(scale_dims);
        scale_resource->bias_handle = bias_handle;
    }
    return TNN_NS::TNN_CONVERT_OK;
}


TNN_NS::DataType GetTnnDataTypeFromTFLite(const tflite::TensorType& tensor_type) {
    switch (tensor_type) {
        case tflite::TensorType_FLOAT32: {
            return TNN_NS::DATA_TYPE_FLOAT;
        }
        case tflite::TensorType_FLOAT16: {
            return TNN_NS::DATA_TYPE_HALF;
        }
        case tflite::TensorType_UINT8:
        case tflite::TensorType_INT8: {
            return TNN_NS::DATA_TYPE_INT8;
        }
        case tflite::TensorType_INT32:
        case tflite::TensorType_INT64: {
            return TNN_NS::DATA_TYPE_INT32;
        }
        default: {
            LOGE("Not support tflite TensorType\n");
            assert(0);
        }
    }
}

}  // namespace TNN_CONVERTER
