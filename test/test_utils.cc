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

#include <math.h>

#include "tnn/core/common.h"
#include "tnn/utils/bfp16.h"

namespace TNN_NS {

DeviceType ConvertDeviceType(std::string device_type) {
    if ("METAL" == device_type) {
        return DEVICE_METAL;
    } else if ("OPENCL" == device_type) {
        return DEVICE_OPENCL;
    } else if ("CUDA" == device_type) {
        return DEVICE_CUDA;
    } else if ("X86" == device_type) {
        return DEVICE_X86;
    } else if ("NAIVE" == device_type) {
        return DEVICE_NAIVE;
    } else if ("HUAWEI_NPU" == device_type) {
        return DEVICE_HUAWEI_NPU;
    } else if ("RKNPU" == device_type) {
        return DEVICE_RK_NPU;
    } else {
        return DEVICE_ARM;
    }
}

ModelType ConvertModelType(std::string model_type) {
    if ("OPENVINO" == model_type) {
        return MODEL_TYPE_OPENVINO;
    } else if ("SNPE" == model_type) {
        return MODEL_TYPE_SNPE;
    } else if ("COREML" == model_type) {
        return MODEL_TYPE_COREML;
    } else if ("NCNN" == model_type) {
        return MODEL_TYPE_NCNN;
    } else {
        return MODEL_TYPE_TNN;
    }
}

NetworkType ConvertNetworkType(std::string network_type) {
    if ("OPENVINO" == network_type) {
        return NETWORK_TYPE_OPENVINO;
    } else if ("SNPE" == network_type) {
        return NETWORK_TYPE_SNPE;
    } else if ("COREML" == network_type) {
        return NETWORK_TYPE_COREML;
    } else if ("HUAWEI_NPU" == network_type) {
        return NETWORK_TYPE_HUAWEI_NPU;
    } else if ("RKNPU" == network_type) {
        return NETWORK_TYPE_RK_NPU;
    } else {
        return NETWORK_TYPE_DEFAULT;
    }
}

Precision ConvertPrecision(std::string precision) {
    if ("HIGH" == precision) {
        return PRECISION_HIGH;
    } else if ("NORMAL" == precision) {
        return PRECISION_NORMAL;
    } else if ("LOW" == precision) {
        return PRECISION_LOW;
    } else {
        return PRECISION_AUTO;
    }
}

int CompareData(const float* ref_data, const float* result_data, size_t n, float ep) {
    for (unsigned long long i = 0; i < n; i++) {
        float diff = static_cast<float>(fabs(result_data[i] - ref_data[i]));
        float sum  = static_cast<float>(fabs(result_data[i]) + fabs(ref_data[i]));
        if (fabs(diff / sum) > ep && fabs(diff) > 1e-4f) {
            printf("ERROR AT %llu result %.6f ref %.6f\n", i, result_data[i], ref_data[i]);
            return -1;
        }
    }

    return 0;
}
int CompareData(const bfp16_t* ref_data, const bfp16_t* result_data, size_t n, float ep) {
    for (unsigned long long i = 0; i < n; i++) {
        float diff = static_cast<float>(fabs(float(result_data[i]) - float(ref_data[i])));
        float sum  = static_cast<float>(fabs(float(result_data[i])) + fabs(float(ref_data[i])));
        if (fabs(diff / sum) > ep && fabs(diff) > 5e-2f) {
            printf("ERROR AT %llu result %.6f ref %.6f\n", i, float(result_data[i]), float(ref_data[i]));
            return -1;
        }
    }

    return 0;
}

int CompareData(const int8_t* ref_data, const int8_t* result_data, size_t n) {
    for (unsigned long long i = 0; i < n; i++) {
        if (abs(result_data[i] - ref_data[i]) > 1) {
            LOGE("ERROR AT %llu result %d ref %d\n", i, result_data[i], ref_data[i]);
            return -1;
        }
    }

    return 0;
}

int CompareData(const uint8_t* ref_data, const uint8_t* result_data, int mat_channel, int channel, size_t n) {
    for (unsigned long long i = 0; i < n; i++) {
        int c = i % mat_channel;
        if (c >= channel)
            continue;
        if (abs(result_data[i] - ref_data[i]) > 1) {
            LOGE("ERROR AT %llu result %d ref %d\n", i, result_data[i], ref_data[i]);
            return -1;
        }
    }

    return 0;
}

}  // namespace TNN_NS
