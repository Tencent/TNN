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

#include "tnn/device/arm/acc/arm_layer_acc.h"

#include <cmath>
#include "tnn/device/arm/arm_common.h"
#include "tnn/device/arm/arm_context.h"
#include "tnn/utils/dims_vector_utils.h"

namespace TNN_NS {

DECLARE_ARM_ACC(PRelu, LAYER_PRELU);

template <typename T>
Status ArmPReluLayerAcc::Exec(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto layer_param = dynamic_cast<PReluLayerParam *>(param_);
    CHECK_PARAM_NULL(layer_param);

    auto layer_res = dynamic_cast<PReluLayerResource *>(resource_);
    CHECK_PARAM_NULL(layer_res);

    const int slope_size     = layer_res->slope_handle.GetBytesSize();
    const DataType data_type = layer_res->slope_handle.GetDataType();

    Blob *input_blob       = inputs[0];
    Blob *output_blob      = outputs[0];
    auto dims              = input_blob->GetBlobDesc().dims;
    const int channel      = dims[1];
    const int height       = dims[2];
    const int width        = dims[3];
    const int count        = dims[0] * ROUND_UP(dims[1], 4) * dims[2] * dims[3];
    const int channel_size = DimsVectorUtils::Count(output_blob->GetBlobDesc().dims, 2);

    const T *slope_data = layer_res->slope_handle.force_to<T *>();

    T *input_data  = reinterpret_cast<T *>(GetBlobHandlePtr(input_blob->GetHandle()));
    T *output_data = reinterpret_cast<T *>(GetBlobHandlePtr(output_blob->GetHandle()));
    if (layer_param->channel_shared) {
        for (int n = 0; n < UP_DIV(count, 4); n++) {
            Float4 v_data = Float4::load(input_data + n * 4);
            Float4 v_res  = Float4::bsl_clt(v_data, Float4(0.f), v_data * slope_data[0], v_data);
            Float4::save(output_data + n * 4, v_res);
        }
    } else {
        for (int batch_idx = 0; batch_idx < dims[0]; ++batch_idx) {
            auto input_ptr  = input_data + batch_idx * width * height * ROUND_UP(channel, 4);
            auto output_ptr = output_data + batch_idx * width * height * ROUND_UP(channel, 4);
            for (int dz = 0; dz < UP_DIV(channel, 4); ++dz) {
                T *src_z       = input_ptr + dz * width * height * 4;
                T *dst_z       = output_ptr + dz * width * height * 4;
                Float4 v_slope = Float4::load(slope_data + dz * 4);
                for (int p = 0; p < width * height; p++) {
                    Float4 v_data = Float4::load(src_z + p * 4);
                    Float4 v_res  = Float4::bsl_clt(v_data, Float4(0.f), v_data * v_slope, v_data);
                    Float4::save(dst_z + p * 4, v_res);
                }
            }
        }
    }

    return TNN_OK;
}

Status ArmPReluLayerAcc::DoForward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto in_data_type = inputs[0]->GetBlobDesc().data_type;
    if (in_data_type == DATA_TYPE_FLOAT) {
        return Exec<float>(inputs, outputs);
    } else if (in_data_type == DATA_TYPE_BFP16) {
        return Exec<bfp16_t>(inputs, outputs);
    } else {
        return TNNERR_LAYER_ERR;
    }
}

REGISTER_ARM_ACC(PRelu, LAYER_PRELU)
}  // namespace TNN_NS
