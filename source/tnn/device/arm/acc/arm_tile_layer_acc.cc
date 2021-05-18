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
#include "tnn/utils/data_type_utils.h"
#include "tnn/utils/dims_vector_utils.h"
#include "tnn/utils/omp_utils.h"

namespace TNN_NS {

DECLARE_ARM_ACC(Tile, LAYER_REPEAT);

template <typename T>
Status ArmTileLayerAcc::Exec(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto input_blob        = inputs[0];
    auto output_blob       = outputs[0];
    auto input_dims        = input_blob->GetBlobDesc().dims;
    auto input_batch       = input_dims[0];
    auto input_channel     = input_dims[1];
    auto input_channel_r4  = ROUND_UP(input_channel, 4);
    auto input_stride      = DimsVectorUtils::Count(input_dims, 2);
    auto output_dims       = output_blob->GetBlobDesc().dims;
    auto output_batch      = output_dims[0];
    auto output_channel    = output_dims[1];
    auto output_channel_r4 = ROUND_UP(output_channel, 4);
    auto output_stride     = DimsVectorUtils::Count(output_dims, 2);
    auto count             = DimsVectorUtils::Count(output_dims, 1);

    T *input_data  = reinterpret_cast<T *>(GetBlobHandlePtr(input_blob->GetHandle()));
    T *output_data = reinterpret_cast<T *>(GetBlobHandlePtr(output_blob->GetHandle()));
    //    T *unpack_buffer = reinterpret_cast<T *>(context_->GetSharedWorkSpace(input_channel * input_stride *
    //    sizeof(T))); T *output_tmp_buffer =
    //        reinterpret_cast<T *>(context_->GetSharedWorkSpace(output_channel * output_stride * sizeof(T)));
    T *unpack_buffer     = reinterpret_cast<T *>(calloc(input_channel * input_stride, sizeof(float)));
    T *output_tmp_buffer = reinterpret_cast<T *>(calloc(output_channel * output_stride, sizeof(float)));
    for (int n = 0; n < output_batch; ++n) {
        auto input_ptr  = input_data + (n % input_batch) * input_channel_r4 * input_stride;
        auto output_ptr = output_data + n * output_channel_r4 * output_stride;
        UnpackC4(unpack_buffer, input_ptr, input_stride, input_channel);
        OMP_PARALLEL_FOR_
        for (int index = 0; index < count; ++index) {
            int offset = 0;
            int prod   = count;
            for (int i = 1; i < input_dims.size(); i++) {
                prod /= output_dims[i];
                int mod = index / prod % input_dims[i];
                offset  = offset * input_dims[i] + mod;
            }
            output_tmp_buffer[index] = unpack_buffer[offset];
        }
        PackC4(output_ptr, output_tmp_buffer, output_stride, output_channel);
    }
    delete unpack_buffer;
    delete output_tmp_buffer;

    return TNN_OK;
}

Status ArmTileLayerAcc::DoForward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto layer_param = dynamic_cast<TileLayerParam *>(param_);
    CHECK_PARAM_NULL(layer_param);
    auto data_type = outputs[0]->GetBlobDesc().data_type;
    if (data_type == DATA_TYPE_FLOAT) {
        Exec<float>(inputs, outputs);
    } else {
        return Status(Status(TNNERR_MODEL_ERR, "ArmTileLayerAcc input has invalid data type"));
    }
    return TNN_OK;
}

REGISTER_ARM_ACC(Tile, LAYER_REPEAT);

}  // namespace TNN_NS