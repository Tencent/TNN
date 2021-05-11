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
#include "tnn/utils/dims_utils.h"
#include "tnn/utils/omp_utils.h"

namespace TNN_NS {

DECLARE_ARM_ACC(Tile, LAYER_REPEAT);

template <typename T>
Status ArmTileLayerAcc::Exec(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto input_blob  = inputs[0];
    auto output_blob = outputs[0];
    auto input_dims  = input_blob->GetBlobDesc().dims;
    auto output_dims = output_blob->GetBlobDesc().dims;
    int count        = DimsVectorUtils::Count(output_dims);
    T *input_data    = reinterpret_cast<T *>(GetBlobHandlePtr(input_blob->GetHandle()));
    T *output_data   = reinterpret_cast<T *>(GetBlobHandlePtr(output_blob->GetHandle()));
    OMP_PARALLEL_FOR_
    for (int index = 0; index < count; ++index) {
        int offset = 0;
        int prod   = count;
        for (int i = 0; i < input_dims.size(); i++) {
            prod /= output_dims[i];
            int mod = index / prod % input_dims[i];
            offset  = offset * input_dims[i] + mod;
        }
        output_data[index] = input_data[offset];
    }

    return TNN_OK;
}

Status ArmTileLayerAcc::DoForward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto layer_param = dynamic_cast<TileLayerParam *>(param_);
    CHECK_PARAM_NULL(layer_param);
    auto data_type = outputs[0]->GetBlobDesc().data_type;
    if (data_type == DATA_TYPE_FLOAT) {
        Exec<float>(inputs, outputs);
    } else if (data_type == DATA_TYPE_INT32) {
        Exec<int32_t>(inputs, outputs);
    } else if (data_type == DATA_TYPE_INT8) {
        Exec<int8_t>(inputs, outputs);
    } else {
        return Status(Status(TNNERR_MODEL_ERR, "ArmTileLayerAcc input has invalid data type"));
    }
    return TNN_OK;
}

REGISTER_ARM_ACC(Tile, LAYER_REPEAT);
REGISTER_ARM_LAYOUT(LAYER_REPEAT, DATA_FORMAT_NCHW);

}  // namespace TNN_NS