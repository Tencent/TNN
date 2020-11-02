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

#include "tnn/device/arm/acc/arm_nchw_layer_acc.h"
#include "tnn/device/arm/arm_common.h"
#include "tnn/utils/data_type_utils.h"
#include "tnn/utils/dims_vector_utils.h"
#include "tnn/utils/naive_compute.h"

namespace TNN_NS {

DECLARE_ARM_NCHW_ACC(Reorg, LAYER_REORG);

Status ArmReorgLayerAcc::DoForward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto param = dynamic_cast<ReorgLayerParam *>(param_);
    CHECK_PARAM_NULL(param);
    DataType data_type = inputs[0]->GetBlobDesc().data_type;
    auto input_dims    = inputs[0]->GetBlobDesc().dims;
    auto output_dims   = outputs[0]->GetBlobDesc().dims;

    int stride  = param->stride;
    int forward = param->forward;
    int mode    = param->mode;

    AllocConvertBuffer(inputs, outputs);

    UnPackInputs(inputs);
    if (data_type == DATA_TYPE_FLOAT) {
        auto *bottom_data = reinterpret_cast<float *>(GetBlobHandlePtr(nchw_blob_in[0]->GetHandle()));
        auto *top_data = reinterpret_cast<float *>(GetBlobHandlePtr(nchw_blob_out[0]->GetHandle()));

        if (forward) {
            int batch             = input_dims[0];
            int channel           = input_dims[1];
            int height            = input_dims[2];
            int width             = input_dims[3];
            NaiveReorg(bottom_data, width, height, channel, batch, stride, forward, mode, top_data);
        } else {
            int batch              = output_dims[0];
            int channel            = output_dims[1];
            int height             = output_dims[2];
            int width              = output_dims[3];
            NaiveReorg(bottom_data, width, height, channel, batch, stride, forward, mode, top_data);
        }
    } else {
        return Status(TNNERR_LAYER_ERR, "NO IMPLEMENT FOR int8/bfp16 shuffle, in todo list");
    }
    PackOutputs(outputs);

    return TNN_OK;
}

REGISTER_ARM_ACC(Reorg, LAYER_REORG)

}  // namespace TNN_NS
