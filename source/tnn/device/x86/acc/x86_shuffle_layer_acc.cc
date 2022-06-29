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

#include "tnn/device/x86/acc/x86_layer_acc.h"
#include "tnn/utils/data_type_utils.h"
#include "tnn/utils/dims_vector_utils.h"

namespace TNN_NS {

static int shuffle_cpu(float *output, const float *input, int group_row, int group_column, int len) {
    int ele_size = sizeof(float);
    for (int i = 0; i < group_row; ++i)  // 2
    {
        for (int j = 0; j < group_column; ++j)  // 3
        {
            const float *p_i = input + (i * group_column + j) * len;
            float *p_o       = output + (j * group_row + i) * len;
            memcpy(p_o, p_i, len * ele_size);
        }
    }

    return 0;
}

DECLARE_X86_ACC(Shuffle, LAYER_SHUFFLE_CHANNEL);

Status X86ShuffleLayerAcc::DoForward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto param = dynamic_cast<ShuffleLayerParam *>(param_);
    if (!param) {
        LOGE("Error: ShuffleLayerParam is nil\n");
        return Status(TNNERR_MODEL_ERR, "Error: ShuffleLayerParam is nil");
    }

    auto input  = inputs[0];
    auto output = outputs[0];
    auto dims   = input->GetBlobDesc().dims;

    const float *bottom_data = handle_ptr<float *>(input->GetHandle());
    float *top_data          = handle_ptr<float *>(output->GetHandle());

    const int num              = dims[0];
    const int feature_map_size = DimsVectorUtils::Count(dims, 1);
    const int sp_sz            = DimsVectorUtils::Count(dims, 2);
    const int chs              = dims[1];

    int group_row    = param->group;
    int group_column = int(chs / group_row);

    assert(chs == (group_column * group_row));

    // Dtype* temp_data = temp_blob_.mutable_cpu_data();
    for (int n = 0; n < num; ++n) {
        shuffle_cpu(top_data + n * feature_map_size, bottom_data + n * feature_map_size, group_row, group_column,
                    sp_sz);
    }

    return TNN_OK;
}

REGISTER_X86_ACC(Shuffle, LAYER_SHUFFLE_CHANNEL);

}  // namespace TNN_NS
