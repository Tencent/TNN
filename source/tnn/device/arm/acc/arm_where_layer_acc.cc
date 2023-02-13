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

#include "tnn/device/arm/acc/arm_where_layer_acc.h"

#include "tnn/device/arm/arm_common.h"
#include "tnn/utils/bfp16.h"
#include "tnn/utils/data_format_converter.h"
#include "tnn/utils/data_type_utils.h"
#include "tnn/utils/dims_utils.h"

namespace TNN_NS {

Status ArmWhereLayerAcc::Init(Context *context, LayerParam *param, LayerResource *resource,
                                const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    Status ret = ArmLayerAcc::Init(context, param, resource, inputs, outputs);
    RETURN_ON_NEQ(ret, TNN_OK);

    return TNN_OK;
}

inline void getBroadCastNext(const DimsVector &input_shape, const DimsVector &output_shape, const DimsVector &sub_size, DimsVector &cur_dims, int &offset, const int &rank) {
    int cur_rank_id = rank - 1;
    cur_dims[cur_rank_id]++;
    bool carry = false;
    while (cur_rank_id > 0 && cur_dims[cur_rank_id] == output_shape[cur_rank_id]) {
        carry = true;
        cur_dims[cur_rank_id] = 0;
        cur_dims[--cur_rank_id]++;
    }
    if (input_shape[cur_rank_id] != 1) {
        offset++;
    } else if (carry) {
        offset -= (sub_size[cur_rank_id] - 1);
    }
}

inline void calculateSubSize(const DimsVector &input_shape, DimsVector &sub_size) {
    for (int i = sub_size.size() - 2; i >= 0; i--) {
        sub_size[i] = input_shape[i + 1] * sub_size[i + 1];
    }
}

template <typename T>
Status ArmWhereLayerAcc::Exec(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto shape_output = outputs[0]->GetBlobDesc().dims;
    const int count = DimsVectorUtils::Count(shape_output);
    T *output_data  = reinterpret_cast<T *>(GetBlobHandlePtr(outputs[0]->GetHandle()));
    

    // LOGD("where 逻辑加速\n");
    T *input_data_0 = reinterpret_cast<T *>(GetBlobHandlePtr(inputs[0]->GetHandle()));
    T *input_data_1 = reinterpret_cast<T *>(GetBlobHandlePtr(inputs[1]->GetHandle()));
    int8_t *input_data_2 = reinterpret_cast<int8_t *>(GetBlobHandlePtr(inputs[2]->GetHandle()));

    int rank = outputs[0]->GetBlobDesc().dims.size();
    DimsVector input0_dims = (inputs[0]->GetBlobDesc().dims.size() == rank) ? inputs[0]->GetBlobDesc().dims : std::vector<int>(rank, 1);
    DimsVector input1_dims = (inputs[1]->GetBlobDesc().dims.size() == rank) ? inputs[1]->GetBlobDesc().dims : std::vector<int>(rank, 1);
    DimsVector input2_dims = (inputs[2]->GetBlobDesc().dims.size() == rank) ? inputs[2]->GetBlobDesc().dims : std::vector<int>(rank, 1);
    DimsVector output_dims = outputs[0]->GetBlobDesc().dims;

    DimsVector sub_size_0(input0_dims.size(), 1);
    DimsVector sub_size_1(input1_dims.size(), 1);
    DimsVector sub_size_2(input2_dims.size(), 1);
    calculateSubSize(input0_dims, sub_size_0);
    calculateSubSize(input1_dims, sub_size_1);
    calculateSubSize(input2_dims, sub_size_2);

    DimsVector cur_dims_0(input0_dims.size(), 0);
    DimsVector cur_dims_1(input1_dims.size(), 0);
    DimsVector cur_dims_2(input2_dims.size(), 0);
    
    int idx_0 = 0, idx_1 = 0, idx_2 = 0;
    for (int offset = 0; offset < count; ++offset) {
        output_data[offset] = input_data_2[idx_2] != 0 ? input_data_0[idx_0] : input_data_1[idx_1];
        getBroadCastNext(input0_dims, output_dims, sub_size_0, cur_dims_0, idx_0, rank);
        getBroadCastNext(input1_dims, output_dims, sub_size_1, cur_dims_1, idx_1, rank);
        getBroadCastNext(input2_dims, output_dims, sub_size_2, cur_dims_2, idx_2, rank);
    }

    return TNN_OK;
}

Status ArmWhereLayerAcc::DoForward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    Blob *output_blob = outputs[0];
    auto data_type = output_blob->GetBlobDesc().data_type;

    if (data_type == DATA_TYPE_INT32) {
        Exec<int>(inputs, outputs);
    }  else if (data_type == DATA_TYPE_FLOAT) {
        Exec<float>(inputs, outputs);
    } else {
        LOGE("Error: ArmEqualLayerAcc don't support data type: %d\n", output_blob->GetBlobDesc().data_type);
        return Status(TNNERR_MODEL_ERR, "Error: ArmEqualLayerAcc don't support data type");
    }

    return TNN_OK;
}

REGISTER_ARM_ACC(Where, LAYER_WHERE);
REGISTER_ARM_LAYOUT(LAYER_WHERE, DATA_FORMAT_NCHW)

}  // namespace TNN_NS
