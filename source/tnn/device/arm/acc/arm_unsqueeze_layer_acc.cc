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
#include "tnn/utils/dims_utils.h"
#include "tnn/utils/data_type_utils.h"

namespace TNN_NS {

DECLARE_ARM_ACC(Unsqueeze, LAYER_UNSQUEEZE);

// 修改处：添加了新的函数TransDataType，用于将T_IN类别数据转化为T_OUT类别存储
template <typename T_IN, typename T_OUT>
Status TransDataType(void *data, const DimsVector &shapes) {
    if (std::is_same<T_IN, T_OUT>::value) {
        return TNN_OK;
    }
    size_t data_size = DimsVectorUtils::Count(shapes);
    T_OUT *output = (T_OUT *)data;
    T_IN *origin  = reinterpret_cast<T_IN *>(data);
    for (int i = 0; i < data_size; i++) {
        auto temp = origin[i];
        output[i] = (T_OUT)temp;
    }
    data = reinterpret_cast<void *>(output);
    return TNN_OK;
}

Status ArmUnsqueezeLayerAcc::DoForward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    void *input_data  = GetBlobHandlePtr(inputs[0]->GetHandle());
    void *output_data = GetBlobHandlePtr(outputs[0]->GetHandle());
    auto dims         = outputs[0]->GetBlobDesc().dims;
    auto count        = DimsVectorUtils::Count(dims);
    auto ele_size     = DataTypeUtils::GetBytesSize(outputs[0]->GetBlobDesc().data_type);

    if (input_data != output_data) {
        memcpy(output_data, input_data, count * ele_size);
    }

    if (inputs[0]->GetBlobDesc().data_type != outputs[0]->GetBlobDesc().data_type) {
        // LOGD("修改处：在 Unsqueeze 算子计算时，可能存在类型转换\n");
        if (inputs[0]->GetBlobDesc().data_type == DATA_TYPE_FLOAT && outputs[0]->GetBlobDesc().data_type == DATA_TYPE_INT32) {
            TransDataType<float, int32_t>(output_data, dims);
        }
        else if (inputs[0]->GetBlobDesc().data_type == DATA_TYPE_INT32 && outputs[0]->GetBlobDesc().data_type == DATA_TYPE_FLOAT) {
            TransDataType<int32_t, float>(output_data, dims);
        }
        else {
            LOGE("Unsupport DataType Transfer in Unsqueeze : %d -> %d\n", (int)inputs[0]->GetBlobDesc().data_type, (int)outputs[0]->GetBlobDesc().data_type);
        }
    }
    return TNN_OK;
}

REGISTER_ARM_ACC(Unsqueeze, LAYER_UNSQUEEZE);
REGISTER_ARM_PRECISION_FP16(LAYER_UNSQUEEZE)
REGISTER_ARM_LAYOUT(LAYER_UNSQUEEZE, DATA_FORMAT_NCHW)

}  // namespace TNN_NS
