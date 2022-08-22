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
#include "tnn/utils/data_format_converter.h"
#include "tnn/utils/data_type_utils.h"
#include "tnn/utils/dims_utils.h"

namespace TNN_NS {

DECLARE_X86_ACC(Reshape, LAYER_RESHAPE);

Status X86ReshapeLayerAcc::DoForward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto &input  = inputs[0];
    auto &output = outputs[0];
    auto param   = (ReshapeLayerParam *)param_;
    ASSERT(param != nullptr);
    if (param->reshape_type == 0) {

        void * in_ptr = handle_ptr<void*>(input->GetHandle());
        void * out_ptr = handle_ptr<void*>(output->GetHandle());
        if (in_ptr != out_ptr){
            auto dims_input    = input->GetBlobDesc().dims;
            int data_byte_size = DataTypeUtils::GetBytesSize(output->GetBlobDesc().data_type);
            auto size_in_bytes = DimsVectorUtils::Count(dims_input) * data_byte_size;
            memcpy(out_ptr, in_ptr, size_in_bytes);
        }
    } else if (param->reshape_type == 1) {
        // tensorflow reshape
        DataFormatConverter::ConvertFromNCHWToNHWC<float>(input, output);
        DataFormatConverter::ConvertFromNHWCToNCHW<float>(output, nullptr);
    } else {
        LOGE("Error: Unsupport reshape type(%d)", param->reshape_type);
        return Status(TNNERR_MODEL_ERR, "Error: X86ReshapeLayerAcc failed!\n");
    }
    return TNN_OK;
}

REGISTER_X86_ACC(Reshape, LAYER_RESHAPE);

}  // namespace TNN_NS
