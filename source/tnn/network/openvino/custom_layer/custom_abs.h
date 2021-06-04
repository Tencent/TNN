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

#include "tnn/network/openvino/custom_layer/custom_implementation.h"
#include "immintrin.h"
#include "time.h"
#include <chrono>

namespace TNN_NS {
    
DECLARE_CUSTOM_OP(Abs);

void CustomAbsOp::validate_and_infer_types()  {
    for (size_t i = 0; i < output_blobs_.size(); i++) {
        auto output_shape = get_input_shape(0);
        auto output_desc = output_blobs_[i]->GetBlobDesc();
        auto dims0 = output_desc.dims;
        for (size_t j = 0; j < dims0.size(); j++) {
            dims0[j] = output_shape[j];
        }
        output_desc.dims = dims0;
        output_blobs_[i]->SetBlobDesc(output_desc);
        set_output_type(i, get_input_element_type(0), ngraph::PartialShape(output_shape));
    }

    for (size_t i = 0; i < input_blobs_.size(); i++) {
        auto input_desc = input_blobs_[i]->GetBlobDesc();
        auto input_dims = input_desc.dims;
        auto input_shape = get_input_shape(i);
        for (size_t j = 0; j < input_dims.size(); j++) {
            input_dims[j] = input_shape[j];
        }
        input_desc.dims = input_dims;
        input_blobs_[i]->SetBlobDesc(input_desc);
    }
}
REGISTER_CUSTOM_OP(Abs);

DECLARE_CUSTOM_IMPLEMENTATION(Abs);
REGISTER_CUSTOM_IMPLEMENTATION(Abs, CustomAbs);

}