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
    
DECLARE_CUSTOM_OP(LayerNorm);
void CustomLayerNormOp::validate_and_infer_types()  {
    for (size_t i = 0; i < output_blobs_.size(); i++) {
        auto dims0 = output_blobs_[i]->GetBlobDesc().dims;
        ngraph::Shape output_shape(dims0.size());
        for (size_t j = 0; j < dims0.size(); j++) {
            output_shape[j] = dims0[j];
        }
        set_output_type(i, get_input_element_type(0), ngraph::PartialShape(output_shape));
    }
}
REGISTER_CUSTOM_OP(LayerNorm);

DECLARE_CUSTOM_IMPLEMENTATION(LayerNorm);
REGISTER_CUSTOM_IMPLEMENTATION(LayerNorm, CustomLayerNorm);

}