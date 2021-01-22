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

#include "base_layer.h"
#include "tnn/utils/dims_vector_utils.h"

namespace TNN_NS {
DECLARE_LAYER(LSTMONNX, LAYER_LSTMONNX);

Status LSTMONNXLayer::InferOutputDataType() {
    return BaseLayer::InferOutputDataType();
}

Status LSTMONNXLayer::InferOutputShape(bool ignore_error) {
    BaseLayer::InferOutputShape(ignore_error);
    
    auto layer_param = dynamic_cast<LSTMONNXLayerParam*>(param_);
    CHECK_PARAM_NULL(layer_param);
    int num_directions = layer_param->direction >=2 ? 2 : 1;
    
    auto input_dims = input_blobs_[0]->GetBlobDesc().dims;
    auto sequence_len = input_dims[0]; // length of sequence
    auto batch = input_dims[1];  // batch_size
    auto input_size = DimsVectorUtils::Count(input_dims, 2); // input dimension
    auto output_size = layer_param->hidden_size;
    
    //[seq_length, batch_size, num_directions*hidden_size], shape after transpose and reshape
    DimsVector output_dims = {sequence_len, batch, num_directions*output_size};
    output_blobs_[0]->GetBlobDesc().dims = output_dims;
    if (output_blobs_.size() >= 3) {
        //[num_directions, batch_size, output_size]
        output_dims = {num_directions, batch, output_size};
        output_blobs_[1]->GetBlobDesc().dims = output_dims;
        output_blobs_[2]->GetBlobDesc().dims = output_dims;
    }
    return TNN_OK;
}

REGISTER_LAYER(LSTMONNX, LAYER_LSTMONNX);

}  // namespace TNN_NS
