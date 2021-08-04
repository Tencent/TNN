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

// author: sanerzheng@tencent.com

#include "tnn/train/grad/grad_manager.h"
#include "tnn/train/operations/op_builder.h"

namespace TNN_NS {
namespace train {
DECLARE_LAYER_GRAD(BinaryCrossEntropy, LAYER_BINARY_CROSSENTROPY);

Status BinaryCrossEntropyLayerGrad::OnGrad(const BaseLayer* layer, TrainContext& context){
    auto inputs = layer->input_blobs_;
    auto outputs = layer->output_blobs_;
    if(inputs.size() != 2 || outputs.size() != 1) {
        return Status(TNN_TRAIN_ERROR, "input size or output size not match in BinaryCrossEntropyLayerGrad");
    }
    auto input0_dims = inputs[0]->GetBlobDesc().dims;
    auto input1_dims = inputs[1]->GetBlobDesc().dims;
    auto output_dims = outputs[0]->GetBlobDesc().dims;
    if(!DimsVectorUtils::Equal(input1_dims, input0_dims) || !DimsVectorUtils::Equal(input1_dims, output_dims)) {
        return Status(TNN_TRAIN_ERROR, "input dims and output dims not match in BinaryCrossEntropyLayerGrad");
    }
    auto input0_data_type = inputs[0]->GetBlobDesc().data_type;
    auto input1_data_type = inputs[1]->GetBlobDesc().data_type;
    auto output_data_type = outputs[0]->GetBlobDesc().data_type;
    ParamWrapper pw0(inputs[0]);
    ParamWrapper pw1(inputs[1]);
    //x0 is logits, x1 is true labels
    //dy/dx0 = -(x1/x0 + (1-x1)/(1-x0))
    //dy/dx1 = log(1-x0) - log(x0)
    ParamWrapper grad0 = _Neg(_Add(_Div(pw1, pw0, context), _Div(_Sub(_Const(ParamWrapper(1.0f), input1_dims), pw1, context),
                                            _Sub(_Const(ParamWrapper(1.0f), input0_dims), pw0, context)
                                            ,context), context), context);
    ParamWrapper grad1 = _Sub(_Log(_Sub(_Const(ParamWrapper(1.0f), input0_dims), pw0, context), context),
                              _Log(pw0, context), context);
    if(!grad0.IsRawbufferSharedPtr() || !grad1.IsRawbufferSharedPtr()) {
        return Status(TNN_TRAIN_ERROR, "Calcute BinaryCrossEntropyLayerGrad error");
    }
    context.backward_grads_resource[inputs[0]] = grad0.GetRawbufferSharedPtr();
    context.backward_grads_resource[inputs[1]] = grad1.GetRawbufferSharedPtr();
    return Status(TNN_OK); 
}
} //namespace train         
} //namspace TNN_NS