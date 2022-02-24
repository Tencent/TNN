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

#include "tnn/network/tensorrt/layer_builder/tensorrt_layer_builder.h"
#include "tnn/network/tensorrt/utils.h"

namespace TNN_NS {

DECLARE_TENSORRT_LAYER_BUILDER(ReshapeTorch, LAYER_RESHAPETORCH);

// In TorchScript model, there may be multiple Reshapes that share the same shape.
// E.g., there are two inputs input_0 and input_1, the input size of input_0 is (1, 3, 16, 16) and
// the input size of input_1 is (1, 3, 32, 32), the shared shape is (1, 3, -1). When inferencing,
// the -1 in shape will be converted to the real value. In LAYER_RESHAPE,
// there is a problem with using input_tensors[1] as shape. Therefore, add LAYER_RESHAPETORCH,
// use the shape stored in param first.
ILayer* ReshapeTorchTRTLayerBuilder::AddToNetwork(INetworkDefinition* network) {
    auto paramlist = dynamic_cast<ReshapeLayerParam*>(param_);

    Blob* output_blob  = output_blobs_[0];
    auto output_dims = output_blob->GetBlobDesc().dims;
    Dims reshape_dims = ConvertToTRTDims(paramlist->shape);
    auto input_tensors = GetInputITensors();
    auto output_tensors = GetOutputITensors();
    IShuffleLayer* layer = network->addShuffle(*input_tensors[0]);
    if (layer != nullptr) {
        layer->setName(layer_name_.c_str());

        if (input_tensors.size() > 1 && paramlist->shape.size() == 0) {
            layer->setInput(1, *input_tensors[1]);
        } else {
            layer->setReshapeDimensions(reshape_dims);
        }
        if (paramlist->reshape_type != 0 && output_dims.size() <= 4) {
            Permutation CHW2HWC;
            const auto& input_dims = input_blobs_[0]->GetBlobDesc().dims;
            CHW2HWC.order[0] = 0;
            CHW2HWC.order[input_dims.size()-1] = 1;
            for(int i=1; i<input_dims.size()-1; ++i) {
                CHW2HWC.order[i] = i+1;
            }
            layer->setFirstTranspose(CHW2HWC);
            Permutation HWC2CHW;
            HWC2CHW.order[0] = 0;
            HWC2CHW.order[1] = output_dims.size()-1;
            for(int i=2; i<output_dims.size(); ++i) {
                HWC2CHW.order[i] = i-1;
            }
            auto permuted_dims = output_dims;
            permuted_dims[output_dims.size()-1] = output_dims[1];
            for(int i=2; i<output_dims.size(); ++i) {
                permuted_dims[i-1] = output_dims[i];
            }
            layer->setReshapeDimensions(ConvertToTRTDynamicDims(permuted_dims));
            layer->setSecondTranspose(HWC2CHW);
        }
    }

    auto input_foreign_tensor = dynamic_cast<ForeignBlob*>(input_blobs_[0])->GetForeignTensor();
    auto output_foreign_tensor = dynamic_cast<ForeignBlob*>(output_blobs_[0])->GetForeignTensor();
    if (std::dynamic_pointer_cast<TensorRTTensor>(input_foreign_tensor)->IsQuantized()) {
        std::dynamic_pointer_cast<TensorRTTensor>(output_foreign_tensor)->SetQuantized();
    }

    return layer;
}

REGISTER_TENSORRT_LAYER_BUILDER(ReshapeTorch, LAYER_RESHAPETORCH);

}  //  namespace TNN_NS
