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

#include "tnn/network/tensorrt/tensorrt_network.h"
#include "tnn/network/tensorrt/layer_builder/tensorrt_layer_builder.h"

namespace TNN_NS {

DECLARE_TENSORRT_LAYER_BUILDER(Concat, LAYER_CONCAT);

ILayer* ConcatTRTLayerBuilder::AddToNetwork(INetworkDefinition* network) {
    auto paramlist = dynamic_cast<ConcatLayerParam*>(param_);
    auto input_foreign_tensor = dynamic_cast<ForeignBlob*>(input_blobs_[0])->GetForeignTensor();
    auto output_foreign_tensor = dynamic_cast<ForeignBlob*>(output_blobs_[0])->GetForeignTensor();
    size_t nbInputs = input_blobs_.size();
    ITensor ** input_tensors = new ITensor*[nbInputs];
    for (int i = 0; i < nbInputs; i++) {
        auto foreign_tensor = dynamic_cast<ForeignBlob*>(input_blobs_[i])->GetForeignTensor();
        auto tensor = std::dynamic_pointer_cast<TensorRTTensor>(foreign_tensor)->GetTensor();
        input_tensors[i] = tensor;
    }

    m_network->m_concat_blob_names.insert(output_blobs_[0]->GetBlobDesc().name);

    ILayer* last_layer;
    IConcatenationLayer* layer = network->addConcatenation(input_tensors, nbInputs);
    if (layer != nullptr) {
        int axis = paramlist->axis;
        if (axis < 0 && input_tensors[0]->getDimensions().nbDims > 0) {
            axis += input_tensors[0]->getDimensions().nbDims;
        } 
        layer->setName(layer_name_.c_str());
        layer->setAxis(axis);
        last_layer = layer;
    }
    delete [] input_tensors;

    return last_layer;
}

REGISTER_TENSORRT_LAYER_BUILDER(Concat, LAYER_CONCAT);

}  //  namespace TNN_NS

