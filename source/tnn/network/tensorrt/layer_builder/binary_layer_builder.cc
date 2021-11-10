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
#include <numeric>
#include "tnn/network/tensorrt/layer_builder/binary_layer_builder.h"
#include "tnn/network/tensorrt/utils.h"

namespace TNN_NS {

BinaryTRTLayerBuilder::BinaryTRTLayerBuilder(LayerType ignore) : TensorRTLayerBuilder(ignore) {
}

ILayer* BinaryTRTLayerBuilder::AddToNetwork(INetworkDefinition* network) {
    IElementWiseLayer* layer;
    if (input_blobs_.size() == 2) {
        auto input_foreign_tensor1 = dynamic_cast<ForeignBlob*>(input_blobs_[0])->GetForeignTensor();
        auto input_foreign_tensor2 = dynamic_cast<ForeignBlob*>(input_blobs_[1])->GetForeignTensor();
        auto output_foreign_tensor = dynamic_cast<ForeignBlob*>(output_blobs_[0])->GetForeignTensor();
        auto input_tensor1 = std::dynamic_pointer_cast<TensorRTTensor>(input_foreign_tensor1)->GetTensor();
        auto input_tensor2 = std::dynamic_pointer_cast<TensorRTTensor>(input_foreign_tensor2)->GetTensor();

        if (input_tensor1->getDimensions().nbDims < input_tensor2->getDimensions().nbDims) {
            std::vector<int> axes(input_tensor2->getDimensions().nbDims - input_tensor1->getDimensions().nbDims);
            std::iota(axes.begin(), axes.end(), 0);
            ILayer* unsqueeze_layer = addUnsqueeze(network, *input_tensor1, axes);
            input_tensor1 = unsqueeze_layer->getOutput(0);
        } else if (input_tensor1->getDimensions().nbDims > input_tensor2->getDimensions().nbDims) {
            std::vector<int> axes(input_tensor1->getDimensions().nbDims - input_tensor2->getDimensions().nbDims);
            std::iota(axes.begin(), axes.end(), 0);
            ILayer* unsqueeze_layer = addUnsqueeze(network, *input_tensor2, axes);
            input_tensor2 = unsqueeze_layer->getOutput(0);
        }

        layer = network->addElementWise(*input_tensor1, *input_tensor2, m_op);
        if (layer != nullptr) {
            layer->setName(layer_name_.c_str());
        }
    } else {
        auto paramlist = dynamic_cast<MultidirBroadcastLayerParam*>(param_);
        auto resource = dynamic_cast<EltwiseLayerResource*>(resource_);

        auto foreign_tensor = dynamic_cast<ForeignBlob*>(input_blobs_[0])->GetForeignTensor();
        auto src_a = std::dynamic_pointer_cast<TensorRTTensor>(foreign_tensor)->GetTensor();

        bool unsqueeze = src_a->getDimensions().nbDims == 0;
        if (unsqueeze) {
            ShapeTensor tmp(*src_a, 0);
            src_a = &(convertTo1D(network, tmp).tensor(network));
        }

        auto const_layer = ConvertWeightToConstLayer(network, &(resource->element_handle),
            resource->element_shape, src_a->getDimensions().nbDims);

        if (const_layer == nullptr) {
            LOGE("BinaryTRTLayerBuilder create weights node failed\n");
            return nullptr;
        }

        auto src_b = const_layer->getOutput(0);
        if (src_a->getDimensions().nbDims < src_b->getDimensions().nbDims) {
            std::vector<int> axes(src_b->getDimensions().nbDims - src_a->getDimensions().nbDims);
            std::iota(axes.begin(), axes.end(), 0);
            ILayer* unsqueeze_layer = addUnsqueeze(network, *src_a, axes);
            src_a = unsqueeze_layer->getOutput(0);
        }

        if (paramlist->weight_input_index == 0) {
            std::swap(src_a, src_b);
        }
        layer = network->addElementWise(*src_a, *src_b, m_op);
        if (layer != nullptr) {
            layer->setName(layer_name_.c_str());
        }
        if (unsqueeze) {
            Dims tmp_dims;
            tmp_dims.nbDims = 0;
            IShuffleLayer* shuffle = network->addShuffle(*layer->getOutput(0));
            shuffle->setReshapeDimensions(tmp_dims);
            return shuffle;
        }
    }
    return layer;
}

}  //  namespace TNN_NS
