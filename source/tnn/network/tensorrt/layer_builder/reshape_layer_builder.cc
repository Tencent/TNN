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

#include "tnn/network/tensorrt/layer_builder/reshape_layer_builder.h"

namespace TNN_NS {

ReshapeTRTLayerBuilder::ReshapeTRTLayerBuilder(LayerType type) : TensorRTLayerBuilder(type) {
}

ReshapeTRTLayerBuilder::~ReshapeTRTLayerBuilder() {
}

Status ReshapeTRTLayerBuilder::Reshape() {
    Blob* output_blob  = output_blobs_[0];
    Status ret = m_layer->Reshape();
    if (ret != TNN_OK) {
        return ret;
    }
    auto output_dims = output_blob->GetBlobDesc().dims;
    printf("%d %d %d %d\n", output_dims[0], output_dims[1], output_dims[2],
        output_dims[3]);
    return TNN_OK;
}

ILayer* ReshapeTRTLayerBuilder::AddToNetwork(INetworkDefinition* network) {
    auto paramlist = dynamic_cast<ReshapeLayerParam*>(param_);
    if (paramlist->reshape_type != 0) {
        LOGE("Error: Unsupport reshape type(%d)", paramlist->reshape_type);
        return nullptr;
    }
    Blob* output_blob  = output_blobs_[0];
    auto output_dims = output_blob->GetBlobDesc().dims;
    Dims reshape_dims;
    reshape_dims.nbDims = 4;
    reshape_dims.d[0] = -1;
    reshape_dims.d[1] = output_dims[1];
    reshape_dims.d[2] = output_dims[2];
    reshape_dims.d[3] = output_dims[3];
    reshape_dims.type[1] = DimensionType::kCHANNEL;
    reshape_dims.type[2] = reshape_dims.type[3] = DimensionType::kSPATIAL;
    auto input_foreign_tensor = dynamic_cast<ForeignBlob*>(input_blobs_[0])->GetForeignTensor();
    auto output_foreign_tensor = dynamic_cast<ForeignBlob*>(output_blobs_[0])->GetForeignTensor();
    auto input_tensor = std::dynamic_pointer_cast<TensorRTTensor>(input_foreign_tensor)->GetTensor();
    IShuffleLayer* layer = network->addShuffle(*input_tensor);
    if (layer != nullptr) {
        layer->setName(layer_name_.c_str());
        layer->setReshapeDimensions(reshape_dims);
    }
    if (std::dynamic_pointer_cast<TensorRTTensor>(input_foreign_tensor)->IsQuantized()) {
        std::dynamic_pointer_cast<TensorRTTensor>(output_foreign_tensor)->SetQuantized();
    }

    return layer;
}

REGISTER_TENSORRT_LAYER_BUILDER(Reshape, LAYER_RESHAPE);

}  //  namespace TNN_NS
