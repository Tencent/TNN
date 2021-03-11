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

namespace TNN_NS {

DECLARE_TENSORRT_LAYER_BUILDER(Upsample, LAYER_UPSAMPLE);

ILayer* UpsampleTRTLayerBuilder::AddToNetwork(INetworkDefinition* network) {
    auto paramlist = dynamic_cast<UpsampleLayerParam*>(param_);
    Blob* output_blob  = output_blobs_[0];
    auto output_dims = output_blob->GetBlobDesc().dims;
    auto input_foreign_tensor = dynamic_cast<ForeignBlob*>(input_blobs_[0])->GetForeignTensor();
    auto output_foreign_tensor = dynamic_cast<ForeignBlob*>(output_blobs_[0])->GetForeignTensor();
    auto input_tensor = std::dynamic_pointer_cast<TensorRTTensor>(input_foreign_tensor)->GetTensor();
    IResizeLayer* layer = network->addResize(*input_tensor);
    if (layer != nullptr) {
        layer->setName(layer_name_.c_str());
        if (input_blobs_.size() == 1) {
            nvinfer1::Dims4 dims(output_dims[0], output_dims[1], output_dims[2], output_dims[3]);
            layer->setOutputDimensions(dims);
        } else {
            auto input_foreign_tensor2 = dynamic_cast<ForeignBlob*>(input_blobs_[input_blobs_.size()-1])->GetForeignTensor();
            auto input_tensor2 = std::dynamic_pointer_cast<TensorRTTensor>(input_foreign_tensor2)->GetTensor();
            layer->setInput(1, *(network->addShape(*input_tensor2)->getOutput(0)));
        }
        layer->setResizeMode(paramlist->mode == 1 ? ResizeMode::kNEAREST : ResizeMode::kLINEAR);
        layer->setAlignCorners(paramlist->align_corners);
    }

    return layer;
}

REGISTER_TENSORRT_LAYER_BUILDER(Upsample, LAYER_UPSAMPLE);

}  //  namespace TNN_NS

