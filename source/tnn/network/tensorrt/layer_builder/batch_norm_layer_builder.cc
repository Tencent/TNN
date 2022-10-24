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

#include "tnn/utils/dims_utils.h"
#include "tnn/utils/data_type_utils.h"
#include "tnn/network/tensorrt/layer_builder/tensorrt_layer_builder.h"
#include "tnn/network/tensorrt/utils.h"


namespace TNN_NS {

DECLARE_TENSORRT_LAYER_BUILDER(BatchNorm, LAYER_BATCH_NORM);

ILayer* BatchNormTRTLayerBuilder::AddToNetwork(INetworkDefinition* network) {
    auto resource = dynamic_cast<BatchNormLayerResource *>(resource_);

    auto foreign_tensor = dynamic_cast<ForeignBlob*>(input_blobs_[0])->GetForeignTensor();
    auto tensor = std::dynamic_pointer_cast<TensorRTTensor>(foreign_tensor)->GetTensor();

    Weights power { nvinfer1::DataType::kFLOAT, nullptr, 0 };
    Weights shift;
    shift = ConvertToWeights(&(resource->bias_handle));

    Weights scale;
    scale = ConvertToWeights(&(resource->scale_handle));

    int dims_size = tensor->getDimensions().nbDims;
    // unsqueeze 
    ILayer* layer;
    if (dims_size == 2 || dims_size == 3) {
        DimsVector unsqueeze_dims;
        for (int i = 0; i < dims_size; i++) {
            unsqueeze_dims.push_back(0);
        }
        for (int i = 0; i < 4 - dims_size; i++) {
            unsqueeze_dims.push_back(1);
        }
        layer = AddReshapeToNetwork(network, tensor, unsqueeze_dims, (layer_name_ + "squeeze").c_str());
        tensor = layer->getOutput(0);
    }

    //add scale
    if (resource->scale_handle.GetBytesSize() == DataTypeUtils::GetBytesSize(resource->scale_handle.GetDataType())) {
        layer = network->addScaleNd(*tensor, ScaleMode::kUNIFORM, shift, scale, power, 1);
    } else {
        layer = network->addScaleNd(*tensor, ScaleMode::kCHANNEL, shift, scale, power, 1);
    }
    if (layer != NULL) {
        layer->setName(layer_name_.c_str());
        tensor = layer->getOutput(0);
    }

    //squeeze
    if(dims_size == 2 || dims_size == 3) {
        DimsVector squeeze_dims;
        for (int i = 0; i < dims_size; i++) {
            squeeze_dims.push_back(0);
        }
        layer = AddReshapeToNetwork(network, tensor, squeeze_dims, (layer_name_ + "unsqueeze").c_str());
    }

    return layer;
}

REGISTER_TENSORRT_LAYER_BUILDER(BatchNorm, LAYER_BATCH_NORM);
REGISTER_TENSORRT_LAYER_BUILDER(BatchNorm, LAYER_SCALE);

}  //  namespace TNN_NS

