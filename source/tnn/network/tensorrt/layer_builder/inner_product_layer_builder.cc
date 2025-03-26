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
#include "NvInfer.h"

namespace TNN_NS {

DECLARE_TENSORRT_LAYER_BUILDER(InnerProduct, LAYER_INNER_PRODUCT);

ILayer* InnerProductTRTLayerBuilder::AddToNetwork(INetworkDefinition* network) {
    auto paramlist = dynamic_cast<InnerProductLayerParam*>(param_);
    auto resource = dynamic_cast<InnerProductLayerResource*>(resource_);

    auto input_foreign_tensor = dynamic_cast<ForeignBlob*>(input_blobs_[0])->GetForeignTensor();
    auto output_foreign_tensor = dynamic_cast<ForeignBlob*>(output_blobs_[0])->GetForeignTensor();
    auto input_tensor = std::dynamic_pointer_cast<TensorRTTensor>(input_foreign_tensor)->GetTensor();

    nvinfer1::ITensor* weight_tensor = nullptr;
    bool weight_as_input = (input_blobs_.size() == 2);
    int weight_count = 1;

    Weights kernelWeights;
    Weights biasWeights;

    if (weight_as_input) {
        auto weight_foreign_tensor = dynamic_cast<ForeignBlob*>(input_blobs_[1])->GetForeignTensor();
        weight_tensor = std::dynamic_pointer_cast<TensorRTTensor>(weight_foreign_tensor)->GetTensor();
        auto dims = weight_tensor->getDimensions();
        paramlist->num_output = dims.d[0];
        for (int i = 0; i < dims.nbDims; i++)
            weight_count *= dims.d[i];
    } else {
        kernelWeights = ConvertToWeights(&(resource->weight_handle));
        weight_count = kernelWeights.count;
    }

    if (paramlist->has_bias) {
        biasWeights = ConvertToWeights(&(resource->bias_handle));
    } else {
        biasWeights = ConvertToWeights(nullptr, true, resource->weight_handle.GetDataType());
    }

    if (!weight_as_input) {
        // Create a constant layer for the weights
        Dims weight_dims;
        weight_dims.nbDims = 2;
        weight_dims.d[0] = paramlist->num_output;
        weight_dims.d[1] = weight_count / paramlist->num_output;
        weight_tensor = network->addConstant(weight_dims, kernelWeights)->getOutput(0);
    }

    // Matrix Multiply
    ILayer* matmul_layer = network->addMatrixMultiply(*input_tensor, MatrixOperation::kNONE, *weight_tensor, MatrixOperation::kTRANSPOSE);
    if (matmul_layer == nullptr) {
        return nullptr;
    }

    ILayer* layer = matmul_layer;

    // Add bias if present
    if (paramlist->has_bias) {
        // Adjust bias tensor dimensions to match the output of matmul_layer
        Dims bias_dims;
        bias_dims.nbDims = 2;
        bias_dims.d[0] = 1; // Broadcast across batch size
        bias_dims.d[1] = paramlist->num_output;

        auto bias_tensor = network->addConstant(bias_dims, biasWeights)->getOutput(0);
        layer = network->addElementWise(*matmul_layer->getOutput(0), *bias_tensor, ElementWiseOperation::kSUM);
    }

    if (layer != nullptr) {
        layer->setName(layer_name_.c_str());
        input_tensor = layer->getOutput(0);
    }

    Dims out_dims;
    out_dims.nbDims = paramlist->axis + 1;
    for (int i = 0; i < out_dims.nbDims; i++) {
        out_dims.d[i] = 0;
    }
    IShuffleLayer* out_reshape_layer = network->addShuffle(*input_tensor);
    out_reshape_layer->setReshapeDimensions(out_dims);
    layer = out_reshape_layer;

    return layer;
}

REGISTER_TENSORRT_LAYER_BUILDER(InnerProduct, LAYER_INNER_PRODUCT);

}  //  namespace TNN_NS

