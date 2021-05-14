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

#include "tnn/core/macro.h"
#include "tnn/network/tensorrt/utils.h"

namespace TNN_NS {

DECLARE_TENSORRT_LAYER_BUILDER(MatMul, LAYER_MATMUL);

nvinfer1::Dims unsqueeze_trt_dims(const nvinfer1::Dims &input_dims, int unsqueeze_len)  {
    nvinfer1::Dims ret;
    ret.nbDims = std::min(input_dims.nbDims + unsqueeze_len, 5);
    int insert_num = ret.nbDims - input_dims.nbDims;
    int i=0;
    for(;i<insert_num;i++) ret.d[i] = 1;
    for(;i<ret.nbDims;i++) ret.d[i] = input_dims.d[i - insert_num];
    return ret;
}

ILayer* MatMulTRTLayerBuilder::AddToNetwork(INetworkDefinition* network) {
    auto paramlist = dynamic_cast<MatMulLayerParam *>(param_);
    auto resource  = dynamic_cast<MatMulLayerResource *>(resource_);

    auto input_foreign_tensor = dynamic_cast<ForeignBlob*>(input_blobs_[0])->GetForeignTensor();
    auto output_foreign_tensor = dynamic_cast<ForeignBlob*>(output_blobs_[0])->GetForeignTensor();
    auto input_tensor = std::dynamic_pointer_cast<TensorRTTensor>(input_foreign_tensor)->GetTensor();
    int batch_size = input_blobs_[0]->GetBlobDesc().dims[0];

    auto input_tensors = GetInputITensors();

    ITensor * matrix_a = nullptr;
    ITensor * matrix_b = nullptr;
    
    if (input_tensors.size() == 2) {
        matrix_a = input_tensors[0];
        matrix_b = input_tensors[1];
    } else {
        auto buf = resource->weight;
        DimsVector buf_dims = buf.GetBufferDims();
        int nbDims = input_tensors[0]->getDimensions().nbDims;
        int diff = nbDims - buf_dims.size();
        for(int i = 0; i < diff; ++i) {
            buf_dims.insert(buf_dims.begin(), 1);
        }
        auto const_layer = ConvertWeightToConstLayer(network, &buf, buf_dims);
        matrix_a    = paramlist->weight_position == 0 ? const_layer->getOutput(0) : input_tensors[0];
        matrix_b    = paramlist->weight_position == 1 ? const_layer->getOutput(0) : input_tensors[0];
    }

    if (matrix_a == nullptr || matrix_b == nullptr) {
        LOGE("MatMulTRTLayerBuilder got null inputs");
        return nullptr;
    }

    // TRT Restrict that : dimsA.nbDims == dimsB.nbDims , when nbDims >= 2
    auto dims_a = matrix_a->getDimensions();
    auto dims_b = matrix_b->getDimensions();
    int nbDimsDiff = std::abs(dims_a.nbDims - dims_b.nbDims);
    if (dims_a.nbDims > dims_b.nbDims)
    {
        nvinfer1::Dims new_dims = unsqueeze_trt_dims(dims_b, nbDimsDiff);
        nvinfer1::IShuffleLayer* unsqueeze = network->addShuffle(*matrix_b);
        unsqueeze->setReshapeDimensions(new_dims);
        unsqueeze->setName((GetLayerName()+"_unqueeze_b").c_str());
        matrix_b = unsqueeze->getOutput(0);
    }

    if (dims_b.nbDims > dims_a.nbDims)
    {
        nvinfer1::Dims new_dims = unsqueeze_trt_dims(dims_a, nbDimsDiff);
        nvinfer1::IShuffleLayer* unsqueeze = network->addShuffle(*matrix_a);
        unsqueeze->setReshapeDimensions(new_dims);
        unsqueeze->setName((GetLayerName()+"_unqueeze_a").c_str());
        matrix_a = unsqueeze->getOutput(0);
    }

    const auto getMatrixOp = [](const nvinfer1::ITensor* input) {
        return (input->getDimensions().nbDims == 1) ? MatrixOperation::kVECTOR
                                                   : MatrixOperation::kNONE;
    };

    MatrixOperation opA = getMatrixOp(matrix_a);
    MatrixOperation opB = getMatrixOp(matrix_b);

    IMatrixMultiplyLayer* layer = network->addMatrixMultiply(*matrix_a, opA, *matrix_b, opB);

    if (layer != nullptr) {
        layer->setName(layer_name_.c_str());
    }

    return layer;
}

REGISTER_TENSORRT_LAYER_BUILDER(MatMul, LAYER_MATMUL);

}  //  namespace TNN_NS

