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
    ILayer* layer;
    nvinfer1::DataType out_dtype = ConvertToTRTDataType(output_blobs_[0]->GetBlobDesc().data_type); 
    
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

        // Get Input, Output DataType
        // Output DataType comes from TNN InferType.
        nvinfer1::DataType in1_dtype = input_tensor1->getType();
        nvinfer1::DataType in2_dtype = input_tensor2->getType();

        // DataType Cast Before Elemwise
        if (m_op==ElementWiseOperation::kAND || m_op==ElementWiseOperation::kOR ||
            m_op==ElementWiseOperation::kXOR) {
            // kAND, kOR, kXOR requires input type == kBOOL
            if (in1_dtype!=nvinfer1::DataType::kBOOL) {
                ILayer* cast_layer = network->addIdentity(*input_tensor1);
                cast_layer->setName((layer_name_+"_a_2bool").c_str());
                cast_layer->setOutputType(0, nvinfer1::DataType::kBOOL);
                input_tensor1 = cast_layer->getOutput(0);
            }
            if (in2_dtype!=nvinfer1::DataType::kBOOL) {
                ILayer* cast_layer = network->addIdentity(*input_tensor2);
                cast_layer->setName((layer_name_+"_b_2bool").c_str());
                cast_layer->setOutputType(0, nvinfer1::DataType::kBOOL);
                input_tensor2 = cast_layer->getOutput(0);
            }
        } else {
            if (in1_dtype==nvinfer1::DataType::kINT32 && 
                (in2_dtype==nvinfer1::DataType::kFLOAT || in2_dtype==nvinfer1::DataType::kHALF)) {
                ILayer* cast_layer = network->addIdentity(*input_tensor1);
                cast_layer->setName((layer_name_+"_a_int2fp").c_str());
                cast_layer->setOutputType(0, in2_dtype);
                input_tensor1 = cast_layer->getOutput(0);
            } else if ((in1_dtype==nvinfer1::DataType::kFLOAT || in1_dtype==nvinfer1::DataType::kHALF) && 
                in2_dtype==nvinfer1::DataType::kINT32) {
                ILayer* cast_layer = network->addIdentity(*input_tensor2);
                cast_layer->setName((layer_name_+"_b_int2fp").c_str());
                cast_layer->setOutputType(0, in1_dtype);
                input_tensor2 = cast_layer->getOutput(0);
            }
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

        // DataType Cast
        //DataType src_a_dtype = input_blobs_[0]->GetBlobDesc().data_type;
        //DataType src_b_dtype = resource->element_handle.GetDataType();
        // Get Input, Output DataType
        // Output DataType comes from TNN InferType.
        nvinfer1::DataType src_a_dtype = src_a->getType();
        nvinfer1::DataType src_b_dtype = src_b->getType();
        if (m_op==ElementWiseOperation::kAND || m_op==ElementWiseOperation::kOR ||
            m_op==ElementWiseOperation::kXOR) {
            // kAND, kOR, kXOR requires input type == kBOOL
            if (src_a_dtype!=nvinfer1::DataType::kBOOL) {
                ILayer* cast_layer = network->addIdentity(*src_a);
                cast_layer->setName((layer_name_+"_a_2bool").c_str());
                cast_layer->setOutputType(0, nvinfer1::DataType::kBOOL);
                src_a = cast_layer->getOutput(0);
            }
            if (src_b_dtype!=nvinfer1::DataType::kBOOL) {
                ILayer* cast_layer = network->addIdentity(*src_b);
                cast_layer->setName((layer_name_+"_b_2bool").c_str());
                cast_layer->setOutputType(0, nvinfer1::DataType::kBOOL);
                src_b = cast_layer->getOutput(0);
            }
        } else {
            if (src_a_dtype==nvinfer1::DataType::kINT32 && 
                (src_b_dtype==nvinfer1::DataType::kFLOAT || src_b_dtype==nvinfer1::DataType::kHALF)) {
                ILayer* cast_layer = network->addIdentity(*src_a);
                cast_layer->setName((layer_name_+"_in1_int2fp").c_str());
                cast_layer->setOutputType(0, src_b_dtype);
                src_a = cast_layer->getOutput(0);
            } else if ((src_a_dtype==nvinfer1::DataType::kFLOAT || src_a_dtype==nvinfer1::DataType::kHALF) && 
                src_b_dtype==nvinfer1::DataType::kINT32) {
                ILayer* cast_layer = network->addIdentity(*src_b);
                cast_layer->setName((layer_name_+"_in2_int2fp").c_str());
                cast_layer->setOutputType(0, src_a_dtype);
                src_b = cast_layer->getOutput(0);
            }
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
            layer = shuffle;
        }
    }
    
    // DataType Cast After Elemwise
    // kAND, kOR, kXOR, kEQUAL, kGREATER, kLESS default out type == kBOOL
    if ((m_op==ElementWiseOperation::kAND || m_op==ElementWiseOperation::kOR ||
        m_op==ElementWiseOperation::kXOR || m_op==ElementWiseOperation::kEQUAL ||
        m_op==ElementWiseOperation::kLESS || m_op==ElementWiseOperation::kGREATER) &&
        out_dtype != nvinfer1::DataType::kINT8) {
        // If Output Type set by TNN::inferOutputType() is DATA_TYPE_INT8, thus, nvinfer1::DataType:kInt8
        // INT8 here means BOOL, not really quantization INT8. Output DataType cast is not needed.
        auto output_tensor = layer->getOutput(0);
        layer = network->addIdentity(*output_tensor);
        layer->setName((layer_name_+"_out_cast").c_str());
        layer->setOutputType(0, out_dtype);
    }

    return layer;
}

}  //  namespace TNN_NS
