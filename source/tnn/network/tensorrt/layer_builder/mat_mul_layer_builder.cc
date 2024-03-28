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

#include "tnn/network/tensorrt/layer_builder/tensorrt_plugin_layer_builder.h"

#include "tnn/core/macro.h"
#include "tnn/network/tensorrt/utils.h"

namespace TNN_NS {

DECLARE_TENSORRT_PLUGIN_LAYER_BUILDER(MatMul, LAYER_MATMUL);

nvinfer1::Dims unsqueeze_trt_dims(const nvinfer1::Dims &input_dims, int unsqueeze_len) {
    nvinfer1::Dims ret;
    ret.nbDims = std::min(input_dims.nbDims + unsqueeze_len, 5);
    int insert_num = ret.nbDims - input_dims.nbDims;
    int i=0;
    for(;i<insert_num;i++) ret.d[i] = 1;
    for(;i<ret.nbDims;i++) ret.d[i] = input_dims.d[i - insert_num];
    return ret;
}

bool MatMulTRTPluginLayerBuilder::supportsFormatCombination(
        int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs, int nbOutputs) noexcept {
    if (pos == 1 && inOut[pos].dims.d[inOut[pos].dims.nbDims-1]==1) {
        // GEMV + reduce sum case, input 1 should be fp32 to keep precision.
        return inOut[pos].type == nvinfer1::DataType::kFLOAT && inOut[pos].format == nvinfer1::TensorFormat::kLINEAR;
    } else {
        return (inOut[pos].type == nvinfer1::DataType::kFLOAT || inOut[pos].type == nvinfer1::DataType::kHALF) &&
        inOut[0].type == inOut[pos].type && inOut[pos].format == nvinfer1::TensorFormat::kLINEAR;
    }
}

Status MatMulTRTPluginLayerBuilder::Reshape() {
    return TNN_OK;
}

const char* MatMulTRTPluginLayerBuilder::getPluginType() const noexcept {
    return "MatMul";
}

nvinfer1::DataType MatMulTRTPluginLayerBuilder::getOutputDataType(int index, const nvinfer1::DataType* inputTypes,
        int nbInputs) const noexcept {
    return inputTypes[0];
}

ILayer* MatMulTRTPluginLayerBuilder::AddToNetwork(INetworkDefinition* network) noexcept {
    auto paramlist = dynamic_cast<MatMulLayerParam *>(param_);
    auto resource  = dynamic_cast<MatMulLayerResource *>(resource_);
    auto input_tensors = GetInputITensors();
    // TODO: Add Ntew Flag in Context to determine if run with Custom Cuda MatMul Acc
    //if (paramlist->dynamic_matrix_a_dim0) {
    //    return TensorRTPluginLayerBuilder::AddToNetwork(network);
    //}
 
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

    if (input_tensors.size() == 1 && dims_a.nbDims == 3) {
        if (paramlist->extra_config.find("ffn") != paramlist->extra_config.end()) {
            LOGD("Layer %s of Dims <%d,%d,%d>, weigth:<%d,%d,%d> goto plugin\n", 
                    layer_name_.c_str(), dims_a.d[0], dims_a.d[1],dims_a.d[2], dims_b.d[0], dims_b.d[1],dims_b.d[2]);
            return TensorRTPluginLayerBuilder::AddToNetwork(network); 
        }
    }

    MatrixOperation opA = getMatrixOp(matrix_a);
    MatrixOperation opB = getMatrixOp(matrix_b);

    // CASEs when Custom Plugin MatMul OP is prefered:
    // case 1: N=1, TRT GEMV with reduce sum, TRT default batched-gemv is slow,
    //         besides, fp16 GEMV has reduce sum OP, reduce should be calculated under fp32.
    // case 2: Batched-GEMM, without unsqueeze, TRT 7,8 may trigger "Unable to find CUBLAS algo" ERROR,
    //         in some corner cases.
    //         Calling Plugin CUBLAS GEMM may hurt performace, so we put a very strict prerequisite.
    //         Ideally, Batched-GEMM plugin should only be called by Models with Transformer Kernels.
    // Update: Disable custom plugin for case 2 above for Myelin optimization to speed-up network.
    // Update: Disable all plugin cases below, plugin should only be called via extra flag "ffn"
    /*
    if (opA == MatrixOperation::kNONE && opB == MatrixOperation::kNONE &&
            input_tensors.size() == 2 &&
            input_tensors[0]->getDimensions().nbDims == input_tensors[1]->getDimensions().nbDims) {
        bool batch_eq = true;
        bool mnk_unknown = true;
        int in0_batch = 1;
        for (int i=0; i<input_tensors[0]->getDimensions().nbDims-2; i++) {
            // dim==-1 would be treated as dim>1 here.
            batch_eq &= (input_tensors[0]->getDimensions().d[i]==input_tensors[1]->getDimensions().d[i]); 
            in0_batch *= input_tensors[0]->getDimensions().d[i]==-1 ? 2 : input_tensors[0]->getDimensions().d[1];
        }
        mnk_unknown &= input_tensors[0]->getDimensions().nbDims==4;
        if (dims_b.d[dims_b.nbDims - 1] == 1 ||
            (batch_eq && in0_batch>1 && mnk_unknown)) {
            return TensorRTPluginLayerBuilder::AddToNetwork(network); 
        }
    }
    */
    IMatrixMultiplyLayer* layer = network->addMatrixMultiply(*matrix_a, opA, *matrix_b, opB);

    if (layer != nullptr) {
        layer->setName(layer_name_.c_str());
    }

    return layer;
}

DimsExprs MatMulTRTPluginLayerBuilder::getOutputDimensions(int index, const nvinfer1::DimsExprs* inputs,
        int nbInput, nvinfer1::IExprBuilder& exprBuilder) noexcept {
    DimsExprs output(inputs[0]);
    int size = inputs[0].nbDims;

    if (nbInput == 1) {
        auto resource  = dynamic_cast<MatMulLayerResource *>(resource_);
        auto buf = resource->weight;
        DimsVector buf_dims = buf.GetBufferDims();
        output.d[size - 1] = exprBuilder.constant(*buf_dims.rbegin());
    } else {
        output.d[size - 1] = inputs[1].d[size - 1];
        output.d[size - 2] = inputs[0].d[size - 2];
        for (int i = size - 3; i >= 0; i--) {
            output.d[i] = exprBuilder.operation(DimensionOperation::kMAX, *inputs[0].d[i], *inputs[1].d[i]);
        }
    }
    return output;
}

const char* MatMulPluginCreator::getPluginName() const noexcept {
    return "MatMul";
}

REGISTER_TENSORRT_PLUGIN_LAYER_BUILDER(MatMul, LAYER_MATMUL);

}  //  namespace TNN_NS
