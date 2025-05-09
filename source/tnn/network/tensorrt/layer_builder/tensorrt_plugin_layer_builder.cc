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

#include <cuda_runtime.h>

#include <sstream>
#include <memory>

#include "tnn/network/tensorrt/layer_builder/tensorrt_plugin_layer_builder.h"
#include "tnn/network/tensorrt/tensorrt_tensor.h"
#include "tnn/network/tensorrt/utils.h"
#include "tnn/utils/dims_utils.h"

namespace TNN_NS {

TensorRTPluginLayerBuilder::TensorRTPluginLayerBuilder(LayerType type) : TensorRTBaseLayerBuilder(type) {
    is_plugin        = true;
    m_maybe_fallback = false;
}

TensorRTPluginLayerBuilder::~TensorRTPluginLayerBuilder() {
}

Status TensorRTPluginLayerBuilder::Init(Context* context, LayerParam* param, LayerResource* resource, std::vector<Blob*>& input_blobs,
        std::vector<Blob*>& output_blobs, AbstractDevice* device, bool enable_const_folder) {
    
    m_layer->SetLayerName(this->GetLayerName());

    Status ret = m_layer->Init(context, param, resource, input_blobs, output_blobs, device, enable_const_folder);
    if (ret != TNN_OK) {
        return ret;
    }
    
    input_blobs_  = m_layer->GetInputBlobs();
    output_blobs_ = m_layer->GetOutputBlobs();

    param_    = param;
    resource_ = resource;
    context_ = context;

    m_format = nvinfer1::TensorFormat::kLINEAR;
    m_type = nvinfer1::DataType::kFLOAT;
    m_has_empty_tensor_input = false;

    return TNN_OK;
}

Status TensorRTPluginLayerBuilder::Forward() {
    return TNN_OK;
}

int TensorRTPluginLayerBuilder::getNbOutputs() const noexcept {
    return output_blobs_.size();
}

DimsExprs TensorRTPluginLayerBuilder::getOutputDimensions(int index, const nvinfer1::DimsExprs* inputs, int nbInputs,
        nvinfer1::IExprBuilder& exprBuilder) noexcept {
    nvinfer1::DimsExprs output(inputs[0]);
    return output;
}

int TensorRTPluginLayerBuilder::initialize() noexcept {
    return 0;
}

void TensorRTPluginLayerBuilder::terminate() noexcept {
}

size_t TensorRTPluginLayerBuilder::getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int nbInputs,
        const nvinfer1::PluginTensorDesc* outputs, int nbOutputs) const noexcept {
    return 0;
}

bool dims_equal(DimsVector dims, nvinfer1::Dims trt_dims) {
    bool same = true;
    same &= (dims.size() == trt_dims.nbDims);
    for(int i=0;i<dims.size();i++) {
        same &= (dims[i] == trt_dims.d[i]);
    }
    return same;
}

int TensorRTPluginLayerBuilder::enqueue(const nvinfer1::PluginTensorDesc* inputDesc,
        const nvinfer1::PluginTensorDesc* outputDesc, const void* const* inputs, void* const* outputs,
        void* workspace, cudaStream_t stream) noexcept {
    bool is_input_zero = false;
    for (int i = 0; i < input_blobs_.size(); i++) {
        Blob* input_blob = input_blobs_[i];
        BlobHandle input_handle;
        input_handle.base = const_cast<void *>(inputs[i]);
        input_handle.bytes_offset = input_blob->GetHandle().bytes_offset;
        input_blob->SetHandle(input_handle);
        DimsVector dims;
        auto foreign_blob = dynamic_cast<ForeignBlob*>(input_blob);
        if (!foreign_blob) return -1;
        for (int j = 0; j < inputDesc[i].dims.nbDims; j++) {
            dims.push_back(inputDesc[i].dims.d[j]);
            // plugin with shape tensor input should be excluded
            if (!m_has_empty_tensor_input) {
                if (inputDesc[i].dims.d[j] == 0) is_input_zero = true;
            }
        }
        input_blob->GetBlobDesc().dims = dims;
        input_blob->GetBlobDesc().data_type = ConvertTRTDataType(inputDesc[i].type);
        input_blob->GetBlobDesc().data_format = ConvertTRTDataFormat(inputDesc[i].format);
    }

    for (int i = 0; i < output_blobs_.size(); i++) {
        Blob* output_blob = output_blobs_[i];
        BlobHandle output_handle;
        output_handle.base = const_cast<void *>(outputs[i]);
        output_handle.bytes_offset = output_blob->GetHandle().bytes_offset;
        output_blob->SetHandle(output_handle);
        DimsVector dims;
        for (int j = 0; j < outputDesc[i].dims.nbDims; j++) {
            dims.push_back(outputDesc[i].dims.d[j]);
        }
        output_blob->GetBlobDesc().dims = dims;
        output_blob->GetBlobDesc().data_type = ConvertTRTDataType(outputDesc[i].type);
        output_blob->GetBlobDesc().data_format = ConvertTRTDataFormat(outputDesc[i].format);
    }

    if (is_input_zero) return 0;

    Status ret = m_layer->Forward();
    if (ret != TNN_OK) return -1;

    return 0;
}

size_t TensorRTPluginLayerBuilder::getSerializationSize() const noexcept {
    return sizeof(m_type) + sizeof(m_format) + sizeof(m_has_empty_tensor_input);
}

void TensorRTPluginLayerBuilder::serialize(void* buffer) const noexcept {
    char* d = reinterpret_cast<char*>(buffer);
    write(d, m_type);
    write(d, m_format);
    write(d, m_has_empty_tensor_input);
}

const char* TensorRTPluginLayerBuilder::getPluginVersion() const noexcept {
    return PLUGIN_VERSION;
}

void TensorRTPluginLayerBuilder::destroy() noexcept {
    delete this;
}

void TensorRTPluginLayerBuilder::setPluginNamespace(const char* libNamespace) noexcept {
    m_plugin_namespace = libNamespace;
}

const char* TensorRTPluginLayerBuilder::getPluginNamespace() const noexcept {
    return m_plugin_namespace.c_str();
}

void TensorRTPluginLayerBuilder::configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs,
        const nvinfer1::DynamicPluginTensorDesc* out, int nbOutputs) noexcept {
    for (int i = 0; i < nbInputs; i++) {
        input_blobs_[i]->GetBlobDesc().data_type = ConvertTRTDataType(in[i].desc.type);
    }

    for (int i = 0; i < nbOutputs; i++) {
        output_blobs_[i]->GetBlobDesc().data_type = ConvertTRTDataType(out[i].desc.type);
    }
}

nvinfer1::IPluginV2DynamicExt* TensorRTPluginLayerBuilder::CreatePlugin() noexcept {
    return this;
}

nvinfer1::IPluginV2DynamicExt* TensorRTPluginLayerBuilder::CreatePlugin(const void* data, size_t length) noexcept {
    const char* d = reinterpret_cast<const char*>(data);
    m_type = read<nvinfer1::DataType>(d);
    m_format = read<TensorFormat>(d);
    m_has_empty_tensor_input = read<bool>(d);
    return this;
}

ILayer* TensorRTPluginLayerBuilder::AddToNetwork(INetworkDefinition* network) noexcept {
    std::vector<ITensor*> tensors = GetInputITensors();
    ILayer* layer = network->addPluginV2(tensors.data(), tensors.size(), *this);
    if (layer != nullptr) {
        layer->setName(layer_name_.c_str());
    }
    return layer;
}

void TensorRTPluginLayerBuilder::ReplaceInputShapeTensor(int index, INetworkDefinition* network) {
    auto foreign_blob = dynamic_cast<ForeignBlob*>(input_blobs_[index]);
    if (foreign_blob->GetReplaceFlag()) {
        m_has_empty_tensor_input = true;
        return;
    }

    auto input_tensors = GetInputITensors();
#if 0
    int rank           = input_tensors[index]->getDimensions().d[0];

    Dims strides{rank};
    std::fill(strides.d, strides.d + strides.nbDims, 0);

    static float dump = 0.f;
    Weights const_weight;
    const_weight.count  = 1;
    const_weight.type   = nvinfer1::DataType::kFLOAT;
    const_weight.values = (void*)&dump;

    nvinfer1::Dims weightDims;
    weightDims.nbDims      = 1;
    weightDims.d[0]        = 1;
    ILayer* constant_layer = network->addConstant(weightDims, const_weight);
    nvinfer1::Dims unsqueezeDims{rank};
    std::fill(unsqueezeDims.d, unsqueezeDims.d + unsqueezeDims.nbDims, 1);
    IShuffleLayer* unsqueeze = network->addShuffle(*constant_layer->getOutput(0));
    unsqueeze->setReshapeDimensions(unsqueezeDims);

    Dims starts;
    starts.nbDims = rank;
    for (int i = 0; i < rank; i++) {
        starts.d[i] = 0;
    }
    ISliceLayer* broadcast_layer = network->addSlice(*unsqueeze->getOutput(0), starts, nvinfer1::Dims{}, strides);
    broadcast_layer->setName((layer_name_ + "_constant_of_shape_slice").c_str());

    if (broadcast_layer != nullptr) {
        broadcast_layer->setInput(2, *input_tensors[index]);
    }

    ILayer* layer = broadcast_layer;
#else
    // https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#comm-shape-tensors-plug-ins
    // the method in above doc has a misstake, empty tensor must have empty value ptr
    // Create an empty-tensor constant with dimensions [1...0].
    int rank = input_tensors[index]->getDimensions().d[0];
    Dims d01;
    d01.nbDims = rank + 1;
    d01.d[rank] = 0;
    std::fill(d01.d, d01.d + rank, 1);
    ITensor *c01 = network->addConstant(d01, {nvinfer1::DataType::kFLOAT, 0x0, 0})->getOutput(0);

    // Create shape tensor that has the value [P,Q...0]
    static int32_t const intZero = 0;
    ITensor* z = network->addConstant({1, {1}}, {nvinfer1::DataType::kINT32, &intZero, 1})->getOutput(0);
    ITensor* concatInputs[] = {input_tensors[index], z};
    IConcatenationLayer* zpq = network->addConcatenation(concatInputs, 2);
    zpq->setAxis(0);

    // Create zero-stride slice with output size [P,Q...0]
    Dims dx;
    dx.nbDims = rank + 1;
    std::fill(dx.d, dx.d + dx.nbDims, 0);
    ISliceLayer* slice = network->addSlice(*c01, dx, dx, dx);
    slice->setInput(2, *zpq->getOutput(0));
    ILayer *layer = slice;

#endif

    auto replace_tensor = std::make_shared<TensorRTTensor>();
    replace_tensor->SetTensor(layer->getOutput(0));

    foreign_blob->SetForeignTensor(replace_tensor, true);
    m_has_empty_tensor_input = true;
}

}  //  namespace TNN_NS

