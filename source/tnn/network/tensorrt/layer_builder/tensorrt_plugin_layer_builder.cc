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
    is_plugin = true;
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

    return TNN_OK;
}

Status TensorRTPluginLayerBuilder::Forward() {
    return TNN_OK;
}

int TensorRTPluginLayerBuilder::getNbOutputs() const {
    return output_blobs_.size();
}

DimsExprs TensorRTPluginLayerBuilder::getOutputDimensions(int index, const nvinfer1::DimsExprs* inputs, int nbInputs,
        nvinfer1::IExprBuilder& exprBuilder) {
    nvinfer1::DimsExprs output(inputs[0]);
    return output;
}

int TensorRTPluginLayerBuilder::initialize() {
    return 0;
}

void TensorRTPluginLayerBuilder::terminate() {
}

size_t TensorRTPluginLayerBuilder::getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int nbInputs,
        const nvinfer1::PluginTensorDesc* outputs, int nbOutputs) const {
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
        void* workspace, cudaStream_t stream) {
    bool is_input_zero = false;
    for (int i = 0; i < input_blobs_.size(); i++) {
        Blob* input_blob = input_blobs_[i];
        BlobHandle input_handle;
        input_handle.base = const_cast<void *>(inputs[i]);
        input_handle.bytes_offset = input_blob->GetHandle().bytes_offset;
        input_blob->SetHandle(input_handle);
        DimsVector dims;
        for (int j = 0; j < inputDesc[i].dims.nbDims; j++) {
            dims.push_back(inputDesc[i].dims.d[j]);
            if (inputDesc[i].dims.d[j] == 0) is_input_zero = true;
        }
        input_blob->GetBlobDesc().dims = dims;
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
    }

    if (is_input_zero) return 0;

    Status ret = m_layer->Forward();
    if (ret != TNN_OK) return -1;

    return 0;
}

size_t TensorRTPluginLayerBuilder::getSerializationSize() const {
    return sizeof(m_type) + sizeof(m_format);
}

void TensorRTPluginLayerBuilder::serialize(void* buffer) const {
    char* d = reinterpret_cast<char*>(buffer);
    write(d, m_type);
    write(d, m_format);
}

const char* TensorRTPluginLayerBuilder::getPluginVersion() const {
    return PLUGIN_VERSION;
}

void TensorRTPluginLayerBuilder::destroy() {
    delete this;
}

void TensorRTPluginLayerBuilder::setPluginNamespace(const char* libNamespace) {
    m_plugin_namespace = libNamespace;
}

const char* TensorRTPluginLayerBuilder::getPluginNamespace() const {
    return m_plugin_namespace.c_str();
}

void TensorRTPluginLayerBuilder::configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs,
        const nvinfer1::DynamicPluginTensorDesc* out, int nbOutputs) {
    for (int i = 0; i < nbInputs; i++) {
        input_blobs_[i]->GetBlobDesc().data_type = ConvertTRTDataType(in[i].desc.type);
    }

    for (int i = 0; i < nbOutputs; i++) {
        output_blobs_[i]->GetBlobDesc().data_type = ConvertTRTDataType(out[i].desc.type);
    }
}

nvinfer1::IPluginV2DynamicExt* TensorRTPluginLayerBuilder::CreatePlugin() {
    return this;
}

nvinfer1::IPluginV2DynamicExt* TensorRTPluginLayerBuilder::CreatePlugin(const void* data, size_t length) {
    const char* d = reinterpret_cast<const char*>(data);
    m_type = read<nvinfer1::DataType>(d);
    m_format = read<TensorFormat>(d);
    return this;
}

ILayer* TensorRTPluginLayerBuilder::AddToNetwork(INetworkDefinition* network) {
    std::vector<ITensor*> tensors = GetInputITensors();
    ILayer* layer = network->addPluginV2(tensors.data(), tensors.size(), *this);
    if (layer != nullptr) {
        layer->setName(layer_name_.c_str());
    }
    return layer;
}

}  //  namespace TNN_NS

