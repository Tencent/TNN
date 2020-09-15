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

#include <memory>

#include "tnn/network/tensorrt/layer_builder/tensorrt_plugin_layer_builder.h"
#include "tnn/network/tensorrt/tensorrt_tensor.h"

#include "tnn/utils/dims_vector_utils.h"

namespace TNN_NS {

TensorRTPluginLayerBuilder::TensorRTPluginLayerBuilder(LayerType type) : TensorRTBaseLayerBuilder(type) {
    is_plugin = true;
}

TensorRTPluginLayerBuilder::~TensorRTPluginLayerBuilder() {
}

Status TensorRTPluginLayerBuilder::Init(Context* context, LayerParam* param, LayerResource* resource, std::vector<Blob*>& input_blobs,
        std::vector<Blob*>& output_blobs, AbstractDevice* device) {
    auto tmp_device = GetDevice(DEVICE_NAIVE);
    auto tmp_context = tmp_device->CreateContext(0);
    Status ret = m_layer->Init(tmp_context, param, resource, input_blobs, output_blobs, tmp_device);
    if (ret != TNN_OK) {
        return ret;
    }

    input_blobs_  = m_layer->GetInputBlobs();
    output_blobs_ = m_layer->GetOutputBlobs();

    param_    = param;
    resource_ = resource;

    layer_acc_ = device->CreateLayerAcc(type_);
    if (layer_acc_ != NULL) {
        return layer_acc_->Init(context, param, resource, input_blobs_, output_blobs_);
    } else {
        LOGE("layer acc of type(%d) is nil\n", type_);
        return Status(TNNERR_LAYER_ERR, "layer acc is nil");
    }

    return TNN_OK;
}

IPluginExt* TensorRTPluginLayerBuilder::CreatePlugin() {
    return this;
}

IPluginExt* TensorRTPluginLayerBuilder::CreatePlugin(const void* data, size_t length) {
    const char* d = reinterpret_cast<const char*>(data);
    m_type = read<nvinfer1::DataType>(d);
    m_format = read<PluginFormat>(d);
    return this;
}

Status TensorRTPluginLayerBuilder::Reshape() {
    return TNN_OK;
}

Status TensorRTPluginLayerBuilder::Forward() {
    return TNN_OK;
}

int TensorRTPluginLayerBuilder::getNbOutputs() const {
    return output_blobs_.size();
}

Dims TensorRTPluginLayerBuilder::getOutputDimensions(int index, const Dims* inputs, int nbInputDims) {
    auto shape = output_blobs_[index]->GetBlobDesc();
    return DimsCHW(shape.dims[1], shape.dims[2], shape.dims[3]);
}

void TensorRTPluginLayerBuilder::configureWithFormat(const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs,
            nvinfer1::DataType type, PluginFormat format, int maxBatchSize) {
    m_type = type;
    m_format = format;
}

int TensorRTPluginLayerBuilder::initialize() {
    return 0;
}

void TensorRTPluginLayerBuilder::terminate() {
}

size_t TensorRTPluginLayerBuilder::getWorkspaceSize(int maxBatchSize) const {
    return 0;
}

int TensorRTPluginLayerBuilder::enqueue(int batchSize, const void* const* inputs, void** outputs,
        void* workspace, cudaStream_t stream) {
    for (int i = 0; i < input_blobs_.size(); i++) {
        Blob* input_blob = input_blobs_[i];
        BlobHandle input_handle;
        input_handle.base = const_cast<void *>(inputs[i]);
        input_handle.bytes_offset = input_blob->GetHandle().bytes_offset;
        input_blob->SetHandle(input_handle);
    }

    for (int i = 0; i < output_blobs_.size(); i++) {
        Blob* output_blob = output_blobs_[i];
        BlobHandle output_handle;
        output_handle.base = const_cast<void *>(outputs[i]);
        output_handle.bytes_offset = output_blob->GetHandle().bytes_offset;
        output_blob->SetHandle(output_handle);
    }

    Status ret = BaseLayer::Forward();
    if (ret != TNN_OK) {
        return -1;
    }

    return 0;
}

size_t TensorRTPluginLayerBuilder::getSerializationSize() {
    return sizeof(m_type) + sizeof(m_format);
}

void TensorRTPluginLayerBuilder::serialize(void* buffer) {
    char* d = reinterpret_cast<char*>(buffer);
    write(d, m_type);
    write(d, m_format);
}

ILayer* TensorRTPluginLayerBuilder::AddToNetwork(INetworkDefinition* network) {
    std::vector<ITensor*> tensors;
    int size = input_blobs_.size();
    for (int i = 0; i < size; ++i) {
        auto foreign_tensor = dynamic_cast<ForeignBlob*>(input_blobs_[i])->GetForeignTensor();
        auto tensor = std::dynamic_pointer_cast<TensorRTTensor>(foreign_tensor)->GetTensor();
        tensors.push_back(tensor);
    }
    ILayer* layer = network->addPlugin(tensors.data(), size, *CreatePlugin());
    if (layer != nullptr) {
        layer->setName(layer_name_.c_str());
    }
    return layer;
}

}  //  namespace TNN_NS