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

#ifndef TNN_SOURCE_TNN_NETWORK_TENSORR_LAYER_BUILDER_TENSORRT_PLUGIN_LAYER_BUILDER_H_
#define TNN_SOURCE_TNN_NETWORK_TENSORR_LAYER_BUILDER_TENSORRT_PLUGIN_LAYER_BUILDER_H_

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "NvInfer.h"
#include "NvUtils.h"
#include "NvInferPlugin.h"

#include "tnn/layer/base_layer.h"
#include "tnn/core/abstract_device.h"
#include "tnn/core/blob.h"
#include "tnn/core/context.h"
#include "tnn/core/layer_type.h"
#include "tnn/core/status.h"
#include "tnn/interpreter/layer_param.h"
#include "tnn/interpreter/layer_resource.h"
#include "tnn/extern_wrapper/base_layer_builder.h"

using namespace nvinfer1;
using namespace plugin;

namespace TNN_NS {

// @brief TensorRTPluginLayer Builder, defines the tensorRT plugin layer builder interface
class TensorRTPluginLayerBuilder : public BaseLayerBuilder {
public:
    explicit TensorRTPluginLayerBuilder(LayerType type);

    // @brief virtual destructor
    virtual ~TensorRTPluginLayerBuilder();

    // @brief virtual layer init
    virtual Status Init(Context* context, LayerParam* param, LayerResource* resource, std::vector<Blob*>& inputs,
                std::vector<Blob*>& outputs, AbstractDevice* device);

    // @brief virtual Reshape recalculate the output tensor dims
    virtual Status Reshape();

    // @brief virtual layer infer
    virtual Status Forward();

    virtual int getNbOutputs() const;

    virtual Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims);

    virtual bool supportFormat(nvinfer1::DataType type, PluginFormat format) const;

    virtual void configureWithFormat(const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs,
            nvinfer1::DataType type, PluginFormat format, int maxBatchSize);

    virtual int initialize();

    virtual void terminate();

    virtual size_t getWorkspaceSize(int maxBatchSize) const;

    virtual int enqueue(int batchSize, const void* const* inputs, void** outputs, void* workspace, cudaStream_t stream) = 0;

    virtual size_t getSerializationSize();

    virtual void serialize(void* buffer);

protected:
    // @brief Build the foreign network
    virtual Status Build() = 0;

    nvinfer1::DataType m_type;
    PluginFormat m_format;

private:
    template<typename T>
    void write(char*& buffer, const T& val) {
        *reinterpret_cast<T*>(buffer) = val;
        buffer += sizeof(T);
    }

    template<typename T>
    void read(const char*& buffer, T& val) {
        val = *reinterpret_cast<const T*>(buffer);
        buffer += sizeof(T);
    }
};

}  //  TNN_NS

#endif  //  TNN_SOURCE_TNN_NETWORK_TENSORR_LAYER_BUILDER_TENSORRT_PLUGIN_LAYER_BUILDER_H_