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

#include "tnn/network/tensorrt/tensorrt_network.h"
#include "tnn/network/tensorrt/layer_builder/tensorrt_base_layer_builder.h"

namespace TNN_NS {

constexpr const char* PLUGIN_VERSION{"1"};

// @brief TensorRTPluginLayer Builder, defines the tensorRT plugin layer builder interface
class TensorRTPluginLayerBuilder : public TensorRTBaseLayerBuilder, public nvinfer1::IPluginV2DynamicExt {
public:
    explicit TensorRTPluginLayerBuilder(LayerType type);

    // @brief virtual destructor
    virtual ~TensorRTPluginLayerBuilder();

    // @brief virtual layer init
    virtual Status Init(Context* context, LayerParam* param, LayerResource* resource, std::vector<Blob*>& inputs,
                std::vector<Blob*>& outputs, AbstractDevice* device);

    // @brief virtual layer infer
    virtual Status Forward();

    // @brief add layer to tensorRT network
    virtual ILayer* AddToNetwork(INetworkDefinition* network);

    virtual int getNbOutputs() const;

    virtual DimsExprs getOutputDimensions(int index, const nvinfer1::DimsExprs* inputs, int nbInputs,
        nvinfer1::IExprBuilder& exprBuilder);

    virtual int initialize();

    virtual void terminate();

    virtual size_t getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int nbInputs,
        const nvinfer1::PluginTensorDesc* outputs, int nbOutputs) const;

    virtual int enqueue(const nvinfer1::PluginTensorDesc* inputDesc, const nvinfer1::PluginTensorDesc* outputDesc,
        const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream);

    virtual size_t getSerializationSize() const;

    virtual void serialize(void* buffer) const;

    virtual const char* getPluginVersion() const;

    virtual void destroy();

    virtual void setPluginNamespace(const char* pluginNamespace);

    virtual const char* getPluginNamespace() const;

    virtual void configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs,
        const nvinfer1::DynamicPluginTensorDesc* out, int nbOutputs);

    nvinfer1::IPluginV2DynamicExt* CreatePlugin();

    nvinfer1::IPluginV2DynamicExt* CreatePlugin(const void* data, size_t length);

protected:
    std::string m_plugin_namespace;
    nvinfer1::DataType m_type;
    TensorFormat m_format;
    Context* context_;

private:
    template<typename T>
    void write(char*& buffer, const T& val) const {
        *reinterpret_cast<T*>(buffer) = val;
        buffer += sizeof(T);
    }

    template<typename T>
    T read(const char*& buffer) const {
        T val = *reinterpret_cast<const T*>(buffer);
        buffer += sizeof(T);
    }
};

//@brief TRTPluginTypeLayerBuilderRegister register TypeLayerBuilderCreator
template <typename T>
class TRTPluginTypeLayerBuilderRegister {
public:
    explicit TRTPluginTypeLayerBuilderRegister(LayerType type) {
        GetTRTPluginLayerBuilderCreatorMap()[type] = shared_ptr<T>(new T(type));
    }
};

#define DECLARE_TENSORRT_PLUGIN_LAYER_BUILDER(type_string, layer_type)                                             \
    class type_string##TRTPluginLayerBuilder : public TensorRTPluginLayerBuilder {                                 \
    public:                                                                                                        \
        type_string##TRTPluginLayerBuilder(LayerType layer_type) : TensorRTPluginLayerBuilder(layer_type) {}       \
        virtual ~type_string##TRTPluginLayerBuilder() {}                                                           \
        virtual bool supportsFormatCombination(int pos, const nvinfer1::PluginTensorDesc* inOut,                   \
            int nbInputs, int nbOutputs);                                                                          \
        virtual DimsExprs getOutputDimensions(int index, const nvinfer1::DimsExprs* inputs, int nbInputs,          \
            nvinfer1::IExprBuilder& exprBuilder);                                                                  \
        virtual const char* getPluginType() const;                                                                 \
        virtual nvinfer1::IPluginV2DynamicExt* clone() const {                                                     \
            auto* plugin = new type_string##TRTPluginLayerBuilder(*this);                                          \
            plugin->setPluginNamespace(this->m_plugin_namespace.c_str());                                          \
            return plugin;                                                                                         \
        }                                                                                                          \
        virtual nvinfer1::DataType getOutputDataType(int index, const nvinfer1::DataType* inputTypes,              \
            int nbInputs) const;                                                                                   \
        virtual ILayer* AddToNetwork(INetworkDefinition* network);                                                 \
    };                                                                                                             \
    class type_string##PluginCreator : public nvinfer1::IPluginCreator {                                           \
    public:                                                                                                        \
        type_string##PluginCreator() {                                                                             \
            m_fc.nbFields = 0;                                                                                     \
            m_fc.fields = nullptr;                                                                                 \
        }                                                                                                          \
        virtual const char* getPluginName() const;                                                                 \
        virtual const char* getPluginVersion() const { return PLUGIN_VERSION; }                                    \
        virtual const nvinfer1::PluginFieldCollection* getFieldNames() { return &m_fc; }                           \
        virtual const char* getPluginNamespace() const { return m_plugin_namespace.c_str(); }                      \
        virtual void setPluginNamespace(const char* libNamespace) { m_plugin_namespace = libNamespace; }           \
        virtual nvinfer1::IPluginV2DynamicExt* createPlugin(const char* name,                                      \
                const nvinfer1::PluginFieldCollection* fc) {                                                       \
            std::unordered_map<std::string, TensorRTPluginLayerBuilder*> layer_map =                               \
                TensorRTNetwork_::GetPluginLayerNameMap();                                                         \
            TensorRTPluginLayerBuilder* layer = layer_map[name];                                                   \
            auto plugin = layer->CreatePlugin();                                                                   \
            plugin->setPluginNamespace(m_plugin_namespace.c_str());                                                \
            return plugin;                                                                                         \
        }                                                                                                          \
        virtual nvinfer1::IPluginV2DynamicExt* deserializePlugin(const char* name,                                 \
                const void* serialData, size_t serialLength) {                                                     \
            std::unordered_map<std::string, TensorRTPluginLayerBuilder*> layer_map =                               \
                TensorRTNetwork_::GetPluginLayerNameMap();                                                         \
            TensorRTPluginLayerBuilder* layer = layer_map[name];                                                   \
            IPluginV2DynamicExt* plugin;                                                                           \
            if (serialLength == 0) {                                                                               \
                plugin = layer->CreatePlugin();                                                                    \
            } else {                                                                                               \
                plugin = layer->CreatePlugin(serialData, serialLength);                                            \
            }                                                                                                      \
            plugin->setPluginNamespace(m_plugin_namespace.c_str());                                                \
            auto new_plugin = plugin->clone();                                                                     \
            return new_plugin;                                                                                     \
        }                                                                                                          \
    private:                                                                                                       \
        nvinfer1::PluginFieldCollection m_fc;                                                                      \
        std::string m_plugin_namespace;                                                                            \
    };                                                                                                             \
    REGISTER_TENSORRT_PLUGIN(type_string##PluginCreator);

#define REGISTER_TENSORRT_PLUGIN_LAYER_BUILDER(type_string, layer_type)                                            \
    TRTPluginTypeLayerBuilderRegister<TypeLayerBuilderCreator<type_string##TRTPluginLayerBuilder>>                 \
        g_##layer_type##_trt_plugin_layer_builder_register(layer_type);

}  //  namespace TNN_NS

#endif  //  TNN_SOURCE_TNN_NETWORK_TENSORR_LAYER_BUILDER_TENSORRT_PLUGIN_LAYER_BUILDER_H_
