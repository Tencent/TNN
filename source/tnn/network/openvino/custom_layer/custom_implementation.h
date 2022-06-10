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

#include <algorithm>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <ie_blob.h>
#include <ie_iextension.h>
#include <ie_layouts.h>

#include <tnn/core/status.h>
#include <tnn/network/openvino/openvino_network.h>
#include <ngraph/ngraph.hpp>
#include <ngraph/opsets/opset.hpp>

#include "tnn/network/openvino/utils.h"

#ifndef TNN_DEVICE_OPENVINO_CUSTOM_OPENVINO_IMPLEMENTAIO_
#define TNN_DEVICE_OPENVINO_CUSTOM_OPENVINO_IMPLEMENTAIO_

namespace TNN_NS {

class CustomOpenvinoOp : public ngraph::op::Op {
public:
    CustomOpenvinoOp() = default;
    explicit CustomOpenvinoOp(const ngraph::OutputVector input_nodes, BaseLayer* baselayer,
                              const std::vector<Blob*> input_blobs, const std::vector<Blob*> output_blobs)
        : Op(input_nodes), base_layer_(baselayer), input_blobs_(input_blobs), output_blobs_(output_blobs) {
        constructor_validate_and_infer_types();
    };

    void validate_and_infer_types() override {
        for (size_t i = 0; i < input_blobs_.size(); i++) {
            auto input_desc = input_blobs_[i]->GetBlobDesc();
            auto dims = input_desc.dims;
            auto input_shape = get_input_shape(i);
            
            dims.resize(input_shape.size());
            for (size_t j = 0; j < input_shape.size(); j++) {
                dims[j] = input_shape[j];
            }
            input_desc.dims = dims;
            input_blobs_[i]->SetBlobDesc(input_desc);
        }
        base_layer_->Reshape();
        for (size_t i = 0; i < output_blobs_.size(); i++) {
            auto desc = output_blobs_[i]->GetBlobDesc();
            auto dims = desc.dims;
            ngraph::Shape output_shape(dims.size());       
            for (size_t j = 0; j < dims.size(); j++) {
                output_shape[j] = dims[j];
            }
            set_output_type(i, ConvertToOVDataType(desc.data_type), ngraph::PartialShape(output_shape));
        }
    };

    bool visit_attributes(ngraph::AttributeVisitor& visitor) override {
        return true;
    }

    BaseLayer* getBaseLayer() {
        return base_layer_;
    }
    std::vector<Blob*> getInputBlobs() {
        return input_blobs_;
    }
    std::vector<Blob*> getOutputBlobs() {
        return output_blobs_;
    }

protected:
    BaseLayer* base_layer_;
    std::vector<Blob*> input_blobs_, output_blobs_;
};

class CustomOpenvinoImpl : public InferenceEngine::ILayerExecImpl {
public:
    explicit CustomOpenvinoImpl(const std::shared_ptr<ngraph::Node>& node) : node_(node) {}

    // @brief get configurations desc of custom node implementation
    InferenceEngine::StatusCode getSupportedConfigurations(std::vector<InferenceEngine::LayerConfig>& conf,
                                                           InferenceEngine::ResponseDesc*) noexcept override {
        InferenceEngine::LayerConfig layerConfig;
        layerConfig.dynBatchSupport = true;

        auto node = dynamic_cast<CustomOpenvinoOp*>(GetNode().get());
        if (node == nullptr) {
            return InferenceEngine::GENERAL_ERROR;
        }
        for (size_t i = 0; i < node_->inputs().size(); i++) {
            auto ov_type   = node->get_input_element_type(i);
            auto precision = ConvertOVTypeToPrecision(ov_type);
            InferenceEngine::DataConfig cfg;
            cfg.constant = false;
            cfg.inPlace  = -1;

            InferenceEngine::SizeVector order;
            auto partialShape = node_->get_input_partial_shape(i);
            if (partialShape.is_dynamic())
                return InferenceEngine::GENERAL_ERROR;

            auto shape = node_->get_input_shape(i);
            for (size_t j = 0; j < shape.size(); j++) {
                order.push_back(j);
            }
            cfg.desc = InferenceEngine::TensorDesc(precision, shape, {shape, order});
            layerConfig.inConfs.push_back(cfg);
        }

        for (size_t i = 0; i < node_->outputs().size(); i++) {
            auto ov_type   = node->get_output_element_type(i);
            auto precision = ConvertOVTypeToPrecision(ov_type);
            InferenceEngine::DataConfig cfg;
            cfg.constant = false;
            cfg.inPlace  = -1;

            InferenceEngine::SizeVector order;
            auto partialShape = node_->get_output_partial_shape(i);
            if (partialShape.is_dynamic())
                return InferenceEngine::GENERAL_ERROR;

            auto shape = node_->get_output_shape(i);
            for (size_t j = 0; j < shape.size(); j++) {
                order.push_back(j);
            }
            cfg.desc = InferenceEngine::TensorDesc(precision, shape, {shape, order});
            layerConfig.outConfs.push_back(cfg);
        }

        conf.push_back(layerConfig);
        return InferenceEngine::OK;
    };

    // @brief init custom node implementaion
    InferenceEngine::StatusCode init(InferenceEngine::LayerConfig&, InferenceEngine::ResponseDesc*) noexcept override {
        return InferenceEngine::StatusCode::OK;
    }

    // @brief custom node execution
    InferenceEngine::StatusCode execute(std::vector<InferenceEngine::Blob::Ptr>& inputs,
                                        std::vector<InferenceEngine::Blob::Ptr>& outputs,
                                        InferenceEngine::ResponseDesc* resp) noexcept {
        const auto node = std::dynamic_pointer_cast<CustomOpenvinoOp>(GetNode());
        auto input_blob = node->getInputBlobs();

        for (size_t i = 0; i < inputs.size(); i++) {
            InferenceEngine::MemoryBlob::CPtr minput = InferenceEngine::as<InferenceEngine::MemoryBlob>(inputs[i]);
            if (!minput) {
                return InferenceEngine::StatusCode::PARAMETER_MISMATCH;
            }

            auto minputHolder = minput->rmap();
            BlobHandle input_handle;

            input_handle.base = minputHolder.as<void*>();
            input_blob[i]->SetHandle(input_handle);
        }

        auto output_blob = node->getOutputBlobs();
        for (size_t i = 0; i < outputs.size(); i++) {
            InferenceEngine::MemoryBlob::Ptr moutput = InferenceEngine::as<InferenceEngine::MemoryBlob>(outputs[i]);
            if (!moutput) {
                return InferenceEngine::StatusCode::PARAMETER_MISMATCH;
            }

            auto moutputHolder = moutput->rmap();
            BlobHandle output_handle;

            output_handle.base = moutputHolder.as<void*>();
            output_blob[i]->SetHandle(output_handle);
        }

        auto base_layer = node->getBaseLayer();
        base_layer->Forward();

        return InferenceEngine::OK;
    }

    // @brief get node_
    const std::shared_ptr<ngraph::Node> GetNode() {
        return node_;
    }

private:
    const std::shared_ptr<ngraph::Node> node_;
};

class CustomOpenvinoLayerManager : public InferenceEngine::IExtension {
public:
    // @brief register impl
    template <typename T>
    static void RegisterCustomOpenvinoLayer(std::string type) {
        std::map<std::string, std::function<InferenceEngine::ILayerImpl::Ptr(const std::shared_ptr<ngraph::Node>)>>&
            custom_openvino_layer_map = GetCustomOpenvinoLayerMap();
        custom_openvino_layer_map[type] =
            [](const std::shared_ptr<ngraph::Node>& node) -> InferenceEngine::ILayerImpl::Ptr {
            return std::make_shared<T>(node);
        };
    }

    // @brief this map is used to create opset into inference engine extension
    static std::map<std::string, std::function<InferenceEngine::ILayerImpl::Ptr(const std::shared_ptr<ngraph::Node>)>>&
    GetCustomOpenvinoLayerMap() {
        static std::map<std::string,
                        std::function<InferenceEngine::ILayerImpl::Ptr(const std::shared_ptr<ngraph::Node>)>>
            custom_openvino_layer_map;
        return custom_openvino_layer_map;
    }

    static std::set<LayerType> &GetCustomLayerTypeSet() {
        static std::set<LayerType> custom_layer_type_set;
        return custom_layer_type_set;
    }

    static void RegisterCustomLayerType(LayerType &type) {
        std::set<LayerType> &layer_type_set = GetCustomLayerTypeSet();
        if (layer_type_set.find(type) == layer_type_set.end()) {
            layer_type_set.insert(type);
        }
    }

    void GetVersion(const InferenceEngine::Version*& versionInfo) const noexcept override {}

    void Unload() noexcept override {}

    void Release() noexcept override {}

    std::vector<std::string> getImplTypes(const std::shared_ptr<ngraph::Node>& node) override {
        auto impls = GetCustomOpenvinoLayerMap();
        if (impls.find(node->description()) == impls.end())
            return {};
        return {"CPU"};
    }

    InferenceEngine::ILayerImpl::Ptr getImplementation(const std::shared_ptr<ngraph::Node>& node,
                                                       const std::string& implType) override {
        auto impls = GetCustomOpenvinoLayerMap();
        if (impls.find(node->description()) == impls.end() || implType != "CPU")
            return nullptr;
        return impls[node->description()](node);
    }

    std::map<std::string, ngraph::OpSet> getOpSets() override {
        static std::mutex g_mutex;
        const std::lock_guard<std::mutex> lock(g_mutex);

        static std::map<std::string, ngraph::OpSet> opsets;
        if (opsets.empty()) {
            opsets["tnnCustom"] = getCustomOpSet();
        }
        return opsets;
    }

    // @brief register ngraph::Op
    template <typename T>
    static void RegisterCustomOp() {
        ngraph::OpSet& opset = getCustomOpSet();
        opset.insert<T>();
    }

    // @brief static op set
    static ngraph::OpSet& getCustomOpSet() {
        static ngraph::OpSet opset;
        return opset;
    }

    static Status status;
};

template <typename T>
class CustomImplementationRegister {
public:
    explicit CustomImplementationRegister(std::string type_string) {
        CustomOpenvinoLayerManager::RegisterCustomOpenvinoLayer<T>(type_string);
    }
};

template <typename T>
class CustomOpRegister {
public:
    explicit CustomOpRegister() {
        CustomOpenvinoLayerManager::RegisterCustomOp<T>();
    }
};

class CustomTypeRegister {
public:
    explicit CustomTypeRegister(LayerType type) {
        CustomOpenvinoLayerManager::RegisterCustomLayerType(type);
    }
};

}  // namespace TNN_NS

#define DECLARE_CUSTOM_IMPLEMENTATION(type)                                                                            \
    class Custom##type##Impl : public TNN_NS::CustomOpenvinoImpl {                                                     \
    public:                                                                                                            \
        explicit Custom##type##Impl(const std::shared_ptr<ngraph::Node>& node) : CustomOpenvinoImpl(node) {}           \
    }

// 注册到一起，方便一次导入，一个 getmap 函数把这些都 get 过来
#define REGISTER_CUSTOM_IMPLEMENTATION(type, type_string)                                                              \
    CustomImplementationRegister<Custom##type##Impl> g_custom_##type##_impl_register(#type_string);

#define REGISTER_CUSTOM_TYPE(type) CustomTypeRegister g_custom_##type##_register(type);

#define REGISTER_CUSTOM_OP(type)                                                                                       \
    CustomOpRegister<Custom##type##Op> g_custom_##type##_op_register();                                                \
    constexpr ngraph::NodeTypeInfo Custom##type##Op::type_info;

#define DECLARE_CUSTOM_OP(type)                                                                                        \
    class Custom##type##Op : public TNN_NS::CustomOpenvinoOp {                                                         \
    public:                                                                                                            \
        static constexpr ngraph::NodeTypeInfo type_info{"Custom" #type, 0};                                            \
        const ngraph::NodeTypeInfo& get_type_info() const {                                                            \
            return type_info;                                                                                          \
        }                                                                                                              \
        Custom##type##Op(const ngraph::OutputVector input_nodes, TNN_NS::BaseLayer* baselayer,                         \
                         const std::vector<TNN_NS::Blob*> input_blobs, const std::vector<TNN_NS::Blob*> output_blobs)  \
            : TNN_NS::CustomOpenvinoOp(input_nodes, baselayer, input_blobs, output_blobs){};                           \
        std::shared_ptr<ngraph::Node> clone_with_new_inputs(const ngraph::OutputVector& new_args) const override {     \
            return std::make_shared<Custom##type##Op>(new_args, base_layer_, input_blobs_, output_blobs_);             \
        }                                                                                                              \
    }

#endif
