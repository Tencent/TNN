// Copyright 2019 Tencent. All Rights Reserved

#include "openvino_network.h"
#include <string.h>

#include <inference_engine.hpp>
#include <ngraph/ngraph.hpp>

#include "tnn/core/blob_int8.h"
#include "tnn/core/profile.h"
#include "tnn/interpreter/default_model_interpreter.h"
#include "tnn/interpreter/layer_param.h"
#include "tnn/interpreter/layer_resource_generator.h"
#include "tnn/optimizer/net_optimizer_manager.h"
#include "tnn/utils/blob_dump_utils.h"
#include "tnn/utils/blob_transfer_utils.h"
#include "tnn/utils/dims_vector_utils.h"
#include "tnn/device/openvino/openvino_types.h"
#include "tnn/extern_wrapper/foreign_blob.h"
#include "tnn/extern_wrapper/foreign_tensor.h"
#include "tnn/device/openvino/layer_builder/openvino_layer_builder.h"

namespace TNN_NS {

NetworkImplFactoryRegister<NetworkImplFactory<OpenVINONetwork_>> g_network_impl_openvino_ngraph_factory_register(NETWORK_TYPE_OPENVINO);

OpenVINONetwork_::~OpenVINONetwork_() {
    DeInit();
}

Status OpenVINONetwork_::Init(NetworkConfig &net_config, ModelConfig &model_config,
                            AbstractModelInterpreter* interpreter,
                            InputShapesMap inputs_shape) {

    Status ret                                   = TNN_OK;

    // RETURN_ON_NEQ(DefaultNetwork::Init(net_config, model_config, interpreter, inputs_shape), TNN_OK);
    DefaultModelInterpreter *default_interpreter = dynamic_cast<DefaultModelInterpreter *>(interpreter);
    CHECK_PARAM_NULL(default_interpreter);

    NetStructure *net_structure = default_interpreter->GetNetStructure();
    NetResource *net_resource   = default_interpreter->GetNetResource();

    if (net_structure == NULL || net_resource == NULL) {
        LOGE("ERROR: network_ is nil, network_type may not support\n");
        return Status(TNNERR_NULL_PARAM, "network_ is nil, network_type may not support");
    }
    device_ = GetDevice(net_config.device_type);
    blob_manager_ = new BlobManager(device_);
    ret = blob_manager_->Init(net_config, net_structure, inputs_shape, GetNetResourceDataType(net_resource));

    //set inputnode
    SetNetInputNode(net_structure, net_resource);
    //init layers and nodes
    InitLayers(net_structure, net_resource);

    device_ = GetDevice(net_config.device_type);
    if (device_ == NULL) {
        return TNNERR_DEVICE_NOT_SUPPORT;
    }

    // build ngraph network
    BuildNgraphNetwork(net_structure);
    //////////////////////////////////////////////////////////////
    ie_.SetConfig({{ CONFIG_KEY(CPU_THREADS_NUM), "1"}}, "CPU");
    
    executable_network_ = ie_.LoadNetwork(*network_, "CPU");
    
    infer_request_ = executable_network_.CreateInferRequest();

    auto input_map = executable_network_.GetInputsInfo();
    for(auto item : input_map) {
        std::string key = item.first;
        auto blob_ptr = infer_request_.GetBlob(key);
        BlobDesc desc;
        desc.data_format = DATA_FORMAT_NCHW;
        desc.name = key;
        auto dims = blob_ptr->getTensorDesc().getDims();
        for(int index = 0; index<dims.size(); index++) {
            desc.dims.push_back(dims[index]);
        }
        BlobHandle handle;
        handle.base = blob_ptr->buffer().as<InferenceEngine::PrecisionTrait<InferenceEngine::Precision::FP32>::value_type*>();
        input_blob_map_[key] = new Blob(desc, handle);  
    }

    auto output_map = executable_network_.GetOutputsInfo();
    for(auto item : output_map) {
        std::string key = item.first;
        auto blob_ptr = infer_request_.GetBlob(key);
        BlobDesc desc;
        desc.data_format = DATA_FORMAT_NCHW;
        desc.name = key;
        auto dims = blob_ptr->getTensorDesc().getDims();
        for(int index = 0; index<dims.size(); index++) {
            desc.dims.push_back(dims[index]);
        }
        BlobHandle handle;
        handle.base = blob_ptr->buffer().as<InferenceEngine::PrecisionTrait<InferenceEngine::Precision::FP32>::value_type*>();
        output_blob_map_[key] = new Blob(desc, handle); 
    }

    return Reshape(inputs_shape);
}

Status OpenVINONetwork_::SetNetInputNode(NetStructure *net_structure, NetResource* net_resource) {
    
    std::string input_name = net_structure->layers.at(0)->inputs.at(0);
    std::vector<int> input_node_shape = net_structure->inputs_shape_map.begin()->second;
    ngraph::Shape ngraphInputShape;
    for (size_t i = 0; i < input_node_shape.size(); i++) {
        ngraphInputShape.push_back(input_node_shape.at(i));
    }
    
    std::shared_ptr<ngraph::op::Parameter> input_node = std::make_shared<ngraph::op::Parameter>(
                        ngraph::element::f32, ngraph::Shape(ngraphInputShape));
    input_node->set_friendly_name(input_name);

    auto blob = blob_manager_->GetBlob(input_name);
    auto foreign_blob = new ForeignBlob(blob);
    auto openvinoTensor = std::make_shared<OpenvinoTensor>();
    
    openvinoTensor->SetNode(input_node);
    foreign_blob->SetForeignTensor(openvinoTensor);
    blob_manager_->ReplaceBlob(input_name, foreign_blob);

    return TNN_OK;
}

Status OpenVINONetwork_::BuildNgraphNetwork(NetStructure *net_structure) {

    auto input_name = net_structure->layers.front()->inputs.front();
    auto input_tensor = dynamic_cast<ForeignBlob*>(blob_manager_->GetBlob(input_name))->GetForeignTensor();
    auto input_openvino_tensor = std::dynamic_pointer_cast<OpenvinoTensor>(input_tensor);
    auto input_node = std::dynamic_pointer_cast<ngraph::op::Parameter>(input_openvino_tensor->GetNode());

    ngraph::NodeVector outputNodes;
    for (auto name : net_structure->outputs) {
        auto output_tensor = dynamic_cast<ForeignBlob*>(blob_manager_->GetBlob(name))->GetForeignTensor();
        auto output_openvino_tensor = std::dynamic_pointer_cast<OpenvinoTensor>(output_tensor);
        auto result_node = std::make_shared<ngraph::op::Result>(output_openvino_tensor->GetNode());
        outputNodes.push_back(result_node);
    } 
    
    std::shared_ptr<ngraph::Function> nodeFunciton = std::make_shared<ngraph::Function>(
         outputNodes, ngraph::ParameterVector{ input_node }, "net");

    network_ =  std::make_shared<InferenceEngine::CNNNetwork>(nodeFunciton);
    return TNN_OK;
}

Status OpenVINONetwork_::GetForwardMemorySize(int &memory_size) {
    memory_size = 0;
    return TNN_OK;
}

Status OpenVINONetwork_::SetForwardMemory(void *memory) {   
    return TNN_OK;
}

Status OpenVINONetwork_::GetAllInputBlobs(BlobMap &blobs) {
    blobs = input_blob_map_;
    return TNN_OK;
}

Status OpenVINONetwork_::GetAllOutputBlobs(BlobMap &blobs) {
    blobs = output_blob_map_;
    return TNN_OK;
}

Status OpenVINONetwork_::Reshape(const InputShapesMap &inputs) {
    RETURN_ON_NEQ(DefaultNetwork::Reshape(inputs), TNN_OK);

    auto network_shapes = network_->getInputShapes();

    for(auto item : inputs) {
        std::string input_name = item.first;
        if (item.second.size() < 4) {
            return TNNERR_PARAM_ERR; 
        }
        if (network_shapes.find(input_name) == network_shapes.end()) {
            return TNNERR_PARAM_ERR;
        }

        InferenceEngine::SizeVector input_shape;
        for(int i=0;i<item.second.size();i++) {
            input_shape.push_back(item.second[i]);
        }
        network_shapes[input_name] = input_shape;

    }

    network_->reshape(network_shapes);

    executable_network_ = ie_.LoadNetwork(*network_, "CPU");
    infer_request_ = executable_network_.CreateInferRequest();

    auto input_map = executable_network_.GetInputsInfo();
    for(auto item : input_map) {
        std::string key = item.first;
        auto blob_ptr = infer_request_.GetBlob(key);

        auto dims = blob_ptr->getTensorDesc().getDims();
        DimsVector blob_dims;
        for(int index = 0; index<dims.size(); index++) {
            blob_dims.push_back(dims[index]);
        }
        input_blob_map_[key]->GetBlobDesc().dims = blob_dims;

        auto handle = input_blob_map_[key]->GetHandle();
        handle.base = blob_ptr->buffer()
            .as<InferenceEngine::PrecisionTrait<InferenceEngine::Precision::FP32>::value_type*>();
        input_blob_map_[key]->SetHandle(handle);
    }

    auto output_map = executable_network_.GetOutputsInfo();
    for(auto item : output_map) {
        std::string key = item.first;
        auto blob_ptr = infer_request_.GetBlob(key);

        auto dims = blob_ptr->getTensorDesc().getDims();
        DimsVector blob_dims;
        for(int index = 0; index<dims.size(); index++) {
            blob_dims.push_back(dims[index]);
        }
        output_blob_map_[key]->GetBlobDesc().dims = blob_dims;

        auto handle = output_blob_map_[key]->GetHandle();
        handle.base = blob_ptr->buffer()
            .as<InferenceEngine::PrecisionTrait<InferenceEngine::Precision::FP32>::value_type*>();
        output_blob_map_[key]->SetHandle(handle);
    }

    return TNN_OK;
}

/*
 * InitLayerBuilders funcion does the following things:
 *  1. Set Blob type accordingly.
 *  2. Set data_tyep accordingly.
 *  3. Infer the blob shapes.
 *  4. Check the weights required.
 *  5. Create Layer Builders.
 */
Status OpenVINONetwork_::InitLayers(NetStructure *net_structure, NetResource *net_resource) {
    Status ret = TNN_OK;

    for (auto layer_info : net_structure->layers) {
        LayerType type       = layer_info->type;
        OpenVINOLayerBuilder *cur_layer = CreateOpenVINOLayerBuilder(type);
        
        if (cur_layer == NULL) {
            LOGE("Error: CreateLayerBuilder failed, type:%d\n", type);
            return Status(TNNERR_PARAM_ERR, "CreateLayerBuilder failed");
        }
        std::string layer_name = layer_info->name;
        cur_layer->SetLayerName(layer_name);
        // set layer nodes
        std::vector<Blob *> inputs;
        std::vector<std::string> &input_names = layer_info->inputs;
        // get input nodes
        for (auto name : input_names) {
            ForeignBlob* blob = dynamic_cast<ForeignBlob*>(blob_manager_->GetBlob(name));
            // auto openvino_tensor = std::dynamic_pointer_cast<OpenvinoTensor>(blob.GetForeignTensor());
            // openvino_tensor->GetNode();
            /*
            // Check for int8
            bool is_int8_blob = layer_info->param->quantized;
            if (is_int8_blob && blob->GetBlobDesc().data_type != DATA_TYPE_INT8) {
                auto new_blob               = new BlobInt8(blob->GetBlobDesc(), blob->GetHandle());
                auto dest                   = blob->GetBlobDesc();
                std::string blob_scale_name = name + "_scale_data_";
#ifdef BENCHMARK
                if (net_resource->resource_map.count(blob_scale_name) == 0) {
                    LayerResource *layer_res  = nullptr;
                    std::vector<Blob *> blobs = {blob};
                    GenerateRandomResource(LAYER_BLOB_SCALE, nullptr, &layer_res, blobs);
                    net_resource->resource_map[blob_scale_name] = std::shared_ptr<LayerResource>(layer_res);
                }
#endif
                new_blob->SetIntResource(
                    reinterpret_cast<IntScaleResource *>(net_resource->resource_map[blob_scale_name].get()));
                blob_manager_->ReplaceBlob(name, new_blob);
                blob = new_blob;
            }
            // Check for bfp16
            if (_config.precision == PRECISION_LOW && blob->GetBlobDesc().data_type != DATA_TYPE_INT8) {
                blob->GetBlobDesc().data_type = DATA_TYPE_BFP16;
            }
            */
            inputs.push_back(blob);
        }
        std::vector<Blob *> outputs;
        std::vector<std::string> &output_names = layer_info->outputs;

#ifdef BENCHMARK
        // generate resource if null
        if (net_resource->resource_map.count(layer_name) == 0) {
            LayerParam *layer_param  = layer_info->param.get();
            LayerResource *layer_res = nullptr;
            GenerateRandomResource(type, layer_param, &layer_res, inputs);
            net_resource->resource_map[layer_name] = std::shared_ptr<LayerResource>(layer_res);
        }

        std::vector<Blob *> outputs_for_shape;
        for (auto name : output_names) {
            outputs_for_shape.push_back(blob_manager_->GetBlob(name));
        }
        cur_layer->InferShapeAhead(inputs, outputs_for_shape, layer_info->param.get(),
                                   net_resource->resource_map[layer_name].get());
#endif
        // init output nodes
        for (auto name : output_names) {
            auto blob = blob_manager_->GetBlob(name);
            auto foreign_blob = new ForeignBlob(blob);
            foreign_blob->SetForeignTensor(std::make_shared<OpenvinoTensor>());
            blob_manager_->ReplaceBlob(name, foreign_blob);
            blob = foreign_blob;
            /*
            // Check for int8
            bool is_int8_blob =
                layer_info->param->quantized ||
                (type == LAYER_REFORMAT &&
                 reinterpret_cast<ReformatLayerParam *>(layer_info->param.get())->dst_type == DATA_TYPE_INT8);

            if (is_int8_blob) {
                auto new_blob               = new BlobInt8(blob->GetBlobDesc(), blob->GetHandle());
                std::string blob_scale_name = name + "_scale_data_";
#ifdef BENCHMARK
                if (net_resource->resource_map.count(blob_scale_name) == 0) {
                    LayerResource *layer_res  = nullptr;
                    std::vector<Blob *> blobs = {blob};
                    GenerateRandomResource(LAYER_BLOB_SCALE, nullptr, &layer_res, blobs);
                    net_resource->resource_map[blob_scale_name] = std::shared_ptr<LayerResource>(layer_res);
                }
#endif
                new_blob->SetIntResource(
                    reinterpret_cast<IntScaleResource *>(net_resource->resource_map[blob_scale_name].get()));
                blob_manager_->ReplaceBlob(name, new_blob);
                blob = new_blob;
            }
            // Check for bfp16
            if (_config.precision == PRECISION_LOW && blob->GetBlobDesc().data_type != DATA_TYPE_INT8) {
                blob->GetBlobDesc().data_type = DATA_TYPE_BFP16;
            }
            */
           outputs.push_back(blob);
        }
        LayerResource *layer_resource = net_resource->resource_map[layer_name].get();
        
        // init node
        ret = cur_layer->Init(context_, layer_info->param.get(), layer_resource, inputs, outputs, device_);
        if (ret != TNN_OK) {
            LOGE("Error Init layer %s (err: %d or 0x%X)\n", cur_layer->GetLayerName().c_str(), (int)ret, (int)ret);
            return ret;
        }
        
        layers_.push_back(cur_layer);
    }
    return ret;
}

Status OpenVINONetwork_::DeInit() {
    for(auto item : input_blob_map_) {
        delete item.second;
    }
    input_blob_map_.clear();
    for(auto item : output_blob_map_) {
        delete item.second;
    }
    output_blob_map_.clear();
    return TNN_OK;
}

Status OpenVINONetwork_::GetCommandQueue(void **command_queue) {
    return TNN_OK;
}

Status OpenVINONetwork_::Forward() {
    infer_request_.Infer();
    return TNN_OK;
}

// @brief openvino instance network infer, it will not wait
Status OpenVINONetwork_::ForwardAsync(Callback call_back) {
    Status result = TNN_OK;
    infer_request_.Infer();
    return result;
}

}  // namespace TNN_NS
