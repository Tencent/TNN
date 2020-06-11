// Copyright 2019 Tencent. All Rights Reserved

#include "openvino_network.h"
#include "openvino_model_interpreter.h"

#include <inference_engine.hpp>

namespace TNN_NS {

NetworkImplFactoryRegister<NetworkImplFactory<OpenVINONetwork>> g_network_impl_openvino_factory_register(NETWORK_TYPE_OPENVINO);

OpenVINONetwork::~OpenVINONetwork() {
    DeInit();
}

Status OpenVINONetwork::Init(NetworkConfig &net_config, ModelConfig &model_config,
                            AbstractModelInterpreter* interpreter,
                            InputShapesMap inputs_shape) {
    Status ret = TNN_OK;

    ie_.SetConfig({{ CONFIG_KEY(CPU_THREADS_NUM), "1"}}, "CPU");

    OpenVINOModelInterpreter* default_interpreter = dynamic_cast<OpenVINOModelInterpreter*>(interpreter);
    network_ = default_interpreter->GetCNNNetwork();

    executable_network_ = ie_.LoadNetwork(network_, "CPU");
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

Status OpenVINONetwork::GetForwardMemorySize(int &memory_size) {
    memory_size = 0;
    return TNN_OK;
}

Status OpenVINONetwork::SetForwardMemory(void *memory) {
    return TNN_OK;
}

Status OpenVINONetwork::GetAllInputBlobs(BlobMap &blobs) {
    blobs = input_blob_map_;
    return TNN_OK;
}

Status OpenVINONetwork::GetAllOutputBlobs(BlobMap &blobs) {
    blobs = output_blob_map_;
    return TNN_OK;
}

Status OpenVINONetwork::Reshape(const InputShapesMap &inputs) {
    auto network_shapes = network_.getInputShapes();

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

    network_.reshape(network_shapes);

    executable_network_ = ie_.LoadNetwork(network_, "CPU");
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

Status OpenVINONetwork::DeInit() {
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

Status OpenVINONetwork::GetCommandQueue(void **command_queue) {
    return TNN_OK;
}

Status OpenVINONetwork::Forward() {
    infer_request_.Infer();
    return TNN_OK;
}

// @brief openvino instance network infer, it will not wait
Status OpenVINONetwork::ForwardAsync(Callback call_back) {
    Status result = TNN_OK;
    infer_request_.Infer();
    return result;
}

}  // namespace TNN_NS
