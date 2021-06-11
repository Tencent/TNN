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

#include "tnn/network/torch/torch_network.h"

#include <memory>
#include <sstream>
#include <fstream>
#include <iostream>
#include <set>

#include "tnn/core/blob.h"
#include "tnn/core/common.h"
#include "tnn/core/status.h"
#include "tnn/extern_wrapper/foreign_blob.h"
#include "tnn/extern_wrapper/foreign_tensor.h"
#include "tnn/interpreter/abstract_model_interpreter.h"
#include "tnn/interpreter/default_model_interpreter.h"
#include "tnn/network/torch/torch_utils.h"
#include "tnn/network/torch/torch_tensor.h"

#include "torch/csrc/jit/passes/freeze_module.h"
#include "torch/csrc/jit/passes/lower_graph.h"

namespace TNN_NS {

NetworkImplFactoryRegister<NetworkImplFactory<TNNTorchNetwork>> g_network_impl_torch_factory_register(NETWORK_TYPE_TNNTORCH);

TNNTorchNetwork::~TNNTorchNetwork() {}

Status TNNTorchNetwork::Init(NetworkConfig &net_config, ModelConfig &model_config, AbstractModelInterpreter *interpreter,
                        InputShapesMap min_inputs_shape, InputShapesMap max_inputs_shape) {

    config_                                      = net_config;
    Status ret                                   = TNN_OK;

    device_ = GetDevice(net_config.device_type);
    if (device_ == nullptr) {
        return TNNERR_DEVICE_NOT_SUPPORT;
    }
    context_ = device_->CreateContext(net_config.device_id);
    if (context_ == nullptr) {
        return TNNERR_DEVICE_CONTEXT_CREATE;
    }

    if (model_config.model_type == MODEL_TYPE_TORCHSCRIPT) {
        std::ifstream model_stream(model_config.params[0], std::ios::binary);
        if (model_config.params.size() > 1) {
            forward_func_name_ = model_config.params[1];
        }
        RETURN_ON_FAIL(LoadModule(model_stream, net_config));
    } else {
        return Status(TNNERR_PARAM_ERR, "Unsupported model type for TNNTorchNetwork");
    }

    at::ArrayRef<torch::jit::Value*> inputs = graph_->block()->inputs();
    at::ArrayRef<torch::jit::Value*> outputs = graph_->block()->outputs();

    #if 0
    printf("graph dump:\n:%s\n", graph_->toString().c_str());

    for(int i=0;i<inputs.size();i++) {
        auto input = inputs[i];
        printf("input[%d] from node [%s] of type:[%s]\n", input->unique(), input->node()->scopeName().c_str(), input->type()->annotation_str().c_str());
    }

    for(int i=0;i<outputs.size();i++) {
        auto output = outputs[i];
        printf("output[%d] from node [%s] of type:[%s]\n", output->unique(), output->node()->scopeName().c_str(), output->type()->annotation_str().c_str());
    }
    #endif

    RETURN_ON_FAIL(CreateIOBinding(min_inputs_shape, max_inputs_shape));

    return TNN_OK;
}

Status TNNTorchNetwork::LoadModule(std::istream& in, NetworkConfig &config) {
    c10::Device device(c10::kCPU);
    RETURN_ON_NEQ(ConvertToTorchDevice(device, config.device_type, config.device_id), TNN_OK);
    auto mod = torch::jit::load(in, device);
    module_ = std::make_shared<torch::jit::Module>(torch::jit::freeze(mod));
    graph_ = module_->get_method(forward_func_name_).graph(); 

    // auto graph_and_ivalues = torch::jit::LowerGraph(*graph_, module_->_ivalue());
    // graph_ = graph_and_ivalues.first;

    return TNN_OK;
}

Status TNNTorchNetwork::CreateIOBinding(InputShapesMap  min_shape, InputShapesMap max_shape) {
    NetStructure fake_netstructure;
    fake_netstructure.inputs_shape_map = max_shape;

    // regardless of those blobs only show once
    for(auto p : min_shape) fake_netstructure.blobs.insert(p.first);
    for(auto p : max_shape) {
        if (fake_netstructure.blobs.find(p.first) == fake_netstructure.blobs.end()) {
            fake_netstructure.blobs.erase(p.first);
        }
    }

    blob_manager_ = new BlobManager(device_);
    if (blob_manager_ == nullptr) {
        return Status(TNNERR_COMMON_ERROR, "create blob_manager_ failed");;
    }

    RETURN_ON_FAIL(blob_manager_->Init(config_, &fake_netstructure, max_shape, DATA_TYPE_FLOAT));

    RETURN_ON_FAIL(blob_manager_->AllocateBlobMemory());

    in_ivalues_.resize(fake_netstructure.blobs.size());

    int i=0;
    for(auto input : fake_netstructure.blobs) {
        auto blob  = blob_manager_->GetBlob(input);
        auto foreign_blob = new ForeignBlob(blob->GetBlobDesc(), blob->GetHandle());

        blob_manager_->ReplaceBlob(input, foreign_blob);
        input_blob_map_[input] = foreign_blob;

        RETURN_ON_FAIL(attachTensor(foreign_blob));
        RETURN_ON_FAIL(ForeignBlobToIValue(in_ivalues_[i++], foreign_blob));
    }

    std::vector<torch::Tensor> out_tensors;

    auto out = module_->forward(in_ivalues_);
    RETURN_ON_FAIL(ConvertIValueToTensors(out_tensors, out));

    i = 0;
    for(auto out : out_tensors) {
        char tensor_name[200];
        snprintf(tensor_name, 200, "output_%d", i++);
        BlobDesc desc;
        desc.name = std::string(tensor_name);
        RETURN_ON_FAIL(GetBlobDescFromTensor(desc, out));

        BlobHandle handle;
        handle.base = out.data_ptr(); 

        auto foreign_blob =  new ForeignBlob(desc, handle);
        foreign_blob->SetForeignTensor(std::make_shared<TorchTensor>(std::make_shared<torch::Tensor>(std::move(out))));
        output_blob_map_[desc.name] = foreign_blob;
    }

    return TNN_OK;
}

Status TNNTorchNetwork::Reshape(const InputShapesMap &inputs) {
    return TNN_OK;
}


Status TNNTorchNetwork::Forward() {


    auto out = module_->forward(in_ivalues_);

    std::vector<torch::Tensor> out_tensors;
    RETURN_ON_FAIL(ConvertIValueToTensors(out_tensors, out));

    auto cpu_out = out_tensors[0].to(torch::kCPU);

    int i=0;
    for(auto p : output_blob_map_ ) {
        auto foreign_blob = dynamic_cast<ForeignBlob*>(p.second);
        BlobHandle handle = foreign_blob->GetHandle();
        handle.base = out_tensors[i].data_ptr();
        auto cpu_out = out_tensors[i].to(torch::kCPU);

        foreign_blob->SetHandle(handle);
        foreign_blob->SetForeignTensor(std::make_shared<TorchTensor>(std::make_shared<torch::Tensor>(std::move(out_tensors[i]))));
        i++;

        #if 0
        auto bytes = torch::jit::pickle_save(cpu_out);
        std::ofstream fout("out.pt", std::ios::out | std::ios::binary);
        fout.write(bytes.data(), bytes.size());
        fout.close();
        #endif
    }

    #if 0
    for(auto iv : in_ivalues_) {
        auto bytes = torch::jit::pickle_save(iv.deepcopy().toTensor());
        std::ofstream fout("in.pt", std::ios::out | std::ios::binary);
        fout.write(bytes.data(), bytes.size());
        fout.close();
    }
    #endif

    return TNN_OK;
}

Status TNNTorchNetwork::ForwardAsync(Callback call_back) {
    return Forward();
}

Status TNNTorchNetwork::GetAllInputBlobs(BlobMap &blobs) {
    blobs = input_blob_map_;
    return TNN_OK;
}

Status TNNTorchNetwork::GetAllOutputBlobs(BlobMap &blobs) {
    blobs = output_blob_map_;
    return TNN_OK;
}

}  // namespace TNN_NS