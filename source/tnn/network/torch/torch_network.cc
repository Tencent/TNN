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
#include "tnn/core/macro.h"
#include "tnn/extern_wrapper/foreign_blob.h"
#include "tnn/extern_wrapper/foreign_tensor.h"
#include "tnn/interpreter/abstract_model_interpreter.h"
#include "tnn/interpreter/default_model_interpreter.h"
#include "tnn/network/torch/torch_utils.h"
#include "tnn/network/torch/torch_tensor.h"
#include "tnn/network/torch/torch_types.h"
#include "tnn/network/torch/partitioning.h"
#include "tnn/network/torch/torch_convert.h"
#include "tnn/network/torch/torch_compile.h"

#include <torch/torch.h>
#include <torch/csrc/jit/passes/freeze_module.h>
#include <torch/csrc/jit/passes/frozen_graph_optimizations.h>
#include "torch/csrc/jit/passes/lower_tuples.h"

#ifdef TNN_TORCHVISION
#include <torchvision/vision.h>
#endif

#include "tnn/utils/blob_dump_utils.h"

namespace TNN_NS {

NetworkImplFactoryRegister<NetworkImplFactory<TNNTorchNetwork>> g_network_impl_torch_factory_register(NETWORK_TYPE_TNNTORCH);

TNNTorchNetwork::~TNNTorchNetwork() {
    // delete output foreign_blobs 
    ClearOutputs();
    if (blob_manager_) {
        delete blob_manager_;
        blob_manager_ = nullptr;
    }
}

Status TNNTorchNetwork::Init(NetworkConfig &net_config, ModelConfig &model_config,
                             AbstractModelInterpreter *interpreter, InputShapesMap min_inputs_shape,
                             InputShapesMap max_inputs_shape, bool enable_const_folder) {
    config_    = net_config;
    Status ret = TNN_OK;

    if (config_.precision == PRECISION_LOW) {
        precision_ = DATA_TYPE_HALF;
    } else {
        precision_ = DATA_TYPE_FLOAT;
    }

    device_ = GetDevice(net_config.device_type);
    if (device_ == nullptr) {
        return TNNERR_DEVICE_NOT_SUPPORT;
    }
    context_ = device_->CreateContext(net_config.device_id);
    if (context_ == nullptr) {
        return TNNERR_DEVICE_CONTEXT_CREATE;
    }

    min_inputs_shape_ = min_inputs_shape;
    max_inputs_shape_ = max_inputs_shape;
    // if share net resource with another net, create io binding later when sharing
    if (net_config.share_memory_mode != SHARE_MEMORY_MODE_SHARE_NET_RESOURCE) {
        if (model_config.model_type == MODEL_TYPE_TORCHSCRIPT) {
            std::ifstream model_stream(model_config.params[0], std::ios::binary);
            if (model_config.params.size() > 1) {
                forward_func_name_ = model_config.params[1];
            }
            
            c10::Device device(c10::kCPU);
            RETURN_ON_NEQ(ConvertToTorchDevice(device, config_.device_type, config_.device_id), TNN_OK);
            auto mod = torch::jit::load(model_stream, device);
            module_ = CompileTorch(mod, max_inputs_shape_, config_, forward_func_name_);
            graph_ = module_.get_method(forward_func_name_).graph();
        } else {
            return Status(TNNERR_PARAM_ERR, "Unsupported model type for TNNTorchNetwork");
        }

        #if 0
        at::ArrayRef<torch::jit::Value*> inputs = graph_->block()->inputs();
        at::ArrayRef<torch::jit::Value*> outputs = graph_->block()->outputs();

        //printf("graph dump:\n:%s\n", graph_->toString().c_str());

        for(int i=0;i<inputs.size();i++) {
            auto input = inputs[i];
            printf("input[%d] from node [%s] of type:[%s]\n", input->unique(), input->node()->scopeName().c_str(), input->type()->annotation_str().c_str());
            printf("input[%d] typekind:[%s]\n", input->unique(), typeKindToString(input->type()->kind()));
        }

        for(int i=0;i<outputs.size();i++) {
            auto output = outputs[i];
            printf("output[%d] from node [%s] of type:[%s]\n", output->unique(), output->node()->scopeName().c_str(), output->type()->annotation_str().c_str());
            printf("output[%d] typekind:[%s]\n", output->unique(), typeKindToString(output->type()->kind()));
        }
        #endif

        RETURN_ON_FAIL(CreateIOBinding(min_inputs_shape, max_inputs_shape));
        init_done_ = true;
    }

    return TNN_OK;
}

Status TNNTorchNetwork::ShareNetResource(AbstractNetwork* network) {
    auto network_target = dynamic_cast<TNNTorchNetwork *>(network);
    if (!network_target) {
        return Status(TNNERR_PARAM_ERR, "network to share resource need to be TNNTorchNetwork");
    }

    module_ = network_target->GetModule();
    graph_ = network_target->GetGraph();

    RETURN_ON_FAIL(CreateIOBinding(min_inputs_shape_, max_inputs_shape_));
    init_done_ = true;
    return TNN_OK;
}

Status TNNTorchNetwork::CreateIOBinding(InputShapesMap  min_shape, InputShapesMap max_shape) {

    std::set<c10::TypeKind> supported_kinds = {
        c10::TypeKind::TensorType,
        c10::TypeKind::TupleType,
        c10::TypeKind::ListType,
        c10::TypeKind::DictType,
    };


    std::vector<torch::jit::Value*> inputs;

    int i=0;
    for(auto input : graph_->block()->inputs()) {
        c10::TypeKind kind = input->type()->kind();
        if (supported_kinds.find(kind) != supported_kinds.end()) {
            inputs.push_back(input);
        }
    }

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

    in_ivalues_.resize(inputs.size());

    // Create Ivalues from types
    for(int i=0;i<inputs.size();i++) {
        // printf("[%d] create ivalue from type:%s kind:%s\n", i, inputs[i]->type()->annotation_str().c_str(), c10::typeKindToString(inputs[i]->type()->kind()));
        RETURN_ON_FAIL(CreateIValueFromTypePtr(in_ivalues_[i], inputs[i]->type()));  
    }

    i = 0;
    for(auto input : fake_netstructure.blobs) {
        auto blob  = blob_manager_->GetBlob(input);
        auto foreign_blob = new ForeignBlob(blob->GetBlobDesc(), blob->GetHandle());

        blob_manager_->ReplaceBlob(input, foreign_blob);
        input_blob_map_[input] = foreign_blob;

        int id = JitTypeMatcher::idFromName(input);
        TNN_CHECK(id >= 0 && id < inputs.size(), "Invalid input id.");

        auto router = IValueRouter::create(inputs[id]->type(), input); 
        std::shared_ptr<at::Tensor> tensor;
        RETURN_ON_FAIL(CreateTensorByBlob(tensor, foreign_blob)); 
        RETURN_ON_FAIL(router->attach(in_ivalues_[id], tensor));

        #if 1
        // in_ivalues_[id].dump();
        // auto cpu_out = out_tensors[i].to(torch::kCPU);
        // auto bytes = torch::jit::pickle_save(ivalue.deepcopy().toTensor());
        // auto bytes = torch::jit::pickle_save(cpu_out);
        // std::ofstream fout("out.pt", std::ios::out | std::ios::binary);
        // fout.write(bytes.data(), bytes.size());
        // fout.close();
        #endif
    }

    // TODO Check integrity of the ivalues

    std::vector<torch::IValue> cur_inputs = in_ivalues_;
    // Convert to half here because blobConverter does not support half precision currently
    if (precision_ == DATA_TYPE_HALF) {
        for(int i=0;i<cur_inputs.size();i++) {
            auto ivalue = cur_inputs[i];
            RETURN_ON_FAIL(IValueTensorTo(ivalue, at::ScalarType::Half));
            cur_inputs[i] = ivalue;
        }
    }

    auto out = module_.forward(cur_inputs);

    // Convert to float here because blobConverter does not support half precision currently
    if (precision_ == DATA_TYPE_HALF) {
        // convert output to fp32
        RETURN_ON_FAIL(IValueTensorTo(out, at::ScalarType::Float));
    }

    std::vector<std::string> out_names;
    RETURN_ON_FAIL(IValueRouter::getAllTensorNames(out, "output_0", out_names));

    for(auto name : out_names) {
        auto router = IValueRouter::create(out.type(), name); 

        std::shared_ptr<at::Tensor> tensor;
        RETURN_ON_FAIL(router->route(out, tensor));
        TNN_CHECK(tensor != nullptr, "Got null tensor from IValueRouter");

        BlobDesc desc;
        RETURN_ON_FAIL(GetBlobDescFromTensor(desc, *tensor));

        BlobHandle handle;
        handle.base = tensor->data_ptr(); 

        auto foreign_blob =  new ForeignBlob(desc, handle);
        RETURN_ON_FAIL(foreign_blob->SetForeignTensor(std::make_shared<TorchTensor>(tensor, router)));
        output_blob_map_[name] = foreign_blob;    
    }

    return TNN_OK;
}

Status TNNTorchNetwork::Reshape(const InputShapesMap &inputs) {
    return TNN_OK;
}


Status TNNTorchNetwork::Forward() {
    // Blob converter may issue async converting, we need to wait for data ready.
    RETURN_ON_FAIL(context_->Synchronize());

    // TNN blob holds torch's output Tensor of previous forward round, so we can access it's data.
    // at this point, we don't need it, so release it to save memory.
    RETURN_ON_FAIL(ReleaseTorchOutputTensors());

    if (!init_done_) {
        return Status(TNNERR_INST_ERR, "Torch Network is not initialized");
    }

    std::vector<torch::IValue> cur_inputs = in_ivalues_;

    // Convert to half here because blobConverter does not support half precision currently
    if (precision_ == DATA_TYPE_HALF) {
        for(int i=0;i<cur_inputs.size();i++) {
            auto ivalue = cur_inputs[i];
            RETURN_ON_FAIL(IValueTensorTo(ivalue, at::ScalarType::Half));
            cur_inputs[i] = ivalue;
        }
    }

    auto out = module_.forward(cur_inputs);

    // Convert to float here because blobConverter does not support half precision currently
    if (precision_ == DATA_TYPE_HALF) {
        RETURN_ON_FAIL(IValueTensorTo(out, at::ScalarType::Float));
    }

    std::vector<std::string> out_names;
    RETURN_ON_FAIL(IValueRouter::getAllTensorNames(out, "output_0", out_names));

    // recreating output_blobs if output names not match
    if (mapKeysEqualTo(output_blob_map_, out_names)) {

        for(auto name : out_names) {
            auto blob =  output_blob_map_[name];
            IValueRouterPtr router;
            RETURN_ON_FAIL(GetIValueRouterFromBlob(blob, router));

            std::shared_ptr<at::Tensor> tensor;
            RETURN_ON_FAIL(router->route(out, tensor));

            TNN_CHECK(tensor != nullptr, "Got null tensor from IValueRouter");

            BlobDesc desc;
            RETURN_ON_FAIL(GetBlobDescFromTensor(desc, *tensor));

            BlobHandle handle;
            handle.base = tensor->data_ptr(); 

            blob->SetHandle(handle);
            blob->SetBlobDesc(desc);

            RETURN_ON_FAIL(SetTensorToBlob(blob, tensor));
        }
    } else {
        RETURN_ON_FAIL(ClearOutputs());

        for(auto name : out_names) {

            auto router = IValueRouter::create(out.type(), name); 

            std::shared_ptr<at::Tensor> tensor;
            RETURN_ON_FAIL(router->route(out, tensor));

            TNN_CHECK(tensor != nullptr, "Got null tensor from IValueRouter");

            BlobDesc desc;
            RETURN_ON_FAIL(GetBlobDescFromTensor(desc, *tensor));

            BlobHandle handle;
            handle.base = tensor->data_ptr(); 

            auto foreign_blob =  new ForeignBlob(desc, handle);
            RETURN_ON_FAIL(foreign_blob->SetForeignTensor(std::make_shared<TorchTensor>(tensor, router)));
            output_blob_map_[name] = foreign_blob;    
        }
    }
#if (DUMP_INPUT_BLOB || DUMP_OUTPUT_BLOB)
    RETURN_ON_FAIL(context_->Synchronize());
    DumpAllOutputBlob();
#endif

    return TNN_OK;
}

Status TNNTorchNetwork::ClearOutputs() {
    auto it = output_blob_map_.begin();
    while(it != output_blob_map_.end()) {
        delete it->second;
        it = output_blob_map_.erase(it);
    }
    return TNN_OK;
}


Status TNNTorchNetwork::ReleaseTorchOutputTensors() {
    auto it = output_blob_map_.begin();
    while(it != output_blob_map_.end()) {
        auto blob = it->second;
        std::shared_ptr<at::Tensor> empty_tensor;
        RETURN_ON_FAIL(SetTensorToBlob(blob, empty_tensor));
        it++;
    }

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

Status TNNTorchNetwork::DumpAllOutputBlob() {
    Status ret;
    for(auto output : output_blob_map_) {
        ret = DumpDeviceBlob(output.second, context_, "torch_" + output.first);
        if(ret != TNN_OK) {
            LOGE("DumpDeviceBlob failed error code: %d, msg: %s \n", (int)ret, ret.description().c_str());
        }
    }
    return TNN_OK;
}

}  // namespace TNN_NS
