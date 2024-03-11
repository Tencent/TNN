// Tencent is pleased to support the open source community by making TNN available.
//
// Copyright (C) 2024 THL A29 Limited, a Tencent company. All rights reserved.
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

#include <time.h>
#include <chrono>
#include "tnn/device/atlas/torchaie/torchaie_network.h"
#include "tnn/device/atlas/torchaie/torchaie_interpreter.h"
#include "tnn/device/atlas/torchaie/torchaie_utils.h"
#include "tnn/utils/dims_vector_utils.h"
#include "tnn/utils/dims_vector_utils.h"
#include "torch_aie.h"

namespace TNN_NS {

NetworkImplFactoryRegister<NetworkImplFactory<TorchAieNetwork>> g_network_impl_torchaie_factory_register(NETWORK_TYPE_TORCHAIE);

TorchAieNetwork::~TorchAieNetwork() {
    if (aie_module_.get() != nullptr) {
        torch_aie::finalize();
    }
}

Status TorchAieNetwork::Init(NetworkConfig &net_config, ModelConfig &model_config, AbstractModelInterpreter *interpreter,
                             InputShapesMap min_inputs_shape, InputShapesMap max_inputs_shape,
                             InputDataTypeMap inputs_data_type, bool enable_const_folder) {

    Status status = TNN_OK;
    network_config_ = net_config;
    TorchAieInterpreter* aie_interpreter = reinterpret_cast<TorchAieInterpreter*>(interpreter);

    if (min_inputs_shape.size() > 1) {
        LOGE("torchaie only support one input now\n");
        return TNNERR_INVALID_INPUT;
    }

    try {
        torch_aie::set_device(network_config_.device_id);

        if (network_config_.extra.find("need_compile") != network_config_.extra.end()) {
            auto mod = aie_interpreter->GetModule(true);
            aie_module_ = Compile(min_inputs_shape, max_inputs_shape, inputs_data_type, *(mod.get()));
        } else {
            aie_module_ = aie_interpreter->GetModule(false);
        }
    } catch(const std::exception& e) {
        status = TNNERR_INVALID_MODEL;
        LOGE("invalid torchaie model\n");
    }

    if (status == TNN_OK) {
        status = UpdateInputBlobMap(max_inputs_shape, inputs_data_type);
    }

    if (status == TNN_OK) {
        status = UpdateOutputBlobMap();
    }

    return status;
}

Status TorchAieNetwork::Forward() {
    LOGD("TorchAie Forward!\n");
    
    try {
        auto aie_results = aie_module_->forward({ input_tensors_.begin()->second }).toTuple()->elements();

	for (int i = 0; i < aie_results.size(); i++) {
            //printf("DEBUG: %s\n", c10::typeKindToString(aie_results[i].type()->kind()));
            auto tensor = aie_results[i].toTensor().contiguous().to("cpu");
            DimsVector dims;

            for (auto i : tensor.sizes()) {
                dims.push_back(i);
            }

            output_tensors_[output_names_[i]] = tensor;
            output_blob_map_[output_names_[i]]->GetBlobDesc().dims = dims;
	}
    } catch(const std::exception& e) {
        std::cout << e.what() << '\n';
        return TNNERR_NO_RESULT;
    }

    return TNN_OK;
}

Status TorchAieNetwork::ForwardAsync(Callback call_back) {
    LOGD("TorchAie Async Forward! (as same as Forward by now)\n");
    return Forward();
}

std::shared_ptr<torch::jit::Module> TorchAieNetwork::Compile(
    InputShapesMap &min_inputs_shape, InputShapesMap &max_inputs_shape,
    InputDataTypeMap &inputs_data_type, torch::jit::Module &mod) {

    Status status = TNN_OK;

    if (min_inputs_shape.size() != max_inputs_shape.size()) {
        return nullptr;
    }

    int input_count = max_inputs_shape.size();
    std::vector<torch_aie::Input> inputs;
    std::shared_ptr<torch::jit::Module> aie_module = nullptr;
    std::vector<std::string> input_names;

    for (auto iter : min_inputs_shape) {
        // std::string key = "input_" + std::to_string(i);
        std::string key = iter.first;
        std::string format_key = key + "_format";
        std::vector<int64_t> min_shapes;
        std::vector<int64_t> max_shapes;

        for (auto val : min_inputs_shape[key]) {
            min_shapes.push_back(static_cast<int64_t>(val));
        }
        for (auto val : max_inputs_shape[key]) {
	    max_shapes.push_back(static_cast<int64_t>(val));
        }

        auto data_type = torch_aie::DataType::FLOAT16;
        auto format = torch_aie::TensorFormat::NCHW;

        if (inputs_data_type.find(key) != inputs_data_type.end()) {
            data_type = AieDataTypeConverter(inputs_data_type[key]);
        }
        if (network_config_.extra.find(format_key) != network_config_.extra.end()) {
            format = AieTensorFormatConverter(network_config_.extra[format_key]);
        }

        if (min_shapes == max_shapes) {
            inputs.emplace_back(torch_aie::Input(max_shapes, data_type, format));
        } else {
            inputs.emplace_back(torch_aie::Input(min_shapes, max_shapes, data_type, format));
        }
    }

    if (status == TNN_OK) {
        torch_aie::torchscript::CompileSpec compile_spec(inputs);

        if (network_config_.extra.find("allow_tensor_replace_int") != network_config_.extra.end()) {
            compile_spec.allow_tensor_replace_int = true;
        }
        if (network_config_.extra.find("require_full_compilation") != network_config_.extra.end()) {
            compile_spec.require_full_compilation = true;
        }
        if (network_config_.extra.find("min_block_size") != network_config_.extra.end()) {
            compile_spec.min_block_size = std::stoi(network_config_.extra["min_block_size"]);
        }
        compile_spec.torch_executed_ops = { };

        aie_module = std::make_shared<torch::jit::Module>(torch_aie::torchscript::compile(mod, compile_spec));

	if (network_config_.extra.find("save_path") != network_config_.extra.end()) {
	    const std::string save_path = network_config_.extra["save_path"];
	    aie_module->save(save_path);
	    LOGD("torchaie model is saved\n");
	}
    }

    return aie_module;
}

Status TorchAieNetwork::UpdateInputBlobMap(InputShapesMap &inputs_shape, InputDataTypeMap &inputs_data_type) {
    Status status = TNN_OK;

    auto g = aie_module_->get_method("forward").graph();

    // for (auto &output : g->inputs()) {
    //     std::cout << "DEBUG input name " << output->debugName() << std::endl;
    //     printf("DEBUG: %s\n", c10::typeKindToString(output->type()->kind()));
    // }

    for (auto iter : inputs_shape) {
        std::string name = iter.first;
        input_names_.push_back(name);

        BlobDesc blob_desc;
        blob_desc.device_type = DEVICE_ATLAS;
        blob_desc.dims = inputs_shape[name];
        blob_desc.name = name;

        if (inputs_data_type.find(name) != inputs_data_type.end()) {
            blob_desc.data_type = inputs_data_type[name];
        } else {
            blob_desc.data_type = DATA_TYPE_HALF;
        }

        BlobHandle blob_handle;
        Blob *blob = new Blob(blob_desc, blob_handle);

        if (blob) {
            input_blob_map_[name] = blob;
        } else {
            status = TNNERR_OUTOFMEMORY;
            break;
        }
    }

    return status;
}

Status TorchAieNetwork::UpdateOutputBlobMap() {
    Status status = TNN_OK;
    auto g = aie_module_->get_method("forward").graph();
    auto outputs = g->outputs();

    if (outputs.size() > 1 || outputs[0]->type()->kind() != c10::TypeKind::TupleType) {
        LOGE("invalid torchaie model, output should be a tuple\n");
        return TNNERR_INVALID_MODEL;
    }

    torch::jit::Node* output_node = nullptr;

    for (auto node : g->nodes()) {
        if (node->outputs().size() > 0 && node->outputs()[0] == outputs[0]) {
            output_node = node;
            break;
        }
    }

    if (output_node == nullptr) {
        LOGE("invalid torchaie model, output node is not found\n");
        return TNNERR_INVALID_MODEL;
    }

    for (auto &output : output_node->inputs()) {
        std::string name = output->debugName();

        BlobDesc blob_desc;
        blob_desc.device_type = DEVICE_ATLAS;
        blob_desc.data_type = DATA_TYPE_HALF;
	blob_desc.name = name;

        BlobHandle blob_handle;
        Blob *blob = new Blob(blob_desc, blob_handle);

        if (blob) {
            output_blob_map_[name] = blob;
        } else {
            status = TNNERR_OUTOFMEMORY;
            break;
        }
        output_names_.push_back(name);
    }

    return status;
}

Status TorchAieNetwork::GetAllInputBlobs(BlobMap &blobs) {
    blobs = input_blob_map_;
    return TNN_OK;
}

Status TorchAieNetwork::GetAllOutputBlobs(BlobMap &blobs) {
    blobs = output_blob_map_;
    return TNN_OK;
}

Status TorchAieNetwork::GetCommandQueue(void **command_queue) {
    // torchaie use network as command queue.
    *command_queue = static_cast<void*>(this);
    return TNN_OK;
}

Status TorchAieNetwork::Reshape(const InputShapesMap &inputs) {
    for (auto iter : inputs) {
        input_blob_map_[iter.first]->GetBlobDesc().dims = iter.second;
    }
    return TNN_OK;
}

Status TorchAieNetwork::SetInputTensor(at::Tensor tensor, std::string name) {
    std::string npu_str = "npu:" + std::to_string(network_config_.device_id);
    input_tensors_[name] = tensor.toType(torch::kFloat16).to(npu_str.c_str());
    return TNN_OK;
}

at::Tensor TorchAieNetwork::GetOutputTensor(std::string name) {
    at::Tensor tensor;
    if (output_tensors_.find(name) != output_tensors_.end()) {
        return output_tensors_[name].to("cpu");
    } else {
        LOGE("TorchAie get invalid output tensor!\n");
    }
    return tensor;
}

Status TorchAieNetwork::DeInit() {
    return TNN_OK;
}

Status TorchAieNetwork::GetForwardMemorySize(size_t &memory_size) {
    return TNN_OK;
}

Status TorchAieNetwork::SetForwardMemory(void *memory) {
    return TNN_OK;
}

}  // namespace TNN_NS
