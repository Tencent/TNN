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

#include "npu_network.h"

#include <tnn/device/huawei_npu/convert/npu_base_layer_convert.h>
#include <tnn/interpreter/layer_resource_generator.h>

#include "HiAiModelManagerService.h"
#include "graph/model.h"
#include "graph/op/array_defs.h"
#include "hiai_ir_build.h"
#include "tnn/core/abstract_device.h"
#include "tnn/device/huawei_npu/convert/npu_utils.h"
#include "tnn/interpreter/default_model_interpreter.h"
#include "tnn/optimizer/net_optimizer_manager.h"
#include "tnn/utils/data_format_converter.h"

namespace TNN_NS {

NetworkImplFactoryRegister<NetworkImplFactory<NpuNetwork>> g_network_impl_npu_factory_register(NETWORK_TYPE_HUAWEI_NPU);

NpuNetwork::NpuNetwork() {
    model_name_ = "";
    client_     = nullptr;
    input_tensor_.clear();
    output_tensor_.clear();
}

NpuNetwork::~NpuNetwork() {
    DeInit();
}

Status NpuNetwork::InitCheck() {
    // Start to load HiAi API
    if (client_ == nullptr) {
        return Status(TNNERR_NPU_HIAI_API_ERROR, "ERROR: HiaiDDK API load error, check ddk");
    }

    // init Ai Model Manager Client
    hiai::AIStatus ret = client_->Init(nullptr);
    if (ret != hiai::AI_SUCCESS) {
        return Status(TNNERR_NPU_LOAD_ERROR, "ERROR: huawei_npu is not installed");
    }
    // get rom version
    const char *version = client_->GetVersion();
    if (version == nullptr) {
        return Status(TNNERR_NPU_LOAD_ERROR,
                      "ERROR: GetRomVersion(ROM): huawei_npu is not installed or rom version is too low");
    }
    // check if NPU version is greater than 300
    version_num_ = NpuUtils::checkNpuVersion(version);
    LOGI("[TNN/NPU]ddk current version: %s", version);
    if (version_num_ < 320) {
        return Status(TNNERR_NPU_LOAD_ERROR, "ERROR: huawei_npu is installed but is below 100.320.xxx.xxx");
    }
    return TNN_OK;
}
Status NpuNetwork::Init(NetworkConfig &net_config, ModelConfig &model_config, AbstractModelInterpreter *interpreter,
                        InputShapesMap inputs_shape) {
    if (net_config.device_type != DEVICE_HUAWEI_NPU || model_config.model_type != MODEL_TYPE_TNN) {
        return Status(TNNERR_NULL_PARAM, "ERROR: Npu not support device_type or model type");
    }
    client_         = std::make_shared<hiai::AiModelMngerClient>();
    Status init_ret = InitCheck();
    if (init_ret != TNN_OK) {
        return init_ret;
    }

    auto *default_interpreter = dynamic_cast<DefaultModelInterpreter *>(interpreter);
    net_structure_            = default_interpreter->GetNetStructure();
    model_name_               = NpuUtils::GetFileHash(model_config);
    std::vector<std::shared_ptr<hiai::AiModelDescription>> model_desc;
    InputShapesMap instance_input_shapes_map = net_structure_->inputs_shape_map;
    InputShapesMap cpu_input_shape;

    // the path that is used to store/find om
    std::string path_to_om = "";
    // check if store the om file
    from_path_ = (net_config.cache_path.compare("") != 0);
    path_to_om = from_path_? net_config.cache_path + "/" : "";

    // modify the inputMap and add a suffix to the model name to create a new model
    std::string model_suffix = NpuUtils::modifyModelInputSize(inputs_shape, instance_input_shapes_map);
    model_name_              = model_name_ + model_suffix + "_" + std::to_string(version_num_);

    std::string model_path = path_to_om + model_name_ + ".om";
    LOGI("[TNN/NPU]The path %s\n", model_path.c_str());
    auto model_builder = std::make_shared<hiai::AiModelBuilder>(client_);

    if (from_path_ && NpuUtils::FileExits(model_path)) {
        LOGI("[TNN/NPU]The om file already exists in %s\n", path_to_om.c_str());
    } else {
        // NPU IR Build
        Status build_ret = IRInitLayers(net_config, interpreter, instance_input_shapes_map);
        if (build_ret != TNN_OK) {
            LOGI("[TNN/NPU] Some layers not support in NPU, switch to ARM\n");
            if (cpu_count_ != net_structure_->layers.size()) {
                // from here load cpu
                default_network_             = std::make_shared<DefaultNetwork>();
                NetworkConfig cpu_net_config = net_config;
                cpu_net_config.device_type   = DEVICE_ARM;
                cpu_net_config.network_type  = NETWORK_TYPE_DEFAULT;

                cpu_input_shape = modifyInterpreterCPU();
                if (cpu_input_shape.empty()) {
                    LOGE("ERROR: When split the network,  the arm can not find input in the huawei_npu visited layers\n");
                    return Status(TNNERR_LAYER_ERR, "ERROR: When split the network,  the arm can not find input in the huawei_npu visited layers");
                }
                net_structure_->inputs_shape_map = cpu_input_shape;
                build_ret = default_network_->Init(cpu_net_config, model_config, interpreter, cpu_input_shape);
                if (build_ret != TNN_OK) {
                    return build_ret;
                } else {
                    useCPU = true;
                }
            }
        }
        // Set Graph
        SetGraphInputsAndOutputs(instance_input_shapes_map, cpu_input_shape);
        // Build Graph
        if (!useCPU) {
            build_ret = from_path_ ? BuildModel(model_path) : TNN_OK;
            if (build_ret != TNN_OK) {
                return build_ret;
            }
        } else {
            from_path_ = false;
        }
    }
    hiai::MemBuffer *model_mem_buffer = nullptr;
    // From here, start load
    domi::HiaiIrBuild ir_build;
    domi::ModelBufferData build_mem_buff;
    if (from_path_) {
        model_mem_buffer = model_builder->InputMemBufferCreate(model_path);
    } else {
        ge::Model model(model_name_, model_name_ + "_v1");
        model.SetGraph(graph_);
        bool build_ret = ir_build.CreateModelBuff(model, build_mem_buff);
        domi::BuildOptions options;
        options.useOriginFormat = true;
        if (!build_ret) {
            return Status(TNNERR_NPU_HIAI_API_ERROR, "HIAI build model, CreateModelBuff() failed");
        }
        build_ret = ir_build.BuildIRModel(model, build_mem_buff, options);
        if (!build_ret) {
            return Status(TNNERR_NPU_HIAI_API_ERROR, "HIAI build model, BuildIRModel() failed");
        }
        model_mem_buffer = model_builder->InputMemBufferCreate(build_mem_buff.data, build_mem_buff.length);
    }
    if (model_mem_buffer == nullptr) {
        return Status(TNNERR_NPU_HIAI_API_ERROR, "ERROR: function InputMemBufferCreate() failed");
    }

    std::shared_ptr<hiai::AiModelDescription> desc = std::make_shared<hiai::AiModelDescription>(
        model_name_ + ".om", hiai::AiModelDescription_Frequency_HIGH, hiai::HIAI_FRAMEWORK_NONE,
        hiai::HIAI_MODELTYPE_ONLINE, hiai::AiModelDescription_DeviceType_NPU);

    desc->SetModelBuffer(model_mem_buffer->GetMemBufferData(), model_mem_buffer->GetMemBufferSize());
    // only load one model
    model_desc.push_back(desc);
    // load model
    hiai::AIStatus ret = client_->Load(model_desc);
    if (ret != hiai::AI_SUCCESS) {
        return Status(TNNERR_NPU_HIAI_API_ERROR, "ERROR: Load model Load() failed");
    }

    // check model
    bool isModelCompatibility = true;
    ret                       = client_->CheckModelCompatibility(*desc, isModelCompatibility);
    LOGI("[TNN/NPU] isModelCompatibility: %s", isModelCompatibility ? "true" : "false");
    LOGI("[TNN/NPU] ret value %d", ret);
    if (ret != hiai::AI_SUCCESS) {
        return Status(TNNERR_NPU_HIAI_API_ERROR, "ERROR: check model CheckModelCompatibility() failed");
    }

    // destroy unused memory
    model_builder->MemBufferDestroy(model_mem_buffer);
    ir_build.ReleaseModelBuff(build_mem_buff);

    input_tensor_.clear();
    output_tensor_.clear();
    std::vector<hiai::TensorDimension> input_dims;
    std::vector<hiai::TensorDimension> output_dims;
    ret = client_->GetModelIOTensorDim(model_name_ + ".om", input_dims, output_dims);
    if (ret != hiai::AI_SUCCESS) {
        return Status(TNNERR_NPU_HIAI_API_ERROR, "ERROR: function GetModelIOTensorDim() failed");
    }
    if (input_dims.size() == 0) {
        return Status(TNNERR_MODEL_ERR, "ERROR: Npu the model input_dims.size() == 0");
    }

    for (auto dim : input_dims) {
        std::shared_ptr<hiai::AiTensor> input = std::make_shared<hiai::AiTensor>();
        ret                                   = input->Init(&dim);

        if (ret != hiai::AI_SUCCESS) {
            return Status(TNNERR_NPU_HIAI_API_ERROR, "ERROR:Get input tensor from loaded model failed");
        }
        input_tensor_.push_back(input);
    }

    for (auto dim : output_dims) {
        std::shared_ptr<hiai::AiTensor> output = std::make_shared<hiai::AiTensor>();
        ret                                    = output->Init(&dim);
        if (ret != hiai::AI_SUCCESS) {
            return Status(TNNERR_NPU_HIAI_API_ERROR, "ERROR:Get output tensor from loaded model failed");
        }
        output_tensor_.push_back(output);
    }

    auto input_it = instance_input_shapes_map.begin();
    // init input buffers
    for (int i = 0; i < input_tensor_.size(); ++i) {
        hiai::TensorDimension dims = input_dims[i];
        int n                      = dims.GetNumber();
        int c                      = dims.GetChannel();
        int h                      = dims.GetHeight();
        int w                      = dims.GetWidth();
        // add blob
        std::string name = input_it->first;
        char layer_name[name.size() + 1];
        strcpy(layer_name, name.c_str());
        BlobDesc desc;
        desc.data_format = DATA_FORMAT_NCHW;
        desc.name        = layer_name;
        desc.dims.push_back(n);
        desc.dims.push_back(c);
        desc.dims.push_back(h);
        desc.dims.push_back(w);
        BlobHandle handle;
        handle.base                = input_tensor_[i]->GetBuffer();
        input_blob_map_[desc.name] = new Blob(desc, handle);
        input_it++;
    }
    // init output buffers
    //init output iterator through the map
    auto output_it = net_structure_->outputs.begin();
    auto end_it = net_structure_->outputs.end();

    std::set<std::string> npu_inter_outputs;
    if (useCPU) {
        for (auto i = cpu_input_shape.begin(); i != cpu_input_shape.end(); i++) {
            npu_inter_outputs.insert(i->first);
        }
        output_it = npu_inter_outputs.begin();
        end_it = npu_inter_outputs.end();

        default_network_->GetAllInputBlobs(cpu_inter_in_blobmap_);
        default_network_->GetAllOutputBlobs(output_blob_map_);
    }
    int count = 0;
    for (; output_it != end_it ; ++output_it) {
        std::string name = *output_it;
        BlobDesc desc;
        BlobHandle handle;
        char layer_name[name.size() + 1];
        strcpy(layer_name, name.c_str());

        if (input_blob_map_.count(name) != 0) {
            //if the input is the output, then use the input tensor
            desc = input_blob_map_[name]->GetBlobDesc();
            handle = input_blob_map_[name]->GetHandle();
        } else {
            hiai::TensorDimension dims = output_dims[count];
            int n                      = dims.GetNumber();
            int c                      = dims.GetChannel();
            int h                      = dims.GetHeight();
            int w                      = dims.GetWidth();
            // add blob
            desc.data_format = DATA_FORMAT_NCHW;
            desc.name        = layer_name;
            desc.dims.push_back(n);
            desc.dims.push_back(c);
            desc.dims.push_back(h);
            desc.dims.push_back(w);
            handle.base = output_tensor_[count]->GetBuffer();
            count++;
        }
        if (useCPU) {
            npu_inter_out_blobmap_[desc.name] = new Blob(desc, handle);
        } else {
            output_blob_map_[desc.name] = new Blob(desc, handle);
        }
    }
    return TNN_OK;
}

Status NpuNetwork::IRInitLayers(NetworkConfig &net_config, AbstractModelInterpreter *interpreter,
                                InputShapesMap &inputs_shape) {
    Status ret                = TNN_OK;
    auto *default_interpreter = dynamic_cast<DefaultModelInterpreter *>(interpreter);
    NetResource *net_resource = default_interpreter->GetNetResource();

    if (net_structure_ == NULL || net_resource == NULL) {
        return Status(TNNERR_NULL_PARAM, "ERROR: network_ is nil, network_type may not support");
    }

    device_ = GetDevice(net_config.device_type);
    if (device_ == NULL) {
        return TNNERR_DEVICE_NOT_SUPPORT;
    }
    context_ = device_->CreateContext(net_config.device_id);
    if (context_ == NULL) {
        return TNNERR_DEVICE_CONTEXT_CREATE;
    }

    ret = context_->LoadLibrary(net_config.library_path);
    if (ret != TNN_OK) {
        return ret;
    }

    ret = optimizer::NetOptimizerManager::Optimize(net_structure_, net_resource, net_config.device_type);
    if (ret != TNN_OK) {
        return ret;
    }

    // Create input operators
    ret = CreateGraphInputs(inputs_shape);
    if (ret != TNN_OK) {
        return ret;
    }
    // Init layers
    ret = ConvertLayers(net_resource);
    if (ret != TNN_OK) {
        return ret;
    }
    return TNN_OK;
}

InputShapesMap NpuNetwork::modifyInterpreterCPU() {
    std::vector<shared_ptr<LayerInfo>> layers;
    // only view input
    InputShapesMap cpu_input_shapes_map;
    for (int i = cpu_count_; i < net_structure_->layers.size(); i++) {
        std::shared_ptr<LayerInfo> &layer_info = net_structure_->layers[i];
        for (std::string &input : layer_info->inputs) {
            // if the cpu input exists in visited
            if (visited_.count(input) > 0) {
                if (cpu_input_shapes_map.count(input) == 0) {
                    cpu_input_shapes_map[input] = global_operator_map_[input]->GetShape();
                }
            }
        }
        layers.push_back(layer_info);
    }
    net_structure_->layers = layers;
    return cpu_input_shapes_map;
}

Status NpuNetwork::CreateGraphInputs(InputShapesMap &input_shape_map) {
    Status ret = TNN_OK;
    // init graph input
    auto iterator = input_shape_map.begin();
    for (; iterator != input_shape_map.end(); iterator++) {
        shared_ptr<ge::op::Data> input_data;
        std::string input_name           = iterator->first;
        DimsVector dims_vector           = iterator->second;
        ret                              = NpuUtils::CreateInputData(input_data, input_name, dims_vector);
        auto input_op                    = std::make_shared<OperatorInfo>(input_data, dims_vector);
        global_operator_map_[input_name] = input_op;
        visited_.insert(input_name);
    }
    return ret;
}

Status NpuNetwork::SetGraphInputsAndOutputs(InputShapesMap &input_shape_map, InputShapesMap &cpu_input_shape_map) {
    // init graph input
    std::vector<ge::Operator> input_ops;
    std::vector<ge::Operator> output_ops;
    auto iterator = input_shape_map.begin();
    for (; iterator != input_shape_map.end(); iterator++) {
        std::string input_name = iterator->first;
        input_ops.push_back(*global_operator_map_[input_name]->GetOperator());
    }
    // init graph output
    if (!useCPU) {
        for (auto &name : net_structure_->outputs) {
            if (input_shape_map.count(name) == 0) {
                output_ops.push_back(*global_operator_map_[name]->GetOperator());
            }
        }
    } else {
        auto iterator = cpu_input_shape_map.begin();
        for (; iterator != cpu_input_shape_map.end(); iterator++) {
            if (input_shape_map.count(iterator->first) == 0) {
                if (global_operator_map_[iterator->first] != nullptr) {
                    output_ops.push_back(*global_operator_map_[iterator->first]->GetOperator());
                } else {
                    return Status(TNNERR_LAYER_ERR, "ERROR: When init the cpu network, some input not found\n");
                }

            }
        }
    }
    graph_.SetInputs(input_ops).SetOutputs(output_ops);
    return TNN_OK;
}

Status NpuNetwork::BuildModel(std::string model_path) {
    // Set model parameters : model name and  model name + version
    ge::Model model(model_name_, model_name_ + "_v1");
    model.SetGraph(graph_);
    // Build the ir model
    domi::HiaiIrBuild ir_build;
    domi::ModelBufferData om_model_buff;
    domi::BuildOptions options;
    options.useOriginFormat = true;
    bool build_ret          = ir_build.CreateModelBuff(model, om_model_buff);
    if (!build_ret) {
        return Status(TNNERR_NPU_HIAI_API_ERROR, "HIAI build model, CreateModelBuff failed");
    }
    build_ret = ir_build.BuildIRModel(model, om_model_buff, options);
    if (!build_ret) {
        return Status(TNNERR_NPU_HIAI_API_ERROR, "HIAI build model, BuildIRModel() failed");
    }

    Status ret = NpuUtils::WriteModelFile(om_model_buff, model_path);
    if (ret != TNN_OK) {
        return ret;
    }
    ir_build.ReleaseModelBuff(om_model_buff);
    return TNN_OK;
}

Status NpuNetwork::ConvertLayers(NetResource *net_resource) {
    Status ret = TNN_OK;
    // loop net_structure
    cpu_count_ = 0;
    for (auto layer_info : net_structure_->layers) {
        LayerType type          = layer_info->type;
        NpuBaseLayer *cur_layer = CreateNpuBaseLayer(type);
        if (cur_layer == nullptr) {
            LOGE("Error Init layer tyep %d, huawei_npu does not support, may switch to arm\n", layer_info->type);
            return Status(TNNERR_LAYER_ERR, "CreateLayer failed");
        }
        std::string layer_name = layer_info->name;
        cur_layer->SetLayerName(layer_name);

        // set layer nodes
        std::vector<std::shared_ptr<OperatorInfo>> input_ops;
#ifdef BENCHMARK
        std::vector<Blob *> input_blobs;
        BlobDesc blob_desc;
#endif
        for (std::string &name : layer_info->inputs) {
            input_ops.push_back(global_operator_map_[name]);
#ifdef BENCHMARK
            blob_desc.dims = global_operator_map_[name]->GetShape();
            Blob *blob     = new Blob(blob_desc);
            input_blobs.push_back(blob);
#endif
        }
#ifdef BENCHMARK
        // generate resource if null
        if (net_resource->resource_map.count(layer_name) == 0) {
            LayerParam *layer_param  = layer_info->param.get();
            LayerResource *layer_res = nullptr;
            GenerateRandomResource(type, layer_param, &layer_res, input_blobs);
            net_resource->resource_map[layer_name] = std::shared_ptr<LayerResource>(layer_res);
        }
        for (auto &blob : input_blobs) {
            delete (blob);
        }
#endif
        LayerResource *layer_resource = net_resource->resource_map[layer_name].get();
        /*
         * cur_layer->convert
         */
        ret =
            cur_layer->Init(context_, layer_info->param.get(), layer_resource, input_ops, device_, layer_info->outputs);
        layers_.push_back(cur_layer);
        if (ret != TNN_OK) {
            LOGE("Error Init layer %s (err: %d or 0x%X)\n", cur_layer->GetLayerName().c_str(), (int)ret, (int)ret);
            return ret;
        }

        for (auto &op : cur_layer->GetOutputOps()) {
            visited_.insert(op->GetOperator()->GetName());
            global_operator_map_[op->GetOperator()->GetName()] = op;
        }
        cpu_count_++;
    }
    return ret;
}

Status NpuNetwork::GetForwardMemorySize(int &memory_size) {
    memory_size = 0;
    return TNN_OK;
}

Status NpuNetwork::SetForwardMemory(void *memory) {
    return TNN_OK;
}

Status NpuNetwork::GetAllInputBlobs(BlobMap &blobs) {
    blobs = input_blob_map_;
    return TNN_OK;
}

Status NpuNetwork::GetAllOutputBlobs(BlobMap &blobs) {
    blobs = output_blob_map_;
    return TNN_OK;
}

Status NpuNetwork::SetDeviceAffinity(const std::vector<int> &) {
    return TNNERR_NPU_UNSUPPORT_ERROR;
}

Status NpuNetwork::Reshape(const InputShapesMap &inputs) {
    return TNNERR_NPU_UNSUPPORT_ERROR;
}

Status NpuNetwork::DeInit() {
    client_->UnLoadModel();
    auto iterator = input_blob_map_.begin();
    while (iterator != input_blob_map_.end()) {
        if (iterator->second != nullptr) {
            delete (iterator->second);
            iterator->second = nullptr;
        }
        iterator++;
    }
    input_blob_map_.clear();

    if (!useCPU) {
        iterator = output_blob_map_.begin();
        while (iterator != output_blob_map_.end()) {
            if (iterator->second != nullptr) {
                delete (iterator->second);
                iterator->second = nullptr;
            }
            iterator++;
        }
        output_blob_map_.clear();
    } else {
        iterator = npu_inter_out_blobmap_.begin();
        while (iterator != npu_inter_out_blobmap_.end()) {
            if (iterator->second != nullptr) {
                delete (iterator->second);
                iterator->second = nullptr;
            }
            iterator++;
        }
        npu_inter_out_blobmap_.clear();
    }

    for (auto &layer : layers_) {
        delete (layer);
    }
    layers_.clear();

    if (context_ != nullptr) {
        delete context_;
        context_ = nullptr;
    }
    return TNN_OK;
}

Status NpuNetwork::GetCommandQueue(void **command_queue) {
    return TNN_OK;
}

Status NpuNetwork::Forward() {
    hiai::AiContext context;
    std::string key   = "model_name";
    std::string value = model_name_ + ".om";
    context.AddPara(key, value);
    int istamp;
    std::chrono::time_point<std::chrono::system_clock> start = std::chrono::system_clock::now();
    hiai::AIStatus ret = client_->Process(context, input_tensor_, output_tensor_, 1000, istamp);
    if (ret != hiai::AI_SUCCESS) {
        return Status(TNNERR_NPU_HIAI_API_ERROR, "Forward failed!");
    }
    if (useCPU) {
        start = std::chrono::system_clock::now();
        for (auto iterator = npu_inter_out_blobmap_.begin(); iterator != npu_inter_out_blobmap_.end(); iterator++) {
            std::string name = iterator->first;
            Blob *npu_blob   = iterator->second;
            Blob *cpu_blob   = cpu_inter_in_blobmap_[name];
            int num          = npu_blob->GetBlobDesc().dims[0];
            int channel      = npu_blob->GetBlobDesc().dims[1];
            int height       = npu_blob->GetBlobDesc().dims[2];
            int width        = npu_blob->GetBlobDesc().dims[3];
            float *src       = reinterpret_cast<float *>(npu_blob->GetHandle().base);
            float *dst       = reinterpret_cast<float *>(reinterpret_cast<char *>(cpu_blob->GetHandle().base) +
                                                   cpu_blob->GetHandle().bytes_offset);
            DataFormatConverter::ConvertFromNCHWToNCHW4Float(src, dst, num, channel, height, width);
        }
        default_network_->Forward();
    }
    return TNN_OK;
}

Status NpuNetwork::ForwardAsync(Callback call_back) {
    return NpuNetwork::Forward();
}

}  // namespace TNN_NS
