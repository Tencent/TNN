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

#include <sys/time.h>
#include <tnn/device/huawei_npu/convert/npu_base_layer_convert.h>
#include <tnn/interpreter/layer_resource_generator.h>

#include "HiAiModelManagerService.h"
#include "graph/model.h"
#include "hiai_ir_build.h"
#include "tnn/core/abstract_device.h"
#include "tnn/device/huawei_npu/convert/npu_utils.h"
#include "tnn/interpreter/default_model_interpreter.h"
#include "tnn/interpreter/tnn/model_interpreter.h"
#include "tnn/optimizer/net_optimizer_manager.h"
#include "tnn/utils/data_format_converter.h"
#include "tnn/utils/npu_common_utils.h"

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

Status NpuNetwork::Init(NetworkConfig &net_config, ModelConfig &model_config, AbstractModelInterpreter *interpreter,
                        InputShapesMap min_inputs_shape, InputShapesMap max_inputs_shape) {
    // config check
    if (InitConfigCheck(net_config, model_config)) {
        return Status(TNNERR_NULL_PARAM, "ERROR: Npu not support device_type or model type");
    }

    // rom version check
    client_        = std::make_shared<hiai::AiModelMngerClient>();
    Status tnn_ret = RomVersionCheck();
    if (tnn_ret != TNN_OK) {
        return tnn_ret;
    }

    // create context
    tnn_ret = InitContext(net_config);
    if (tnn_ret != TNN_OK) {
        return tnn_ret;
    }

    // get interpreter
    auto *default_interpreter = dynamic_cast<DefaultModelInterpreter *>(interpreter);
    net_structure_            = default_interpreter->GetNetStructure();

    // check if store the om file
    use_path_ = (net_config.cache_path.compare("") != 0);

    // modify the inputShapeMap. if reshape, add a suffix to the model name to create a new model
    InputShapesMap input_shapes_map_temp = net_structure_->inputs_shape_map;
    std::string model_suffix             = NpuCommonUtils::modifyModelInputSize(max_inputs_shape, input_shapes_map_temp);

    // init the path to store/read om
    model_name_            = NpuCommonUtils::GetFileHash(model_config);
    model_name_            = model_name_ + model_suffix + "_" + version_str_;
    std::string model_path = use_path_ ? net_config.cache_path + "/" + model_name_ + ".om" : "";
    LOGI("[TNN/NPU]The path %s\n", model_path.c_str());

    // hiai model init
    InputShapesMap cpu_inputs_shape;
    tnn_ret = HiAIModelInit(model_path, net_config, model_config, default_interpreter, input_shapes_map_temp,
                            cpu_inputs_shape);
    if (tnn_ret != TNN_OK) {
        return tnn_ret;
    }

    // create tnn input/output blobs
    tnn_ret = InitBlobs(input_shapes_map_temp, cpu_inputs_shape);
    if (tnn_ret != TNN_OK) {
        return tnn_ret;
    }

    return TNN_OK;
}

bool NpuNetwork::InitConfigCheck(NetworkConfig &net_config, ModelConfig &model_config) {
    return net_config.device_type != DEVICE_HUAWEI_NPU || model_config.model_type != MODEL_TYPE_TNN;
}

// check Npu init situation
Status NpuNetwork::RomVersionCheck() {
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
        return Status(
            TNNERR_NPU_LOAD_ERROR,
            "ERROR: GetRomVersion(ROM): huawei npu is not match (only support DaVinci NPU) or rom version is too low");
    }
    version_str_ = version;

    // check if NPU version is greater than 300
    LOGI("[TNN/NPU]ddk current version: %s\n", version_str_.c_str());
    if (!NpuUtils::VersionCompare(version_str_, "100.320.xxx.xxx", VCT_BIGEQUAL)) {
        return Status(TNNERR_NPU_LOAD_ERROR, "ERROR: huawei_npu is installed but is below 100.320.xxx.xxx");
    }
    return TNN_OK;
}

Status NpuNetwork::InitContext(NetworkConfig &net_config) {
    Status ret = TNN_OK;
    device_    = GetDevice(net_config.device_type);
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
    return TNN_OK;
}

Status NpuNetwork::HiAIModelInit(std::string model_path, NetworkConfig &net_config, ModelConfig &model_config,
                                 DefaultModelInterpreter *interpreter, InputShapesMap inputs_shape,
                                 InputShapesMap &cpu_inputs_shape) {
    // hiai variables
    std::vector<std::shared_ptr<hiai::AiModelDescription>> model_desc;
    auto model_builder                = std::make_shared<hiai::AiModelBuilder>(client_);
    hiai::MemBuffer *model_mem_buffer = nullptr;
    // hiai ir variables
    domi::HiaiIrBuild ir_build;
    domi::ModelBufferData om_model_buff;

    if (use_path_ && NpuCommonUtils::FileExits(model_path)) {
        LOGI("[TNN/NPU]The om file already exists in %s\n", model_path.c_str());
        model_mem_buffer = model_builder->InputMemBufferCreate(model_path);
    } else {
        // NPU IR build
        Status ir_ret = IRInitLayers(net_config, interpreter, inputs_shape);
        if (ir_ret != TNN_OK) {
            LOGI("[TNN/NPU] Some layers not support in NPU, switch to ARM\n");
            if (cpu_count_ != net_structure_->layers.size()) {
                // create sub_network_interp_
                sub_network_interp_.reset(new ModelInterpreter());
                *sub_network_interp_->GetNetStructure() = *interpreter->GetNetStructure();
                *sub_network_interp_->GetNetResource()  = *interpreter->GetNetResource();

                ir_ret = InitSubNetwork(net_config, model_config, sub_network_interp_.get(), cpu_inputs_shape);
                if (ir_ret != TNN_OK) {
                    return ir_ret;
                }
            }
        }
        // update use path, if the network is split, then don't save om file
        use_path_ = use_path_ && !use_subnet_;
        // set Graph
        ir_ret = SetGraphInputsAndOutputs(inputs_shape, cpu_inputs_shape);
        if (ir_ret != TNN_OK) {
            return ir_ret;
        }
        // build Graph
        ir_ret = BuildGraph(ir_build, om_model_buff);
        if (ir_ret != TNN_OK) {
            return ir_ret;
        }
        // if path is specified, then first write to file, load from file later
        if (use_path_) {
            ir_ret = NpuUtils::WriteModelFile(om_model_buff, model_path);
            if (ir_ret != TNN_OK) {
                return ir_ret;
            }
        }

        // finish build, start to load the model
        model_mem_buffer = model_builder->InputMemBufferCreate(om_model_buff.data, om_model_buff.length);
    }

    if (model_mem_buffer == nullptr) {
        return Status(TNNERR_NPU_HIAI_API_ERROR, "ERROR: function InputMemBufferCreate() failed");
    }

    // set model attribute
    std::shared_ptr<hiai::AiModelDescription> desc = std::make_shared<hiai::AiModelDescription>(
        model_name_, hiai::AiModelDescription_Frequency_HIGH, hiai::HIAI_FRAMEWORK_NONE, hiai::HIAI_MODELTYPE_ONLINE,
        hiai::AiModelDescription_DeviceType_NPU);

    desc->SetModelBuffer(model_mem_buffer->GetMemBufferData(), model_mem_buffer->GetMemBufferSize());

    // only load one model
    model_desc.push_back(desc);

    // load model
    hiai::AIStatus ret = client_->Load(model_desc);
    if (ret != hiai::AI_SUCCESS) {
        return Status(TNNERR_NPU_HIAI_API_ERROR, "ERROR: Load model Load() failed");
    }

    // destroy unused memory
    model_builder->MemBufferDestroy(model_mem_buffer);
    ir_build.ReleaseModelBuff(om_model_buff);

    // check model
    bool is_compatible = true;
    ret                = client_->CheckModelCompatibility(*desc, is_compatible);
    LOGI("[TNN/NPU] is model compatible: %s", is_compatible ? "true" : "false");
    LOGI("[TNN/NPU] ret value %d", ret);
    if (ret != hiai::AI_SUCCESS) {
        return Status(TNNERR_NPU_HIAI_API_ERROR, "ERROR: check model CheckModelCompatibility() failed");
    }

    return TNN_OK;
}

Status NpuNetwork::IRInitLayers(NetworkConfig &net_config, DefaultModelInterpreter *interpreter,
                                InputShapesMap &inputs_shape) {
    Status ret                = TNN_OK;
    NetResource *net_resource = interpreter->GetNetResource();

    if (net_structure_ == NULL || net_resource == NULL) {
        return Status(TNNERR_NULL_PARAM, "ERROR: network_ is nil, network_type may not support");
    }

    ret = optimizer::NetOptimizerManager::Optimize(net_structure_, net_resource, net_config);
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

Status NpuNetwork::InitSubNetwork(NetworkConfig &net_config, ModelConfig &model_config,
                                  DefaultModelInterpreter *interpreter, InputShapesMap &cpu_inputs_shape) {
    // from here load cpu
    sub_network_                 = std::make_shared<DefaultNetwork>();
    NetworkConfig cpu_net_config = net_config;
    cpu_net_config.device_type   = DEVICE_ARM;
    cpu_net_config.network_type  = NETWORK_TYPE_DEFAULT;
    // change the network_structure for split
    NpuUtils::SplitNetwork(cpu_count_, interpreter->GetNetStructure(), visited_, global_operator_map_);
    cpu_inputs_shape = interpreter->GetNetStructure()->inputs_shape_map;
    if (cpu_inputs_shape.empty()) {
        LOGE(
            "ERROR: When split the network,  the arm can not find input in the huawei_npu visited "
            "layers\n");
        return Status(TNNERR_LAYER_ERR,
                      "ERROR: When split the network,  the arm can not find input in the huawei_npu visited layers");
    }
    Status ret = sub_network_->Init(cpu_net_config, model_config, interpreter, cpu_inputs_shape);
    if (ret != TNN_OK) {
        return ret;
    }
    use_subnet_ = true;
    return TNN_OK;
}

Status NpuNetwork::ConvertLayers(NetResource *net_resource) {
    Status ret = TNN_OK;
    // loop net_structure
    cpu_count_ = 0;
    for (auto layer_info : net_structure_->layers) {
        LOGI("convert layer (type: %d, name: %s)\n", layer_info->type, layer_info->name.c_str());
        LayerType type          = layer_info->type;
        NpuBaseLayer *cur_layer = CreateNpuBaseLayer(type);
        if (cur_layer == nullptr) {
            LOGE("Error Init layer tyep %d, huawei_npu does not support, may switch to arm\n", layer_info->type);
            return Status(TNNERR_LAYER_ERR, "CreateLayer failed");
        }
        std::string layer_name = layer_info->name;
        cur_layer->SetLayerName(layer_name);
        cur_layer->SetNpuVersion(version_str_);

        // set layer nodes
        std::vector<std::shared_ptr<OperatorInfo>> input_ops;
#ifdef GENERATE_RESOURCE
        std::vector<Blob *> input_blobs;
        BlobDesc blob_desc;
#endif
        for (std::string &name : layer_info->inputs) {
            input_ops.push_back(global_operator_map_[name]);
#ifdef GENERATE_RESOURCE
            blob_desc.dims = global_operator_map_[name]->GetShape();
            Blob *blob     = new Blob(blob_desc);
            input_blobs.push_back(blob);
#endif
        }
#ifdef GENERATE_RESOURCE
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
        if (ret != TNN_OK) {
            LOGE("Error Init layer %s (%s), may switch to arm\n", cur_layer->GetLayerName().c_str(),
                 ret.description().c_str());
            return ret;
        }
        layers_.push_back(cur_layer);

        for (auto &op : cur_layer->GetOutputOps()) {
            visited_.insert(op->GetOperator()->GetName());
            global_operator_map_[op->GetOperator()->GetName()] = op;
        }
        cpu_count_++;
    }
    return ret;
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
    if (use_subnet_) {
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
    } else {
        for (auto &name : net_structure_->outputs) {
            if (input_shape_map.count(name) == 0) {
                output_ops.push_back(*global_operator_map_[name]->GetOperator());
            }
        }
    }
    graph_.SetInputs(input_ops).SetOutputs(output_ops);
    return TNN_OK;
}

Status NpuNetwork::BuildGraph(domi::HiaiIrBuild &ir_build, domi::ModelBufferData &om_model_buff) {
    ge::Model model(model_name_, model_name_ + "_v1");
    model.SetGraph(graph_);
    // build options
    domi::BuildOptions options;
    options.useOriginFormat = true;
    bool build_ret          = ir_build.CreateModelBuff(model, om_model_buff);
    if (!build_ret) {
        return Status(TNNERR_NPU_HIAI_API_ERROR, "HIAI build model, CreateModelBuff() failed");
    }
    build_ret = ir_build.BuildIRModel(model, om_model_buff, options);
    if (!build_ret) {
        return Status(TNNERR_NPU_HIAI_API_ERROR, "HIAI build model, BuildIRModel() failed");
    }
    return TNN_OK;
}

Status NpuNetwork::InitBlobs(InputShapesMap &inputs_shape, InputShapesMap &cpu_inputs_shape) {
    input_tensor_.clear();
    output_tensor_.clear();
    std::vector<hiai::TensorDimension> input_dims;
    std::vector<hiai::TensorDimension> output_dims;
    hiai::AIStatus ret = client_->GetModelIOTensorDim(model_name_, input_dims, output_dims);
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

    // init input blobs
    int input_idx = 0;
    for (auto item : inputs_shape) {
        auto name             = item.first;
        auto npu_blob         = CreateNpuBlob(input_dims[input_idx], name, input_tensor_[input_idx]->GetBuffer());
        input_blob_map_[name] = npu_blob;
        input_idx++;
    }

    // init output blobs
    if (use_subnet_) {
        sub_network_->GetAllInputBlobs(cpu_inter_in_blobmap_);
        // create sub-network input blob-converter
        for (auto blob_item : cpu_inter_in_blobmap_) {
            cpu_blob_converter_map_[blob_item.first] = std::make_shared<BlobConverter>(blob_item.second);
        }

        int output_idx = 0;
        for (auto item : cpu_inputs_shape) {
            auto name = item.first;
            if (input_blob_map_.count(name) != 0) {
                auto desc                    = input_blob_map_[name]->GetBlobDesc();
                auto handle                  = input_blob_map_[name]->GetHandle();
                npu_inter_out_blobmap_[name] = new Blob(desc, handle);
            } else {
                auto npu_blob = CreateNpuBlob(output_dims[output_idx], name, output_tensor_[output_idx]->GetBuffer());
                npu_inter_out_blobmap_[name] = npu_blob;
                output_idx++;
            }
        }

        // get the final output
        sub_network_->GetAllOutputBlobs(output_blob_map_);
    } else {
        // get the final output
        int output_idx = 0;
        for (auto name : net_structure_->outputs) {
            auto npu_blob = CreateNpuBlob(output_dims[output_idx], name, output_tensor_[output_idx]->GetBuffer());
            output_blob_map_[name] = npu_blob;
            output_idx++;
        }
    }

    LOGI("Init NPU Blobs Done!\n");
    return TNN_OK;
}

Blob *NpuNetwork::CreateNpuBlob(hiai::TensorDimension dims, std::string name, void *data) {
    int n = dims.GetNumber();
    int c = dims.GetChannel();
    int h = dims.GetHeight();
    int w = dims.GetWidth();
    // add blob
    BlobDesc desc;
    desc.device_type = DEVICE_HUAWEI_NPU;
    desc.data_format = DATA_FORMAT_NCHW;
    desc.name        = name;
    desc.dims.push_back(n);
    desc.dims.push_back(c);
    desc.dims.push_back(h);
    desc.dims.push_back(w);
    BlobHandle handle;
    handle.base = data;
    return new Blob(desc, handle);
}

Status NpuNetwork::GetForwardMemorySize(int &memory_size) {
    memory_size = 0;
    return TNNERR_NPU_UNSUPPORT_ERROR;
}

Status NpuNetwork::SetForwardMemory(void *memory) {
    return TNNERR_NPU_UNSUPPORT_ERROR;
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

    if (use_subnet_) {
        iterator = npu_inter_out_blobmap_.begin();
        while (iterator != npu_inter_out_blobmap_.end()) {
            if (iterator->second != nullptr) {
                delete (iterator->second);
                iterator->second = nullptr;
            }
            iterator++;
        }
        npu_inter_out_blobmap_.clear();
    } else {
        iterator = output_blob_map_.begin();
        while (iterator != output_blob_map_.end()) {
            if (iterator->second != nullptr) {
                delete (iterator->second);
                iterator->second = nullptr;
            }
            iterator++;
        }
        output_blob_map_.clear();
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
    std::string value = model_name_;
    context.AddPara(key, value);
    int istamp;
#if TNN_PROFILE
    struct timeval start, end;
    gettimeofday(&start, NULL);
#endif
    hiai::AIStatus ret = client_->Process(context, input_tensor_, output_tensor_, 1000, istamp);
    if (ret != hiai::AI_SUCCESS) {
        return Status(TNNERR_NPU_HIAI_API_ERROR, "Forward failed!");
    }
#if TNN_PROFILE
    gettimeofday(&end, NULL);
    float delta = (end.tv_sec - start.tv_sec) * 1000.f + (end.tv_usec - start.tv_usec) / 1000.f;
    std::shared_ptr<ProfilingData> pdata(new ProfilingData());
    pdata->kernel_time = delta;
    pdata->layer_name  = "NPU Forward";
    pdata->op_name     = "NPU Execute";
    context_->AddProfilingData(pdata);
#endif

    if (use_subnet_) {
        for (auto iterator = npu_inter_out_blobmap_.begin(); iterator != npu_inter_out_blobmap_.end(); iterator++) {
            std::string name = iterator->first;
            Blob *npu_blob   = iterator->second;
            Blob *cpu_blob   = cpu_inter_in_blobmap_[name];
            if (cpu_blob_converter_map_.count(name) == 0) {
                LOGE("cpu blob convert for sub-network not found!\n");
                return Status(TNNERR_NULL_PARAM, "cpu blob convert for sub-network not found!");
            }
            Mat input_mat(DEVICE_NAIVE, NCHW_FLOAT, npu_blob->GetBlobDesc().dims,
                          (char *)npu_blob->GetHandle().base + npu_blob->GetHandle().bytes_offset);
            MatConvertParam param;
            cpu_blob_converter_map_[name]->ConvertFromMat(input_mat, param, nullptr);
        }
        sub_network_->Forward();
    }
    return TNN_OK;
}

Status NpuNetwork::ForwardAsync(Callback call_back) {
    return NpuNetwork::Forward();
}

#if TNN_PROFILE
void NpuNetwork::StartProfile() {
    context_->StartProfile();
    if (nullptr != sub_network_) {
        sub_network_->StartProfile();
    }
}

std::shared_ptr<ProfileResult> NpuNetwork::FinishProfile() {
    auto result = context_->FinishProfile();
    if (nullptr != sub_network_) {
        auto sub_result = sub_network_->FinishProfile();
        result->AddProfileResult(sub_result);
    }
    return result;
}
#endif

}  // namespace TNN_NS
