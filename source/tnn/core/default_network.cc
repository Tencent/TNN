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

#include "tnn/core/default_network.h"

#include <string.h>

#include "tnn/core/blob_int8.h"
#include "tnn/core/profile.h"
#include "tnn/interpreter/default_model_interpreter.h"
#include "tnn/interpreter/layer_param.h"
#include "tnn/interpreter/layer_resource_generator.h"
#include "tnn/memory_manager/blob_memory_pool_factory.h"
#include "tnn/optimizer/net_optimizer_manager.h"
#include "tnn/utils/blob_dump_utils.h"
#include "tnn/utils/blob_transfer_utils.h"
#include "tnn/utils/cpu_utils.h"
#include "tnn/utils/data_flag_utils.h"
#include "tnn/utils/dims_utils.h"
#include "tnn/utils/md5.h"
#include "tnn/utils/string_utils_inner.h"

namespace TNN_NS {

//reserved for incompatible
const std::string CACHE_TAG = "d1";

NetworkImplFactoryRegister<NetworkImplFactory<DefaultNetwork>> g_network_impl_default_factory_register(
    NETWORK_TYPE_DEFAULT);

std::mutex DefaultNetwork::optimize_mtx_;

DefaultNetwork::DefaultNetwork()
    : device_(nullptr), context_(nullptr), blob_manager_(nullptr), net_structure_(nullptr) {}

DefaultNetwork::~DefaultNetwork() {
    DeInit();
}

Status DefaultNetwork::SetCpuNumThreads(int num_threads) {
    if (context_)
        return context_->SetNumThreads(num_threads);
    else
        return Status(TNNERR_CONTEXT_ERR, "context is nil");
}

/*
 * The Network holds blob, blobmanager, layers etc.
 * Those object is initialized in this function.
 */
Status DefaultNetwork::Init(NetworkConfig &net_config, ModelConfig &model_config, AbstractModelInterpreter *interpreter,
                        InputShapesMap min_inputs_shape, InputShapesMap max_inputs_shape, bool enable_const_folder) {
    config_                                      = net_config;
    Status ret                                   = TNN_OK;
    DefaultModelInterpreter *default_interpreter = dynamic_cast<DefaultModelInterpreter *>(interpreter);
    CHECK_PARAM_NULL(default_interpreter);

    NetStructure *net_structure = default_interpreter->GetNetStructure();
    NetResource *net_resource   = default_interpreter->GetNetResource();

    if (net_structure == NULL || net_resource == NULL) {
        LOGE("ERROR: network_ is nil, network_type may not support\n");
        return Status(TNNERR_NULL_PARAM, "network_ is nil, network_type may not support");
    }

    device_ = GetDevice(net_config.device_type);
    RETURN_VALUE_ON_NEQ(device_ != NULL, true, TNNERR_DEVICE_NOT_SUPPORT);

    context_ = device_->CreateContext(net_config.device_id);
    RETURN_VALUE_ON_NEQ(context_ != NULL, true, TNNERR_DEVICE_CONTEXT_CREATE);

#ifdef DEBUG
    {
        static bool cpu_support_fp16 = CpuUtils::CpuSupportFp16();
        LOGD("support fp 16: %d\n", cpu_support_fp16 ? 1 : 0);
    }
#endif
    context_->SetPrecision(net_config.precision);
    context_->SetEnableTuneKernel(net_config.enable_tune_kernel);

    if(!net_config.cache_path.empty()) {
        auto params_md5 = default_interpreter->GetParamsMd5();
        if (params_md5.size() < 1) {
            return Status(TNNERR_PARAM_ERR, "model params md5 missing");
        }
        context_->SetCachePath(net_config.cache_path);
        context_->SetCacheFilePath(GenerateCacheFileName(model_config, params_md5[0]));
    }

    ret = context_->LoadLibrary(net_config.library_path);
    RETURN_ON_NEQ(ret, TNN_OK);

    /*
     * The NetOptimizeManager holds a list of network optimization processes.
     * The optimization process may change the network structure accoundingly.
     * eg. fuse conv+bn, conv+relu.
     */
    if (runtime_model_ == RUNTIME_MODE_NORMAL) {
        // use mutex to protect net_resource and net_structure in multi-thread
        std::unique_lock<std::mutex> lck(optimize_mtx_);
        ret = optimizer::NetOptimizerManager::Optimize(net_structure, net_resource, net_config);
        RETURN_ON_NEQ(ret, TNN_OK);
    }

    blob_manager_ = new BlobManager(device_);

    ret = blob_manager_->Init(net_config, net_structure, max_inputs_shape, GetNetResourceDataType(net_resource));
    RETURN_ON_NEQ(ret, TNN_OK);

    ret = InitLayers(net_structure, net_resource);
    RETURN_ON_NEQ(ret, TNN_OK);

    ret = AllocateBlobMemory();
    RETURN_ON_NEQ(ret, TNN_OK);

    net_structure_ = net_structure;
    net_resource_ = net_resource;
    
    ret = context_->OnInstanceReshapeBegin();
    RETURN_ON_NEQ(ret, TNN_OK);

    ret = ReshapeLayers();
    RETURN_ON_NEQ(ret, TNN_OK);

    ret = context_->OnInstanceReshapeEnd();
    return ret;
}

static inline bool IsLayoutReformatLayer(std::shared_ptr<LayerInfo> layer) {
    if (layer->type == LAYER_REFORMAT) {
        auto param = dynamic_cast<ReformatLayerParam *>(layer->param.get());
        if (param->src_format != param->dst_format && param->src_type == param->dst_type) {
            return true;
        }
    }
    return false;
}

/*
 * InitLayer function does the following things:
 *  1. Set Blob type accordingly.
 *  2. Set data_type accordingly.
 *  3. Infer the blob shapes.
 *  4. Check the weights required.
 */
Status DefaultNetwork::InitLayers(NetStructure *net_structure, NetResource *net_resource) {
    Status ret            = TNN_OK;
    bool is_quantized_net = GetQuantizedInfoFromNetStructure(net_structure);
    
    // mark const blobs and blob data type
    auto const_blobs = net_resource->constant_map;
    for (auto layer_info : net_structure->layers) {
        std::vector<std::string> &input_names  = layer_info->inputs;
        for (auto name : input_names) {
            auto blob = blob_manager_->GetBlob(name);
            if (const_blobs.find(name) != const_blobs.end()) {
                if (runtime_model_ == RUNTIME_MODE_NORMAL) {
                    blob->SetFlag(DATA_FLAG_CHANGE_NEVER);
                }
                blob->GetBlobDesc().data_type = const_blobs[name]->GetDataType();
            }
        }
    }
    

    auto const_layers = net_resource->constant_layers;
    // update blob precision, alloc new blob required
    for (auto layer_info : net_structure->layers) {
        if (runtime_model_ == RUNTIME_MODE_NORMAL && const_layers.find(layer_info->name) != const_layers.end()) {
            continue;
        }
        
        // set layer nodes
        std::vector<std::string> &input_names  = layer_info->inputs;
        std::vector<std::string> &output_names = layer_info->outputs;

        DataFormat input_fmt = DATA_FORMAT_AUTO;
        for (auto name : input_names) {
            auto blob = blob_manager_->GetBlob(name);
            // skip const blobs
            if (const_blobs.count(name) == 0) {
                input_fmt = blob->GetBlobDesc().data_format;
                auto ret  = UpdateBlobPrecision(layer_info, true, is_quantized_net, name, net_resource, &blob);
                RETURN_ON_NEQ(ret, TNN_OK);
            }
        }

        // output layout equals to input layout except for layout_reformat layer
        DataFormat output_fmt = layer_info->type == LAYER_REFORMAT ?
            dynamic_cast<ReformatLayerParam *>(layer_info->param.get())->dst_format : input_fmt;

#ifdef GENERATE_RESOURCE
        if (runtime_model_ == RUNTIME_MODE_NORMAL || runtime_model_ == RUNTIME_MODE_CONST_FOLD) {
            LayerType type       = layer_info->type;
            BaseLayer *cur_layer = CreateLayer(type);
            if (cur_layer == NULL) {
                LOGE("Error: CreateLayer failed, type:%d\n", type);
                return Status(TNNERR_PARAM_ERR, "CreateLayer failed");
            }
            std::string layer_name = layer_info->name;
            cur_layer->SetLayerName(layer_name);
            cur_layer->SetRuntimeMode(runtime_model_);
            cur_layer->SetConstantResource(&net_resource->constant_map);
            cur_layer->SetConstantResourceFlag(&net_resource->constant_blob_flags);

            std::vector<Blob *> inputs;
            std::vector<Blob *> outputs_for_shape;
            for (auto name : input_names) {
                inputs.push_back(blob_manager_->GetBlob(name));
            }

            for (auto name : output_names) {
                outputs_for_shape.push_back(blob_manager_->GetBlob(name));
            }

            // generate resource if null
            if (net_resource->resource_map.count(layer_name) == 0) {
                LayerParam *layer_param  = layer_info->param.get();
                LayerResource *layer_res = nullptr;
                GenerateRandomResource(type, layer_param, &layer_res, inputs, &net_resource->constant_map);
                net_resource->resource_map[layer_name] = std::shared_ptr<LayerResource>(layer_res);
            }

            cur_layer->InferShapeAhead(inputs, outputs_for_shape, layer_info->param.get(),
                                       net_resource->resource_map[layer_name].get());

            delete cur_layer;
        }
#endif

        for (auto name : output_names) {
            auto blob = blob_manager_->GetBlob(name);
            // skip const blobs
            if (const_blobs.count(name) == 0) {
                blob->GetBlobDesc().data_format = output_fmt;
                auto ret = UpdateBlobPrecision(layer_info, false, is_quantized_net, name, net_resource, &blob);
                RETURN_ON_NEQ(ret, TNN_OK);
            }
        }
    }

    // init layer
    for (auto layer_info : net_structure->layers) {
        if (runtime_model_ == RUNTIME_MODE_NORMAL && const_layers.find(layer_info->name) != const_layers.end()) {
            continue;
        }
        
        LayerType type       = layer_info->type;
        BaseLayer *cur_layer = CreateLayer(type);
        if (cur_layer == NULL) {
            LOGE("Error: CreateLayer failed, type:%d\n", type);
            return Status(TNNERR_PARAM_ERR, "CreateLayer failed");
        }
        std::string layer_name = layer_info->name;
        cur_layer->SetLayerName(layer_name);
        // set layer nodes
        std::vector<Blob *> inputs;
        std::vector<std::string> &input_names = layer_info->inputs;

        for (auto name : input_names) {
            auto blob = blob_manager_->GetBlob(name);
            if (blob == nullptr) {
                delete cur_layer;
                LOGE("Input of layer(%s) are invalid", layer_name.c_str());
                return Status(TNNERR_PARAM_ERR, "Input of layer are invalid");
           }
            // update layout reformat layer's param and blob datatype
            if (IsLayoutReformatLayer(layer_info)) {
                // only need to update model's input blob datatype
                // others are already updated in UpdateBlobPrecision method
                const auto src_data_type = blob->GetBlobDesc().data_type;
                bool update_precision = (src_data_type == DATA_TYPE_FLOAT || src_data_type == DATA_TYPE_HALF || 
                    src_data_type == DATA_TYPE_BFP16);
                if (net_structure->inputs_shape_map.find(name) != net_structure->inputs_shape_map.end() && update_precision) {
                    auto dtype = blob_manager_->GetBlob(layer_info->outputs[0])->GetBlobDesc().data_type;
                    LOGD("DefaultNetwork::InitLayers LayoutReformat set input: %s datatype as: %d\n",
                         name.c_str(), dtype);
                    blob->GetBlobDesc().data_type = dtype;
                }
                auto param      = dynamic_cast<ReformatLayerParam *>(layer_info->param.get());
                param->src_type = blob->GetBlobDesc().data_type;
                param->dst_type = param->src_type;
            }
            inputs.push_back(blob);
        }

        std::vector<Blob *> outputs;
        std::vector<std::string> &output_names = layer_info->outputs;

        for (auto name : output_names) {
            auto blob = blob_manager_->GetBlob(name);
            if (blob == nullptr) {
                delete cur_layer;
                LOGE("Output of layer(%s) are invalid", layer_name.c_str());
                return Status(TNNERR_PARAM_ERR, "Output of layer are invalid");
            }
            outputs.push_back(blob);
        }

        LayerResource *layer_resource = nullptr;
        if (net_resource->resource_map.count(layer_name) != 0) {
            layer_resource = net_resource->resource_map[layer_name].get();
        }
        
        cur_layer->SetRuntimeMode(runtime_model_);
        cur_layer->SetConstantResource(&net_resource->constant_map);
        cur_layer->SetConstantResourceFlag(&net_resource->constant_blob_flags);
        ret = cur_layer->Init(context_, layer_info->param.get(), layer_resource, inputs, outputs, device_);
        if (ret != TNN_OK) {
            LOGE("Error Init layer %s (err: %d or 0x%X)\n", cur_layer->GetLayerName().c_str(), (int)ret, (int)ret);
            // release layer if Init failed
            delete cur_layer;
            return ret;
        }
        cur_layer->SetRuntimeBlobMemoryPool(runtime_blob_pool_);

        layers_.push_back(cur_layer);
    }
    return ret;
}

Status DefaultNetwork::AllocateBlobMemory() {
    return blob_manager_->AllocateBlobMemory(DATA_FLAG_CHANGE_ALWAYS);
}

Status DefaultNetwork::GenerateInt8Blob(const std::string &name, NetResource *net_resource, Blob **blob) {
    auto new_blob = new BlobInt8((*blob)->GetBlobDesc(), (*blob)->GetHandle());
    CHECK_PARAM_NULL(new_blob);

    std::string blob_scale_name = name + "_scale_data_";
#ifdef GENERATE_RESOURCE
    if (net_resource->resource_map.count(blob_scale_name) == 0) {
        LayerResource *layer_res  = nullptr;
        std::vector<Blob *> blobs = {*blob};
        GenerateRandomResource(LAYER_BLOB_SCALE, nullptr, &layer_res, blobs);
        net_resource->resource_map[blob_scale_name] = std::shared_ptr<LayerResource>(layer_res);
    }
#endif
    if (net_resource->resource_map.find(blob_scale_name) == net_resource->resource_map.end()) {
        LOGE("Error Init layer, can not get output blob scale %s \n", blob_scale_name.c_str());
        return TNNERR_NULL_PARAM;
    }

    new_blob->SetIntResource(reinterpret_cast<IntScaleResource *>(net_resource->resource_map[blob_scale_name].get()));
    blob_manager_->ReplaceBlob(name, new_blob);
    *blob = new_blob;

    return TNN_OK;
}

Status DefaultNetwork::UpdateBlobPrecision(std::shared_ptr<LayerInfo> layer_info, bool is_input, bool is_quantized_net,
                                           const std::string &name, NetResource *net_resource, Blob **blob) {
    if (device_->GetDeviceType() != DEVICE_ARM && device_->GetDeviceType() != DEVICE_NAIVE &&
        device_->GetDeviceType() != DEVICE_X86) {
        return TNN_OK;
    }

    auto &desc      = (*blob)->GetBlobDesc();
    auto layer_type = layer_info->type;

    if (layer_type != LAYER_REFORMAT) {
        // non-reformat layer
        if (is_quantized_net) {
            // update blob of quantized network by layer info
            auto int8_blob = dynamic_cast<BlobInt8*>(*blob);
            if (layer_info->param->quantized && int8_blob == nullptr) {
                RETURN_ON_NEQ(GenerateInt8Blob(name, net_resource, blob), TNN_OK);
            }
        } else {
            // update blob of non-quantized network by precision
            auto original_data_type = desc.data_type;
            if (original_data_type == DATA_TYPE_FLOAT || original_data_type == DATA_TYPE_HALF ||
                original_data_type == DATA_TYPE_BFP16) {
                if (config_.precision == PRECISION_NORMAL || config_.precision == PRECISION_AUTO) {
                    static bool cpu_support_fp16 = CpuUtils::CpuSupportFp16();
                    bool layer_implemented_fp16  = device_->GetImplementedPrecision(layer_type)->fp16_implemented;
                    desc.data_type = (cpu_support_fp16 && layer_implemented_fp16) ? DATA_TYPE_HALF : DATA_TYPE_FLOAT;
                } else if (config_.precision == PRECISION_LOW) {
                    if (device_->GetDeviceType() == DEVICE_ARM) {
                        desc.data_type = DATA_TYPE_BFP16;
                    } else if (device_->GetDeviceType() == DEVICE_NAIVE ||
                               device_->GetDeviceType() == DEVICE_X86) {
                        desc.data_type = DATA_TYPE_FLOAT;
                    }
                } else if (config_.precision == PRECISION_HIGH) {
                    desc.data_type = DATA_TYPE_FLOAT;
                } else {
                    return Status(TNNERR_PARAM_ERR, "invalid precision");
                }
            }
        }
    } else {
        // layout reformat, update later
        if (IsLayoutReformatLayer(layer_info)) {
            return TNN_OK;
        }
        // datatype reformat, update by layer param
        if (is_input) {
            auto src_type = reinterpret_cast<ReformatLayerParam *>(layer_info->param.get())->src_type;
            if (src_type == DATA_TYPE_INT8) {
                RETURN_ON_NEQ(GenerateInt8Blob(name, net_resource, blob), TNN_OK);
            } else {
                desc.data_type = src_type;
            }
        } else {
            auto dst_type = reinterpret_cast<ReformatLayerParam *>(layer_info->param.get())->dst_type;
            if (dst_type == DATA_TYPE_INT8) {
                RETURN_ON_NEQ(GenerateInt8Blob(name, net_resource, blob), TNN_OK);
            } else {
                desc.data_type = dst_type;
            }
        }
    }

    return TNN_OK;
}

Status DefaultNetwork::GetForwardMemorySize(int &memory_size) {
    memory_size = blob_manager_->GetAllBlobMemorySize();
    return TNN_OK;
}

Status DefaultNetwork::SetForwardMemory(void *memory) {
    return blob_manager_->SetForwardMemory(memory);
}

Status DefaultNetwork::GetAllInputBlobs(BlobMap &blobs) {
    blob_manager_->GetAllInputBlobs(blobs);
    return TNN_OK;
}

/*
 * Returns the default output blobs in the network.
 * Additional output blob may be assigned with TNN::AddOutput function
 */
Status DefaultNetwork::GetAllOutputBlobs(BlobMap &blobs) {
    blob_manager_->GetAllOutputBlobs(blobs);
    return TNN_OK;
}

/*
 * Reshape function is called when the input shape changes.
 * Memory allocation may be involved in Reshape function.
 */
Status DefaultNetwork::Reshape(const InputShapesMap &inputs) {
    Status ret = TNN_OK;
    bool shape_changed = false;
    ret = PrepareDoReshape(inputs, shape_changed);
    if(ret != TNN_OK) {
        return ret; 
    }
    if(shape_changed) {
        return DoReshape();
    }
    return ret;
}

Status DefaultNetwork::PrepareDoReshape(const InputShapesMap& inputs, bool& shape_changed) {
    shape_changed = false;
    for (auto iter : inputs) {
        Blob *blob = blob_manager_->GetBlob(iter.first);
        if (blob == nullptr) {
            LOGE("DefaultNetwork reshape blob is empty, maybe the blob name is wrong\n");
            return Status(TNNERR_PARAM_ERR, "DefaultNetwork reshape blob is empty, maybe the blob name is wrong");
        }
        if(!DimsVectorUtils::Equal(blob->GetBlobDesc().dims, iter.second)) {
            blob->GetBlobDesc().dims = iter.second;
            shape_changed = true;
        }
    }
    return TNN_OK;
}

Status DefaultNetwork::DoReshape() {
    Status ret = TNN_OK;
    ret = context_->OnInstanceReshapeBegin();
    if (ret != TNN_OK) {
        return ret;
    }

    ret = ReshapeLayers();
    if (ret != TNN_OK) {
        return ret;
    }

    ret = context_->OnInstanceReshapeEnd();

    return ret;
}

Status DefaultNetwork::DeInit() {
    for (size_t i = 0; i < layers_.size(); i++) {
        if (layers_[i] != NULL) {
            delete layers_[i];
        }
    }
    layers_.clear();

    if (blob_manager_ != NULL) {
        delete blob_manager_;
        blob_manager_ = NULL;
    }
    
    if (runtime_blob_pool_ != nullptr) {
        delete runtime_blob_pool_;
        runtime_blob_pool_ = nullptr;
    }

    if (context_ != NULL) {
        delete context_;
        context_ = NULL;
    }

    return TNN_OK;
}
/*
 * CommandQueue is an abstract object.
 * The actual object maybe:
 *  1. OpenCl commnadqueue.
 *  2. Metal command buffer.
 *  3. Cuda Stream
 *  ...
 */
Status DefaultNetwork::GetCommandQueue(void **command_queue) {
    if (context_ == NULL) {
        return TNNERR_DEVICE_CONTEXT_CREATE;
    }
    return context_->GetCommandQueue(command_queue);
}

Status DefaultNetwork::ShareCommandQueue(AbstractNetwork *network) {
    if (context_ == NULL) {
        return TNNERR_DEVICE_CONTEXT_CREATE;
    }

    auto network_target = dynamic_cast<DefaultNetwork *>(network);
    if (!network_target) {
        return Status(TNNERR_DEVICE_CONTEXT_CREATE, "inpute network is DefaultNetwork");
    }
    return context_->ShareCommandQueue(network_target->GetContext());
}

Context* DefaultNetwork::GetContext() {
    return context_;
}

Status DefaultNetwork::Forward() {
    auto status = blob_manager_->CheckBlobMemoryState();
    RETURN_ON_NEQ(status, TNN_OK);
    
    if (runtime_blob_pool_) {
        //now we allocate blob eachtime when running acc, so clear blob pool to avoid memory leak
        runtime_blob_pool_->ClearBlobMemoryPool();
    }
    
    status = context_->OnInstanceForwardBegin();
    RETURN_ON_NEQ(status, TNN_OK);
    
    int cnt = 0;
    for (auto layer : layers_) {
        std::vector<Blob *> inputs  = layer->GetInputBlobs();
        std::vector<Blob *> outputs = layer->GetOutputBlobs();

        {
            
#if DUMP_INPUT_BLOB
            if (runtime_model_ == RUNTIME_MODE_NORMAL) {
                // InputBlob data in dumped into files in NCHW_FLOAT format as default
                std::string filename = layer->GetLayerName();
                std::replace(filename.begin(), filename.end(), '/', '_');
                for (int i = 0; i < inputs.size(); i++) {
                    char ss[1000];
                    if (g_tnn_dump_directory.length() > 0) {
                        snprintf(ss, 1000, "%s/%05d-%s-in-%d", g_tnn_dump_directory.c_str(), cnt, filename.c_str(), i);
                    } else {
                        snprintf(ss, 1000, "%05d-%s-in-%d", cnt, filename.c_str(), i);
                    }

                    auto ret = DumpDeviceBlob(inputs[i], context_, std::string(ss));
                    if (ret != TNN_OK) {
                        LOGE("dump blob failed\n");
                        return ret;
                    }
                }
            }
#endif  // DUMP_INPUT_BLOB
            
            status = layer->Forward();
            LOGD("layer name: %s, forward result: %d \n", layer->GetLayerName().c_str(), (int)status);
            LOGD("Output Shape: [%s]\n", layer->GetOutputBlobs()[0]->GetBlobDesc().description().c_str());
            if (status != TNN_OK) {
                LOGE("Forward error %s, exit\n", status.description().c_str());
                return status;
            }

#if DUMP_OUTPUT_BLOB
            if (runtime_model_ == RUNTIME_MODE_NORMAL) {
                // OutBlob data in dumped into files in NCHW_FLOAT format as default
                std::string out_file_name = layer->GetLayerName();
                std::replace(out_file_name.begin(), out_file_name.end(), '/', '_');
                for (int i = 0; i < outputs.size(); i++) {
                    char ss[1000];
                    if (g_tnn_dump_directory.length() > 0) {
                        snprintf(ss, 1000, "%s/%05d-%s-out-%d", g_tnn_dump_directory.c_str(), cnt, out_file_name.c_str(), i);
                    } else {
                        snprintf(ss, 1000, "%05d-%s-out-%d", cnt, out_file_name.c_str(), i);
                    }

                    auto ret = DumpDeviceBlob(outputs[i], context_, std::string(ss));
                    if (ret != TNN_OK) {
                        LOGE("dump blob failed\n");
                        return ret;
                    }
                }
            }
#endif  // DUMP_OUTPUT_BLOB
        }
        
        cnt++;
    }
    context_->OnInstanceForwardEnd();
    context_->Synchronize();
    return status;
}

#ifdef FORWARD_CALLBACK_ENABLE
Status DefaultNetwork::ForwardWithCallback(BlobStatisticCallback before, BlobStatisticCallback after) {
    Status result = TNN_OK;
    result        = blob_manager_->CheckBlobMemoryState();
    if (result != TNN_OK) {
        return result;
    }

    context_->OnInstanceForwardBegin();
    int cnt = 0;
    for (auto layer : layers_) {
        std::vector<Blob *> inputs  = layer->GetInputBlobs();
        std::vector<Blob *> outputs = layer->GetOutputBlobs();

        auto layer_info = GetLayerInfoFromName(net_structure_, layer->GetLayerName());
        if (before != nullptr)
            before(inputs, layer_info.get());

        result = layer->Forward();
        if (result != TNN_OK) {
            LOGE("Forward error %s, exit\n", result.description().c_str());
            return result;
        }
        context_->Synchronize();

        if (after != nullptr)
            after(outputs, layer_info.get());

        cnt++;
    }
    context_->OnInstanceForwardEnd();
    return result;
}
#endif  // end of FORWARD_CALLBACK_ENABLE

// @brief tnn instance network infer, it will not wait
// blob dump is not implement in this funciton.
Status DefaultNetwork::ForwardAsync(Callback call_back) {
    Status result = TNN_OK;
    result        = blob_manager_->CheckBlobMemoryState();
    if (result != TNN_OK) {
        return result;
    }

    context_->OnInstanceForwardBegin();
    for (auto layer : layers_) {
        result = layer->Forward();
        RETURN_ON_NEQ(result, TNN_OK);
    }
    context_->OnInstanceForwardEnd();
    return result;
}

#if TNN_PROFILE
void DefaultNetwork::StartProfile() {
    context_->StartProfile();
}

std::shared_ptr<ProfileResult> DefaultNetwork::FinishProfile() {
    return context_->FinishProfile();
}
#endif

std::string DefaultNetwork::GenerateCacheFileName(ModelConfig &model_config, std::string& md5_str) {
    return CACHE_TAG + "_" + ToString(config_.device_type) + "_" + ToString(config_.device_id)
        + "_" + ToString(config_.precision) + "_" + ToString(model_config.model_type) +
        "_" + md5_str;
}

Status DefaultNetwork::ReshapeLayers() {
    for (auto cur_layer : layers_) {
        auto status = cur_layer->Reshape();
        RETURN_ON_NEQ(status, TNN_OK);
        //Note output shape may not change after reshape for const folder, but will do change after forward because shape may be determined at rumtime
        LOGD("ReshapeLayers Output Shape: [%s]\n", cur_layer->GetOutputBlobs()[0]->GetBlobDesc().description().c_str());
    }
    return TNN_OK;
}

}  // namespace TNN_NS
