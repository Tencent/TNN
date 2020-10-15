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
#include "tnn/optimizer/net_optimizer_manager.h"
#include "tnn/utils/blob_dump_utils.h"
#include "tnn/utils/blob_transfer_utils.h"
#include "tnn/utils/dims_vector_utils.h"

namespace TNN_NS {

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
                            InputShapesMap inputs_shape) {
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
    if (device_ == NULL) {
        return TNNERR_DEVICE_NOT_SUPPORT;
    }

    context_ = device_->CreateContext(net_config.device_id);
    if (context_ == NULL) {
        return TNNERR_DEVICE_CONTEXT_CREATE;
    }

    ret = context_->SetPrecision(net_config.precision);
    if (ret != TNN_OK) {
        return ret;
    }

    ret = context_->LoadLibrary(net_config.library_path);
    if (ret != TNN_OK) {
        return ret;
    }

    /*
     * The NetOptimizeManager holds a list of network optimization processes.
     * The optimization process may change the network structure accoundingly.
     * eg. fuse conv+bn, conv+relu.
     */
    {
        // use mutex to protect net_resource and net_structure in multi-thread
        std::unique_lock<std::mutex> lck(optimize_mtx_);
        ret = optimizer::NetOptimizerManager::Optimize(net_structure, net_resource, net_config.device_type);
        if (ret != TNN_OK) {
            return ret;
        }
    }

    blob_manager_ = new BlobManager(device_);

    ret = blob_manager_->Init(net_config, net_structure, inputs_shape, GetNetResourceDataType(net_resource));
    if (ret != TNN_OK) {
        return ret;
    }

    ret = InitLayers(net_structure, net_resource);
    if (ret != TNN_OK) {
        return ret;
    }

    ret = blob_manager_->AllocateBlobMemory();
    if (ret != TNN_OK) {
        return ret;
    }

    net_structure_ = net_structure;

    InputShapesMap input_shape_map;
    return Reshape(input_shape_map);
}

/*
 * InitLayer funcion does the following things:
 *  1. Set Blob type accordingly.
 *  2. Set data_tyep accordingly.
 *  3. Infer the blob shapes.
 *  4. Check the weights required.
 */
Status DefaultNetwork::InitLayers(NetStructure *net_structure, NetResource *net_resource) {
    Status ret = TNN_OK;
    for (auto layer_info : net_structure->layers) {
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
            // Check for int8
            bool is_int8_blob = layer_info->param->quantized && blob->GetBlobDesc().data_type != DATA_TYPE_INT8;
            if (is_int8_blob) {
                RETURN_ON_NEQ(GenerateInt8Blob(name, net_resource, &blob), TNN_OK);
            }
            // Check for bfp16
            if (config_.precision == PRECISION_LOW && blob->GetBlobDesc().data_type != DATA_TYPE_INT8) {
                blob->GetBlobDesc().data_type = DATA_TYPE_BFP16;
            }
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

        for (auto name : output_names) {
            auto blob = blob_manager_->GetBlob(name);
            bool is_int8_blob =
                layer_info->param->quantized ||
                (type == LAYER_REFORMAT &&
                 reinterpret_cast<ReformatLayerParam *>(layer_info->param.get())->dst_type == DATA_TYPE_INT8);
            // Check for int8
            if (is_int8_blob) {
                RETURN_ON_NEQ(GenerateInt8Blob(name, net_resource, &blob), TNN_OK);
            }
            // Check for bfp16
            if (config_.precision == PRECISION_LOW && blob->GetBlobDesc().data_type != DATA_TYPE_INT8) {
                blob->GetBlobDesc().data_type = DATA_TYPE_BFP16;
            }
            outputs.push_back(blob);
        }

        LayerResource *layer_resource = nullptr;
        if (net_resource->resource_map.count(layer_name) != 0) {
            layer_resource = net_resource->resource_map[layer_name].get();
        }

        ret = cur_layer->Init(context_, layer_info->param.get(), layer_resource, inputs, outputs, device_);
        if (ret != TNN_OK) {
            LOGE("Error Init layer %s (err: %d or 0x%X)\n", cur_layer->GetLayerName().c_str(), (int)ret, (int)ret);
            return ret;
        }

        layers_.push_back(cur_layer);
    }
    return ret;
}

Status DefaultNetwork::GenerateInt8Blob(const std::string &name, NetResource *net_resource, Blob **blob) {
    auto new_blob = new BlobInt8((*blob)->GetBlobDesc(), (*blob)->GetHandle());
    CHECK_PARAM_NULL(new_blob);

    std::string blob_scale_name = name + "_scale_data_";
#ifdef BENCHMARK
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
    for (auto iter : inputs) {
        Blob *blob = blob_manager_->GetBlob(iter.first);
        if (blob == nullptr) {
            LOGE("DefaultNetwork reshape blob is empty\n");
            return Status(TNNERR_PARAM_ERR, "DefaultNetwork reshape blob is empty");
        }
        blob->GetBlobDesc().dims = iter.second;
    }

    Status ret = TNN_OK;
    for (auto cur_layer : layers_) {
        ret = cur_layer->Reshape();
        if (ret != TNN_OK) {
            return ret;
        }
    }
    return ret;
}

Status DefaultNetwork::DeInit() {
    for (int i = 0; i < layers_.size(); i++) {
        if (layers_[i] != NULL) {
            delete layers_[i];
        }
    }
    layers_.clear();

    if (blob_manager_ != NULL) {
        delete blob_manager_;
        blob_manager_ = NULL;
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

Status DefaultNetwork::Forward() {
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

#if DUMP_INPUT_BLOB
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
#endif  // DUMP_INPUT_BLOB

        result = layer->Forward();
        LOGD("layer name: %s, forward result: %d \n", layer->GetLayerName().c_str(), (int)result);
        if (result != TNN_OK) {
            LOGE("Forward error %s, exit\n", result.description().c_str());
            return result;
        }

#if DUMP_OUTPUT_BLOB
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
#endif  // DUMP_OUTPUT_BLOB

        cnt++;
    }
    context_->OnInstanceForwardEnd();
    context_->Synchronize();
    return result;
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
        if (result != TNN_OK) {
            LOGE("Forward error %s, exit\n", result.description().c_str());
            return result;
        }
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

}  // namespace TNN_NS
