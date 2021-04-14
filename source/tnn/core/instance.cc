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

#include "tnn/core/instance.h"

#include <memory>

#include "tnn/core/abstract_network.h"
#include "tnn/core/common.h"
#include "tnn/core/macro.h"
#include "tnn/core/profile.h"
#include "tnn/core/status.h"
#include "tnn/interpreter/abstract_model_interpreter.h"
#include "tnn/utils/dims_vector_utils.h"

namespace TNN_NS {

/*
 * The Instance Object mainly holds the network object now.
 * It wraps the network object to keep consistency of the header.
 */

Instance::Instance(NetworkConfig &net_config, ModelConfig &model_config) {
    net_config_   = net_config;
    model_config_ = model_config; // note that, the params in model_config is empty, don't use it
}
Instance::~Instance() {
    DeInit();
}

Status Instance::Init(std::shared_ptr<AbstractModelInterpreter> interpreter, InputShapesMap inputs_shape) {
    interpreter_ = interpreter;

    /*
     * NetworkImpl is register by each Impl.
     * TNN model runs with the default_network.
     */
    network_ = NetworkImplManager::GetNetworkImpl(net_config_.network_type);
    if (!network_) {
        LOGE("ERROR: network_ is nil, network_type may not support\n");
        return Status(TNNERR_NET_ERR, "network_ is nil, network_type may not support");
    }
    auto ret = network_->Init(net_config_, model_config_, interpreter.get(), inputs_shape);
    RETURN_ON_NEQ(ret, TNN_OK);

    return TNN_OK;
}

Status Instance::DeInit() {
    network_ = nullptr;
    return TNN_OK;
}

Status Instance::GetForwardMemorySize(int &memory_size) {
    return network_->GetForwardMemorySize(memory_size);
}

Status Instance::SetForwardMemory(void *memory) {
    return network_->SetForwardMemory(memory);
}

Status Instance::Reshape(const InputShapesMap &inputs) {
    return (Status)network_->Reshape(inputs);
}

Status Instance::GetCommandQueue(void **command_queue) {
    return (Status)network_->GetCommandQueue(command_queue);
}

Status Instance::ShareCommandQueue(Instance *instance) {
    return network_->ShareCommandQueue(instance->GetNetwork());
}

AbstractNetwork *Instance::GetNetwork() {
    return network_.get();
}

Status Instance::Forward() {
    output_mats_convert_status_.clear();
    return (Status)network_->Forward();
}

#ifdef FORWARD_CALLBACK_ENABLE
Status Instance::ForwardWithCallback(BlobStatisticCallback before, BlobStatisticCallback after) {
    output_mats_convert_status_.clear();
    return (Status)network_->ForwardWithCallback(before, after);
}
#endif  // end of FORWARD_CALLBACK_ENABLE

Status Instance::ForwardAsync(Callback call_back) {
    output_mats_convert_status_.clear();
    return (Status)network_->ForwardAsync(call_back);
}

Status Instance::GetAllInputBlobs(BlobMap &blobs) {
    return network_->GetAllInputBlobs(blobs);
}

Status Instance::GetAllOutputBlobs(BlobMap &blobs) {
    return network_->GetAllOutputBlobs(blobs);
}

Status Instance::SetCpuNumThreads(int num_threads) {
    return network_->SetCpuNumThreads(num_threads);
}

// set input Mat
Status Instance::SetInputMat(std::shared_ptr<Mat> mat, MatConvertParam param, std::string input_name) {
    if (!mat) {
        LOGE("input mat is empty ,please check!\n");
        return Status(TNNERR_PARAM_ERR, "input mat is empty ,please check!");
    }

    // get input blobs
    BlobMap input_blobs;
    auto status = network_->GetAllInputBlobs(input_blobs);
    if (status != TNN_OK || input_blobs.size() <= 0) {
        LOGE("instance.GetAllInputBlobs Error: %s\n", status.description().c_str());
        return status;
    }

    // insure name is valid, take the first input name for default
    if (input_name.length() <= 0) {
        input_name = input_blobs.begin()->first;
    } else {
        if (input_blobs.find(input_name) == input_blobs.end()) {
            LOGE("instance dont have the input with name: %s\n", input_name.c_str());
            return Status(TNNERR_MODEL_ERR, "instance dont have the input with name");
        }
    }

    // check blob convert
    std::shared_ptr<BlobConverter> blob_converter = nullptr;
    if (input_converters_.size() > 0 && input_converters_.find(input_name) != input_converters_.end()) {
        blob_converter = input_converters_[input_name];
    } else {
        auto input_blob               = input_blobs[input_name];
        blob_converter                = std::make_shared<BlobConverter>(input_blob);
        input_converters_[input_name] = blob_converter;
    }

    // get command queue
    void *command_queue = nullptr;
    network_->GetCommandQueue(&command_queue);

    status = blob_converter->ConvertFromMatAsync(*(mat.get()), param, command_queue);
    if (status != TNN_NS::TNN_OK) {
        LOGE("input_blob_convert.ConvertFromMatAsync Error: %s\n", status.description().c_str());
        return status;
    }

    return TNN_OK;
}

// get output Mat
Status Instance::GetOutputMat(std::shared_ptr<Mat> &mat, MatConvertParam param, std::string output_name,
                              DeviceType device, MatType mat_type) {
    // get output blobs
    BlobMap output_blobs;
    auto status = network_->GetAllOutputBlobs(output_blobs);
    if (status != TNN_OK || output_blobs.size() <= 0) {
        LOGE("instance.GetAllOutputBlobs Error: %s\n", status.description().c_str());
        return status;
    }

    // insure name is valid, take the first output name for default
    if (output_name.length() <= 0) {
        output_name = output_blobs.begin()->first;
    } else {
        if (output_blobs.find(output_name) == output_blobs.end()) {
            LOGE("instance dont have the output with name: %s\n", output_name.c_str());
            return Status(TNNERR_MODEL_ERR, "instance dont have the output with name");
        }
    }

    // check if it has been converted
    if (output_mats_convert_status_.find(output_name) != output_mats_convert_status_.end() &&
        output_mats_.find(output_name) != output_mats_.end()) {
        mat = output_mats_[output_name];
        return TNN_OK;
    }

    // check if it has been allocated or reallocated for dims change.
    // allocate output mat
    bool need_allocate = true;
    if (output_mats_.find(output_name) != output_mats_.end()) {
        auto mat_dims  = output_mats_[output_name]->GetDims();
        auto blob_dims = output_blobs[output_name]->GetBlobDesc().dims;
        if (DimsVectorUtils::Equal(mat_dims, blob_dims)) {
            need_allocate = false;
        }
    }

    if (need_allocate) {
        auto dims                 = output_blobs[output_name]->GetBlobDesc().dims;
        auto output_mat           = std::make_shared<TNN_NS::Mat>(device, mat_type, dims);
        output_mats_[output_name] = output_mat;
    }

    mat = output_mats_[output_name];

    // check blob convert
    std::shared_ptr<BlobConverter> blob_converter = nullptr;
    if (output_converters_.size() > 0 && output_converters_.find(output_name) != output_converters_.end()) {
        blob_converter = output_converters_[output_name];
    } else {
        auto input_blob                 = output_blobs[output_name];
        blob_converter                  = std::make_shared<BlobConverter>(input_blob);
        output_converters_[output_name] = blob_converter;
    }

    // get command queue
    void *command_queue = nullptr;
    network_->GetCommandQueue(&command_queue);
    status = blob_converter->ConvertToMat(*(mat.get()), param, command_queue);
    if (status == TNN_NS::TNN_OK) {
        // set output mat convert status
        output_mats_convert_status_[output_name] = 1;
    } else {
        LOGE("output_blob_convert.ConvertFromMat Error: %s\n", status.description().c_str());
    }

    return status;
}

#if TNN_PROFILE
void Instance::StartProfile() {
    network_->StartProfile();
}

std::string Instance::FinishProfile(bool do_print) {
    std::shared_ptr<ProfileResult> profile_result = network_->FinishProfile();
    std::string result_str                        = " ";
    if (profile_result) {
        result_str = profile_result->GetProfilingDataInfo();
        if (do_print) {
            printf("%s", result_str.c_str());
        }
    }

    return result_str;
}

#endif

}  // namespace TNN_NS
