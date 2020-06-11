// Copyright 2019 Tencent. All Rights Reserved

#include "hiai_network.h"
#include <stdlib.h>
#include "hiai_model_interpreter.h"
#include "hiai_utils.h"

namespace TNN_NS {

NetworkImplFactoryRegister<NetworkImplFactory<HiaiNetwork>>
    g_network_impl_hiai_factory_register(NETWORK_TYPE_HIAI);

HiaiNetwork::HiaiNetwork() {
    model_buffer_     = nullptr;
    model_manager_    = nullptr;
    model_tensorinfo_ = nullptr;
}

HiaiNetwork::~HiaiNetwork() {
    DeInit();
}

Status HiaiNetwork::Init(NetworkConfig &net_config, ModelConfig &model_config,
                         AbstractModelInterpreter *interpreter,
                         InputShapesMap inputs_shape) {
    HiaiModelInterpreter *hiai_interpreter =
        dynamic_cast<HiaiModelInterpreter *>(interpreter);

    model_name_            = model_config.params[0];
    std::string model_path = model_config.params[1];

    model_buffer_ = HIAI_MixModelBuffer_Create_From_File(
        model_name_.c_str(), model_path.c_str(), HIAI_MIX_DEVPREF_HIGH, false);
    if (model_buffer_ == nullptr) {
        LOGE("create mix model buffer falied!\n");
        return TNNERR_INVALID_MODEL;
    }

    HIAI_MixModelBuffer *model_buffer_array[] = {model_buffer_};
    model_manager_ = HIAI_MixModelManager_Create(NULL);
    if (model_manager_ == nullptr) {
        LOGE("create mix model manager falied!\n");
        return TNNERR_INVALID_MODEL;
    }

    int ret = HIAI_MixModel_LoadFromModelBuffers(model_manager_,
                                                 model_buffer_array, 1);
    if (ret != 0) {
        LOGE("model manager load model falied!\n");
        return TNNERR_INVALID_MODEL;
    }

    model_tensorinfo_ =
        HIAI_MixModel_GetModelTensorInfo(model_manager_, model_name_.c_str());

    if (model_tensorinfo_->input_shape == nullptr ||
        model_tensorinfo_->output_shape == nullptr) {
        LOGE("Get input or output shape falied!\n");
        return TNNERR_HIAI_API_ERROR;
    }

    // init input buffers
    for (int i = 0, pos = 0; i < model_tensorinfo_->input_cnt; ++i) {
        int n = model_tensorinfo_->input_shape[pos++];
        int c = model_tensorinfo_->input_shape[pos++];
        int h = model_tensorinfo_->input_shape[pos++];
        int w = model_tensorinfo_->input_shape[pos++];

        HIAI_MixTensorBuffer *input = HIAI_MixTensorBuffer_Create(n, c, h, w);
        if (input == nullptr) {
            LOGE("hiai allocate buffer failed!\n");
            return TNNERR_HIAI_API_ERROR;
        }
        input_buffers_.push_back(input);

        // add blob
        char layer_name[16];
        sprintf(layer_name, "%d", i);
        BlobDesc desc;
        desc.data_format = DATA_FORMAT_NCHW;
        desc.name        = layer_name;
        desc.dims.push_back(n);
        desc.dims.push_back(c);
        desc.dims.push_back(h);
        desc.dims.push_back(w);
        BlobHandle handle;
        handle.base                = HIAI_MixTensorBuffer_GetRawBuffer(input);
        input_blob_map_[desc.name] = new Blob(desc, handle);
    }

    // init output buffers
    for (int i = 0, pos = 0; i < model_tensorinfo_->output_cnt; ++i) {
        int n = model_tensorinfo_->output_shape[pos++];
        int c = model_tensorinfo_->output_shape[pos++];
        int h = model_tensorinfo_->output_shape[pos++];
        int w = model_tensorinfo_->output_shape[pos++];

        HIAI_MixTensorBuffer *output = HIAI_MixTensorBuffer_Create(n, c, h, w);
        if (output == nullptr) {
            LOGE("hiai allocate buffer failed!\n");
            return TNNERR_HIAI_API_ERROR;
        }
        output_buffers_.push_back(output);

        // add blob
        char layer_name[16];
        sprintf(layer_name, "%d", i);
        BlobDesc desc;
        desc.data_format = DATA_FORMAT_NCHW;
        desc.name        = layer_name;
        desc.dims.push_back(n);
        desc.dims.push_back(c);
        desc.dims.push_back(h);
        desc.dims.push_back(w);
        BlobHandle handle;
        handle.base                 = HIAI_MixTensorBuffer_GetRawBuffer(output);
        output_blob_map_[desc.name] = new Blob(desc, handle);
    }

    return TNN_OK;
}

Status HiaiNetwork::GetForwardMemorySize(int &memory_size) {
    memory_size = 0;
    return TNN_OK;
}

Status HiaiNetwork::SetForwardMemory(void *memory) {
    return TNN_OK;
}

Status HiaiNetwork::GetAllInputBlobs(BlobMap &blobs) {
    blobs = input_blob_map_;
    return TNN_OK;
}

Status HiaiNetwork::GetAllOutputBlobs(BlobMap &blobs) {
    blobs = output_blob_map_;
    return TNN_OK;
}

Status HiaiNetwork::Reshape(const InputShapesMap &inputs) {
    LOGE("Hiai Reshape!\n");
    return TNN_OK;
}

Status HiaiNetwork::DeInit() {
    for (auto buffer : input_buffers_) {
        if (buffer != nullptr)
            HIAI_MixTensorBufferr_Destroy(buffer);
    }

    for (auto buffer : output_buffers_) {
        if (buffer != nullptr)
            HIAI_MixTensorBufferr_Destroy(buffer);
    }

    if (model_buffer_ != nullptr) {
        HIAI_MixModelBuffer_Destroy(model_buffer_);
        model_buffer_ = nullptr;
    }

    if (model_tensorinfo_ != nullptr) {
        HIAI_MixModel_ReleaseModelTensorInfo(model_tensorinfo_);
    }

    HIAI_MixModel_UnLoadModel(model_manager_);
    if (model_manager_ != nullptr) {
        HIAI_MixModelManager_Destroy(model_manager_);
        model_manager_ = nullptr;
    }

    for (auto item : input_blob_map_) {
        delete item.second;
    }
    input_blob_map_.clear();
    for (auto item : output_blob_map_) {
        delete item.second;
    }
    output_blob_map_.clear();
    return TNN_OK;
}

Status HiaiNetwork::GetCommandQueue(void **command_queue) {
    return TNN_OK;
}

Status HiaiNetwork::Forward() {
    LOGE("Hiai Forward!\n");

    int ret = HIAI_MixModel_RunModel(
        model_manager_, input_buffers_.data(), model_tensorinfo_->input_cnt,
        output_buffers_.data(), model_tensorinfo_->output_cnt, 1000,
        model_name_.c_str());

    if (ret != 0) {
        LOGE("Forward failed!\n");
        return TNNERR_HIAI_API_ERROR;
    }

    return TNN_OK;
}

Status HiaiNetwork::ForwardAsync(Callback call_back) {
    LOGE("Hiai Async Forward! (as same as Forward by now)\n");

    int ret = HIAI_MixModel_RunModel(
        model_manager_, input_buffers_.data(), model_tensorinfo_->input_cnt,
        output_buffers_.data(), model_tensorinfo_->output_cnt, 1000,
        model_name_.c_str());

    if (ret != 0) {
        LOGE("Forward failed!\n");
        return TNNERR_HIAI_API_ERROR;
    }

    return TNN_OK;
}

}  // namespace TNN_NS
