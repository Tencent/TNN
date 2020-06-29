// Copyright 2019 Tencent. All Rights Reserved

#include "atlas_network.h"
#include <time.h>
#include <chrono>
#include "atlas_common_types.h"
#include "atlas_model_interpreter.h"
#include "atlas_runtime.h"
#include "atlas_utils.h"
#include "tnn/utils/dims_vector_utils.h"

namespace TNN_NS {

NetworkImplFactoryRegister<NetworkImplFactory<AtlasNetwork>> g_network_impl_atlas_factory_register(NETWORK_TYPE_ATLAS);

AtlasNetwork::~AtlasNetwork() {
    DeInit();
    AtlasRuntime::DecreaseRef();
}

Status AtlasNetwork::Init(NetworkConfig &net_config, ModelConfig &model_config, AbstractModelInterpreter *interpreter,
                          InputShapesMap inputs_shape) {
    AtlasModelInterpreter *atlas_interpreter = dynamic_cast<AtlasModelInterpreter *>(interpreter);

    atlas_config_ = atlas_interpreter->GetModelConfig();

    // Init ACL
    Status ret = AtlasRuntime::GetInstance()->Init();
    if (ret != TNN_OK) {
        LOGE("acl init falied\n");
        return ret;
    }
    AtlasRuntime::IncreaseRef();

    // Set Device
    ret = AtlasRuntime::GetInstance()->SetDevice(net_config.device_id);
    if (ret != TNN_OK) {
        LOGE("acl set device falied\n");
        return ret;
    }

    // Create Context
    aclError acl_ret = aclrtCreateContext(&context_, net_config.device_id);
    if (acl_ret != ACL_ERROR_NONE) {
        LOGE("acl create context failed (acl error code: %d)\n", acl_ret);
        return Status(TNNERR_ATLAS_RUNTIME_ERROR, "acl create context falied");
    }

    // Create Stream
    acl_ret = aclrtCreateStream(&stream_);
    if (acl_ret != ACL_ERROR_NONE) {
        LOGE("acl create stream failed (acl error code: %d)\n", acl_ret);
        return Status(TNNERR_ATLAS_RUNTIME_ERROR, "acl create stream falied");
    }

    command_queue_.reset(new AtlasCommandQueue());
    command_queue_->context = context_;
    command_queue_->stream  = stream_;

    // Load model
    ret = LoadModelFromFile(atlas_config_.om_path);
    if (ret != TNN_OK)
        return ret;

    // allocate input and output
    ret = AllocateDataset(&input_, true);
    if (ret != TNN_OK)
        return ret;
    ret = AllocateDataset(&output_, false);
    if (ret != TNN_OK)
        return ret;

    // add model info
    AtlasModelInfo model_info;
    model_info.model_desc    = model_desc_;
    model_info.model_id      = model_id_;
    model_info.input_dataset = input_;
    for (auto item : input_blob_map_) {
        AtlasRuntime::GetInstance()->AddModelInfo(item.second, model_info);
    }

    return TNN_OK;
}

Status AtlasNetwork::GetForwardMemorySize(int &memory_size) {
    memory_size = model_mem_size_;
    return TNN_OK;
}

Status AtlasNetwork::SetForwardMemory(void *memory) {
    LOGE("Not support setting forward memory in Atlas!\n");
    return Status(TNNERR_DEVICE_NOT_SUPPORT, "Not support setting forward memory in Atlas!");
}

Status AtlasNetwork::GetAllInputBlobs(BlobMap &blobs) {
    blobs = input_blob_map_;
    return TNN_OK;
}

Status AtlasNetwork::GetAllOutputBlobs(BlobMap &blobs) {
    blobs = output_blob_map_;
    return TNN_OK;
}

Status AtlasNetwork::Reshape(const InputShapesMap &inputs) {
    LOGE("Not support reshape in Atlas!\n");
    return Status(TNNERR_DEVICE_NOT_SUPPORT, "Not support reshape in Atlas!");
}

Status AtlasNetwork::DeInit() {
    for (auto item : input_blob_map_) {
        if (nullptr != item.second) {
            // delete model info
            AtlasRuntime::GetInstance()->DelModelInfo(item.second);
            delete item.second;
        }
    }
    input_blob_map_.clear();
    for (auto item : output_blob_map_) {
        if (nullptr != item.second) {
            delete item.second;
        }
    }
    output_blob_map_.clear();

    LOGD("acl destroy input dataset\n");
    DestroyDataset(input_);
    LOGD("acl destroy output dataset\n");
    DestroyDataset(output_);

    UnloadModel();

    aclError ret;
    if (nullptr != stream_) {
        ret = aclrtDestroyStream(stream_);
        LOGD("acl destroy stream\n");
        if (ret != ACL_ERROR_NONE) {
            LOGE("destroy stream failed\n");
        }
        stream_ = nullptr;
    }

    if (nullptr != context_) {
        ret = aclrtDestroyContext(context_);
        LOGD("acl destroy context\n");
        if (ret != ACL_ERROR_NONE) {
            LOGE("destroy context failed\n");
        }
        context_ = nullptr;
    }

    return TNN_OK;
}

Status AtlasNetwork::GetCommandQueue(void **command_queue) {
    *command_queue = command_queue_.get();
    return TNN_OK;
}

Status AtlasNetwork::Forward() {
    LOGD("Atlas Forward!\n");

    aclError ret = aclrtSetCurrentContext(context_);
    if (ret != ACL_ERROR_NONE) {
        LOGE("set context failed\n");
        return Status(TNNERR_ATLAS_RUNTIME_ERROR, "set context failed");
    }

    ret = aclmdlExecute(model_id_, input_, output_);
    if (ret != ACL_ERROR_NONE) {
        LOGE("execute model failed, modelId is %u\n", model_id_);
        return Status(TNNERR_ATLAS_RUNTIME_ERROR, "execute model failed");
    }

    return TNN_OK;
}

Status AtlasNetwork::ForwardAsync(Callback call_back) {
    LOGD("Atlas Async Forward! (as same as Forward by now)\n");
    return Forward();
}

Status AtlasNetwork::LoadModelFromFile(std::string om_file) {
    aclError ret = aclmdlQuerySize(om_file.c_str(), &model_mem_size_, &model_weight_size_);
    if (ret != ACL_ERROR_NONE) {
        LOGE("query model failed, model file is %s\n", om_file.c_str());
        return Status(TNNERR_ATLAS_RUNTIME_ERROR, "query model failed");
    }

    ret = aclrtMalloc(&model_mem_ptr_, model_mem_size_, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_ERROR_NONE) {
        LOGE("malloc buffer for mem failed, require size is %zu\n", model_mem_size_);
        return Status(TNNERR_ATLAS_RUNTIME_ERROR, "malloc buffer for mem failed");
    }

    ret = aclrtMalloc(&model_weight_ptr_, model_weight_size_, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_ERROR_NONE) {
        LOGE("malloc buffer for weight failed, require size is %zu\n", model_weight_size_);
        return Status(TNNERR_ATLAS_RUNTIME_ERROR, "malloc buffer for weight failed");
    }

    ret = aclmdlLoadFromFileWithMem(om_file.c_str(), &model_id_, model_mem_ptr_, model_mem_size_, model_weight_ptr_,
                                    model_weight_size_);
    if (ret != ACL_ERROR_NONE) {
        LOGE("load model from file failed, model file is %s\n", om_file.c_str());
        return Status(TNNERR_ATLAS_RUNTIME_ERROR, "load model from file failed");
    }

    // create model desc to get model info
    model_desc_ = aclmdlCreateDesc();
    if (nullptr == model_desc_) {
        LOGE("create model description failed\n");
        return Status(TNNERR_ATLAS_RUNTIME_ERROR, "create model description failed");
    }

    ret = aclmdlGetDesc(model_desc_, model_id_);
    if (ret != ACL_ERROR_NONE) {
        LOGE("get model description failed\n");
        return Status(TNNERR_ATLAS_RUNTIME_ERROR, "get model description failed");
    }

    return TNN_OK;
}

void AtlasNetwork::UnloadModel() {
    aclError ret = aclmdlUnload(model_id_);
    LOGD("acl unload model\n");
    if (ret != ACL_ERROR_NONE) {
        LOGE("unload model failed, modelId is %u\n", model_id_);
    }

    if (nullptr != model_desc_) {
        (void)aclmdlDestroyDesc(model_desc_);
        LOGD("acl destroy model desc\n");
        model_desc_ = nullptr;
    }

    if (nullptr != model_mem_ptr_) {
        aclrtFree(model_mem_ptr_);
        LOGD("acl free model mem ptr\n");
        model_mem_ptr_  = nullptr;
        model_mem_size_ = 0;
    }

    if (nullptr != model_weight_ptr_) {
        aclrtFree(model_weight_ptr_);
        LOGD("acl free model weight ptr\n");
        model_weight_ptr_  = nullptr;
        model_weight_size_ = 0;
    }
}

Status AtlasNetwork::AllocateDataset(aclmdlDataset **data_set, bool is_input) {
    if (nullptr == model_desc_) {
        LOGE("no model description, create ouput failed\n");
        return Status(TNNERR_ATLAS_RUNTIME_ERROR, "no model description, create ouput failed");
    }

    *data_set = aclmdlCreateDataset();
    if (nullptr == *data_set) {
        LOGE("can't create dataset, create output failed\n");
        return Status(TNNERR_ATLAS_RUNTIME_ERROR, "can't create dataset, create output failed");
    }

    size_t count = 0;
    if (is_input) {
        count = aclmdlGetNumInputs(model_desc_);
    } else {
        count = aclmdlGetNumOutputs(model_desc_);
    }
    for (size_t i = 0; i < count; ++i) {
        size_t buffer_size = 0;
        if (is_input) {
            buffer_size = aclmdlGetInputSizeByIndex(model_desc_, i);
        } else {
            buffer_size = aclmdlGetOutputSizeByIndex(model_desc_, i);
        }

        void *buffer     = nullptr;
        aclError acl_ret = aclrtMalloc(&buffer, buffer_size, ACL_MEM_MALLOC_NORMAL_ONLY);
        if (acl_ret != ACL_ERROR_NONE) {
            LOGE("can't malloc buffer, size is %zu\n", buffer_size);
            return Status(TNNERR_ATLAS_RUNTIME_ERROR, "can't malloc buffer");
        }

        aclDataBuffer *data_buffer = aclCreateDataBuffer(buffer, buffer_size);
        if (acl_ret != ACL_ERROR_NONE) {
            LOGE("can't create data buffer\n");
            aclrtFree(buffer);
            return Status(TNNERR_ATLAS_RUNTIME_ERROR, "can't create data buffer");
        }

        acl_ret = aclmdlAddDatasetBuffer(*data_set, data_buffer);
        if (acl_ret != ACL_ERROR_NONE) {
            LOGE("can't add data buffer, create output failed\n");
            aclrtFree(buffer);
            aclDestroyDataBuffer(data_buffer);
            return Status(TNNERR_ATLAS_RUNTIME_ERROR, "can't add data buffer");
        }

        Status ret = AddBlobToMap(i, buffer, is_input);
        if (TNN_OK != ret) {
            return ret;
        }
    }

    return TNN_OK;
}

Status AtlasNetwork::AddBlobToMap(size_t index, void *data, bool is_input) {
    if (nullptr == model_desc_) {
        LOGE("no model description\n");
        return Status(TNNERR_ATLAS_RUNTIME_ERROR, "no model description");
    }

    std::string blob_name = "";
    aclmdlIODims acl_dims;
    aclDataType data_type;
    aclFormat data_format;

    if (is_input) {
        // get blob name
        blob_name = aclmdlGetInputNameByIndex(model_desc_, index);
        // get dims info
        aclError acl_ret = aclmdlGetInputDims(model_desc_, index, &acl_dims);
        if (acl_ret != ACL_ERROR_NONE) {
            LOGE("can't get input dims\n");
            return Status(TNNERR_ATLAS_RUNTIME_ERROR, "can't get input dims");
        }
        // get data type
        data_type = aclmdlGetInputDataType(model_desc_, index);
        // get data format
        data_format = aclmdlGetInputFormat(model_desc_, index);
        LOGD("input data type: %d  input data format: %d\n", data_type, data_format);
    } else {
        // get blob name
        blob_name = aclmdlGetOutputNameByIndex(model_desc_, index);
        // get dims info
        aclError acl_ret = aclmdlGetOutputDims(model_desc_, index, &acl_dims);
        if (acl_ret != ACL_ERROR_NONE) {
            LOGE("can't get output dims\n");
            return Status(TNNERR_ATLAS_RUNTIME_ERROR, "can't get output dims");
        }
        // get data type
        data_type = aclmdlGetOutputDataType(model_desc_, index);
        // get data format
        data_format = aclmdlGetOutputFormat(model_desc_, index);
        LOGD("output data type: %d  output data format: %d\n", data_type, data_format);
    }

    Status ret = TNN_OK;
    BlobDesc blob_desc;
    blob_desc.device_type = DEVICE_ATLAS;
    ret                   = ConvertFromAclDataTypeToTnnDataType(data_type, blob_desc.data_type);
    if (TNN_OK != ret) {
        LOGE("convert from acl data type to tnn data type falied\n");
        return ret;
    }
    ret = ConvertFromAclDataFormatToTnnDataFormat(data_format, blob_desc.data_format);
    if (TNN_OK != ret) {
        LOGE("convert from acl data format to tnn data format falied\n");
        return ret;
    }
    for (int i = 0; i < acl_dims.dimCount; ++i) {
        blob_desc.dims.push_back((int)acl_dims.dims[i]);
    }
    for (int i = acl_dims.dimCount; i < 4; ++i) {
        blob_desc.dims.push_back(1);
    }
    blob_desc.name = blob_name;

    BlobHandle blob_handle;
    blob_handle.base = data;

    Blob *blob = new Blob(blob_desc, blob_handle);

    if (is_input) {
        input_blob_map_[blob_name] = blob;
    } else {
        output_blob_map_[blob_name] = blob;
    }

    return TNN_OK;
}

void AtlasNetwork::DestroyDataset(aclmdlDataset *data_set) {
    if (nullptr == data_set) {
        return;
    }

    for (size_t i = 0; i < aclmdlGetDatasetNumBuffers(data_set); ++i) {
        aclDataBuffer *data_buffer = aclmdlGetDatasetBuffer(data_set, i);
        void *data                 = aclGetDataBufferAddr(data_buffer);
        (void)aclrtFree(data);
        (void)aclDestroyDataBuffer(data_buffer);
    }

    (void)aclmdlDestroyDataset(data_set);
}

}  // namespace TNN_NS
