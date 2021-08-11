// Copyright 2019 Tencent. All Rights Reserved

#include "tnn/device/atlas/atlas_network.h"
#include <time.h>
#include <chrono>
#include "tnn/device/atlas/atlas_common_types.h"
#include "tnn/device/atlas/atlas_model_interpreter.h"
#include "tnn/device/atlas/atlas_runtime.h"
#include "tnn/device/atlas/atlas_utils.h"
#include "tnn/utils/dims_vector_utils.h"

namespace TNN_NS {

NetworkImplFactoryRegister<NetworkImplFactory<AtlasNetwork>> g_network_impl_atlas_factory_register(NETWORK_TYPE_ATLAS);

AtlasNetwork::~AtlasNetwork() {
    if (need_to_deinit) {
        DeInit();
    }
}

Status AtlasNetwork::Init(NetworkConfig &net_config, ModelConfig &model_config, AbstractModelInterpreter *interpreter,
                          InputShapesMap min_inputs_shape, InputShapesMap max_inputs_shape, bool enable_const_folder) {
    need_to_deinit = true;

    AtlasModelInterpreter *atlas_interpreter = dynamic_cast<AtlasModelInterpreter *>(interpreter);
    model_weight_size_                       = atlas_interpreter->GetModelWeightsBufferSize();

    // Init ACL
    Status ret = AtlasRuntime::Init();
    if (ret != TNN_OK) {
        LOGE("acl init falied\n");
        return ret;
    }

    // Set Device
    ret = AtlasRuntime::GetInstance()->SetDevice(net_config.device_id);
    if (ret != TNN_OK) {
        LOGE("acl set device falied\n");
        return ret;
    }

    // Get model weights buffer ptr
    model_weight_ptr_ = atlas_interpreter->GetModelWeightsBufferPtr(net_config.device_id);
    if (model_weight_ptr_ == nullptr) {
        LOGE("get model weight buffer ptr falied\n");
        return Status(TNNERR_ATLAS_RUNTIME_ERROR, "get model weight buffer ptr falied");
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
    if (atlas_interpreter->GetModelConfig().is_path) {
        LOGD("load model form file\n");
        ret = LoadModelFromFile(atlas_interpreter->GetModelConfig().om_str);
    } else {
        LOGD("load model form memory\n");
        ret = LoadModelFromMemory(atlas_interpreter->GetModelConfig().om_str);
    }
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
    model_info.has_aipp      = has_aipp_;
    for (auto item : input_blob_map_) {
        if (aipp_input_format_map_.find(item.first) != aipp_input_format_map_.end())
            model_info.aipp_input_format = aipp_input_format_map_[item.first];
        else
            model_info.aipp_input_format = ACL_AIPP_RESERVED;
        AtlasRuntime::GetInstance()->AddModelInfo(item.second, model_info);
    }

    // set dynamic batch size if needed. must do if input is dynamic batch
    for (auto item : input_blob_map_) {
        ret = SetDynamicBatchSize(item.first, item.second->GetBlobDesc().dims[0]);
        if (ret != TNN_OK)
            return ret;
    }

    // reshape if needed
    ret = Reshape(max_inputs_shape);
    if (ret != TNN_OK)
        return ret;

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
    aclError ret = aclrtSetCurrentContext(context_);
    if (ret != ACL_ERROR_NONE) {
        LOGE("set context failed\n");
        return Status(TNNERR_ATLAS_RUNTIME_ERROR, "set context failed");
    }

    for (auto item : inputs) {
        if (input_blob_map_.find(item.first) != input_blob_map_.end()) {
            auto dims_org = input_blob_map_[item.first]->GetBlobDesc().dims;
            auto dims     = item.second;
            LOGD("reshape input %s form [%d,%d,%d,%d] to [%d,%d,%d,%d]\n", item.first.c_str(), dims_org[0], dims_org[1],
                 dims_org[2], dims_org[3], dims[0], dims[1], dims[2], dims[3]);
            input_blob_map_[item.first]->GetBlobDesc().dims = dims;

            if (dims_org[0] == dims[0] && dims_org[1] == dims[1] && dims_org[2] == dims[2] && dims_org[3] == dims[3]) {
                LOGD("input shape is same, no need to do reshape!\n");
                continue;
            }

            Status tnn_ret = SetDynamicBatchSize(item.first, dims[0]);
            if (TNN_OK != tnn_ret)
                return tnn_ret;
        }
    }

    return TNN_OK;
}

Status AtlasNetwork::DeInit() {
    aclError ret = ACL_ERROR_NONE;
    if (nullptr != context_) {
        ret = aclrtSetCurrentContext(context_);
        if (ret != ACL_ERROR_NONE) {
            LOGE("set context failed\n");
            return Status(TNNERR_ATLAS_RUNTIME_ERROR, "set context failed");
        }
    }

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
    if (nullptr != input_) {
        DestroyDataset(input_);
    }
    LOGD("acl destroy output dataset\n");
    if (nullptr != output_) {
        DestroyDataset(output_);
    }

    UnloadModel();

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

    AtlasRuntime::DecreaseRef();
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

Status AtlasNetwork::LoadModelFromFile(const std::string &om_file) {
    size_t temp_size;
    aclError ret = aclmdlQuerySize(om_file.c_str(), &model_mem_size_, &temp_size);
    if (ret != ACL_ERROR_NONE) {
        LOGE("query model failed (ret=%d), model file is %s\n", ret, om_file.c_str());
        return Status(TNNERR_ATLAS_RUNTIME_ERROR, "query model failed");
    }
    LOGD("atlas model mem size: %d\n", model_mem_size_);

    ret = aclrtMalloc(&model_mem_ptr_, model_mem_size_, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_ERROR_NONE) {
        LOGE("malloc buffer for mem failed, require size is %zu\n", model_mem_size_);
        return Status(TNNERR_ATLAS_RUNTIME_ERROR, "malloc buffer for mem failed");
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

Status AtlasNetwork::LoadModelFromMemory(const std::string &om_content) {
    size_t temp_size;
    aclError ret = aclmdlQuerySizeFromMem(om_content.data(), om_content.length(), &model_mem_size_, &temp_size);
    if (ret != ACL_ERROR_NONE) {
        LOGE("query model failed (ret=%d)\n", ret);
        return Status(TNNERR_ATLAS_RUNTIME_ERROR, "query model failed");
    }
    LOGD("atlas model mem size: %d\n", model_mem_size_);

    ret = aclrtMalloc(&model_mem_ptr_, model_mem_size_, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_ERROR_NONE) {
        LOGE("malloc buffer for mem failed, require size is %zu\n", model_mem_size_);
        return Status(TNNERR_ATLAS_RUNTIME_ERROR, "malloc buffer for mem failed");
    }

    ret = aclmdlLoadFromMemWithMem(om_content.data(), om_content.length(), &model_id_, model_mem_ptr_, model_mem_size_,
                                   model_weight_ptr_, model_weight_size_);
    if (ret != ACL_ERROR_NONE) {
        LOGE("load model from file failed\n");
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
        LOGD("AllocateDataset for input (count=%d)\n", count);
    } else {
        count = aclmdlGetNumOutputs(model_desc_);
        LOGD("AllocateDataset for output (count=%d)\n", count);
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
        LOGD("acl malloc buffer size: %zu  addr: 0x%lx\n", buffer_size, (long long)buffer);

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

    Status ret            = TNN_OK;
    std::string blob_name = "";
    std::vector<int> io_dims;
    aclDataType data_type;
    aclFormat data_format;

    io_dims.clear();
    if (is_input) {
        // get blob name
        blob_name = aclmdlGetInputNameByIndex(model_desc_, index);
        // skip dynamic aipp input
        if (blob_name.find(ACL_DYNAMIC_AIPP_NAME) != std::string::npos) {
            LOGD("find dynamic aipp input (%s) and skip...\n", blob_name.c_str());
            return TNN_OK;
        }
        // skip dynamic batch input
        if (blob_name.find(ACL_DYNAMIC_TENSOR_NAME) != std::string::npos) {
            LOGD("find dynamic batch input (%s) and skip...\n", blob_name.c_str());
            dynamic_batch_name_.push_back(blob_name);
            return TNN_OK;
        }

        // get dims info and data format
        ret = GetInputInfo(index, io_dims, data_format, data_type);
        if (TNN_OK != ret) {
            return ret;
        }

        LOGD("input data type: %d  input data format: %d\n", data_type, data_format);
        LOGD("input shape:\n");
        for (int i = 0; i < io_dims.size(); ++i) {
            LOGD("[%d]\n", io_dims[i]);
        }
    } else {
        // get blob name
        blob_name = aclmdlGetOutputNameByIndex(model_desc_, index);
        // get dims info
        aclmdlIODims acl_dims;
        aclError acl_ret = aclmdlGetOutputDims(model_desc_, index, &acl_dims);
        if (acl_ret != ACL_ERROR_NONE) {
            LOGE("can't get output dims\n");
            return Status(TNNERR_ATLAS_RUNTIME_ERROR, "can't get output dims");
        }
        // get dims0
        int max_batch = GetMaxBatchSize(model_desc_, 1);
        if (0 == max_batch) {
            LOGE("get batch size failed\n");
            return Status(TNNERR_ATLAS_RUNTIME_ERROR, "get batch size failed");
        }
        output_dim0_map_[blob_name] = (int)acl_dims.dims[0] / max_batch;
        // get data type
        data_type = aclmdlGetOutputDataType(model_desc_, index);
        // get data format
        data_format = aclmdlGetOutputFormat(model_desc_, index);

        LOGD("output data type: %d  output data format: %d\n", data_type, data_format);
        LOGD("output shape:\n");
        for (int i = 0; i < acl_dims.dimCount; ++i) {
            io_dims.push_back((int)acl_dims.dims[i]);
            LOGD("[%d]\n", (int)acl_dims.dims[i]);
        }
    }

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
    for (int i = 0; i < io_dims.size(); ++i) {
        blob_desc.dims.push_back((int)io_dims[i]);
    }
    for (int i = io_dims.size(); i < 4; ++i) {
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

Status AtlasNetwork::GetInputInfo(size_t index, std::vector<int> &input_dims, aclFormat &input_format,
                                  aclDataType &input_data_type) {
    std::string blob_name = aclmdlGetInputNameByIndex(model_desc_, index);
    aclAippInfo aipp_info;
    aclError acl_ret = aclmdlGetFirstAippInfo(model_id_, index, &aipp_info);

    input_dims.clear();
    if (ACL_ERROR_NONE == acl_ret) {
        // has static aipp
        has_aipp_ = true;
        LOGD("shapeCount: %d   srcDimNum: %d\n", aipp_info.shapeCount, aipp_info.srcDimNum);
        // get aipp input format
        aipp_input_format_map_[blob_name] = aipp_info.inputFormat;

        // get data format
        input_format = aipp_info.srcFormat;

        // get data type
        input_data_type = aipp_info.srcDatatype;

        if (aipp_info.shapeCount < 1) {
            LOGE("model input is less than 1\n");
            return Status(TNNERR_ATLAS_RUNTIME_ERROR, "model input is less than 1");
        }
        // get the max input dims
        aclmdlIODims acl_dims = aipp_info.outDims[0].srcDims;
        for (int i = 0; i < acl_dims.dimCount; ++i) {
            input_dims.push_back((int)acl_dims.dims[i]);
        }

        for (int i = 1; i < aipp_info.shapeCount; ++i) {
            acl_dims = aipp_info.outDims[i].srcDims;
            for (int i = 0; i < acl_dims.dimCount; ++i) {
                input_dims[i] = std::max((int)acl_dims.dims[i], input_dims[i]);
            }
        }
    } else {
        LOGE("get aipp info failed (ret=%d), use input info directly\n", acl_ret);
        // get aipp input format
        aipp_input_format_map_[blob_name] = ACL_AIPP_RESERVED;

        // get data format
        input_format = aclmdlGetInputFormat(model_desc_, index);

        // get data type
        input_data_type = aclmdlGetInputDataType(model_desc_, index);

        // get dims info
        aclmdlIODims acl_dims;
        aclError acl_ret = aclmdlGetInputDims(model_desc_, index, &acl_dims);
        if (acl_ret != ACL_ERROR_NONE) {
            LOGE("can't get input dims\n");
            return Status(TNNERR_ATLAS_RUNTIME_ERROR, "can't get input dims");
        }
        // in dynamic batch input, reset batch
        if (-1 == acl_dims.dims[0]) {
            auto buffer_size = aclmdlGetInputSizeByIndex(model_desc_, index);
            int chw_size     = aclDataTypeSize(input_data_type);
            for (int i = 1; i < acl_dims.dimCount; ++i) {
                chw_size *= acl_dims.dims[i];
            }
            acl_dims.dims[0] = buffer_size / chw_size;

            LOGD("dynamic batch input, batch is set to %d\n", acl_dims.dims[0]);
        }
        for (int i = 0; i < acl_dims.dimCount; ++i) {
            input_dims.push_back((int)acl_dims.dims[i]);
        }
    }

    return TNN_OK;
}

Status AtlasNetwork::SetDynamicBatchSize(std::string blob_name, int batch_size) {
    if (IsDynamicBatch(model_desc_, blob_name) && dynamic_batch_name_.size() > 0) {
        // set dynamic batch
        size_t index     = 0;
        aclError acl_ret = aclmdlGetInputIndexByName(model_desc_, dynamic_batch_name_[0].c_str(), &index);
        if (acl_ret != ACL_ERROR_NONE) {
            LOGE("get dynamic batch input index falied!\n");
            return Status(TNNERR_ATLAS_RUNTIME_ERROR, "get dynamic batch input index falied");
        }
        acl_ret = aclmdlSetDynamicBatchSize(model_id_, input_, index, batch_size);
        if (acl_ret != ACL_ERROR_NONE) {
            LOGE("set batch size (%s) in reshape failed\n", blob_name.c_str());
            return Status(TNNERR_ATLAS_RUNTIME_ERROR, "set batch size in reshape failed");
        }
        LOGD("input (%s) set dynamic batch size %d (index: %d)\n", blob_name.c_str(), batch_size, index);

        // set output batch size
        for (auto output_item : output_blob_map_) {
            output_item.second->GetBlobDesc().dims[0] = output_dim0_map_[output_item.first] * batch_size;
        }
    } else {
        LOGD("not dymamic batch input, skip\n");
    }

    return TNN_OK;
}

void AtlasNetwork::DestroyDataset(aclmdlDataset *&data_set) {
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
    data_set = nullptr;
}

}  // namespace TNN_NS
