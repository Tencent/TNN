// Copyright 2019 Tencent. All Rights Reserved

#include "tnn/device/atlas/atlas_network.h"
#include <time.h>
#include <chrono>
#include "tnn/device/atlas/atlas_common_types.h"
#include "tnn/device/atlas/atlas_model_interpreter.h"
#include "tnn/device/atlas/atlas_runtime.h"
#include "tnn/device/atlas/atlas_utils.h"
#include "tnn/utils/dims_vector_utils.h"
#include "tnn/utils/dims_vector_utils.h"

namespace TNN_NS {

NetworkImplFactoryRegister<NetworkImplFactory<AtlasNetwork>> g_network_impl_atlas_factory_register(NETWORK_TYPE_ATLAS);

AtlasNetwork::~AtlasNetwork() {
    if (need_to_deinit) {
        DeInit();
    }
}

Status AtlasNetwork::Init(NetworkConfig &net_config, ModelConfig &model_config, AbstractModelInterpreter *interpreter,
                          InputShapesMap min_inputs_shape, InputShapesMap max_inputs_shape, InputDataTypeMap inputs_data_type, 
                          bool enable_const_folder) {
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

    // deduce if dynamic input exists
    // get type of dynamic input if exists
    // type 1: Traditional Types
    //     --dynamic_batch_size
    //     --dynamic_image_size (hw)
    //     --dynamic_dims
    // type 2: Flexible Dynamic
    //     --input_shape_range
    ret = DeduceDynamicInputType();
    if (ret != TNN_OK)
        return ret;

    // allocate input and output
    ret = AllocateDatasetCreateBlob(&input_, max_inputs_shape, true);
    if (ret != TNN_OK)
        return ret;
    ret = AllocateDatasetCreateBlob(&output_, max_inputs_shape, false);
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

    // set dynamic batch size
    // must do if input is dynamic batch
    if (this->atc_mode_dynamic_batch_hw_dim_) {
        for (auto item : input_blob_map_) {
            ret = SetDynamicBatchSize(item.first, item.second->GetBlobDesc().dims[0]);
            if (ret != TNN_OK)
                return ret;
        }
    }

    // reshape if needed
    ret = Reshape(max_inputs_shape);
    if (ret != TNN_OK)
        return ret;

    return TNN_OK;
}

Status AtlasNetwork::GetForwardMemorySize(size_t &memory_size) {
    memory_size = model_mem_size_;
    return TNN_OK;
}

Status AtlasNetwork::SetCommandQueue(void *command_queue) {
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

            bool all_dims_equal = true;
            for (int d = 0; d < dims.size(); d++) {
                if (dims_org[d] != dims[d]) {
                    all_dims_equal = false;
                }
            }
            if (all_dims_equal) {
                LOGD("input '%s' shape is same, no need to do reshape.\n",
                     input_blob_map_[item.first]->GetBlobDesc().name.c_str());
                continue;
            }

            // Traditional Dynamic Batch, Set Input/Output Blob Shape.
            if (this->atc_mode_dynamic_batch_) {
                Status tnn_ret = SetDynamicBatchSize(item.first, dims[0]);
                if (TNN_OK != tnn_ret)
                    return tnn_ret;
            }

            // Range input for Model Converted with --input_shape_range
            // Range input output shape cannot be infered from input shape.
            // Output Shape will be deduced after ACL Forward() API is called.
            if (this->dynamic_input_shape_range_names_.find(item.first) !=
                this->dynamic_input_shape_range_names_.end()) {
                Status tnn_ret = SetRangeDynamicInputDim(item.first, dims);
                if (TNN_OK != tnn_ret)
                    return tnn_ret;
            }
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
        LOGE("set context & synchronize stream failed\n");
        return Status(TNNERR_ATLAS_RUNTIME_ERROR, "set context & synchronized failed");
    }

    ret = aclmdlExecute(model_id_, input_, output_);
    if (ret != ACL_ERROR_NONE) {
        LOGE("execute model failed, modelId is %u\n", model_id_);
        return Status(TNNERR_ATLAS_RUNTIME_ERROR, "execute model failed");
    }

    // For Range Dynamic Models with --input_shape_range
    // Update Output Blob Shapes here.
    if (!this->dynamic_input_shape_range_names_.empty()) {
        Status tnn_ret = UpdateRangeDynamicOutputDims();
        if (TNN_OK != tnn_ret) {
            return tnn_ret;
        }
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

    // Some model, e.g model Converted with atc config: --input_shape_range,
    // Does not have model_mem_size, aclrtMalloc EMPTY mem is NOT ALLOWED.
    if (model_mem_size_) {
        ret = aclrtMalloc(&model_mem_ptr_, model_mem_size_, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != ACL_ERROR_NONE) {
            LOGE("malloc buffer for mem failed, require size is %zu\n", model_mem_size_);
            return Status(TNNERR_ATLAS_RUNTIME_ERROR, "malloc buffer for mem failed");
        }

        ret = aclmdlLoadFromFileWithMem(om_file.c_str(), &model_id_, model_mem_ptr_, model_mem_size_, model_weight_ptr_,
                                        model_weight_size_);
        if (ret != ACL_ERROR_NONE) {
            LOGE("load model from file with mem failed, model file is %s\n", om_file.c_str());
            return Status(TNNERR_ATLAS_RUNTIME_ERROR, "load model from file with mem failed");
        }
    } else {
        ret = aclmdlLoadFromFile(om_file.c_str(), &model_id_);
        if (ret != ACL_ERROR_NONE) {
            LOGE("load model from file without mem failed, model file is %s\n", om_file.c_str());
            return Status(TNNERR_ATLAS_RUNTIME_ERROR, "load model from file without mem failed");
        }
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

    // Some model, e.g model Converted with atc config: --input_shape_range,
    // Does not need model_mem_size,
    if (model_mem_size_) {
        ret = aclrtMalloc(&model_mem_ptr_, model_mem_size_, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != ACL_ERROR_NONE) {
            LOGE("malloc buffer for mem failed, require size is %zu\n", model_mem_size_);
            return Status(TNNERR_ATLAS_RUNTIME_ERROR, "malloc buffer for mem failed");
        }

        ret = aclmdlLoadFromMemWithMem(om_content.data(), om_content.length(), &model_id_, model_mem_ptr_,
                                       model_mem_size_, model_weight_ptr_, model_weight_size_);
        if (ret != ACL_ERROR_NONE) {
            LOGE("load model from file with mem failed\n");
            return Status(TNNERR_ATLAS_RUNTIME_ERROR, "load model from file with mem failed");
        }
    } else {
        ret = aclmdlLoadFromMem(om_content.data(), om_content.length(), &model_id_);
        if (ret != ACL_ERROR_NONE) {
            LOGE("load model from file without mem failed\n");
            return Status(TNNERR_ATLAS_RUNTIME_ERROR, "load model from file without mem failed");
        }
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

Status AtlasNetwork::AllocateDatasetCreateBlob(aclmdlDataset **data_set, const InputShapesMap &max_input_shapes_map,
                                               bool is_input) {
    // This Function should be called twice.
    // Input should be called first, then output should also be called.

    if (nullptr == model_desc_) {
        LOGE("no model description, create ouput failed\n");
        return Status(TNNERR_ATLAS_RUNTIME_ERROR, "no model description, create ouput failed");
    }

    *data_set = aclmdlCreateDataset();
    if (nullptr == *data_set) {
        LOGE("can't create dataset, create output failed\n");
        return Status(TNNERR_ATLAS_RUNTIME_ERROR, "can't create dataset, create output failed");
    }

    bool infer_output_shape_required = false;

    size_t count = 0;
    if (is_input) {
        count = aclmdlGetNumInputs(model_desc_);
        LOGD("AllocateDataset for input (count=%d)\n", count);
    } else {
        count = aclmdlGetNumOutputs(model_desc_);
        LOGD("AllocateDataset for output (count=%d)\n", count);
        
        //ret = InferOutputShapeIfNecessery();
        //if (ret != TNN_OK)
        //    return ret;
    }


    for (size_t i = 0; i < count; ++i) {
        size_t buffer_size = 0;
        // OM Model Converted with atc config "--input_shape_range"
        // does not have buffer_size info. buffer_size should be provided externally
        // from MAX_INPUTS_SHAPE in "tnn::CreateInst() API"
        if (is_input) {
            buffer_size = aclmdlGetInputSizeByIndex(model_desc_, i);
            if (buffer_size == 0) {
                std::string input_name = aclmdlGetInputNameByIndex(model_desc_, i);
                auto iter = max_input_shapes_map.find(input_name);
                if (iter == max_input_shapes_map.end()) {
                    LOGE("Shape of dynamic input: %s, not found in max_input_shapes_map.\n", input_name.c_str());
                    return Status(TNNERR_ATLAS_RUNTIME_ERROR,
                                  "Shape of dynamic input not found in max_input_shapes_map.");
                }
                buffer_size = sizeof(int64_t)*DimsVectorUtils::Count(iter->second);
            }
        } else {
            buffer_size = aclmdlGetOutputSizeByIndex(model_desc_, i);
            if (buffer_size == 0) {
                std::string output_name = aclmdlGetOutputNameByIndex(model_desc_, i);
                auto iter = max_input_shapes_map.find(output_name);
                if (iter == max_input_shapes_map.end()) {
                    LOGE("Shape of dynamic output: %s, not found in max_input_shapes_map.\n", output_name.c_str());
                    return Status(TNNERR_ATLAS_RUNTIME_ERROR,
                                  "Shape of dynamic output not found in max_input_shapes_map.");
                } 
                buffer_size = sizeof(int64_t)*DimsVectorUtils::Count(iter->second);
            }
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

        // Add Blob to TNN Blob Map
        Status ret = AddBlobToMap(max_input_shapes_map, i, buffer, is_input);
        if (TNN_OK != ret) {
            return ret;
        }
        
        // for type 2 --input_shape_range input,
        // Call ATC Dynamic Input API
        // Create Tensor Desc for dynamic Input
        // https://www.hiascend.com/document/detail/zh/canncommercial/601/inferapplicationdev/atctool/atctool_0053.html
        if (is_input) {
            std::string input_name = aclmdlGetInputNameByIndex(this->model_desc_, i);
            if (this->dynamic_input_shape_range_names_.find(input_name) !=
                this->dynamic_input_shape_range_names_.end()) {
                auto iter = max_input_shapes_map.find(input_name);
                if (iter == max_input_shapes_map.end()) {
                    LOGE("MAX shape of Dynamic Input Range input '%s' not found.\n", input_name.c_str());
                    return Status(TNNERR_ATLAS_RUNTIME_ERROR, "MAX shape of Dynamic Input Range input not found");
                }

                int64_t dim_arr[iter->second.size()];
                for (int d = 0; d < iter->second.size(); d++) {
                    dim_arr[d] = iter->second[d];
                }
                // Input TensorDesc should only be created ONCE.
                // It will be destroyed in DeInit()
                aclTensorDesc *input_desc =
                    aclCreateTensorDesc(aclmdlGetInputDataType(this->model_desc_, i), iter->second.size(), dim_arr,
                                                aclmdlGetInputFormat(this->model_desc_, i));
                acl_ret = aclmdlSetDatasetTensorDesc(*data_set, input_desc, i);
                if (acl_ret != ACL_ERROR_NONE) {
                    LOGE("API aclmdlSetDatasetTensorDesc failed for input '%s'.\n", input_name.c_str());
                    return Status(TNNERR_ATLAS_RUNTIME_ERROR, "API aclmdlSetDatasetTensorDesc failed.");
                }
            }
        }
    }

    return TNN_OK;
}

Status AtlasNetwork::AddBlobToMap(const InputShapesMap &max_input_shapes_map, size_t index, void *data, bool is_input) {
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
            LOGD("find dynamic batch/hw/dims input (%s) and skip...\n", blob_name.c_str());
            atc_mode_dynamic_batch_hw_dim_ = true;
            //dynamic_batch_name_.push_back(blob_name);
            return TNN_OK;
        }
        // get dims info and data format
        ret = GetInputInfo(index, io_dims, data_format, data_type);
        if (TNN_OK != ret) {
            return ret;
        }

        // If "max_input_shapes" is externally provided.
        // Set io_dims to max_input_shape.
        auto max_input_shape_iter = max_input_shapes_map.find(blob_name);
        auto max_input_range_iter = this->dynamic_input_shape_range_names_.find(blob_name);
        if (max_input_range_iter != this->dynamic_input_shape_range_names_.end() &&
            max_input_shape_iter == max_input_shapes_map.end()) {
            // For MODELS with '--input_shape_range' type dynamic input,
            // external "max_input_shapes" is REQUIRED.
            LOGE("Max Input Shape is REQUIRED for dynamic input : '%s'.\n", blob_name.c_str());
            return Status(TNNERR_ATLAS_RUNTIME_ERROR, "Max Input Shape is REQUIRED for dynamic input.");
        }
        if (max_input_shape_iter != max_input_shapes_map.end()) {
            io_dims.clear();
            for (const auto& dim : max_input_shape_iter->second) {
                io_dims.push_back(dim);
            }
        }

        LOGD("input data type: %d, input data format: %d\n", data_type, data_format);
        LOGD("input '%s' shape:\n", blob_name.c_str());
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

        if (this->atc_mode_dynamic_batch_) {
            // get dims0
            int max_batch = GetMaxBatchSize(model_desc_, 1);
            if (0 == max_batch) {
                LOGE("get batch size failed\n");
                return Status(TNNERR_ATLAS_RUNTIME_ERROR, "get batch size failed");
            }
            output_dim0_map_[blob_name] = (int)acl_dims.dims[0] / max_batch;
        }
        // get data type
        data_type = aclmdlGetOutputDataType(model_desc_, index);
        // get data format
        data_format = aclmdlGetOutputFormat(model_desc_, index);
        for (int i = 0; i < acl_dims.dimCount; ++i) {
            io_dims.push_back((int)acl_dims.dims[i]);
        }

        // In rare cases like Detection Models
        // When Max Output Dims cannot be infered directly from Max Input Dims
        // TNN ATLAS Allow pre-defined "max_output_shapes" in "max_input_shapes"
        // If "max_output_shapes" is externally provided in "max_input_shapes"
        // Set io_dims to value in max_input_shape.
        auto max_input_shape_iter = max_input_shapes_map.find(blob_name);
        if (max_input_shape_iter != max_input_shapes_map.end()) {
            LOGI("WARNING!!! Set MAX output shape of output '%s' to externally defined values in 'max_input_shapes'.\n",
                 blob_name.c_str());
            io_dims.clear();
            for (const auto& dim : max_input_shape_iter->second) {
                io_dims.push_back(dim);
            }
        }

        LOGD("output data type: %d, output data format: %d\n", data_type, data_format);
        LOGD("output '%s' shape:\n", blob_name.c_str());
        for (int i = 0; i < io_dims.size(); ++i) {
            LOGD("[%d]\n", (int)io_dims[i]);
        }
    }

    BlobDesc blob_desc;
    blob_desc.device_type = DEVICE_ATLAS;
    ret                   = ConvertFromAclDataTypeToTnnDataType(data_type, blob_desc.data_type);
    if (TNN_OK != ret) {
        LOGE("convert from acl data type to tnn data type falied\n");
        return ret;
    }
    ret = ConvertAclDataFormatToTnnDataFormat(data_format, blob_desc.data_format);
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
        LOGD("get aipp info failed (ret=%d), use input info directly\n", acl_ret);
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

Status AtlasNetwork::SetRangeDynamicInputDim(std::string input_name, const DimsVector& target_input_shape) {
    size_t index = 0;
    aclError acl_ret = aclmdlGetInputIndexByName(model_desc_, input_name.c_str(), &index);
    if (acl_ret != ACL_ERROR_NONE) {
        LOGE("get dynamic batch input index falied!\n");
        return Status(TNNERR_ATLAS_RUNTIME_ERROR, "get dynamic batch input index falied");
    }

    // Get & Destroy Old Output TensorDesc
    aclTensorDesc* old_input_desc = aclmdlGetDatasetTensorDesc(this->input_, index);
    if (old_input_desc == nullptr) {
        LOGE("failed to get existing TensorDesc for input '%s'.\n", input_name.c_str());
        return Status(TNNERR_ATLAS_RUNTIME_ERROR, "failed to get existing TensorDesc for dynamic input.");
    }
    aclDestroyTensorDesc(old_input_desc);

    // Create & Set New Output TensorDesc
    int64_t dim_arr[target_input_shape.size()];
    for (int d = 0; d < target_input_shape.size(); d++) {
        dim_arr[d] = target_input_shape[d];
    }
    aclTensorDesc *new_input_desc =
        aclCreateTensorDesc(aclmdlGetInputDataType(this->model_desc_, index), target_input_shape.size(), dim_arr,
                                        aclmdlGetInputFormat(this->model_desc_, index));
    acl_ret = aclmdlSetDatasetTensorDesc(this->input_, new_input_desc, index);
    if (acl_ret != ACL_ERROR_NONE) {
        LOGE("API aclmdlSetDatasetTensorDesc failed for input '%s'.\n", input_name.c_str());
        return Status(TNNERR_ATLAS_RUNTIME_ERROR, "API aclmdlSetDatasetTensorDesc failed.");
    }

    return TNN_OK;
}

Status AtlasNetwork::UpdateRangeDynamicOutputDims() {
    int out_count = aclmdlGetNumOutputs(model_desc_);
    for (int i=0; i<out_count; i++) {
        aclTensorDesc* desc_i = aclmdlGetDatasetTensorDesc(this->output_, i);
        std::string output_name = aclmdlGetOutputNameByIndex(this->model_desc_, i);
        if (output_blob_map_.find(output_name) == output_blob_map_.end()) {
            LOGE("Unable to find output '%s' in output blob map.\n", output_name.c_str());
            return Status(TNNERR_ATLAS_RUNTIME_ERROR, "Unable to find output in output blob map.");
        }

        int num_dims = aclGetTensorDescNumDims(desc_i);
        if (num_dims > output_blob_map_[output_name]->GetBlobDesc().dims.size()) {
            // Some default-created output blob come with dim = 4. Checking non-equity here will not work.
            LOGE("Output '%s' ACL num_dim=%d, not equal with stored blob num_dim=%d\n", output_name.c_str(), num_dims,
                 output_blob_map_[output_name]->GetBlobDesc().dims.size());
            return Status(TNNERR_ATLAS_RUNTIME_ERROR, "Output num_dim not equal with stored blob num_dim.");
        }

        for (int d=0; d<num_dims; d++) {
            int64_t cur_dim = -1;
            aclError acl_ret = aclGetTensorDescDimV2(desc_i, d, &cur_dim);
            if (acl_ret != ACL_ERROR_NONE || cur_dim < 0) {
                LOGE("API aclGetTensorDescDimV2 failed for output '%s'::dim[%d].\n", output_name.c_str(), d);
                return Status(TNNERR_ATLAS_RUNTIME_ERROR, "API aclGetTensorDescDimV2 failed for output '%s' dim[%d].");
            }
            if (cur_dim != output_blob_map_[output_name]->GetBlobDesc().dims[d]) {
                LOGD("Update output '%s'::dim[%d] from %d to %d.\n", output_name.c_str(), d,
                     output_blob_map_[output_name]->GetBlobDesc().dims[d], cur_dim);
                output_blob_map_[output_name]->GetBlobDesc().dims[d] = cur_dim;
            }
        }
    }

    return TNN_OK;
}

Status AtlasNetwork::SetDynamicBatchSize(std::string blob_name, int batch_size) {
    if (IsDynamicBatch(model_desc_, blob_name) && atc_mode_dynamic_batch_hw_dim_) {
        // set dynamic batch
        size_t index     = 0;
        aclError acl_ret = aclmdlGetInputIndexByName(model_desc_, ACL_DYNAMIC_TENSOR_NAME, &index);
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


Status AtlasNetwork::DeduceDynamicInputType() {
    // ATC Converted HUAWEI atlas .om dynamic Models are devided into:
    //
    // 1. Traditional dynamic models with only 1 dynamic inputs.
    //    Min/Max value of the dynamic dim has been explicitly defined in ATC Conversion.
    // ---- 1.1:
    //      dynmaic batch
    //      --input_shape="img:-1,2,224,224;img_info:-1,4"
    //      --dynamic_batch_size="1,2,4,8"
    // ---- 1.2:
    //      dynamic hw size
    //      --input_shape="data:8,3,-1,-1;img_info:8,4,-1,-1"
    //      --dynamic_image_size="416,416;832,832"
    // ---- 1.3
    //      dynamic dims
    //      --input_shape="data:-1,1,256,256", --dynamic_dims="1,2"
    //
    // 2. More flexible dynamic input models.
    //    Min/Max Value is not explictly defined in ATC Conversion.
    // ---- 2.1:
    //      input_shape_range
    //      --input_shape_range="input1:[8~20,3,5,-1];input2:[5,3~9,10,-1]"
    
    // Get Number of Inputs by Calling ACL API
    int count = aclmdlGetNumInputs(this->model_desc_);
    LOGD("Network have %d inputs.\n", count);

    // Type 1 OM model has an extra input called "ascend_mbatch_shape_data"
    // Check if the input exists.
    for (int i = 0; i < count; i++) {
        std::string input_name = aclmdlGetInputNameByIndex(model_desc_, i);
        if (input_name.find(ACL_DYNAMIC_TENSOR_NAME) != std::string::npos) {
            LOGD("Network is converted with dynamic batch/hw/dims.\n");
            atc_mode_dynamic_batch_hw_dim_ = true;
        }
    }

    // Traditional Type 1 Dynamic
    if (atc_mode_dynamic_batch_hw_dim_) {
        if (count != 2) {
            // TODO: SUPPORT Type 1 Model with more than ONE input in the future.
            LOGD("Dynamic batch/hw/dims ATLAS with more than ONE input not supported yet.\n");
            return Status(TNNERR_ATLAS_RUNTIME_ERROR,
                          "Dynamic batch/hw/dims ATLAS with more than ONE input not supported yet.");
        }
        
        // TODO: Update this part for multiple inputs
        for (int i = 0; i < count; i++) {
            std::string input_name = aclmdlGetInputNameByIndex(model_desc_, i);
            if (input_name.find(ACL_DYNAMIC_TENSOR_NAME) == std::string::npos) {
                aclmdlIODims acl_dims;
                aclError acl_ret = aclmdlGetInputDims(this->model_desc_, i, &acl_dims);
                if (ACL_ERROR_NONE != acl_ret) {
                    LOGE("ACL API Call aclmdlGetInputDims falied!\n");
                    return Status(TNNERR_ATLAS_RUNTIME_ERROR, "ACL API Call aclmdlGetInputDims falied.");
                }

                int minus_one_count = 0;
                for (int d = 0; d < acl_dims.dimCount; d++) {
                    if (acl_dims.dims[d] == -1) {
                        minus_one_count++;
                    }
                }
                if (minus_one_count == 0) {
                    LOGE("The Only Input %s is not dynamic But Model is dynamic. Not Supported.\n", input_name.c_str());
                    return Status(TNNERR_ATLAS_RUNTIME_ERROR,
                                  "The Only Input is not dynamic But Model is dynamic. Not Supported..");
                }

                if (minus_one_count == 1 && acl_dims.dims[0] == -1) {
                    LOGD("Deduced Dynamic Batch Mode from input: %s.\n", input_name.c_str());
                    this->atc_mode_dynamic_batch_ = true;
                    return TNN_OK;
                }
                if (minus_one_count == 2 && acl_dims.dimCount == 4 && acl_dims.dims[2] == -1 &&
                    acl_dims.dims[3] == -1) {
                    LOGD("Deduced Dynamic HW Mode from input: %s.\n", input_name.c_str());
                    this->atc_mode_dynamic_hw_ = true;
                    return TNN_OK;
                }
                // ELSE
                LOGD("Deduced Dynamic Dim Mode from input: %s.\n", input_name.c_str());
                this->atc_mode_dynamic_dim_ = true;
                return TNN_OK;
            }
        }
    }

    // No Dynamic Or Type 2 Dynamic Input by --input_shape_range
    for (int i = 0; i < count; i++) {
        aclmdlIODims acl_dims;
        aclError acl_ret = aclmdlGetInputDims(this->model_desc_, i, &acl_dims);
        if (ACL_ERROR_NONE != acl_ret) {
            LOGE("ACL API Call aclmdlGetInputDims falied!\n");
            return Status(TNNERR_ATLAS_RUNTIME_ERROR, "ACL API Call aclmdlGetInputDims falied.");
        }
        
        int minus_one_count = 0;
        for (int d = 0; d < acl_dims.dimCount; d++) {
            if (acl_dims.dims[d] == -1) {
                minus_one_count++;
            }
        }

        if (minus_one_count > 0) {
            std::string input_name = aclmdlGetInputNameByIndex(model_desc_, i);
            LOGD("Input: '%s' is dynamic by --input_shape_range.\n", input_name.c_str());
            this->dynamic_input_shape_range_names_.insert(input_name);
        }
    }

    if (this->dynamic_input_shape_range_names_.empty()) {
        LOGD("No Dynamic Input.\n");
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
