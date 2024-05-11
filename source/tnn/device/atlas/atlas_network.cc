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
#include "tnn/device/atlas/atlas_common_types.h"
#include "tnn/device/atlas/atlas_network.h"
#include "tnn/device/atlas/atlas_om_model_interpreter.h"
#include "tnn/device/atlas/atlas_utils.h"
#include "tnn/utils/dims_vector_utils.h"
#include "tnn/utils/dims_vector_utils.h"


namespace TNN_NS {

NetworkImplFactoryRegister<NetworkImplFactory<AtlasNetwork>> g_network_impl_atlas_factory_register(NETWORK_TYPE_ATLAS);

// Default initialize global variable defined in "atlas_common_types.h"
std::map<Blob*, std::shared_ptr<AtlasOMModelInfo>> global_blob_om_model_info_map;
std::map<aclrtStream, aclrtContext> global_stream_context_map;

AtlasNetwork::~AtlasNetwork() {
    if (!this->network_init_called_) {
        LOGD("TNN ATLAS Network DeInit() called without Inited, do nothing.\n");
    }
    this->network_init_called_ = false;

    for (auto item : input_blob_map_) {
        if (nullptr != item.second) {
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
    
    LOGD("TNN AtlasNetwork Destructor: aclmdl destroy input dataset\n");
    if (this->aclmdl_input_dataset_ != nullptr) {
        DestroyDataset(this->aclmdl_input_dataset_);
    }
    LOGD("TNN AtlasNetwork Destructor: aclmdl destroy output dataset\n");
    if (this->aclmdl_output_dataset_ != nullptr) {
        DestroyDataset(this->aclmdl_output_dataset_);
    }

    if (this->model_type_ == MODEL_TYPE_ATLAS) {
        // Release OM model related classes and resources.
        aclError acl_ret;
        if (this->om_model_info_->model_id != INT_MAX) {
            LOGD("Unload ATLAS ACL Model id & Model Desc.\n");
            acl_ret = aclmdlUnload(this->om_model_info_->model_id);
            if (acl_ret != ACL_ERROR_NONE) {
                LOGE("unload model failed, modelId is %u\n", this->om_model_info_->model_id);
            }
        }
        
        if (nullptr != this->om_model_info_->model_desc) {
            (void)aclmdlDestroyDesc(this->om_model_info_->model_desc);
            this->om_model_info_->model_desc = nullptr;
        }
        
        AtlasContext* tnn_atlas_context = dynamic_cast<AtlasContext*>(context_);
        if(tnn_atlas_context == nullptr) {
            LOGE("TNN ATLAS Network: fail to cast to tnn atlas context\n");
        }
        if (tnn_atlas_context->GetAclrtStream() != nullptr) {
            acl_ret = aclrtSetCurrentContext(om_model_info_->aclrt_context);
            if (acl_ret != ACL_ERROR_NONE) {
                LOGE("TNN ATLAS Network: on destroy stream set context failed\n");
            }
            acl_ret = aclrtDestroyStream(tnn_atlas_context->GetAclrtStream());
            LOGD("aclrt destroy stream\n");
            if (acl_ret != ACL_ERROR_NONE) {
                LOGE("TNN ATLAS Network: destroy stream failed\n");
            }
            tnn_atlas_context->SetAclrtStream(nullptr);
        }
        
        if (om_model_info_->aclrt_context != nullptr) {
            acl_ret = aclrtDestroyContext(om_model_info_->aclrt_context);
            LOGD("aclrt destroy aclrt context\n");
            if (acl_ret != ACL_ERROR_NONE) {
                LOGE("TNN ATLAS Network: destroy context failed\n");
            }
            om_model_info_->aclrt_context = nullptr;
        }
        
        if (nullptr != this->om_model_memory_ptr_) {
            aclrtFree(this->om_model_memory_ptr_);
            LOGD("Unload ATLAS ACL Model Memory.\n");
            this->om_model_memory_ptr_  = nullptr;
            this->om_model_info_->memory_size = 0;
        }
        
        if (nullptr != this->om_model_weight_ptr_) {
            aclrtFree(this->om_model_weight_ptr_);
            LOGD("Unload ATLAS ACL Model Weight.\n");
            this->om_model_weight_ptr_  = nullptr;
            this->om_model_info_->weight_size = 0;
        }
    }

    // Call DeInit() of DefaultNetwork
    DeInit();
}

Status AtlasNetwork::LoadOMModelFromFile(const std::string &om_file) {
    // Step 1: Query Model Weight And Memory Size
    aclError acl_ret = aclmdlQuerySize(om_file.c_str(), &(om_model_info_->memory_size), &(om_model_info_->weight_size));
    if (acl_ret != ACL_ERROR_NONE) {
        LOGE("Atlas API: aclmdlQuerySize failed with Error Code: (%d)\n", acl_ret);
        return Status(TNNERR_ATLAS_RUNTIME_ERROR, "Atlas API: aclmdlQuerySize failed");
    }
    LOGD("Load Atlas OM Model From FILE. Weight Size: %d, Memory Size: %d\n", om_model_info_->weight_size, om_model_info_->memory_size);

    // Step 2: Load Model & Alloc Model Memory
    if (om_model_info_->memory_size > 0) {
        acl_ret = aclrtMalloc(&om_model_memory_ptr_, om_model_info_->memory_size, ACL_MEM_MALLOC_HUGE_FIRST);
        if (acl_ret != ACL_ERROR_NONE) {
            LOGE("Atlas API: aclrtMalloc for model memory failed, require size is %zu\n", om_model_info_->memory_size);
            return Status(TNNERR_ATLAS_RUNTIME_ERROR, "Atlas API: aclrtMalloc for model memory failed");
        }

        acl_ret = aclmdlLoadFromFileWithMem(om_file.c_str(), &(om_model_info_->model_id), om_model_memory_ptr_, om_model_info_->memory_size,
                                            om_model_weight_ptr_, om_model_info_->weight_size);
        if (acl_ret != ACL_ERROR_NONE) {
            LOGE("Atlas API: aclmdlLoadFromFileWithMem failed, model file is %s\n", om_file.c_str());
            return Status(TNNERR_ATLAS_RUNTIME_ERROR, "Atlas API: aclmdlLoadFromFileWithMem failed");
        }
    } else {
        // Some model, e.g model Converted with atc config: --input_shape_range,
        // Does not have model_mem_size, aclrtMalloc EMPTY mem is NOT ALLOWED.
        acl_ret = aclmdlLoadFromFile(om_file.c_str(), &(om_model_info_->model_id));
        if (acl_ret != ACL_ERROR_NONE) {
            LOGE("Atlas API: aclmdlLoadFromFile without memory failed, model file is %s\n", om_file.c_str());
            return Status(TNNERR_ATLAS_RUNTIME_ERROR, "Atlas API: aclmdlLoadFromFile without memory failed");
        }
    }

    // Step 3: Create Model Desc to get Model Info
    om_model_info_->model_desc = aclmdlCreateDesc();
    if (nullptr == om_model_info_->model_desc) {
        LOGE("Atlas API: aclmdlCreateDesc failed\n");
        return Status(TNNERR_ATLAS_RUNTIME_ERROR, "create model description failed");
    }

    acl_ret = aclmdlGetDesc(om_model_info_->model_desc, om_model_info_->model_id);
    if (acl_ret != ACL_ERROR_NONE) {
        LOGE("Atlas API: aclmdlGetDesc failed\n");
        return Status(TNNERR_ATLAS_RUNTIME_ERROR, "get model description failed");
    }

    return TNN_OK;
}

Status AtlasNetwork::LoadOMModelFromMemory(const std::string &om_content) {
    // Step 1: Query Model Weight And Memory Size
    aclError acl_ret = aclmdlQuerySizeFromMem(om_content.data(), om_content.length(), &(om_model_info_->memory_size), &(om_model_info_->weight_size));
    if (acl_ret != ACL_ERROR_NONE) {
        LOGE("Atlas API: aclmdlQuerySizeFromMem failed with Error Code: (%d)\n", acl_ret);
        return Status(TNNERR_ATLAS_RUNTIME_ERROR, "Atlas API: aclmdlQuerySizeFromMem failed");
    }
    LOGD("Load Atlas OM Model From MEMORY. Weight Size: %d, Memory Size: %d\n", om_model_info_->weight_size, om_model_info_->memory_size);

    // Step 2: Load Model & Alloc Model Memory
    if (om_model_info_->memory_size > 0) {
        acl_ret = aclrtMalloc(&om_model_memory_ptr_, om_model_info_->memory_size, ACL_MEM_MALLOC_HUGE_FIRST);
        if (acl_ret != ACL_ERROR_NONE) {
            LOGE("Atlas API: aclrtMalloc for model memory failed, require size is %zu\n", om_model_info_->memory_size);
            return Status(TNNERR_ATLAS_RUNTIME_ERROR, "Atlas API: aclrtMalloc for model memory failed");
        }

        acl_ret = aclmdlLoadFromMemWithMem(om_content.data(), om_content.length(), &(om_model_info_->model_id), om_model_memory_ptr_,
                                           om_model_info_->memory_size, om_model_weight_ptr_, om_model_info_->weight_size);
        if (acl_ret != ACL_ERROR_NONE) {
            LOGE("Atlas API: aclmdlLoadFromMemWithMem, load om content from memory with model memory failed\n");
            return Status(TNNERR_ATLAS_RUNTIME_ERROR, "Load om content from memory with model memory");
        }
    } else {
        // Some model, e.g model Converted with atc config: --input_shape_range,
        // Does not need model_mem_size,
        acl_ret = aclmdlLoadFromMem(om_content.data(), om_content.length(), &(om_model_info_->model_id));
        if (acl_ret != ACL_ERROR_NONE) {
            LOGE("Atlas API: aclmdlLoadFromMem, load model from file without model memory failed\n");
            return Status(TNNERR_ATLAS_RUNTIME_ERROR, "Load model from file without model memory failed");
        }
    }

    // Step 3: Create Model Desc to get Model Info
    om_model_info_->model_desc = aclmdlCreateDesc();
    if (nullptr == om_model_info_->model_desc) {
        LOGE("Atlas API: aclmdlCreateDesc failed\n");
        return Status(TNNERR_ATLAS_RUNTIME_ERROR, "create model description failed");
    }

    acl_ret = aclmdlGetDesc(om_model_info_->model_desc, om_model_info_->model_id);
    if (acl_ret != ACL_ERROR_NONE) {
        LOGE("Atlas API: aclmdlGetDesc failed\n");
        return Status(TNNERR_ATLAS_RUNTIME_ERROR, "get model description failed");
    }

    return TNN_OK;
}

Status AtlasNetwork::DeduceOMModelDynamicMode() {
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
    // ---- 2.1:
    //      input_shape (without "dynamic_batch_size" or "dynamic_image_size")
    //      --input_shape="input1:[8~20,3,5,-1];input2:[5,3~9,10,-1]"
    
    // Get Number of Inputs by Calling ACL API
    int count = aclmdlGetNumInputs(this->om_model_info_->model_desc);
    LOGD("TNN Atlas Loaded OM Model have %d inputs.\n", count);

    // Type 1 OM model has an extra input called "ascend_mbatch_shape_data"
    // Check if the input exists.
    bool is_om_model_dynamic = false;
    
    for (int i = 0; i < count; i++) {
        std::string input_name = aclmdlGetInputNameByIndex(om_model_info_->model_desc, i);
        if (input_name.find(ACL_DYNAMIC_TENSOR_NAME) != std::string::npos) {
            LOGD("Network is converted with dynamic batch/hw/dims.\n");
            is_om_model_dynamic = true;
        }
    }

    // Traditional Type 1 Dynamic
    if (is_om_model_dynamic) {
        if (count != 2) {
            // TODO: SUPPORT Type 1 Model with more than ONE input in the future.
            LOGD("Dynamic batch/hw/dims ATLAS with more than ONE input not supported yet.\n");
            return Status(TNNERR_ATLAS_RUNTIME_ERROR,
                          "Dynamic batch/hw/dims ATLAS with more than ONE input not supported yet.");
        }
        
        // TODO: Update this part for multiple inputs
        for (int i = 0; i < count; i++) {
            std::string input_name = aclmdlGetInputNameByIndex(om_model_info_->model_desc, i);
            if (input_name.find(ACL_DYNAMIC_TENSOR_NAME) == std::string::npos) {
                aclmdlIODims acl_dims;
                aclError acl_ret = aclmdlGetInputDims(this->om_model_info_->model_desc, i, &acl_dims);
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
                    this->om_model_info_->dynamic_mode = AtlasOmModelDynamicMode::DynamicBatch;
                    return TNN_OK;
                }
                if (minus_one_count == 2 && acl_dims.dimCount == 4 && acl_dims.dims[2] == -1 &&
                    acl_dims.dims[3] == -1) {
                    LOGD("Deduced Dynamic HW Mode from input: %s.\n", input_name.c_str());
                    this->om_model_info_->dynamic_mode = AtlasOmModelDynamicMode::DynamicHW;
                    return TNN_OK;
                }
                // ELSE
                LOGD("Deduced Generic Dynamic Dim Mode from input: %s.\n", input_name.c_str());
                this->om_model_info_->dynamic_mode = AtlasOmModelDynamicMode::GenericDynamic;
                return TNN_OK;
            }
        }
    }

    // No Dynamic Or Type 2 Dynamic Input by --input_shape_range
    for (int i = 0; i < count; i++) {
        aclmdlIODims acl_dims;
        aclError acl_ret = aclmdlGetInputDims(this->om_model_info_->model_desc, i, &acl_dims);
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
            std::string input_name = aclmdlGetInputNameByIndex(om_model_info_->model_desc, i);
            LOGD("Input: '%s' is dynamic by --input_shape_range.\n", input_name.c_str());
            this->om_model_info_->generic_dynamic_input_names.insert(input_name);
        }
    }

    if (this->om_model_info_->generic_dynamic_input_names.empty()) {
        LOGD("No Dynamic Input.\n");
    }
    return TNN_OK;
}

Status AtlasNetwork::DeduceOMModelAIPPInputFormat() {
    // Get Number of Inputs by Calling ACL API
    int count = aclmdlGetNumInputs(this->om_model_info_->model_desc);

    for (int i = 0; i < count; i++) {
        std::string input_name = aclmdlGetInputNameByIndex(om_model_info_->model_desc, i);
        aclAippInfo aipp_info;
        aclError acl_ret = aclmdlGetFirstAippInfo(this->om_model_info_->model_id, i, &aipp_info);
        if (acl_ret == ACL_ERROR_NONE) {
            LOGD("Found AIPP Input, shapeCount: %d srcDimNum: %d\n", aipp_info.shapeCount, aipp_info.srcDimNum);
            this->om_model_info_->aipp_input_format_map[input_name] = aipp_info.inputFormat;
        }
    }
    return TNN_OK;
}

Status AtlasNetwork::InitOMModel(ModelConfig &model_config, AbstractModelInterpreter *interpreter,
                                 InputShapesMap min_inputs_shape, InputShapesMap max_inputs_shape,
                                 InputDataTypeMap inputs_data_type, bool enable_const_folder) {
    AtlasOMModelInterpreter *om_interpreter = dynamic_cast<AtlasOMModelInterpreter *>(interpreter);
    CHECK_PARAM_NULL(om_interpreter);
    AtlasContext* atlas_context = dynamic_cast<AtlasContext *>(context_);
    CHECK_PARAM_NULL(atlas_context);

    std::string& om_str = om_interpreter->GetOmString();

    // Part 1: Load(Interpret) Model. Aclrt load OM model will directly load model onto Device
    //         So it can only be called in AtlasNetwork, not ModelInterpreter
    // Step 1: Create OM Model Info, aclrt load_model_context & load model stream
    this->om_model_info_ = std::make_shared<AtlasOMModelInfo>();

    aclError acl_ret = aclrtSetDevice(this->config_.device_id);
    if (acl_ret != ACL_ERROR_NONE) {
        LOGE("acl open device %d failed (acl error code: %d)\n", this->config_.device_id, acl_ret);
        return Status(TNNERR_ATLAS_RUNTIME_ERROR, "acl open device falied");
    }
    acl_ret = aclrtCreateContext(&(om_model_info_->aclrt_context), this->config_.device_id);
    if (acl_ret != ACL_ERROR_NONE) {
        LOGE("acl create context failed (acl error code: %d)\n", acl_ret);
        return Status(TNNERR_ATLAS_RUNTIME_ERROR, "acl create context falied");
    }
    acl_ret = aclrtSetCurrentContext(om_model_info_->aclrt_context);
    if (acl_ret != ACL_ERROR_NONE) {
        LOGE("TNN ATLAS OM Model Interpreter: on destroy stream set context failed\n");
    }
    aclrtStream aclrt_stream;
    acl_ret = aclrtCreateStream(&aclrt_stream);
    if (acl_ret != ACL_ERROR_NONE) {
        LOGE("acl create stream failed (acl error code: %d)\n", acl_ret);
        return Status(TNNERR_ATLAS_RUNTIME_ERROR, "acl create stream falied");
    }
    atlas_context->SetAclrtStream(aclrt_stream);
    global_stream_context_map[atlas_context->GetAclrtStream()] = om_model_info_->aclrt_context;

    // Step 2: Load Model From Path or From Memory
    // Determine OM string is model path or model content
    Status tnn_ret;
    if (om_str.length() < 1024) {
        std::ifstream om_file(om_str);
        if (!om_file) {
            LOGE("Invalied om file path! (om_str : %s) maybe as memory content\n", om_str.c_str());
            return Status(TNNERR_ATLAS_RUNTIME_ERROR, "Invalied om file Path, cannot determine if om_str is Path or Model Content.");
        }
        tnn_ret = LoadOMModelFromFile(om_str);
        if (tnn_ret != TNN_OK) {
            LOGE("TNN Atlas Load OM Model from File Failed.\n");
            return tnn_ret;
        }
    } else {
        tnn_ret = LoadOMModelFromMemory(om_str);
        if (tnn_ret != TNN_OK) {
            LOGE("TNN Atlas Load OM Model from Model Content Failed.\n");
            return tnn_ret;
        }
    }
    // Synchronize Device and Destroy Model Load Stream
    acl_ret = aclrtSynchronizeDevice();
    if (acl_ret != ACL_ERROR_NONE) {
        LOGE("acl device synchronize failed (acl error code: %d)\n", acl_ret);
        return Status(TNNERR_ATLAS_RUNTIME_ERROR, "acl device synchronize falied");
    }

    // Step 3: Deduce Atlas OM Model Dynamic Type
    tnn_ret = DeduceOMModelDynamicMode();
    if (tnn_ret != TNN_OK) {
        LOGE("TNN Atlas Deduce Model Dynamic Mode Failed.\n");
        return tnn_ret;
    }

    // Step 4: Deduce Atlas OM Model AIPP input format if input is AIPP Mode.
    tnn_ret = DeduceOMModelAIPPInputFormat();
    if (tnn_ret != TNN_OK) {
        LOGE("TNN Atlas Deduce Model AIPP input format Failed.\n");
        return tnn_ret;
    }


    // Part 2: Allocate Input/Output, Reshape etc.
    // Step 5: allocate input and output
    tnn_ret = AllocateDatasetCreateBlob(&aclmdl_input_dataset_, max_inputs_shape, true);
    if (tnn_ret != TNN_OK)
        return tnn_ret;
    tnn_ret = AllocateDatasetCreateBlob(&aclmdl_output_dataset_, max_inputs_shape, false);
    if (tnn_ret != TNN_OK)
        return tnn_ret;

    // Step 6: set dynamic batch size
    //         must do if input is dynamic batch
    if (this->om_model_info_->dynamic_mode != AtlasOmModelDynamicMode::Static) {
        for (auto item : input_blob_map_) {
            tnn_ret = SetDynamicBatchSize(item.first, item.second->GetBlobDesc().dims[0]);
            if (tnn_ret != TNN_OK)
                return tnn_ret;
        }
    }

    // Step 7: reshape if needed
    tnn_ret = Reshape(max_inputs_shape);
    if (tnn_ret != TNN_OK)
        return tnn_ret;

    return TNN_OK;
}

Status AtlasNetwork::Init(NetworkConfig &net_config, ModelConfig &model_config, AbstractModelInterpreter *interpreter,
                          InputShapesMap min_inputs_shape, InputShapesMap max_inputs_shape, InputDataTypeMap inputs_data_type, 
                          bool enable_const_folder) {
    this->network_init_called_ = true;
    this->config_ = net_config;
    this->model_type_ = model_config.model_type;

    // GetDevice and Context
    this->device_ = GetDevice(net_config.device_type);
    CHECK_PARAM_NULL(this->device_);
    this->context_ = device_->CreateContext(net_config.device_id);
    CHECK_PARAM_NULL(this->context_);

    // Set AtlasContext model type
    AtlasContext* atlas_context = dynamic_cast<AtlasContext *>(context_);
    CHECK_PARAM_NULL(atlas_context);
    atlas_context->SetModelType(model_config.model_type);

    // Init Model For different Model Types
    if (model_config.model_type == MODEL_TYPE_TORCHSCRIPT) {
        LOGE("Fail to init AtlasNetwork, MODEL_TYPE_TORCHSCRIPT not supported YET.\n");
        return Status(TNNERR_ATLAS_RUNTIME_ERROR, "Fail to init AtlasNetwork, MODEL_TYPE_TORCHSCRIPT not supported YET");
    } else if (model_config.model_type == MODEL_TYPE_TNN ||
               model_config.model_type == MODEL_TYPE_RAPIDNET) {
        LOGE("Fail to init AtlasNetwork, MODEL_TYPE_TNN not supported YET.\n");
        return Status(TNNERR_ATLAS_RUNTIME_ERROR, "Fail to init AtlasNetwork, MODEL_TYPE_TNN not supported YET");
    } else if (model_config.model_type == MODEL_TYPE_ATLAS) {
        return InitOMModel(model_config, interpreter, min_inputs_shape, max_inputs_shape,
                           inputs_data_type, enable_const_folder);
    } else {
        LOGE("Fail to init AtlasNetwork, model type not supported.\n");
        return Status(TNNERR_ATLAS_RUNTIME_ERROR, "Fail to init AtlasNetwork, model type not supported");
    }
}


Status AtlasNetwork::GetForwardMemorySize(size_t &memory_size) {
    if (model_type_ == MODEL_TYPE_ATLAS) {
        if (!om_model_info_) {
            LOGE("Unable to Get ForwardMemorySize, ATLAS om ModelInfo Missing.\n");
            return Status(TNNERR_DEVICE_NOT_SUPPORT, "Unable to Get ForwardMemorySize, ATLAS om ModelInfo Missing.");
        }
        memory_size = om_model_info_->memory_size + om_model_info_->weight_size;
    }
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

std::shared_ptr<AtlasOMModelInfo> AtlasNetwork::GetOMModelInfo() {
    return this->om_model_info_;
}

Status AtlasNetwork::ReshapeOMModel(const InputShapesMap &inputs) {
    AtlasContext* atlas_context = dynamic_cast<AtlasContext *>(context_);
    CHECK_PARAM_NULL(atlas_context);

    aclError acl_ret = aclrtSetCurrentContext(om_model_info_->aclrt_context);
    if (acl_ret != ACL_ERROR_NONE) {
        LOGE("ReshapeOMModel set context failed\n");
        return Status(TNNERR_ATLAS_RUNTIME_ERROR, "ReshapeOMModel set context failed");
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
            if (this->om_model_info_->dynamic_mode == AtlasOmModelDynamicMode::DynamicBatch) {
                Status tnn_ret = SetDynamicBatchSize(item.first, dims[0]);
                if (TNN_OK != tnn_ret)
                    return tnn_ret;
            }

            // Range input for Model Converted with --input_shape_range
            // Range input output shape cannot be infered from input shape.
            // Output Shape will be deduced after ACL Forward() API is called.
            if (this->om_model_info_->generic_dynamic_input_names.find(item.first) !=
                this->om_model_info_->generic_dynamic_input_names.end()) {
                Status tnn_ret = SetRangeDynamicInputDim(item.first, dims);
                if (TNN_OK != tnn_ret)
                    return tnn_ret;
            }
        }
    }

    return TNN_OK;
}


Status AtlasNetwork::Reshape(const InputShapesMap &inputs) {
    // Reshape Model For different Model Types
    if (this->model_type_ == MODEL_TYPE_TORCHSCRIPT) {
        LOGE("Fail to reshape AtlasNetwork, MODEL_TYPE_TORCHSCRIPT not supported YET.\n");
        return Status(TNNERR_ATLAS_RUNTIME_ERROR, "Fail to reshape AtlasNetwork, MODEL_TYPE_TORCHSCRIPT not supported YET");
    } else if (this->model_type_ == MODEL_TYPE_TNN || this->model_type_ == MODEL_TYPE_RAPIDNET) {
        LOGE("Fail to reshape AtlasNetwork, MODEL_TYPE_TNN not supported YET.\n");
        return Status(TNNERR_ATLAS_RUNTIME_ERROR, "Fail to reshape AtlasNetwork, MODEL_TYPE_TNN not supported YET");
    } else if (this->model_type_ == MODEL_TYPE_ATLAS) {
        return ReshapeOMModel(inputs);
    } else {
        LOGE("Fail to reshape AtlasNetwork, model type not supported.\n");
        return Status(TNNERR_ATLAS_RUNTIME_ERROR, "Fail to reshape AtlasNetwork, model type not supported");
    }
}

Status AtlasNetwork::GetCommandQueue(void **command_queue) {
    return context_->GetCommandQueue(command_queue);
}

Status AtlasNetwork::Forward() {
    // Reshape Model For different Model Types
    if (this->model_type_ == MODEL_TYPE_TORCHSCRIPT) {
        LOGE("Fail to execute AtlasNetwork, MODEL_TYPE_TORCHSCRIPT not supported YET.\n");
        return Status(TNNERR_ATLAS_RUNTIME_ERROR, "Fail to execute AtlasNetwork, MODEL_TYPE_TORCHSCRIPT not supported YET");
    } else if (this->model_type_ == MODEL_TYPE_TNN || this->model_type_ == MODEL_TYPE_RAPIDNET ||
               this->model_type_ == MODEL_TYPE_ATLAS) {
        LOGD("Atlas Forward!\n");
        AtlasContext* atlas_context = dynamic_cast<AtlasContext *>(context_);
        CHECK_PARAM_NULL(atlas_context);

        aclError acl_ret = aclrtSetCurrentContext(om_model_info_->aclrt_context);
        if (acl_ret != ACL_ERROR_NONE) {
            LOGE("ReshapeOMModel set context failed\n");
            return Status(TNNERR_ATLAS_RUNTIME_ERROR, "ReshapeOMModel set context failed");
        }

        acl_ret = aclrtSynchronizeStream(atlas_context->GetAclrtStream());
        if (acl_ret != ACL_ERROR_NONE) {
            LOGE("before forward synchronize stream failed\n");
            return Status(TNNERR_ATLAS_RUNTIME_ERROR, "before forward synchronize stream failed");
        }

        acl_ret = aclmdlExecute(this->om_model_info_->model_id, aclmdl_input_dataset_, aclmdl_output_dataset_);
        if (acl_ret != ACL_ERROR_NONE) {
            LOGE("execute model failed, modelId is %u\n", this->om_model_info_->model_id);
            return Status(TNNERR_ATLAS_RUNTIME_ERROR, "execute model failed");
        }

        // For Range Dynamic Models with --input_shape_range
        // Update Output Blob Shapes here.
        if (!this->om_model_info_->generic_dynamic_input_names.empty()) {
            Status tnn_ret = UpdateRangeDynamicOutputDims();
            if (TNN_OK != tnn_ret) {
                return tnn_ret;
            }
        }
    } else {
        LOGE("Fail to reshape AtlasNetwork, model type not supported.\n");
        return Status(TNNERR_ATLAS_RUNTIME_ERROR, "Fail to reshape AtlasNetwork, model type not supported");
    }

    return TNN_OK;
}

Status AtlasNetwork::ForwardAsync(Callback call_back) {
    LOGD("Atlas Async Forward! (as same as Forward by now)\n");
    return Forward();
}


Status AtlasNetwork::AllocateDatasetCreateBlob(aclmdlDataset **data_set, const InputShapesMap &max_input_shapes_map,
                                               bool is_input) {
    // This Function should be called twice.
    // Input should be called first, then output should also be called.

    if (nullptr == om_model_info_->model_desc) {
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
        count = aclmdlGetNumInputs(om_model_info_->model_desc);
        LOGD("AllocateDataset for input (count=%d)\n", count);
    } else {
        count = aclmdlGetNumOutputs(om_model_info_->model_desc);
        LOGD("AllocateDataset for output (count=%d)\n", count);
    }

    for (size_t i = 0; i < count; ++i) {
        size_t buffer_size = 0;
        // OM Model Converted with atc config "--input_shape_range"
        // does not have buffer_size info. buffer_size should be provided externally
        // from MAX_INPUTS_SHAPE in "tnn::CreateInst() API"
        if (is_input) {
            buffer_size = aclmdlGetInputSizeByIndex(om_model_info_->model_desc, i);
            if (buffer_size == 0) {
                std::string input_name = aclmdlGetInputNameByIndex(om_model_info_->model_desc, i);
                auto iter = max_input_shapes_map.find(input_name);
                if (iter == max_input_shapes_map.end()) {
                    LOGE("Shape of dynamic input: %s, not found in max_input_shapes_map.\n", input_name.c_str());
                    return Status(TNNERR_ATLAS_RUNTIME_ERROR,
                                  "Shape of dynamic input not found in max_input_shapes_map.");
                }
                buffer_size = sizeof(int64_t)*DimsVectorUtils::Count(iter->second);
            }
        } else {
            buffer_size = aclmdlGetOutputSizeByIndex(om_model_info_->model_desc, i);
            if (buffer_size == 0) {
                std::string output_name = aclmdlGetOutputNameByIndex(om_model_info_->model_desc, i);
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
            std::string input_name = aclmdlGetInputNameByIndex(om_model_info_->model_desc, i);
            if (om_model_info_->generic_dynamic_input_names.find(input_name) !=
                om_model_info_->generic_dynamic_input_names.end()) {
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
                    aclCreateTensorDesc(aclmdlGetInputDataType(om_model_info_->model_desc, i), iter->second.size(), dim_arr,
                                                aclmdlGetInputFormat(om_model_info_->model_desc, i));
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
    if (om_model_info_->model_desc == nullptr) {
        LOGE("no model description\n");
        return Status(TNNERR_ATLAS_RUNTIME_ERROR, "no model description");
    }

    Status ret = TNN_OK;
    std::string blob_name = "";
    std::vector<int> io_dims;
    aclDataType data_type;
    aclFormat data_format;

    io_dims.clear();
    if (is_input) {
        // get blob name
        blob_name = aclmdlGetInputNameByIndex(om_model_info_->model_desc, index);
        // skip dynamic aipp input
        if (blob_name.find(ACL_DYNAMIC_AIPP_NAME) != std::string::npos) {
            LOGD("find dynamic aipp input (%s) and skip...\n", blob_name.c_str());
            return TNN_OK;
        }
        // skip dynamic batch input
        if (blob_name.find(ACL_DYNAMIC_TENSOR_NAME) != std::string::npos) {
            LOGD("find dynamic batch/hw/dims input (%s) and skip...\n", blob_name.c_str());
            //atc_mode_dynamic_batch_hw_dim_ = true;
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
        auto max_input_range_iter = om_model_info_->generic_dynamic_input_names.find(blob_name);
        if (max_input_range_iter != om_model_info_->generic_dynamic_input_names.end() &&
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
        blob_name = aclmdlGetOutputNameByIndex(om_model_info_->model_desc, index);
        // get dims info
        aclmdlIODims acl_dims;
        aclError acl_ret = aclmdlGetOutputDims(om_model_info_->model_desc, index, &acl_dims);
        if (acl_ret != ACL_ERROR_NONE) {
            LOGE("can't get output dims\n");
            return Status(TNNERR_ATLAS_RUNTIME_ERROR, "can't get output dims");
        }

        if (om_model_info_->dynamic_mode == AtlasOmModelDynamicMode::DynamicBatch) {
            // get dims0
            int max_batch = GetMaxBatchSize(om_model_info_->model_desc, 1);
            if (0 == max_batch) {
                LOGE("get batch size failed\n");
                return Status(TNNERR_ATLAS_RUNTIME_ERROR, "get batch size failed");
            }
            output_dim0_map_[blob_name] = std::max(1, (int)acl_dims.dims[0] / max_batch);
        }
        // get data type
        data_type = aclmdlGetOutputDataType(om_model_info_->model_desc, index);
        // get data format
        data_format = aclmdlGetOutputFormat(om_model_info_->model_desc, index);
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
    
    // Add Blob To global_blob_om_model_map
    global_blob_om_model_info_map[blob] = om_model_info_;
    LOGD("Added Blob to global_blob_model_info_map, map.size = %d\n", global_blob_om_model_info_map.size());

    if (is_input) {
        input_blob_map_[blob_name] = blob;
    } else {
        output_blob_map_[blob_name] = blob;
    }

    return TNN_OK;
}

Status AtlasNetwork::GetInputInfo(size_t index, std::vector<int> &input_dims, aclFormat &input_format,
                                  aclDataType &input_data_type) {
    std::string blob_name = aclmdlGetInputNameByIndex(om_model_info_->model_desc, index);
    aclAippInfo aipp_info;
    aclError acl_ret = aclmdlGetFirstAippInfo(om_model_info_->model_id, index, &aipp_info);

    input_dims.clear();
    if (ACL_ERROR_NONE == acl_ret) {
        // has static aipp
        LOGD("shapeCount: %d   srcDimNum: %d\n", aipp_info.shapeCount, aipp_info.srcDimNum);

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

        // get data format
        input_format = aclmdlGetInputFormat(om_model_info_->model_desc, index);

        // get data type
        input_data_type = aclmdlGetInputDataType(om_model_info_->model_desc, index);

        // get dims info
        aclmdlIODims acl_dims;
        aclError acl_ret = aclmdlGetInputDims(om_model_info_->model_desc, index, &acl_dims);
        if (acl_ret != ACL_ERROR_NONE) {
            LOGE("can't get input dims\n");
            return Status(TNNERR_ATLAS_RUNTIME_ERROR, "can't get input dims");
        }
        // in dynamic batch input, reset batch
        if (-1 == acl_dims.dims[0]) {
            auto buffer_size = aclmdlGetInputSizeByIndex(om_model_info_->model_desc, index);
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
    aclError acl_ret = aclmdlGetInputIndexByName(om_model_info_->model_desc, input_name.c_str(), &index);
    if (acl_ret != ACL_ERROR_NONE) {
        LOGE("get dynamic batch input index falied!\n");
        return Status(TNNERR_ATLAS_RUNTIME_ERROR, "get dynamic batch input index falied");
    }

    // Deprecated in CANN 7+ Version
    // Get & Destroy Old Output TensorDesc
    //aclTensorDesc* old_input_desc = aclmdlGetDatasetTensorDesc(this->aclmdl_input_dataset_, index);
    //if (old_input_desc == nullptr) {
    //    LOGE("failed to get existing TensorDesc for input '%s'.\n", input_name.c_str());
    //    return Status(TNNERR_ATLAS_RUNTIME_ERROR, "failed to get existing TensorDesc for dynamic input.");
    //}
    //aclDestroyTensorDesc(old_input_desc);

    // Create & Set New Output TensorDesc
    int64_t dim_arr[target_input_shape.size()];
    for (int d = 0; d < target_input_shape.size(); d++) {
        dim_arr[d] = target_input_shape[d];
    }
    aclTensorDesc *new_input_desc =
        aclCreateTensorDesc(aclmdlGetInputDataType(om_model_info_->model_desc, index), target_input_shape.size(), dim_arr,
                                        aclmdlGetInputFormat(om_model_info_->model_desc, index));
    acl_ret = aclmdlSetDatasetTensorDesc(this->aclmdl_input_dataset_, new_input_desc, index);
    if (acl_ret != ACL_ERROR_NONE) {
        LOGE("API aclmdlSetDatasetTensorDesc failed for input '%s'.\n", input_name.c_str());
        return Status(TNNERR_ATLAS_RUNTIME_ERROR, "API aclmdlSetDatasetTensorDesc failed.");
    }

    return TNN_OK;
}

Status AtlasNetwork::UpdateRangeDynamicOutputDims() {
    int out_count = aclmdlGetNumOutputs(this->om_model_info_->model_desc);
    for (int i=0; i<out_count; i++) {
        aclTensorDesc* desc_i = aclmdlGetDatasetTensorDesc(this->aclmdl_output_dataset_, i);
        std::string output_name = aclmdlGetOutputNameByIndex(this->om_model_info_->model_desc, i);
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
    if (IsDynamicBatch(this->om_model_info_->model_desc, blob_name) &&
        om_model_info_->dynamic_mode != AtlasOmModelDynamicMode::Static) {
        // set dynamic batch
        size_t index     = 0;
        aclError acl_ret = aclmdlGetInputIndexByName(om_model_info_->model_desc, ACL_DYNAMIC_TENSOR_NAME, &index);
        if (acl_ret != ACL_ERROR_NONE) {
            LOGE("get dynamic batch input index falied!\n");
            return Status(TNNERR_ATLAS_RUNTIME_ERROR, "get dynamic batch input index falied");
        }
        acl_ret = aclmdlSetDynamicBatchSize(om_model_info_->model_id, aclmdl_input_dataset_, index, batch_size);
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
