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

#include <sys/utsname.h>
#include "coreml_network.h"
#include "tnn/utils/data_type_utils.h"
#include "tnn/utils/dims_utils.h"
#include "tnn/interpreter/default_model_interpreter.h"

namespace TNN_NS {

// utsname.machine has device identifier. For example, identifier for iPhone Xs is "iPhone11,2".
// Since Neural Engine is only available for use on A12 and later, major device version in the
// identifier is checked for these models:
// A12: iPhone XS (11,2), iPad Mini - 5th Gen (11,1)
// A12X: iPad Pro - 3rd Gen (8,1)
// For more information, see https://www.theiphonewiki.com/wiki/Models
bool HasAppleNPU() {
#if TNN_COREML_TEST
    return true;
#else
    return true;
    //check hardware
    struct utsname system_info;
    uname(&system_info);

    if (strncmp("iPad", system_info.machine, 4) == 0) {
    const int major_version = atoi(system_info.machine + 4);
    return major_version >= 8;  // There are no device between iPad 8 and 11.
    }
    else if (strncmp("iPhone", system_info.machine, 6) == 0) {
    const int major_version = atoi(system_info.machine + 6);
    return major_version >= 11;
    }
    else if (strncmp("MacBookPro", system_info.machine, 10) == 0) {
      const int major_version = atoi(system_info.machine + 10);
      return major_version >= 17;
    }
    else if (strncmp("MacBookAir", system_info.machine, 10) == 0) {
      const int major_version = atoi(system_info.machine + 10);
      return major_version >= 10;
    }
    else if (strncmp("iMac", system_info.machine, 4) == 0) {
      const int major_version = atoi(system_info.machine + 4);
      return major_version >= 21;
    }
    else if (strncmp("Macmini", system_info.machine, 7) == 0) {
      const int major_version = atoi(system_info.machine + 7);
      return major_version >= 9;
    }
    return false;
#endif
}

NetworkImplFactoryRegister<NetworkImplFactory<CoreMLNetwork>>
    g_network_impl_coreml_factory_register(NETWORK_TYPE_COREML);

CoreMLNetwork::CoreMLNetwork() {}

CoreMLNetwork::~CoreMLNetwork() {
    DeInit();
    
    blob_input_map_ = {};
    blob_output_map_ = {};
    
    if (context_ != nullptr) {
        delete context_;
        context_ = nullptr;
    }
}

Status CoreMLNetwork::Init(NetworkConfig &net_config, ModelConfig &model_config, AbstractModelInterpreter *interpreter,
                           InputShapesMap min_inputs_shape, InputShapesMap max_inputs_shape, bool enable_const_folder) {
    if (!HasAppleNPU()) {
        return Status(TNNERR_COMMON_ERROR, "Apple device dont have NeuralEngine");
    }
    
    model_config_ = model_config;
    
    if (@available(iOS 12.0, macOS 10.14, *)) {
        Status ret = TNN_OK;
        
        auto type = net_config.device_type;
        if(type == DEVICE_APPLE_NPU) {
            //use DEVICE_ARM OR DEVICE_X86 according to hardware
#if defined(__arm__) || defined(__arm64__)
            type = DEVICE_ARM;
#else
            type = DEVICE_X86;
#endif
        }
        device_ = GetDevice(type);

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
        
        if (model_config.model_type == MODEL_TYPE_TNN) {
            auto default_interpreter = dynamic_cast<DefaultModelInterpreter *>(interpreter);
            CHECK_PARAM_NULL(default_interpreter);
            auto md5s = default_interpreter->GetParamsMd5();
            std::string md5 = md5s.size() > 0 ? md5s[0]: "unit_test_default_md5";  // default md5 for unit_test
            if (md5s.size() >= 2) {
                md5 = md5 + "-" + md5s[1];
            }

            net_structure_ = default_interpreter->GetNetStructure();
            net_resource_   = default_interpreter->GetNetResource();
            if (net_structure_ == NULL || net_resource_ == NULL) {
                LOGE("ERROR: network_ is nil, network_type may not support\n");
                return Status(TNNERR_NULL_PARAM, "network_ is nil, network_type may not support");
            }
            
            //init coreml model
            coreml_executor_ = [[CoreMLModel alloc] initWithCachePath:net_config.cache_path ID:md5];
            
            if ([coreml_executor_ buildFromCache] != TNN_OK) {
                RETURN_ON_NEQ(InitCoreMLModel(net_structure_, net_resource_), TNN_OK);
                
                RETURN_ON_NEQ(ConvertCoreMLModel(net_structure_, net_resource_), TNN_OK);
                
                RETURN_ON_NEQ(CompileModel(coreml_model_.get()), TNN_OK);
            }
        } else if (model_config.model_type == MODEL_TYPE_COREML) {
            //init coreml model
            coreml_executor_ = [[CoreMLModel alloc] initWithModelPath:model_config.params[0]];
            RETURN_ON_NEQ([coreml_executor_ buildFromModelPath], TNN_OK);
        } else {
            return Status(TNNERR_MODEL_ERR, "CoreMLNetwork dont support the mode type in model_config");
        }

        BlobMap blobs;
        RETURN_ON_NEQ(GetAllInputBlobs(blobs), TNN_OK);
        return GetAllOutputBlobs(blobs);
        
    } else {
        return Status(TNNERR_IOS_VERSION_ERROR, "The operate system is not iOS 12+ or macOS 10.14+");
    }
}

Status CoreMLNetwork::InitCoreMLModel(NetStructure *net_structure, NetResource *net_resource) {
    Status ret = TNN_OK;
    //  Init CoreML Model
    if (coreml_model_ == nullptr) {
        coreml_model_.reset(new _CoreML__Specification__Model);
        core_ml__specification__model__init(coreml_model_.get());
        coreml_model_->specificationversion = 4;
        coreml_model_->type_case = CORE_ML__SPECIFICATION__MODEL__TYPE_NEURAL_NETWORK;
        coreml_neural_network_ = std::shared_ptr<CoreML__Specification__NeuralNetwork>(new CoreML__Specification__NeuralNetwork);
        coreml_model_->neuralnetwork = (CoreML__Specification__NeuralNetwork *)coreml_neural_network_.get();
        core_ml__specification__neural_network__init(coreml_model_->neuralnetwork);
        coreml_model_->neuralnetwork->arrayinputshapemapping =  CORE_ML__SPECIFICATION__NEURAL_NETWORK_MULTI_ARRAY_SHAPE_MAPPING__EXACT_ARRAY_MAPPING;
    }
    //  Set CoreML Model Input&Output
    BlobMap input_blobs;
    BlobMap output_blobs;
    RETURN_ON_NEQ(GetAllInputBlobs(input_blobs),  TNN_OK);
    RETURN_ON_NEQ(GetAllOutputBlobs(output_blobs),  TNN_OK);
    
    coreml_model_description_ = std::shared_ptr<CoreML__Specification__ModelDescription>(new CoreML__Specification__ModelDescription);
    coreml_model_->description = (CoreML__Specification__ModelDescription *)coreml_model_description_.get();
    core_ml__specification__model_description__init(coreml_model_->description);
    coreml_model_->description->n_input = input_blobs.size();
    coreml_input_arr_ = std::shared_ptr<CoreML__Specification__FeatureDescription*>(new CoreML__Specification__FeatureDescription* [input_blobs.size()], [](CoreML__Specification__FeatureDescription** p) { delete[] p; });
    coreml_model_->description->input = coreml_input_arr_.get();
    int idx = 0;
    for(const auto& iter : input_blobs) {
        auto name = iter.first.c_str();
        auto shape = iter.second->GetBlobDesc().dims;
        auto type = iter.second->GetBlobDesc().data_type;
        RETURN_ON_NEQ(SetInput(coreml_model_->description->input + idx++, name, shape, type), TNN_OK);
    }
    coreml_model_->description->n_output = output_blobs.size();
    coreml_output_arr_ = std::shared_ptr<CoreML__Specification__FeatureDescription*>(new CoreML__Specification__FeatureDescription* [output_blobs.size()], [](CoreML__Specification__FeatureDescription** p) { delete[] p; });
    coreml_model_->description->output = coreml_output_arr_.get();
    idx = 0;
    for(const auto& iter : output_blobs) {
        auto name = iter.first.c_str();
        auto shape = iter.second->GetBlobDesc().dims;
        auto type = iter.second->GetBlobDesc().data_type;
        RETURN_ON_NEQ(SetOutput(coreml_model_->description->output + idx++, name, shape, type), TNN_OK);
    }
    
    return ret;
}

Status CoreMLNetwork::SetInput(CoreML__Specification__FeatureDescription** describe, std::string name, std::vector<int> shape, DataType type) {
    Status ret = TNN_OK;
    coreml_input_feature_description_.push_back(std::shared_ptr<CoreML__Specification__FeatureDescription>(new CoreML__Specification__FeatureDescription));
    (*describe) = (CoreML__Specification__FeatureDescription *)coreml_input_feature_description_.back().get();
    core_ml__specification__feature_description__init(*describe);
    input_name_.push_back(NullTerminatedCString(name));
    (*describe)->name = input_name_.back().get();
    coreml_input_feature_type_.push_back(std::shared_ptr<CoreML__Specification__FeatureType>(new CoreML__Specification__FeatureType));
    (*describe)->type = coreml_input_feature_type_.back().get();
    core_ml__specification__feature_type__init((*describe)->type);
    (*describe)->type->type_case = CORE_ML__SPECIFICATION__FEATURE_TYPE__TYPE_MULTI_ARRAY_TYPE;
    coreml_input_array_feature_type_.push_back(std::shared_ptr<CoreML__Specification__ArrayFeatureType>(new CoreML__Specification__ArrayFeatureType));
    (*describe)->type->multiarraytype = coreml_input_array_feature_type_.back().get();
    core_ml__specification__array_feature_type__init((*describe)->type->multiarraytype);
    switch (type) {
        case DATA_TYPE_FLOAT:
            (*describe)->type->multiarraytype->datatype = CORE_ML__SPECIFICATION__ARRAY_FEATURE_TYPE__ARRAY_DATA_TYPE__FLOAT32;
            break;
        case DATA_TYPE_INT32:
            (*describe)->type->multiarraytype->datatype = CORE_ML__SPECIFICATION__ARRAY_FEATURE_TYPE__ARRAY_DATA_TYPE__INT32;
            break;
        default:
            return Status(TNNERR_MODEL_ERR, "CoreML dont support this input array data type");
            break;
    }
    (*describe)->type->multiarraytype->n_shape = shape.size();
    coreml_input_shape_.push_back(std::shared_ptr<int64_t>(new int64_t [shape.size()], [](int64_t* p) { delete[] p; }));
    (*describe)->type->multiarraytype->shape = coreml_input_shape_.back().get();
    for (int i = 0; i < shape.size(); i++) {
        (*describe)->type->multiarraytype->shape[i] = shape[i];
    }
    return ret;
}

Status CoreMLNetwork::SetOutput(CoreML__Specification__FeatureDescription** describe, std::string name, std::vector<int> shape, DataType type) {
    Status ret = TNN_OK;
    coreml_output_feature_description_.push_back(std::shared_ptr<CoreML__Specification__FeatureDescription>(new CoreML__Specification__FeatureDescription));
    (*describe) = (CoreML__Specification__FeatureDescription *)coreml_output_feature_description_.back().get();
    core_ml__specification__feature_description__init(*describe);
    output_name_.push_back(NullTerminatedCString(name));
    (*describe)->name = output_name_.back().get();
    coreml_output_feature_type_.push_back(std::shared_ptr<CoreML__Specification__FeatureType>(new CoreML__Specification__FeatureType));
    (*describe)->type = coreml_output_feature_type_.back().get();
    core_ml__specification__feature_type__init((*describe)->type);
    (*describe)->type->type_case = CORE_ML__SPECIFICATION__FEATURE_TYPE__TYPE_MULTI_ARRAY_TYPE;
    coreml_output_array_feature_type_.push_back(std::shared_ptr<CoreML__Specification__ArrayFeatureType>(new CoreML__Specification__ArrayFeatureType));
    (*describe)->type->multiarraytype = coreml_output_array_feature_type_.back().get();
    core_ml__specification__array_feature_type__init((*describe)->type->multiarraytype);
    switch (type) {
        case DATA_TYPE_FLOAT:
            (*describe)->type->multiarraytype->datatype = CORE_ML__SPECIFICATION__ARRAY_FEATURE_TYPE__ARRAY_DATA_TYPE__FLOAT32;
            break;
        case DATA_TYPE_INT32:
            (*describe)->type->multiarraytype->datatype = CORE_ML__SPECIFICATION__ARRAY_FEATURE_TYPE__ARRAY_DATA_TYPE__INT32;
            break;
        default:
            return Status(TNNERR_MODEL_ERR, "CoreML dont support this output array data type");
            break;
    }
    (*describe)->type->multiarraytype->n_shape = shape.size();
    coreml_output_shape_.push_back(std::shared_ptr<int64_t>(new int64_t [shape.size()], [](int64_t* p) { delete[] p; }));
    (*describe)->type->multiarraytype->shape = coreml_output_shape_.back().get();
    for (int i = 0; i < shape.size(); i++) {
        (*describe)->type->multiarraytype->shape[i] = shape[i];
    }
    return ret;
}

Status CoreMLNetwork::ConvertCoreMLModel(NetStructure *net_structure, NetResource *net_resource) {
    Status ret = TNN_OK;
    
    //convert each layer
    coreml_layers_.clear();
    coreml_layer_ptrs_.clear();
    for (auto layer_info : net_structure->layers) {
        LayerType type            = layer_info->type;
        auto cur_layer = CreateCoreMLBaseLayer(type);
        if (cur_layer == nullptr) {
            LOGE("Error: CreateCoreMLBaseLayer failed, dont support type:%d\n", type);
            return Status(TNNERR_PARAM_ERR, "CreateCoreMLBaseLayer failed, dont support op");
        }
        cur_layer->SetNetResource(net_resource);
        
        auto resource = net_resource->resource_map[layer_info->name];
        // cur_layer->convert
        ret = cur_layer->Init(layer_info.get(), resource.get());
        if (ret != TNN_OK) {
            LOGE("Error Init CoreML layer %s (err: %s)\n", cur_layer->GetLayerName().c_str(), ret.description().c_str());
            return ret;
        }
        
        coreml_layers_.push_back(cur_layer);
        
        auto layer_ptrs = cur_layer->GetCoreMLLayerPtrs();
        coreml_layer_ptrs_.insert(coreml_layer_ptrs_.end(), layer_ptrs.begin(), layer_ptrs.end());
    }
    
    //set coreml layers
    coreml_model_->neuralnetwork->layers = coreml_layer_ptrs_.data();
    coreml_model_->neuralnetwork->n_layers = coreml_layer_ptrs_.size();
    
    return ret;
}

Status CoreMLNetwork::CompileModel(CoreML__Specification__Model* model) {
    
    if (@available(iOS 12.0, macOS 10.14, *)) {
        RETURN_ON_NEQ([coreml_executor_ buildFromProtoBuf:model], TNN_OK);
        
        return TNN_OK;
    } else {
        LOGE("Error: CoreML only support iOS 12+.\n");
        return Status(TNNERR_IOS_VERSION_ERROR, "CoreML only support iOS 12+.");
    }
}

Status CoreMLNetwork::GetForwardMemorySize(int &memory_size) {
    memory_size = 0;
    return Status(TNNERR_INST_ERR, "CoreML do not support GetForwardMemorySize");
}

Status CoreMLNetwork::SetForwardMemory(void *memory) {
    return Status(TNNERR_INST_ERR, "CoreML do not support SetForwardMemory");
}

Status CoreMLNetwork::CheckCoreMLStatus() {
    return TNN_OK;
}

Status CoreMLNetwork::GetAllInputBlobs(BlobMap &blobs) {
    if (blob_input_map_.size() > 0) {
        blobs = blob_input_map_;
        return TNN_OK;
    }
    
    auto status = CheckCoreMLStatus();
    if (status != TNN_OK) {
        return status;
    }
    
    BlobShapesMap inputs_shape_map;
    BlobDataTypeMap inputs_type_map;
    if (model_config_.model_type == MODEL_TYPE_TNN) {
        //filter out const input for layer unitest such as layernorm
        for (auto iter : net_structure_->inputs_shape_map) {
            if (net_resource_ && net_resource_->constant_map.find(iter.first) == net_resource_->constant_map.end()) {
                inputs_shape_map[iter.first] = iter.second;
                inputs_type_map[iter.first] = net_structure_->input_data_type_map[iter.first];
            }
        }
    } else if (model_config_.model_type == MODEL_TYPE_COREML) {
        RETURN_ON_NEQ([coreml_executor_ getInputShapesMap:inputs_shape_map
                                             dataTypesMap:inputs_type_map], TNN_OK);
    } else {
        return Status(TNNERR_MODEL_ERR, "CoreMLNetwork dont support the mode type in model_config");
    }
    
    for (const auto & iter : inputs_shape_map) {
        const auto & input_name = iter.first;
        const auto & input_dims = iter.second;
        const auto data_type = inputs_type_map[input_name];

        BlobDesc desc;
        {
            //use DEVICE_ARM OR DEVICE_X86 according to hardware
#if defined(__arm__) || defined(__arm64__)
            desc.device_type = DEVICE_ARM;
#else
            desc.device_type = DEVICE_X86;
#endif
            
            desc.data_type   = data_type;
            // data_format describes data order nchw, nhwc, ...
            desc.data_format = DATA_FORMAT_NCHW;
            desc.dims        = input_dims;
            desc.name       = input_name;
        };
        
        auto blob = std::make_shared<Blob>(desc, true);
        blob_input_.push_back(blob);
        blob_input_map_[desc.name] = blob.get();
    }

    blobs = blob_input_map_;
    return TNN_OK;
}

Status CoreMLNetwork::GetAllOutputBlobs(BlobMap &blobs) {
    if (blob_output_map_.size() > 0) {
        blobs = blob_output_map_;
        return TNN_OK;
    }
    
    auto status = CheckCoreMLStatus();
    if (status != TNN_OK) {
        return status;
    }
    
    BlobShapesMap outputs_shape_map;
    BlobDataTypeMap outputs_type_map;
    std::set<std::string> output_names;
    
    if (model_config_.model_type == MODEL_TYPE_TNN) {
        output_names = net_structure_->outputs;
        outputs_shape_map = net_resource_->blob_shapes_map;
        outputs_type_map = net_resource_->blob_datatype_map;
    } else if (model_config_.model_type == MODEL_TYPE_COREML) {
        RETURN_ON_NEQ([coreml_executor_ getOutputShapesMap:outputs_shape_map
                                             dataTypesMap:outputs_type_map], TNN_OK);
        output_names.clear();
        for (auto const& element : outputs_shape_map) {
            output_names.insert(element.first);
        }
    } else {
        return Status(TNNERR_MODEL_ERR, "CoreMLNetwork dont support the mode type in model_config");
    }
    
    for (const auto &output_name : output_names) {
        auto output_dims = outputs_shape_map[output_name];
        const auto data_type = outputs_type_map[output_name];
        
        BlobDesc desc;
        {
            //use DEVICE_ARM OR DEVICE_X86 according to hardware
#if defined(__arm__) || defined(__arm64__)
            desc.device_type = DEVICE_ARM;
#else
            desc.device_type = DEVICE_X86;
#endif
            
            desc.data_type   = data_type;
            // data_format describes data order nchw, nhwc, ...
            desc.data_format = DATA_FORMAT_NCHW;
            desc.dims        = output_dims;
            desc.name       = output_name;
        };

        auto blob = std::make_shared<Blob>(desc, true);
        blob_output_.push_back(blob);
        blob_output_map_[desc.name] = blob.get();
    }

    blobs = blob_output_map_;
    return TNN_OK;
}

Status CoreMLNetwork::Reshape(const InputShapesMap &inputs) {
    return Status(TNNERR_INST_ERR, "CoreML do not support Reshape");
}

Status CoreMLNetwork::DeInit() {
    coreml_executor_ = nil;
    
    return TNN_OK;
}

Status CoreMLNetwork::GetCommandQueue(void **command_queue) {
    if (context_ == NULL) {
        return Status(TNNERR_DEVICE_CONTEXT_CREATE, "CoreML GetCommandQueue is nil");
    }
    return context_->GetCommandQueue(command_queue);
}

Status CoreMLNetwork::Forward() {
    if (!HasAppleNPU()) {
        return Status(TNNERR_COMMON_ERROR, "Apple device dont have NeuralEngine");
    }
    
    if (@available(iOS 12.0, macOS 10.14, *)) {
        BlobMap blob_output_map;
        auto status = GetAllOutputBlobs(blob_output_map);
        if (status != TNN_OK) {
            return status;
        }
        NSError *error = nil;
        
        //construct coreml inputs
        NSMutableDictionary *input_dict = [NSMutableDictionary dictionary];

        for (auto iter = blob_input_map_.begin(); iter != blob_input_map_.end(); ++iter) {
            auto input_blob = blob_input_map_[iter->first];
            const auto data_type = input_blob->GetBlobDesc().data_type;
            
            auto input_name = [NSString stringWithCString:iter->first.c_str()
                                                 encoding:[NSString defaultCStringEncoding]];

            auto input_data =  (void *)input_blob->GetHandle().base;
            auto input_dims = input_blob->GetBlobDesc().dims;
            
            DimsVector input_stridess;
            for(int i=0; i<input_dims.size(); i++){
                int strides = 1;
                for(int j=i+1; j<input_dims.size(); j++){
                    strides = strides*DimsFunctionUtils::GetDim(input_dims, j);
                }
                input_stridess.push_back(strides);
            }

            NSMutableArray * shape_ = [[NSMutableArray alloc] init];
            NSMutableArray * strides_ = [[NSMutableArray alloc] init];
            for(int i=0; i<input_dims.size(); i++){
                [shape_ addObject:@(input_dims[i])];
                [strides_ addObject:@(input_stridess[i])];
            }
            
            MLMultiArrayDataType data_type_;
            switch (data_type) {
                case DATA_TYPE_FLOAT:
                    data_type_ = MLMultiArrayDataTypeFloat32;
                    break;
                case DATA_TYPE_INT32:
                    data_type_ = MLMultiArrayDataTypeInt32;
                    break;
                default:
                    return Status(TNNERR_PARAM_ERR, "CoreML can not support this input data type.");
                    break;
            }
            
            auto input_array = [[MLMultiArray alloc]  initWithDataPointer:input_data
                                                                    shape:shape_
                                                                 dataType:data_type_
                                                                  strides:strides_
                                                              deallocator:^(void *_Nonnull bytes) {}
                                                                    error:&error];
            LOGE_IF(error, "coreml alloc input array error: %s\n", error.debugDescription.UTF8String);
            RETURN_VALUE_ON_NEQ(error, nil, Status(TNNERR_MODEL_ERR, "coreml alloc input array error"));
            auto input_feat_value = [MLFeatureValue featureValueWithMultiArray:input_array];
            [input_dict setObject:input_feat_value forKey:input_name];
        }

        auto input  = [[MLDictionaryFeatureProvider alloc] initWithDictionary:input_dict
                                                                        error:&error];
        LOGE_IF(error, "coreml alloc input error: %s\n", error.debugDescription.UTF8String);
        RETURN_VALUE_ON_NEQ(error, nil, Status(TNNERR_MODEL_ERR, "coreml alloc input error"));
        
        //coreml forword
        auto output = (MLDictionaryFeatureProvider *)[coreml_executor_.model predictionFromFeatures:input
                                                                                                error:&error];
        LOGE_IF(error, "coreml predict error: %s\n", error.debugDescription.UTF8String);
        RETURN_VALUE_ON_NEQ(error, nil, Status(TNNERR_MODEL_ERR, "coreml predict error"));
        
        //copy output map
        for (auto iter = blob_output_map_.begin(); iter != blob_output_map_.end(); ++iter) {
            auto output_blob = blob_output_map[iter->first];
            
            auto output_name = [NSString stringWithCString:iter->first.c_str()
                                                  encoding:[NSString defaultCStringEncoding]];

            auto output_array = [output objectForKeyedSubscript:output_name].multiArrayValue;
            if (!output_array) {
                LOGE("The compiled CoreML model is invalid, you should try other devices\n");
                return Status(TNNERR_MODEL_ERR, "The compiled CoreML model is invalid, you should try other devices");
            }
            int out_data_count = DimsVectorUtils::Count(iter->second->GetBlobDesc().dims);

            
            auto output_data = (void *)output_blob->GetHandle().base;
            auto output_dims = output_blob->GetBlobDesc().dims;
            int bytes_count = out_data_count * DataTypeUtils::GetBytesSize(output_blob->GetBlobDesc().data_type);

            memcpy(output_data, output_array.dataPointer, bytes_count);

        }
        return TNN_OK;
    } else {
        return Status(TNNERR_IOS_VERSION_ERROR, "The operate system is not iOS 12+ or macOS 10.14+");
    }
}

// @brief tnn instance network infer, it will not wait
Status CoreMLNetwork::ForwardAsync(Callback call_back) {
    return Forward();
}
} // namespace TNN_NS
