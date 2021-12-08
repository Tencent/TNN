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
#include "tnn/device/metal/metal_context.h"
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
}

NetworkImplFactoryRegister<NetworkImplFactory<CoreMLNetwork>>
    g_network_impl_coreml_factory_register(NETWORK_TYPE_COREML);

CoreMLNetwork::CoreMLNetwork() {}

CoreMLNetwork::~CoreMLNetwork() {
    DeInit();
    for (auto iter : blob_input_map_) {
        if (iter.second && iter.second->GetHandle().base) {
            CFBridgingRelease(iter.second->GetHandle().base);
            iter.second->SetHandle(BlobHandle());
        }
    }
    blob_input_map_ = {};

    for (auto iter : blob_output_map_) {
        if (iter.second && iter.second->GetHandle().base) {
            CFBridgingRelease(iter.second->GetHandle().base);
            iter.second->SetHandle(BlobHandle());
        }
    }
    blob_output_map_ = {};
    
    if (blob_manager_ != NULL) {
        delete blob_manager_;
        blob_manager_ = NULL;
    }
    
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
    
    if (@available(iOS 12.0, macOS 10.14, *)) {
        Status ret = TNN_OK;

        DefaultModelInterpreter *default_interpreter = dynamic_cast<DefaultModelInterpreter *>(interpreter);
        CHECK_PARAM_NULL(default_interpreter);

        NetStructure *net_structure = default_interpreter->GetNetStructure();
        NetResource *net_resource   = default_interpreter->GetNetResource();
        if (net_structure == NULL || net_resource == NULL) {
            LOGE("ERROR: network_ is nil, network_type may not support\n");
            return Status(TNNERR_NULL_PARAM, "network_ is nil, network_type may not support");
        }
        
        auto type = net_config.device_type;
        if(type == DEVICE_APPLE_NPU){  // DEVICE_APPLE_NPU reuse DEVICE_METAL
            type = DEVICE_METAL;
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
        
        blob_manager_ = new BlobManager(device_);
        ret = blob_manager_->Init(net_config, net_structure, max_inputs_shape, GetNetResourceDataType(net_resource));
        
        RETURN_ON_NEQ(InitCoreMLModel(net_structure, net_resource), TNN_OK);

        RETURN_ON_NEQ(ConvertCoreMLModel(net_structure, net_resource), TNN_OK);

        RETURN_ON_NEQ(CompileModel(coreml_model_.get()), TNN_OK);
        
        NSError *error = nil;
        NSString *model_dir = compiled_model_file_path;
        NSData *data_net =
            [NSData dataWithContentsOfFile:[model_dir stringByAppendingPathComponent:@"model.espresso.net"]];
        NSData *data_shape =
            [NSData dataWithContentsOfFile:[model_dir stringByAppendingPathComponent:@"model.espresso.shape"]];
        if (!data_net || !data_shape) {
            LOGE("Error: CoreML net or shape file is invalid\n");
            return Status(TNNERR_INST_ERR, "CoreML net or shape file is invalid");
        }
        
        mlmodel_net_ = [NSJSONSerialization JSONObjectWithData:data_net
                                                      options:NSJSONReadingAllowFragments
                                                        error:&error];
        if (error || !mlmodel_net_ || [mlmodel_net_[@"layers"] count] <= 0) {
            LOGE("Error: MLModel modelWithContentsOfURL failed: invalid net file\n");
            return Status(TNNERR_INST_ERR, "MLModel modelWithContentsOfURL failed: invalid net file");
        }
        mlmodel_shape_ = [NSJSONSerialization JSONObjectWithData:data_shape
                                                        options:NSJSONReadingAllowFragments
                                                          error:&error];
        if (error || !mlmodel_shape_ || [mlmodel_shape_[@"layer_shapes"] count] <= 0) {
            LOGE("Error: MLModel modelWithContentsOfURL failed: invalid shape file\n");
            return Status(TNNERR_INST_ERR, "MLModel modelWithContentsOfURL failed: invalid shape file");
        }
        
        //  Configure ComputeUnits
        MLModelConfiguration* config = [[MLModelConfiguration alloc] init];
        if(net_config.device_type == DEVICE_ARM){
            config.computeUnits = MLComputeUnitsCPUOnly;
        } else if(net_config.device_type == DEVICE_METAL){
            config.computeUnits = MLComputeUnitsCPUAndGPU;
        } else if(net_config.device_type == DEVICE_APPLE_NPU){
            config.computeUnits = MLComputeUnitsAll;
        }

        mlmodel_ = [MLModel modelWithContentsOfURL:[NSURL fileURLWithPath:model_dir]
                                           configuration:config
                                                   error:&error];
        
        if (error || !mlmodel_) {
            LOGE("Error: MLModel modelWithContentsOfURL failed\n");
            return Status(TNNERR_INST_ERR, "MLModel modelWithContentsOfURL failed");
        }

//        return ret;
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
    blob_manager_->GetAllInputBlobs(input_blobs);
    blob_manager_->GetAllOutputBlobs(output_blobs);
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
        SetInput(coreml_model_->description->input + idx++, name, shape);
    }
    coreml_model_->description->n_output = output_blobs.size();
    coreml_output_arr_ = std::shared_ptr<CoreML__Specification__FeatureDescription*>(new CoreML__Specification__FeatureDescription* [output_blobs.size()], [](CoreML__Specification__FeatureDescription** p) { delete[] p; });
    coreml_model_->description->output = coreml_output_arr_.get();
    idx = 0;
    for(const auto& iter : output_blobs) {
        auto name = iter.first.c_str();
        auto shape = iter.second->GetBlobDesc().dims;
        SetOutput(coreml_model_->description->output + idx++, name, shape);
    }
    
    return ret;
}

void CoreMLNetwork::SetInput(CoreML__Specification__FeatureDescription** describe, std::string name, std::vector<int> shape) {
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
    (*describe)->type->multiarraytype->datatype = CORE_ML__SPECIFICATION__ARRAY_FEATURE_TYPE__ARRAY_DATA_TYPE__FLOAT32;
    (*describe)->type->multiarraytype->n_shape = shape.size();
    coreml_input_shape_.push_back(std::shared_ptr<int64_t>(new int64_t [shape.size()], [](int64_t* p) { delete[] p; }));
    (*describe)->type->multiarraytype->shape = coreml_input_shape_.back().get();
    for (int i = 0; i < shape.size(); i++) {
        (*describe)->type->multiarraytype->shape[i] = shape[i];
    }
}

void CoreMLNetwork::SetOutput(CoreML__Specification__FeatureDescription** describe, std::string name, std::vector<int> shape) {
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
    (*describe)->type->multiarraytype->datatype = CORE_ML__SPECIFICATION__ARRAY_FEATURE_TYPE__ARRAY_DATA_TYPE__FLOAT32;
    (*describe)->type->multiarraytype->n_shape = shape.size();
    coreml_output_shape_.push_back(std::shared_ptr<int64_t>(new int64_t [shape.size()], [](int64_t* p) { delete[] p; }));
    (*describe)->type->multiarraytype->shape = coreml_output_shape_.back().get();
    for (int i = 0; i < shape.size(); i++) {
        (*describe)->type->multiarraytype->shape[i] = shape[i];
    }
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

Status CoreMLNetwork::InitCoreMLExecutor() {
    Status ret = TNN_OK;
    
    if (coreml_executor_ == nullptr) {
        coreml_executor_ = [[CoreMLExecutor alloc] init];
    }
    if(coreml_executor_ == nullptr) {
        LOGE("Error: Failed to Init CoreML Executor.\n");
        return Status(TNNERR_ANE_EXECUTOR_ERROR, "Failed to Init CoreML Executor.");
    }
    
    return ret;
}

Status CoreMLNetwork::CompileModel(CoreML__Specification__Model* model) {
    RETURN_ON_NEQ(InitCoreMLExecutor(), TNN_OK);
    
    if (@available(iOS 12.0, macOS 10.14, *)) {
        auto executor = coreml_executor_;
        RETURN_ON_NEQ([executor saveModel:model], TNN_OK);
        NSURL* model_url = [executor getMLModelUrl];
        RETURN_ON_NEQ([executor build:model_url], TNN_OK);
        compiled_model_file_path = [executor getMLModelFilePath];

//        [executor cleanup];
        
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
    if (!mlmodel_net_ || [mlmodel_net_[@"layers"] count] <= 0) {
        LOGE("Error: MLModel modelWithContentsOfURL failed: invalid net file\n");
        return Status(TNNERR_INST_ERR, "MLModel modelWithContentsOfURL failed: invalid net file");
    }
    
    if (!mlmodel_shape_ || [mlmodel_shape_[@"layer_shapes"] count] <= 0) {
        LOGE("Error: MLModel modelWithContentsOfURL failed: invalid shape file\n");
        return Status(TNNERR_INST_ERR, "MLModel modelWithContentsOfURL failed: invalid shape file");
    }
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

    MetalContext *context              = dynamic_cast<MetalContext *>(context_);
    TNNMMetalContextImpl *context_impl = context->getMetalContextImpl();
    BlobMap input_blobs;
    blob_manager_->GetAllInputBlobs(input_blobs);
    
    for (auto iter = input_blobs.begin(); iter != input_blobs.end(); ++iter) {
        
        auto input_name = iter->first.c_str();
        auto input_shape = iter->second->GetBlobDesc().dims;

        DimsVector input_dims;
        for(int i=0; i<input_shape.size(); i++){
            input_dims.push_back(input_shape[i]);
        }
        
//        coreml_input_dims_    = input_dims;

        BlobDesc desc;
        {
            desc.device_type = DEVICE_METAL;
            desc.data_type   = DATA_TYPE_FLOAT;
            // data_format describes data order nchw, nhwc, ...
            desc.data_format = DATA_FORMAT_NCHW;
            desc.dims        = input_dims;
            desc.name        = input_name;
        };
        const int data_count = DimsFunctionUtils::GetDim(input_dims, 0) * (((DimsFunctionUtils::GetDim(input_dims, 1) + 3) / 4 * 4)) * DimsFunctionUtils::GetDim(input_dims, 2) * DimsFunctionUtils::GetDim(input_dims, 3) * DimsFunctionUtils::GetDim(input_dims, 4);

        int bytes_count      = data_count * DataTypeUtils::GetBytesSize(desc.data_type);
        id<MTLBuffer> buffer = [context_impl.device newBufferWithLength:bytes_count
                                                                options:MTLResourceCPUCacheModeDefaultCache];

        BlobHandle handle;
        {
            handle.base         = (void *)CFBridgingRetain(buffer);
            handle.bytes_offset = 0;
        };
        
        blob_input_.push_back(std::make_shared<Blob>(desc, handle));
        blob_input_map_[desc.name] = blob_input_.back().get();
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

    MetalContext *context              = dynamic_cast<MetalContext *>(context_);
    TNNMMetalContextImpl *context_impl = context->getMetalContextImpl();
    
    BlobMap output_blobs;
    blob_manager_->GetAllOutputBlobs(output_blobs);
    
    NSDictionary *layer_shapes = mlmodel_shape_[@"layer_shapes"];
    
    for (auto iter = output_blobs.begin(); iter != output_blobs.end(); ++iter) {
        
        auto output_name = iter->first.c_str();
        NSDictionary *output_shape = layer_shapes[@(output_name)];
        
        DimsVector output_dims;
        if([output_shape[@"seq"] intValue]){
            output_dims = {[output_shape[@"seq"] intValue],
                           [output_shape[@"n"] intValue],
                           [output_shape[@"k"] intValue],
                           [output_shape[@"h"] intValue],
                           [output_shape[@"w"] intValue]};
        } else {
            output_dims = {[output_shape[@"n"] intValue],
                           [output_shape[@"k"] intValue],
                           [output_shape[@"h"] intValue],
                           [output_shape[@"w"] intValue]};
        }
        coreml_output_dims_[output_name]    = output_dims;

        BlobDesc desc;
        {
            desc.device_type = DEVICE_METAL;
            desc.data_type   = DATA_TYPE_FLOAT;
            // data_format describes data order nchw, nhwc, ...
            desc.data_format = DATA_FORMAT_NCHW;
            desc.dims        = output_dims;
            desc.name        = output_name;
        };
        const int data_count = DimsFunctionUtils::GetDim(output_dims, 0) * (((DimsFunctionUtils::GetDim(output_dims, 1) + 3) / 4 * 4)) * DimsFunctionUtils::GetDim(output_dims, 2) * DimsFunctionUtils::GetDim(output_dims, 3) * DimsFunctionUtils::GetDim(output_dims, 4);
        int bytes_count      = data_count * DataTypeUtils::GetBytesSize(desc.data_type);
        id<MTLBuffer> buffer = [context_impl.device newBufferWithLength:bytes_count
                                                                options:MTLResourceCPUCacheModeDefaultCache];

        BlobHandle handle;
        {
            handle.base         = (void *)CFBridgingRetain(buffer);
            handle.bytes_offset = 0;
        };
        
        blob_output_.push_back(std::make_shared<Blob>(desc, handle));
        blob_output_map_[desc.name] = blob_output_.back().get();
    }

    blobs = blob_output_map_;
    return TNN_OK;
}

Status CoreMLNetwork::Reshape(const InputShapesMap &inputs) {
    return Status(TNNERR_INST_ERR, "CoreML do not support Reshape");
}

Status CoreMLNetwork::DeInit() {
    mlmodel_ = nil;
    mlmodel_net_   = nil;
    mlmodel_shape_ = nil;
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
      
        NSMutableDictionary *input_dict = [NSMutableDictionary dictionary];
        NSError *error = nil;

        for (auto iter = blob_input_map_.begin(); iter != blob_input_map_.end(); ++iter) {

            NSString *input_name = [NSString stringWithCString:iter->first.c_str() encoding:[NSString defaultCStringEncoding]];
            Blob *input_blob          = blob_input_map_[string(input_name.UTF8String)];
            auto input_mtl_buffer     = (__bridge id<MTLBuffer>)(void *)input_blob->GetHandle().base;
            auto input_dims           = input_blob->GetBlobDesc().dims;

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
            MLMultiArray * input_array = [[MLMultiArray alloc]
            initWithDataPointer:input_mtl_buffer.contents
                          shape:shape_
                       dataType:MLMultiArrayDataTypeFloat32
                        strides:strides_
                    deallocator:^(void *_Nonnull bytes) {}
                          error:&error];
            MLFeatureValue *input_feat_value = [MLFeatureValue featureValueWithMultiArray:input_array];
            [input_dict setObject:input_feat_value forKey:input_name];
        }

        auto input  = [[MLDictionaryFeatureProvider alloc] initWithDictionary:input_dict
                                                                        error:&error];
        auto output = (MLDictionaryFeatureProvider *)[(MLModel *)mlmodel_ predictionFromFeatures:input
                                                                                                error:&error];
  
        for (auto iter = blob_output_map_.begin(); iter != blob_output_map_.end(); ++iter) {
            auto output_name = iter->first.c_str();
            MLMultiArray *output_array = [output objectForKeyedSubscript:@(output_name)].multiArrayValue;
            int out_data_count         = DimsVectorUtils::Count(coreml_output_dims_[output_name]);
            Blob *output_blob      = blob_output_map[output_name];
            auto output_mtl_buffer = (__bridge id<MTLBuffer>)(void *)output_blob->GetHandle().base;
            auto output_dims       = output_blob->GetBlobDesc().dims;
            int bytes_count        = out_data_count * DataTypeUtils::GetBytesSize(output_blob->GetBlobDesc().data_type);
            memcpy(output_mtl_buffer.contents, output_array.dataPointer, bytes_count);
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
