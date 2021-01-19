#include "tnn/device/metal/coreml_network.h"
#include "tnn/device/metal/metal_context.h"
#include "tnn/utils/data_type_utils.h"
#include "tnn/utils/dims_vector_utils.h"

namespace TNN_NS {

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
}

Status CoreMLNetwork::Init(NetworkConfig &net_config, ModelConfig &model_config, AbstractModelInterpreter *interpreter,
                           InputShapesMap min_inputs_shape, InputShapesMap max_inputs_shape) {
    if (@available(iOS 11.0, macOS 10.13, *)) {
        Status ret = TNN_OK;

        device_ = GetDevice(net_config.device_type);
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

        if (model_config.params.size() < 1) {
            LOGE("Error: ModelConfig.params[0] is not a directory of MLModel\n");
            return Status(TNNERR_INST_ERR, "Error: ModelConfig.params[0] is not a directory of MLModel");
        }

        NSError *error = nil;

        NSString *model_dir = [NSString stringWithUTF8String:model_config.params[0].c_str()];
        NSData *data_net =
            [NSData dataWithContentsOfFile:[model_dir stringByAppendingPathComponent:@"model.espresso.net"]];
        NSData *data_shape =
            [NSData dataWithContentsOfFile:[model_dir stringByAppendingPathComponent:@"model.espresso.shape"]];
        if (!data_net || !data_shape) {
            LOGE("Error: CoreML net or shape file is invalid\n");
            return Status(TNNERR_INST_ERR, "CoreML net or shape file is invalid");
        }

        coreml_net_ = [NSJSONSerialization JSONObjectWithData:data_net
                                                      options:NSJSONReadingAllowFragments
                                                        error:&error];
        if (error || !coreml_net_ || [coreml_net_[@"layers"] count] <= 0) {
            LOGE("Error: MLModel modelWithContentsOfURL failed: invalid net file\n");
            return Status(TNNERR_INST_ERR, "MLModel modelWithContentsOfURL failed: invalid net file");
        }
        coreml_shape_ = [NSJSONSerialization JSONObjectWithData:data_shape
                                                        options:NSJSONReadingAllowFragments
                                                          error:&error];
        if (error || !coreml_shape_ || [coreml_shape_[@"layer_shapes"] count] <= 0) {
            LOGE("Error: MLModel modelWithContentsOfURL failed: invalid shape file\n");
            return Status(TNNERR_INST_ERR, "MLModel modelWithContentsOfURL failed: invalid shape file");
        }

        coreml_model_ = [MLModel modelWithContentsOfURL:[NSURL fileURLWithPath:model_dir] error:&error];

        if (error || !coreml_model_) {
            LOGE("Error: MLModel modelWithContentsOfURL failed\n");
            return Status(TNNERR_INST_ERR, "MLModel modelWithContentsOfURL failed");
        }

        return ret;
    } else {
        return Status(TNNERR_INST_ERR, "CoreML only support iOS 11+");
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
    if (!coreml_net_ || [coreml_net_[@"layers"] count] <= 0) {
        LOGE("Error: MLModel modelWithContentsOfURL failed: invalid net file\n");
        return Status(TNNERR_INST_ERR, "MLModel modelWithContentsOfURL failed: invalid net file");
    }
    
    if (!coreml_shape_ || [coreml_shape_[@"layer_shapes"] count] <= 0) {
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
    NSDictionary *layer_shapes = coreml_shape_[@"layer_shapes"];
    NSArray *layeres = coreml_net_[@"layers"];
    
    {
        NSString *input_name       = layeres[0][@"bottom"];
        NSDictionary *inuput_shape = layer_shapes[input_name];

        DimsVector input_dims = {[inuput_shape[@"n"] intValue],
                                 [inuput_shape[@"k"] intValue],
                                 [inuput_shape[@"h"] intValue],
                                 [inuput_shape[@"w"] intValue]};
        coreml_input_dims_    = input_dims;

        BlobDesc desc;
        {
            desc.device_type = DEVICE_METAL;
            desc.data_type   = DATA_TYPE_FLOAT;
            // data_format describes data order nchw, nhwc, ...
            desc.data_format = DATA_FORMAT_NCHW;
            desc.dims        = input_dims;
            desc.name        = input_name.UTF8String;
        };
        const int data_count = input_dims[0] * (((input_dims[1] + 3) / 4 * 4)) * input_dims[2] * input_dims[3];

        int bytes_count      = data_count * DataTypeUtils::GetBytesSize(desc.data_type);
        id<MTLBuffer> buffer = [context_impl.device newBufferWithLength:bytes_count
                                                                options:MTLResourceCPUCacheModeDefaultCache];

        BlobHandle handle;
        {
            handle.base         = (void *)CFBridgingRetain(buffer);
            handle.bytes_offset = 0;
        };

        blob_input_map_[desc.name] = new Blob(desc, handle);
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
    
    NSDictionary *layer_shapes = coreml_shape_[@"layer_shapes"];
    NSArray *layeres = coreml_net_[@"layers"];
    
    {
        NSString *output_name      = layeres[layeres.count - 1][@"top"];
        NSDictionary *output_shape = layer_shapes[output_name];

        DimsVector output_dims = {[output_shape[@"n"] intValue],
                                  [output_shape[@"k"] intValue],
                                  [output_shape[@"h"] intValue],
                                  [output_shape[@"w"] intValue]};
        coreml_output_dims_    = output_dims;

        BlobDesc desc;
        {
            desc.device_type = DEVICE_METAL;
            desc.data_type   = DATA_TYPE_FLOAT;
            // data_format describes data order nchw, nhwc, ...
            desc.data_format = DATA_FORMAT_NCHW;
            desc.dims        = output_dims;
            desc.name        = output_name.UTF8String;
        };
        const int data_count = output_dims[0] * (((output_dims[1] + 3) / 4 * 4)) * output_dims[2] * output_dims[3];

        int bytes_count      = data_count * DataTypeUtils::GetBytesSize(desc.data_type);
        id<MTLBuffer> buffer = [context_impl.device newBufferWithLength:bytes_count
                                                                options:MTLResourceCPUCacheModeDefaultCache];

        BlobHandle handle;
        {
            handle.base         = (void *)CFBridgingRetain(buffer);
            handle.bytes_offset = 0;
        };

        blob_output_map_[desc.name] = new Blob(desc, handle);
    }

    blobs = blob_output_map_;
    return TNN_OK;
}

Status CoreMLNetwork::Reshape(const InputShapesMap &inputs) {
    return Status(TNNERR_INST_ERR, "CoreML do not support Reshape");
}

Status CoreMLNetwork::DeInit() {
    coreml_model_ = nil;
    return TNN_OK;
}

Status CoreMLNetwork::GetCommandQueue(void **command_queue) {
    if (context_ == NULL) {
        return Status(TNNERR_DEVICE_CONTEXT_CREATE, "CoreML GetCommandQueue is nil");
    }
    return context_->GetCommandQueue(command_queue);
}

Status CoreMLNetwork::Forward() {
    if (@available(iOS 11.0, macOS 10.13, *)) {
        BlobMap blob_output_map;
        auto status = GetAllOutputBlobs(blob_output_map);
        if (status != TNN_OK) {
            return status;
        }

        if (!coreml_net_ || [coreml_net_[@"layers"] count] <= 0) {
            LOGE("Error: MLModel modelWithContentsOfURL failed: invalid net file\n");
            return Status(TNNERR_INST_ERR, "MLModel modelWithContentsOfURL failed: invalid net file");
        }
        NSArray *layeres      = coreml_net_[@"layers"];
        NSString *input_name  = layeres[0][@"bottom"];
        NSString *output_name = layeres[layeres.count - 1][@"top"];

        Blob *input_blob          = blob_input_map_[string(input_name.UTF8String)];
        auto input_mtl_buffer     = (__bridge id<MTLBuffer>)(void *)input_blob->GetHandle().base;
        auto input_dims           = input_blob->GetBlobDesc().dims;
        DimsVector input_stridess = {input_dims[1] * input_dims[2] * input_dims[3], input_dims[2] * input_dims[3],
                                     input_dims[3], 1};
        NSError *error            = nil;
        MLMultiArray *input_array = [[MLMultiArray alloc]
            initWithDataPointer:input_mtl_buffer.contents
                          shape:@[ @(input_dims[1]), @(input_dims[2]), @(input_dims[3]) ]
                       dataType:MLMultiArrayDataTypeFloat32
                        strides:@[ @(input_stridess[1]), @(input_stridess[2]), @(input_stridess[3]) ]
                    deallocator:^(void *_Nonnull bytes) {
                    }
                          error:&error];
        MLFeatureValue *input_feat_value = [MLFeatureValue featureValueWithMultiArray:input_array];
        auto input  = [[MLDictionaryFeatureProvider alloc] initWithDictionary:@{input_name : input_feat_value}
                                                                       error:&error];
        auto output = (MLDictionaryFeatureProvider *)[(MLModel *)coreml_model_ predictionFromFeatures:input
                                                                                                error:&error];
        MLMultiArray *output_array = [output objectForKeyedSubscript:output_name].multiArrayValue;
        int out_data_count         = DimsVectorUtils::Count(coreml_output_dims_);

        Blob *output_blob      = blob_output_map[string(output_name.UTF8String)];
        auto output_mtl_buffer = (__bridge id<MTLBuffer>)(void *)output_blob->GetHandle().base;
        auto output_dims       = output_blob->GetBlobDesc().dims;
        int bytes_count        = out_data_count * DataTypeUtils::GetBytesSize(output_blob->GetBlobDesc().data_type);

        memcpy(output_mtl_buffer.contents, output_array.dataPointer, bytes_count);
        return TNN_OK;
    } else {
        return Status(TNNERR_INST_ERR, "CoreML only support iOS 11+");
    }
}

// @brief tnn instance network infer, it will not wait
Status CoreMLNetwork::ForwardAsync(Callback call_back) {
    return Forward();
}
} // namespace TNN_NS
