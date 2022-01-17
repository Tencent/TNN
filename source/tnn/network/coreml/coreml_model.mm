#import <CoreML/CoreML.h>
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#import "coreml_model.h"

#include <fstream>
#include <iostream>

using namespace TNN_NS;

@interface CoreMLModel()
@property(nonatomic, strong) NSString *cachePath;
@property(nonatomic, strong) NSString *ID;
@property(nonatomic, strong) MLModel* model API_AVAILABLE(ios(12.0), macosx(10.14));
@property(nonatomic, strong) NSURL *mlmodelPath;
@property(nonatomic, strong) NSURL *mlmodelcPath;

- (NSURL *)mlmodelPath;

- (TNN_NS::Status)saveModel:(CoreML__Specification__Model*)model;
- (TNN_NS::Status)buildFromPathMLModel;
- (TNN_NS::Status)buildFromPathMLModelC;

- (TNN_NS::Status)getShapesMap:(TNN_NS::BlobShapesMap &) shapesMap
                                dataTypesMap:(TNN_NS::BlobDataTypeMap &) typesMap
fromFeature:(NSDictionary<NSString *, MLFeatureDescription *> *) featureDict API_AVAILABLE(ios(12.0), macosx(10.14));

- (TNN_NS::Status)cleanupMLModel;
- (TNN_NS::Status)cleanupMLModelC;

@end

@implementation CoreMLModel
- (instancetype)initWithCachePath:(std::string)path ID:(std::string) ID {
    if (self = [super init]) {
        _cachePath = [NSString stringWithUTF8String:path.c_str()];
        if (![[NSFileManager defaultManager] fileExistsAtPath:_cachePath]) {
            LOGE("The input cache path (%s) is invalid, automatically try NSTemporaryDirectory\n", path.c_str());
            _cachePath = NSTemporaryDirectory();
#if TNN_COREML_TEST
            LOGE("Input cache path: %s\n", _cachePath.UTF8String);
#endif
        }
        _ID = [NSString stringWithUTF8String:ID.c_str()];
    }
    return self;
}

- (instancetype)initWithModelPath:(std::string)path {
    if (self = [super init]) {
        auto spath = [NSString stringWithUTF8String:path.c_str()];
        if ([spath hasSuffix:@".mlmodel"]) {
            _mlmodelPath = [NSURL fileURLWithPath:spath];
        } else if ([spath hasSuffix:@".mlmodelc"]) {
            _mlmodelcPath = [NSURL fileURLWithPath:spath];
        }
    }
    return self;
}

- (NSURL *)mlmodelPath {
    if ([[_mlmodelPath path] length] >0) {
        return _mlmodelPath;
    }
    
    if (_cachePath && _cachePath.length>0 && _ID && _ID.length > 0) {
        auto tempURL = [NSURL fileURLWithPath:_cachePath isDirectory:YES];
        _mlmodelPath =  [tempURL URLByAppendingPathComponent:_ID];
        return _mlmodelPath;
    } else {
        return nil;
    }
}

- (NSURL *)mlmodelcPath {
    if ([[_mlmodelcPath path] length] >0) {
        return _mlmodelcPath;
    }
    
    if (_cachePath && _cachePath.length>0 && _ID && _ID.length > 0) {
        auto tempURL = [NSURL fileURLWithPath:_cachePath isDirectory:YES];
        _mlmodelcPath=  [tempURL URLByAppendingPathComponent:[_ID stringByAppendingString:@".mlmodelc"]];
        return _mlmodelcPath;
    } else {
        return nil;
    }
}

- (TNN_NS::Status)buildFromCache {
#if TNN_COREML_TEST
    return TNN_NS::Status(TNN_NS::TNNERR_ANE_COMPILE_MODEL_ERROR, "buildFromCache is disabled when TNN_COREML_TEST is ON");
#else
    return [self buildFromPathMLModelC];
#endif
}

- (TNN_NS::Status)buildFromModelPath {
    auto mlmodelcURL = [self mlmodelcPath];
    if ([[NSFileManager defaultManager] fileExistsAtPath:mlmodelcURL.path]) {
        return [self buildFromPathMLModelC];
    }
    
    auto mlmodelURL = [self mlmodelPath];
    if ([[NSFileManager defaultManager] fileExistsAtPath:mlmodelURL.path]) {
        return [self buildFromPathMLModel];
    }
    
    LOGE("The input mlmodel path (%s) does not exist\n", mlmodelURL.path.UTF8String);
    return TNN_NS::Status(TNN_NS::TNNERR_MODEL_ERR, "The input mlmodel path does not exist");
}

- (TNN_NS::Status)buildFromProtoBuf:(CoreML__Specification__Model*)model {
#ifdef DEBUG
    auto time_start = CFAbsoluteTimeGetCurrent();
#endif
    
    //mlmodel path
    auto mlmodelURL = [self mlmodelPath];
    
    //save mlmodel
    auto status = [self saveModel:model];
    RETURN_ON_NEQ(status, TNN_NS::TNN_OK);
    
    //build mlmodelc
    status = [self buildFromPathMLModel];
    
    //remove mlmodel, no need to check error
    [self cleanupMLModel];
    
#if TNN_COREML_TEST
    [self cleanupMLModelC];
#endif
    
#ifdef DEBUG
    LOGD("TNN buildFromProtoBuf time: %f ms\n", (CFAbsoluteTimeGetCurrent() - time_start) * 1000.0);
#endif
    return status;
}

- (TNN_NS::Status)cleanupMLModel {
    NSError* error = nil;
    
    //remove mlmodel, no need to check error
    auto mlmodelURL = [self mlmodelPath];
    [[NSFileManager defaultManager] removeItemAtURL:mlmodelURL error:&error];
    return TNN_NS::TNN_OK;
}

- (TNN_NS::Status)cleanupMLModelC {
    NSError* error = nil;
    
    //remove mlmodelc, no need to check error
    auto mlmodelcURL = [self mlmodelcPath];
    [[NSFileManager defaultManager] removeItemAtURL:mlmodelcURL error:&error];
    return TNN_NS::TNN_OK;
}

- (TNN_NS::Status)cleanup {
    [self cleanupMLModel];
    [self cleanupMLModelC];
    return TNN_NS::TNN_OK;
}

- (TNN_NS::Status)saveModel:(CoreML__Specification__Model*)model {
    //mlmodel path
    auto mlmodelURL = [self mlmodelPath];
    
    if (!(model->specificationversion == 3 || model->specificationversion == 4)) {
        LOGE("Only Core ML models with specification version 3 or 4 are supported.\n");
        return TNN_NS::Status(TNN_NS::TNNERR_COREML_VERSION_ERROR, "Error: Only Core ML models with specification version 3 or 4 are supported.");
    }
    
    size_t modelSize = core_ml__specification__model__get_packed_size(model);
    std::unique_ptr<uint8_t> writeBuffer(new uint8_t[modelSize]);
    core_ml__specification__model__pack(model, writeBuffer.get());
    std::ofstream file_stream([[mlmodelURL path] UTF8String], std::ios::out | std::ios::binary);
    if (!file_stream || !file_stream.is_open() || !file_stream.good()) {
        file_stream.close();
        LOGE("CoreML models file can not be written.\n");
        return TNN_NS::Status(TNN_NS::TNNERR_ANE_SAVE_MODEL_ERROR, "Error: CoreML models file can not be written.");
    }
    const char* ptr = reinterpret_cast<const char*>(writeBuffer.get());
    if (ptr) {
        file_stream.write(ptr, modelSize);
        file_stream.close();
    } else {
        file_stream.close();
        LOGE("CoreML models file is empty.\n");
        return TNN_NS::Status(TNN_NS::TNNERR_ANE_SAVE_MODEL_ERROR, "Error: CoreML models file is empty.");
    }
    
    return TNN_NS::TNN_OK;
}

- (TNN_NS::Status)buildFromPathMLModel {
    if (@available(iOS 12.0, macOS 10.14, *)) {
        //mlmodel path
        auto mlmodelURL = [self mlmodelPath];
        
        NSError* error = nil;
        NSURL* mlmodelcURL = [MLModel compileModelAtURL:mlmodelURL error:&error];
        if (error != nil) {
            [self cleanupMLModelC];
            LOGE("Error compiling model %s.\n", [error localizedDescription].UTF8String);
            return TNN_NS::Status(TNN_NS::TNNERR_ANE_COMPILE_MODEL_ERROR, "Error: Failed compiling model.");
        }
        //To update modelcURL, it may differ from the default value, for example mlmodel is packed in the app, mlmodelc will be stored in NSTemporaryDirectory
        self.mlmodelcPath = mlmodelcURL;
        
        return [self buildFromPathMLModelC];
    } else {
        LOGE("Error: CoreML only support iOS 12+.\n");
        return TNN_NS::Status(TNN_NS::TNNERR_ANE_COMPILE_MODEL_ERROR, "CoreML only support iOS 12+.");
    }
}

- (TNN_NS::Status)buildFromPathMLModelC {
    //mlmodelc path
    auto mlmodelcURL = [self mlmodelcPath];
    
    if (@available(iOS 12.0, macOS 10.14, *)) {
#ifdef DEBUG
    auto time_start = CFAbsoluteTimeGetCurrent();
#endif
        
        NSError* error = nil;
        MLModelConfiguration* config = [MLModelConfiguration alloc];
#if TNN_COREML_TEST
        config.computeUnits = MLComputeUnitsCPUOnly;
#else
        config.computeUnits = MLComputeUnitsAll;
#endif
        _model = [MLModel modelWithContentsOfURL:mlmodelcURL configuration:config error:&error];
        
        if (error != nil) {
            [self cleanupMLModelC];
            LOGE("Error Creating MLModel %s.\n", [error localizedDescription].UTF8String);
            return TNN_NS::Status(TNN_NS::TNNERR_ANE_COMPILE_MODEL_ERROR, "Error: Failed Creating MLModel.");
        }
        
#ifdef DEBUG
    LOGD("TNN buildFromPathMLModelC time: %f ms\n", (CFAbsoluteTimeGetCurrent() - time_start) * 1000.0);
#endif
        return TNN_NS::TNN_OK;
    } else {
        LOGE("Error: CoreML only support iOS 12+.\n");
        return TNN_NS::Status(TNN_NS::TNNERR_ANE_COMPILE_MODEL_ERROR, "CoreML only support iOS 12+.");
    }
}

- (TNN_NS::Status)getInputShapesMap:(TNN_NS::BlobShapesMap &) shapesMap
                                        dataTypesMap:(TNN_NS::BlobDataTypeMap &) typesMap {
    if (@available(iOS 12.0, macOS 10.14, *)) {
        auto description = [_model modelDescription];
        auto feature_dict = [description inputDescriptionsByName];
        return [self getShapesMap:shapesMap
                     dataTypesMap:typesMap
                      fromFeature:feature_dict];
    }
    return TNN_OK;
}

- (TNN_NS::Status)getOutputShapesMap:(TNN_NS::BlobShapesMap &) shapesMap
                                           dataTypesMap:(TNN_NS::BlobDataTypeMap &) typesMap {
    if (@available(iOS 12.0, macOS 10.14, *)) {
        auto description = [_model modelDescription];
        auto feature_dict = [description outputDescriptionsByName];
        return [self getShapesMap:shapesMap
                     dataTypesMap:typesMap
                      fromFeature:feature_dict];
    }
    return TNN_OK;
}

- (TNN_NS::Status)getShapesMap:(TNN_NS::BlobShapesMap &) shapesMap
                                dataTypesMap:(TNN_NS::BlobDataTypeMap &) typesMap
                                    fromFeature:(NSDictionary<NSString *, MLFeatureDescription *>*) featureDict {
    shapesMap.clear();
    typesMap.clear();

    for (NSString *name in featureDict) {
        std::string input_name = name.UTF8String;
        
        auto feature = featureDict[name];
        if (feature.type != MLFeatureTypeMultiArray) {
            LOGE("CoreMLNetwork dont support MLFeatureDescription with type %d", (int)feature.type);
            return Status(TNNERR_MODEL_ERR, "TNN only supports input and out with type of MLFeatureTypeMultiArray");
        }
        auto shapes = feature.multiArrayConstraint.shape;
        
        DimsVector input_dims;
        for (NSNumber *iter in shapes) {
            input_dims.push_back([iter intValue]);
        }
        
        DataType data_type;
        if (feature.multiArrayConstraint.dataType == MLMultiArrayDataTypeFloat32) {
            data_type = DATA_TYPE_FLOAT;
        } else if (feature.multiArrayConstraint.dataType == MLMultiArrayDataTypeInt32) {
            data_type = DATA_TYPE_INT32;
        } else {
            LOGE("CoreMLNetwork dont support MLFeatureTypeMultiArray withdata type %d", (int)feature.multiArrayConstraint.dataType);
            return Status(TNNERR_MODEL_ERR, "TNNCoreMLNetwork only supports input and out with data type Float32 or Int32");
        }

        shapesMap[input_name] = input_dims;
        typesMap[input_name] = data_type;
    }
    return TNN_OK;
}

@end
