#import <CoreML/CoreML.h>
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#import "coreml_model.h"

#include <fstream>
#include <iostream>

@interface CoreMLModel()
@property(nonatomic, strong) NSString *cachePath;
@property(nonatomic, strong) NSString *ID;
@property(nonatomic, strong) MLModel* model API_AVAILABLE(ios(12.0), macosx(10.14));
@property(nonatomic, strong) NSURL *mlmodelcPath;

- (NSURL *)mlmodelPath;

- (TNN_NS::Status)saveModel:(CoreML__Specification__Model*)model;
- (TNN_NS::Status)build:(NSURL*)modelUrl;
@end

@implementation CoreMLModel
- (instancetype)initWithCachePath:(std::string)path ID:(std::string) ID {
    if (self = [super init]) {
        _cachePath = [NSString stringWithUTF8String:path.c_str()];
        if (![[NSFileManager defaultManager] fileExistsAtPath:_cachePath]) {
            LOGE("The input cache path (%s) is invalid, automatically try NSTemporaryDirectory\n", path.c_str());
            _cachePath = NSTemporaryDirectory();
        }
        _ID = [NSString stringWithUTF8String:ID.c_str()];
    }
    return self;
}

- (NSURL *)mlmodelPath {
    auto tempURL = [NSURL fileURLWithPath:_cachePath isDirectory:YES];
    return [tempURL URLByAppendingPathComponent:_ID];
}
- (NSURL *)mlmodelcPath {
    if ([[_mlmodelcPath path] length] >0) {
        return _mlmodelcPath;
    }
    auto tempURL = [NSURL fileURLWithPath:_cachePath isDirectory:YES];
    _mlmodelcPath=  [tempURL URLByAppendingPathComponent:[_ID stringByAppendingString:@".mlmodelc"]];
    return _mlmodelcPath;
}

- (TNN_NS::Status)buildFromCache {
    //mlmodelc path
    auto mlmodelcURL = [self mlmodelcPath];
    
    if (@available(iOS 12.0, macOS 10.14, *)) {
#ifdef DEBUG
    auto time_start = CFAbsoluteTimeGetCurrent();
#endif
        
        NSError* error = nil;
        MLModelConfiguration* config = [MLModelConfiguration alloc];
        config.computeUnits = MLComputeUnitsAll;
        _model = [MLModel modelWithContentsOfURL:mlmodelcURL configuration:config error:&error];
        
        if (error != nil) {
            LOGE("Error Creating MLModel %s.\n", [error localizedDescription].UTF8String);
            return TNN_NS::Status(TNN_NS::TNNERR_ANE_COMPILE_MODEL_ERROR, "Error: Failed Creating MLModel.");
        }
        
#ifdef DEBUG
    LOGD("TNN buildFromCache time: %f ms\n", (CFAbsoluteTimeGetCurrent() - time_start) * 1000.0);
#endif
        return TNN_NS::TNN_OK;
    } else {
        LOGE("Error: CoreML only support iOS 12+.\n");
        return TNN_NS::Status(TNN_NS::TNNERR_ANE_COMPILE_MODEL_ERROR, "CoreML only support iOS 12+.");
    }
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
    status = [self build:mlmodelURL];
    
    //remove mlmodel, no need to check error
    NSError* error = nil;
    [[NSFileManager defaultManager] removeItemAtURL:mlmodelURL error:&error];
    
#ifdef DEBUG
    LOGD("TNN buildFromProtoBuf time: %f ms\n", (CFAbsoluteTimeGetCurrent() - time_start) * 1000.0);
#endif
    return status;
}

- (TNN_NS::Status)cleanup {
    NSError* error = nil;
    
    //remove mlmodel, no need to check error
    auto mlmodelURL = [self mlmodelPath];
    [[NSFileManager defaultManager] removeItemAtURL:mlmodelURL error:&error];
    
    //remove mlmodelc, no need to check error
    auto mlmodelcURL = [self mlmodelPath];
    [[NSFileManager defaultManager] removeItemAtURL:mlmodelcURL error:&error];
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

- (TNN_NS::Status)build:(NSURL*)mlmodelURL {
    if (@available(iOS 12.0, macOS 10.14, *)) {
        NSError* error = nil;
        NSURL* mlmodelcURL = [MLModel compileModelAtURL:mlmodelURL error:&error];
        if (error != nil) {
            LOGE("Error compiling model %s.\n", [error localizedDescription].UTF8String);
            return TNN_NS::Status(TNN_NS::TNNERR_ANE_COMPILE_MODEL_ERROR, "Error: Failed compiling model.");
        }
        //To update modelcURL, it may differ from the default value
        self.mlmodelcPath = mlmodelcURL;
    
        MLModelConfiguration* config = [MLModelConfiguration alloc];
        config.computeUnits = MLComputeUnitsAll;
        _model = [MLModel modelWithContentsOfURL:mlmodelcURL configuration:config error:&error];
        
        if (error != nil) {
            LOGE("Error Creating MLModel %s.\n", [error localizedDescription].UTF8String);
            return TNN_NS::Status(TNN_NS::TNNERR_ANE_COMPILE_MODEL_ERROR, "Error: Failed Creating MLModel.");
        }
        
        return TNN_NS::TNN_OK;
    } else {
        LOGE("Error: CoreML only support iOS 12+.\n");
        return TNN_NS::Status(TNN_NS::TNNERR_ANE_COMPILE_MODEL_ERROR, "CoreML only support iOS 12+.");
    }
}

@end
