#import <CoreML/CoreML.h>
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#import "coreml_executor.h"

#include <fstream>
#include <iostream>

@interface CoreMLExecutor()
@property(nonatomic, strong) NSString *cachePath;
@property(nonatomic, strong) NSString *ID;
@property(nonatomic, strong) MLModel* model API_AVAILABLE(ios(12.0), macosx(10.14));

- (NSURL *)mlmodelPath;
- (NSURL *)mlmodelcPath;

- (TNN_NS::Status)saveModel:(CoreML__Specification__Model*)model;
- (TNN_NS::Status)build:(NSURL*)modelUrl;
@end

@implementation CoreMLExecutor
- (instancetype)initWithCachePath:(std::string)path ID:(std::string) ID {
    if (self = [super init]) {
        _cachePath = [NSString stringWithUTF8String:path.c_str()];
        _ID = [NSString stringWithUTF8String:ID.c_str()];
    }
    return self;
}

- (NSURL *)mlmodelPath {
    auto tempURL = [NSURL fileURLWithPath:_cachePath isDirectory:YES];
    return [tempURL URLByAppendingPathComponent:_ID];
}
- (NSURL *)mlmodelcPath {
    auto tempURL = [NSURL fileURLWithPath:_cachePath isDirectory:YES];
    return [tempURL URLByAppendingPathComponent:[_ID stringByAppendingString:@".mlmodelc"]];
}

- (TNN_NS::Status)buildFromCache {
    //mlmodelc path
    auto mlmodelcURL = [self mlmodelcPath];
    
    if (@available(iOS 12.0, macOS 10.14, *)) {
        NSError* error = nil;
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

- (TNN_NS::Status)buildFromProtoBuf:(CoreML__Specification__Model*)model {
    //mlmodel path
    auto mlmodelURL = [self mlmodelPath];
    
    auto status = [self saveModel:model];
    RETURN_ON_NEQ(status, TNN_NS::TNN_OK);
    
    status = [self build:mlmodelURL];
    return status;
}

- (TNN_NS::Status)cleanup {
//    NSError* error = nil;
//    [[NSFileManager defaultManager] removeItemAtPath:_mlModelFilePath error:&error];
//    if (error != nil) {
//        LOGE("Failed cleaning up model: %s.\n", [error localizedDescription].UTF8String);
//        return TNN_NS::Status(TNN_NS::TNNERR_ANE_CLEAN_ERROR, "Error: Failed cleaning up model.");
//    }
//    [[NSFileManager defaultManager] removeItemAtPath:_compiledModelFilePath error:&error];
//    if (error != nil) {
//        LOGE("Failed cleaning up compiled model: %s.\n", [error localizedDescription].UTF8String);
//        return TNN_NS::Status(TNN_NS::TNNERR_ANE_CLEAN_ERROR, "Error: Failed cleaning up compiled model.");
//    }
    return TNN_NS::TNN_OK;
}

- (TNN_NS::Status)saveModel:(CoreML__Specification__Model*)model {
#ifdef DEBUG
    auto time_start = CFAbsoluteTimeGetCurrent();
#endif
    
    //mlmodel path
    auto mlmodelURL = [self mlmodelPath];
    
    if (model->specificationversion == 3) {
        _coreMlVersion = 2;
    } else if (model->specificationversion == 4) {
        _coreMlVersion = 3;
    } else {
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
    } else {
        LOGE("CoreML models file is empty.\n");
        return TNN_NS::Status(TNN_NS::TNNERR_ANE_SAVE_MODEL_ERROR, "Error: CoreML models file is empty.");
    }
    file_stream.close();
    
#ifdef DEBUG
    LOGD("MLModel save time: %f ms\n", (CFAbsoluteTimeGetCurrent() - time_start) * 1000.0);
#endif
    
    return TNN_NS::TNN_OK;
}

- (TNN_NS::Status)build:(NSURL*)mlmodelURL {
    auto time_start = CFAbsoluteTimeGetCurrent();
    
    if (@available(iOS 12.0, macOS 10.14, *)) {
        
        NSError* error = nil;
        NSURL* mlmodelcURL = [MLModel compileModelAtURL:mlmodelURL error:&error];
        if (error != nil) {
            LOGE("Error compiling model %s.\n", [error localizedDescription].UTF8String);
            return TNN_NS::Status(TNN_NS::TNNERR_ANE_COMPILE_MODEL_ERROR, "Error: Failed compiling model.");
        }
    
        MLModelConfiguration* config = [MLModelConfiguration alloc];
        config.computeUnits = MLComputeUnitsAll;
        _model = [MLModel modelWithContentsOfURL:mlmodelcURL configuration:config error:&error];
        
        if (error != nil) {
            LOGE("Error Creating MLModel %s.\n", [error localizedDescription].UTF8String);
            return TNN_NS::Status(TNN_NS::TNNERR_ANE_COMPILE_MODEL_ERROR, "Error: Failed Creating MLModel.");
        }
        
        NSLog(@"MLModel build time: %f ms", (CFAbsoluteTimeGetCurrent() - time_start) * 1000.0);
        return TNN_NS::TNN_OK;
    } else {
        LOGE("Error: CoreML only support iOS 12+.\n");
        return TNN_NS::Status(TNN_NS::TNNERR_ANE_COMPILE_MODEL_ERROR, "CoreML only support iOS 12+.");
    }
}

@end
