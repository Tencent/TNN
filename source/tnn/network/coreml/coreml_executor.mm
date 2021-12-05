#import <CoreML/CoreML.h>
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#import "coreml_executor.h"

#include <fstream>
#include <iostream>

namespace {

NSURL* createTemporaryFile() {
    NSURL* temporaryDirectoryURL = [NSURL fileURLWithPath:NSTemporaryDirectory() isDirectory:YES];
    NSString* temporaryFilename = [[NSProcessInfo processInfo] globallyUniqueString];
    NSURL* temporaryFileURL = [temporaryDirectoryURL URLByAppendingPathComponent:temporaryFilename];
    return temporaryFileURL;
    }
}

@implementation CoreMLExecutor

- (bool)cleanup {
    NSError* error = nil;
    [[NSFileManager defaultManager] removeItemAtPath:_mlModelFilePath error:&error];
    if (error != nil) {
        NSLog(@"Failed cleaning up model: %@", [error localizedDescription]);
        return NO;
    }
    [[NSFileManager defaultManager] removeItemAtPath:_compiledModelFilePath error:&error];
    if (error != nil) {
        NSLog(@"Failed cleaning up compiled model: %@", [error localizedDescription]);
        return NO;
    }
    return YES;
}

- (NSURL*)saveModel:(CoreML__Specification__Model*)model {
    NSURL* modelUrl = createTemporaryFile();
    NSString* modelPath = [modelUrl path];
    
    if (model->specificationversion == 3) {
        _coreMlVersion = 2;
    } else if (model->specificationversion == 4) {
        _coreMlVersion = 3;
    } else {
        NSLog(@"Only Core ML models with specification version 3 or 4 are supported");
        return nil;
    }
    size_t modelSize = core_ml__specification__model__get_packed_size(model);
    std::unique_ptr<uint8_t> writeBuffer(new uint8_t[modelSize]);
    core_ml__specification__model__pack(model, writeBuffer.get());
    // TODO: Can we mmap this instead of actual writing it to phone ?
    std::ofstream file_stream([modelPath UTF8String], std::ios::out | std::ios::binary);
    const char* ptr = reinterpret_cast<const char*>(writeBuffer.get());
    file_stream.write(ptr, modelSize);
    return modelUrl;
}

- (bool)build:(NSURL*)modelUrl {
    NSError* error = nil;
    NSURL* compileUrl = [MLModel compileModelAtURL:modelUrl error:&error];
    if (error != nil) {
        NSLog(@"Error compiling model %@", [error localizedDescription]);
        return NO;
    }
    _mlModelFilePath = [modelUrl path];
    _compiledModelFilePath = [compileUrl path];

    if (@available(iOS 12.0, *)) {
        MLModelConfiguration* config = [MLModelConfiguration alloc];
        config.computeUnits = MLComputeUnitsAll;
        _model = [MLModel modelWithContentsOfURL:compileUrl configuration:config error:&error];
    } else {
        _model = [MLModel modelWithContentsOfURL:compileUrl error:&error];
    }
    if (error != NULL) {
        NSLog(@"Error Creating MLModel %@", [error localizedDescription]);
        return NO;
    }
    return YES;
}

- (NSString*) getMLModelFilePath{
    return _compiledModelFilePath;
}

@end
