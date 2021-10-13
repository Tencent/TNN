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

#include "tnn/core/profile.h"
#include "tnn/device/metal/metal_context.h"
#import "tnn/device//metal/metal_command_queue.h"
#import <Foundation/Foundation.h>
#import <QuartzCore/QuartzCore.h>
#if TNN_PROFILE
#define kMetalCommandBufferDepth 1
#else
#define kMetalCommandBufferDepth 10
#endif

static NSUInteger smallest_log2(NSUInteger integer) {
    if (integer == 0)
        return 0;
    NSUInteger power = 0;
    while ((integer & 0b1) == 0) {
        integer = integer >> 1;
        power++;
    }
    return power;
}

namespace TNN_NS {
MetalContext::MetalContext() {
    metal_context_impl_         = [TNNMMetalContextImpl new];
    metal_context_impl_.context = this;
}

MetalContext::~MetalContext() {
    metal_context_impl_ = nil;
}

Status MetalContext::GetCommandQueue(void **command_queue) {
    if (!metal_context_impl_) {
        return Status(TNNERR_DEVICE_LIBRARY_LOAD, "metal context is nil");
    }
    if (command_queue) {
        *command_queue = (__bridge void *)metal_context_impl_.commandQueue;
    }
    return TNN_OK;
}

Status MetalContext::ShareCommandQueue(Context* context) {
    if (!metal_context_impl_) {
        return Status(TNNERR_DEVICE_LIBRARY_LOAD, "metal context is nil");
    }
    auto context_target = dynamic_cast<MetalContext *>(context);
    if (!context_target) {
        return Status(TNNERR_DEVICE_LIBRARY_LOAD, "inpute context is not metal context");
    }
    
    metal_context_impl_ = context_target->getMetalContextImpl();
    
    return TNN_OK;
}

Status MetalContext::LoadLibrary(std::vector<std::string> path) {
    if (path.size() <= 0) {
        return Status(TNNERR_DEVICE_LIBRARY_LOAD, "library path is empty");
    }
    return [metal_context_impl_ loadLibrary:[NSString stringWithUTF8String:path[0].c_str()]];
}
Status MetalContext::OnInstanceForwardBegin() {
    Context::OnInstanceForwardBegin();
    return [metal_context_impl_ onInstanceForwardBegin];
}

Status MetalContext::OnInstanceForwardEnd() {
    return [metal_context_impl_ onInstanceForwardEnd];
}
TNNMMetalContextImpl *MetalContext::getMetalContextImpl() {
    return metal_context_impl_;
}
Status MetalContext::Synchronize() {
    if (metal_context_impl_) {
        [metal_context_impl_ waitUntilCompleted:nullptr];
        return TNN_OK;
    } else {
        return Status(TNNERR_INST_ERR, "metal context is nil");
    }
}
} // namespace TNN_NS

@implementation TNNMetalDeviceImpl
+ (id<MTLDevice>)sharedDevice {
    static id<MTLDevice> g_shared_device = nil;
    static dispatch_once_t onceToken;
    dispatch_once(&onceToken, ^{
        g_shared_device = MTLCreateSystemDefaultDevice();
    });
    return g_shared_device;
}
@end

@interface TNNMMetalContextImpl ()
@property(strong, nonatomic) id<MTLLibrary> library;
@property(strong, nonatomic) NSMutableDictionary<NSString *, id<MTLComputePipelineState>> *pipeLineCaches;
@property(strong, nonatomic) NSMutableArray<id<MTLCommandBuffer>> *waitingCommandBufferes;
@end

@implementation TNNMMetalContextImpl

- (instancetype)init {
    self = [super init];
    if (self) {
        if (@available(iOS 9.0, *)) {
            _device       = [TNNMetalDeviceImpl sharedDevice];
            _commandQueue = [[TNNMetalCommandQueueImpl alloc] initWithCommandQueue:[_device newCommandQueue]];
            _commandQueue.metalContextImpl = self;
            _commandBuffer = [_commandQueue commandBuffer];
            _pipeLineCaches                = [[NSMutableDictionary alloc] init];
            _waitingCommandBufferes        = [[NSMutableArray alloc] init];
            _commitCount                   = 0;
        } else {
            LOGE("Error: only support iOS 9.0+\n");
            self = nil;
            return nil;
        }
    }
    return self;
}

- (Status)onInstanceForwardBegin {
    _commitCount = 0;
    if (!_commandBuffer || _commandBuffer.status >= MTLCommandBufferStatusCommitted) {
        _commandBuffer = [_commandQueue commandBuffer];
        [_commandBuffer enqueue];
    }
    //    NSLog(@"onInstanceForwardBegin: %p", _commandBuffer);
    return TNN_OK;
}

- (Status)onInstanceForwardEnd {
    [self commit:YES];
    //    [self waitUntilCompleted];
    return TNN_OK;
}

- (Status)onCommandBufferScheduled:(id<MTLCommandBuffer>)commandBuffer {
    return TNN_OK;
}

- (Status)onCommandBufferCompletedFor:(id<MTLCommandBuffer>)commandBuffer {
    //    NSLog(@"ytfq final: %.6f ms", CACurrentMediaTime()*1000.0f);
    return TNN_OK;
}

- (id<MTLComputeCommandEncoder>)encoder {
    if (!_commandBuffer) {
        LOGE("Error: _commandBuffer in TNNMMetalContextImpl is nil\n");
        return nil;
    }
    auto result = [_commandBuffer computeCommandEncoder];
#if TNN_METAL_DEBUG || TNN_PROFILE
    result.label = nil;
#endif
    return result;
}

- (Status)loadLibrary:(NSString *)path {
    auto library = [_device newLibraryWithFile:path error:nil];
    if (!library) {
        return Status(TNNERR_DEVICE_LIBRARY_LOAD, "library load failed");
    }
    _library = library;
    return TNN_OK;
}

- (Status)load:(NSString *)name
       encoder:(id<MTLComputeCommandEncoder>)encoder
     bandwidth:(TNN_NS::MetalBandwidth &)bandwidth {
    id<MTLComputePipelineState> pipeline = [self pipelineWithName:name];
    if (!pipeline) {
        LOGE("Error: pipelineWithName nil: %s\n", name.UTF8String);
        return Status(TNNERR_INST_ERR, "Error: pipelineWithName return nil");
    }
    [encoder setComputePipelineState:pipeline];
#if TNN_METAL_DEBUG || TNN_PROFILE
    if (!name) {
    } else if (!encoder.label) {
        encoder.label = name;
    } else {
        NSArray *components = [encoder.label componentsSeparatedByString:@","];
        if (![components containsObject:name]) {
            components = [components arrayByAddingObject:name];
        }
        encoder.label = [components componentsJoinedByString:@","];
    }
#endif
    
    bandwidth.thread_execution_width = pipeline.threadExecutionWidth;
    bandwidth.max_threads_per_group = pipeline.maxTotalThreadsPerThreadgroup;
    bandwidth.z_axis_protected = NO;
    return TNN_OK;
}

- (id<MTLComputePipelineState>)pipelineWithName:(NSString *)name {
    if (!name)
        return nil;
    
    id<MTLComputePipelineState> result = _pipeLineCaches[name];
    if (result)
        return result;

    id<MTLFunction> function = [self functionWithName:name];
    if (!function)
        return nil;

    NSError *error = nil;
    result         = [_device newComputePipelineStateWithFunction:function error:&error];

    if (error) {
        LOGE("Error: create pipeline error: %s\n", error.localizedDescription.UTF8String);
    }

    if (result)
        _pipeLineCaches[name] = result;
    return result;
}

- (id<MTLFunction>)functionWithName:(NSString *)name {
    if (!name)
        return nil;
    id<MTLFunction> result = [_library newFunctionWithName:name];
#if TNN_METAL_DEBUG || TNN_PROFILE
    if (@available(iOS 10.0, *))
        result.label = name;
#endif
    return result;
}

- (Status)dispatchEncoder:(id<MTLComputeCommandEncoder>)encoder
                  threads:(MTLSize)threads
                bandwidth:(TNN_NS::MetalBandwidth)bandwidth {
    return [self dispatchEncoder:encoder
                         threads:threads
                 threadsPerGroup:[self threadsPerGroupWithThreads:threads bandwidth:bandwidth]
                       bandwidth:bandwidth];
}

- (Status)dispatchEncoder:(id<MTLComputeCommandEncoder>)encoder
                  threadsPerGroup:(MTLSize)threadsPerGroup
                  groups:(MTLSize)groups
                bandwidth:(TNN_NS::MetalBandwidth)bandwidth {
    MTLSize totalThreads = MTLSizeMake(
        threadsPerGroup.width  * groups.width,
        threadsPerGroup.height * groups.height,
        threadsPerGroup.depth  * groups.depth
    );
    return [self dispatchEncoder:encoder
                         threads:totalThreads
                 threadsPerGroup:threadsPerGroup
                       bandwidth:bandwidth];
}

- (Status)dispatchEncoder:(id<MTLComputeCommandEncoder>)encoder
                  threads:(MTLSize)threads
          threadsPerGroup:(MTLSize)threadsPerGroup
                bandwidth:(TNN_NS::MetalBandwidth)bandwidth {
    if (threads.width == 0 || threads.height == 0 || threads.depth == 0 || threadsPerGroup.width == 0 ||
        threadsPerGroup.height == 0 || threadsPerGroup.depth == 0) {
        LOGE("Error: dispatch error %td %td %td / %td %td %td\n", threads.width, threads.height, threads.depth,
             threadsPerGroup.width, threadsPerGroup.height, threadsPerGroup.depth);
        return Status(TNNERR_INST_ERR, "dispatch threads or threadsPerGroup is invalid");
    }
    threadsPerGroup.width  = MIN(threadsPerGroup.width, bandwidth.max_threads_per_group);
    threadsPerGroup.height = MIN(threadsPerGroup.height, bandwidth.max_threads_per_group);
    threadsPerGroup.depth  = MIN(threadsPerGroup.depth, bandwidth.max_threads_per_group);
    //#ifdef TNN_TARGET_IPHONE
    //    if (@available(iOS 11.0, *)) {
    //        if ([_device supportsFeatureSet:MTLFeatureSet_iOS_GPUFamily4_v1])
    //        {
    //            [encoder dispatchThreads:threads
    //            threadsPerThreadgroup:threadsPerGroup]; return;
    //        }
    //    }
    //#endif
    MTLSize groups = {
        static_cast<NSUInteger>(UP_DIV(threads.width, threadsPerGroup.width)),
        static_cast<NSUInteger>(UP_DIV(threads.height, threadsPerGroup.height)),
        static_cast<NSUInteger>(UP_DIV(threads.depth, threadsPerGroup.depth)),
    };
#if TNN_PROFILE
    LOGD("max_threads_per_group: %d\n", (int)bandwidth.max_threads_per_group);
    LOGD("groups:(%d %d %d)\n", (int)groups.width, (int)groups.height, (int)groups.depth);
    LOGD("threadsPerGroup:(%d %d %d)\n", (int)threadsPerGroup.width, (int)threadsPerGroup.height,
         (int)threadsPerGroup.depth);
#endif
    [encoder dispatchThreadgroups:groups threadsPerThreadgroup:threadsPerGroup];
    return TNN_OK;
}

- (MTLSize)threadsPerGroupWithThreads:(MTLSize)t bandwidth:(TNN_NS::MetalBandwidth)bw {
    auto pwarp = smallest_log2(bw.thread_execution_width);
    auto px = smallest_log2(t.width), sx = (NSUInteger)ceil(log2(t.width));
    auto py = smallest_log2(t.height), sy = (NSUInteger)ceil(log2(t.height));

    // accurately match on x
    if (px >= pwarp) {
        return {bw.thread_execution_width, 1, 1};
    }
    // accurately match on xy
    else if (px + py >= pwarp && sx < pwarp / 2) {
        NSUInteger x = pow(2, px);
        return {x, bw.thread_execution_width / x, 1};
    }
    // similarly match on x
    else if (sx >= pwarp) {
        return {bw.thread_execution_width, 1, 1};
    }
    // similarly match on xy
    else if (sx + sy >= pwarp) {
        NSUInteger x = pow(2, sx);
        return {x, bw.thread_execution_width / x, 1};
    }

    // on xyz (for most shaders do not protect gid.z, z axis must be accurately
    // match)
    auto pz = smallest_log2(t.depth);
    auto sz = bw.z_axis_protected ? ceil(log2(t.depth)) : pz;
    if (px + py + pz >= pwarp) {
        NSUInteger x = pow(2, px), y = pow(2, py);
        return {x, y, bw.thread_execution_width / x / y};
    } else if (sx + sy + sz >= pwarp) {
        NSUInteger x = pow(2, sx), z = pow(2, MIN(sz, pwarp - sx));
        return {x, bw.thread_execution_width / x / z, z};
    } else {
        NSUInteger z = pow(2, sz);
        return {t.width, t.height, z};
    }
}

- (void)commit {
#if TNN_METAL_DEBUG && TNN_METAL_BENCHMARK
    [self commit:YES];
#else
    [self commit:NO];
#endif
}

- (void)commit:(BOOL)force_commit {
    _commitCount++;
    if (!force_commit && _commitCount % kMetalCommandBufferDepth != 0) {
        return;
    }
    _commitCount = (!force_commit) ? _commitCount : 0;

    if (_commandBuffer.status < MTLCommandBufferStatusCommitted) {
        /*Note: addScheduledHandler or addCompletedHandler may cause crash*/
        /*
         @weakify(self);
        [_commandBuffer
            addScheduledHandler:^(id<MTLCommandBuffer> _Nonnull commandBuffer) {
                @strongify(self);
                [self onCommandBufferScheduled:commandBuffer];
            }];
        [_commandBuffer
            addCompletedHandler:^(id<MTLCommandBuffer> _Nonnull commandBuffer) {
                @strongify(self);
                [self onCommandBufferCompletedFor:commandBuffer];
            }];
         */
        [_commandBuffer commit];
        if (_commandBuffer) {
            _waitingCommandBufferes = [NSMutableArray arrayWithObject:_commandBuffer];
        }
        // create a new command buffer
        _commandBuffer = [_commandQueue commandBuffer];
        //for safety, dont enqueue
        //[_commandBuffer enqueue];
    }
}

- (void)waitUntilCompleted:(std::shared_ptr<ProfilingData>)profile_data {
    //    NSLog(@"waitUntilCompleted");
    NSArray *buffers        = _waitingCommandBufferes.copy;
    _waitingCommandBufferes = [NSMutableArray new];

    for (id<MTLCommandBuffer> buffer in buffers) {
        if (buffer.status >= MTLCommandBufferStatusCompleted)
            continue;

//        NSLog(@"waitUntilCompleted: %p", buffer);
#if TNN_METAL_DEBUG || TNN_PROFILE
        NSTimeInterval begin = [NSDate timeIntervalSinceReferenceDate];
        [buffer waitUntilCompleted];
        NSTimeInterval end = [NSDate timeIntervalSinceReferenceDate];
#if TNN_TARGET_IPHONE || TARGET_OS_OSX
        if (@available(iOS 10.3, macos 10.15, *)) {
            if (profile_data) {
                profile_data->kernel_time = (buffer.GPUEndTime - buffer.GPUStartTime) * 1000.f;
                LOGD("commit costs: %.3fms (kernel: %.3fms, GPU: %.3fms)\n", (end - begin) * 1000.f,
                     (buffer.kernelEndTime - buffer.kernelStartTime) * 1000.f,
                     (buffer.GPUEndTime - buffer.GPUStartTime) * 1000.f);
            }
        } else
#endif
        {
            if (profile_data) {
                profile_data->kernel_time = (end - begin) * 1000.f;
                LOGD("commit costs: %.3fms\n", (end - begin) * 1000.f);
            }
        }
#if TNN_PROFILE
        if (profile_data) {
            self.context->AddProfilingData(profile_data);
        }
#endif
#else
        [buffer waitUntilCompleted];
#endif
        if (buffer.error) {
            LOGE("Error: %s\n", buffer.error.localizedDescription.UTF8String);
        }
    }
}

#if TNN_METAL_DEBUG || TNN_PROFILE
- (void)printEncoder:(id<MTLCommandEncoder>)encoder {
    LOGD("Encoder: %s encoded.\n", encoder.label.UTF8String);
}
#endif
@end
