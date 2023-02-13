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

#import "tnn/device//metal/metal_command_queue.h"
#import <Foundation/Foundation.h>
#import <QuartzCore/QuartzCore.h>

@implementation TNNMetalCommandQueueImpl
- (instancetype)initWithCommandQueue:(id<MTLCommandQueue>)commandQueue {
    self = [super init];
    if (self) {
        mtl_command_queue_ = commandQueue;
    }
    return self;
}

- (id<MTLDevice>)device {
    return mtl_command_queue_.device;
}

- (NSString *)label {
    return mtl_command_queue_.label;
}

- (void)setLabel:(NSString *)label {
    mtl_command_queue_.label = label;
}

- (nullable id<MTLCommandBuffer>)commandBuffer {
    return [mtl_command_queue_ commandBuffer];
}

- (nullable id <MTLCommandBuffer>)commandBufferWithDescriptor:(MTLCommandBufferDescriptor*)descriptor API_AVAILABLE(macos(11.0), ios(14.0)) {
    return [mtl_command_queue_ commandBufferWithDescriptor:descriptor];
}

- (nullable id<MTLCommandBuffer>)commandBufferWithUnretainedReferences {
    return [mtl_command_queue_ commandBufferWithUnretainedReferences];
}

- (void)insertDebugCaptureBoundary {
    [mtl_command_queue_ insertDebugCaptureBoundary];
}
@end
