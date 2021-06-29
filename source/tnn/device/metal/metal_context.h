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

#ifndef TNN_SOURCE_TNN_DEVICE_METAL_METAL_CONTEXT_H_
#define TNN_SOURCE_TNN_DEVICE_METAL_METAL_CONTEXT_H_

#include <string>

#include "tnn/core/context.h"
#include "tnn/core/profile.h"
#include "tnn/device/metal/metal_macro.h"

using namespace std;
using namespace TNN_NS;

#if TNN_PROFILE
#define TNN_PRINT_ENCODER(context, encoder, layer_acc)                                                                 \
    if (context_->profile_layer) {                                                                                     \
        auto pdata       = std::make_shared<ProfilingData>();                                                          \
        auto dims_input  = inputs[0]->GetBlobDesc().dims;                                                              \
        auto dims_output = outputs[0]->GetBlobDesc().dims;                                                             \
        layer_acc->UpdateProfilingData(pdata.get(), layer_acc->param_, dims_input, dims_output);                       \
        auto context_metal = context_->getMetalContextImpl();                                                          \
        [context_metal printEncoder:encoder];                                                                          \
        [context_metal waitUntilCompleted:pdata];                                                                      \
    }
#else
#define TNN_PRINT_ENCODER(context, encoder, layer_acc) ((void)0)
#endif

TNN_OBJC_CLASS(TNNMMetalContextImpl);
TNN_OBJC_CLASS(TNNMetalCommandQueueImpl);

namespace TNN_NS {

typedef struct {
    /** wrap size */
    NSUInteger thread_execution_width;
    /** max threads per thread group */
    NSUInteger max_threads_per_group;
    /** run conbcurrently on z axis or not */
    BOOL z_axis_protected;
} MetalBandwidth;

class MetalContext : public Context {
public:
    MetalContext();
    virtual ~MetalContext();

    // @brief get tnn command queue
    // @param command_queue device command queue for forward
    Status GetCommandQueue(void **command_queue);
    // @brief share tnn command queue to another context
    Status ShareCommandQueue(Context* context);

    virtual Status LoadLibrary(std::vector<std::string> path);
    virtual Status OnInstanceForwardBegin();
    virtual Status OnInstanceForwardEnd();
    virtual Status Synchronize();
    TNNMMetalContextImpl *getMetalContextImpl();

private:
    __strong TNNMMetalContextImpl *metal_context_impl_;
};

} // namespace TNN_NS

@interface TNNMetalDeviceImpl : NSObject
+ (id<MTLDevice>)sharedDevice;
@end

@interface TNNMMetalContextImpl : NSObject
@property(assign, nonatomic) NSUInteger commitCount;
@property(assign, nonatomic) MetalContext *context;
/** metal device */
@property(strong, nonatomic, readonly) id<MTLDevice> device;
@property(strong, nonatomic, readonly) id<MTLLibrary> library;
@property(strong, nonatomic) TNNMetalCommandQueueImpl *commandQueue;
@property(strong, nonatomic) id<MTLCommandBuffer> commandBuffer;

/**
 * @brief load metal library
 * @param path      library full path
 */
- (Status)loadLibrary:(NSString *)path;

/**
 * @brief load encoder with function name. returns maxTotalThreadsPerThreadgroup
 * of pipeline.
 * @param name      pipline name
 * @param encoder   command encoder
 * @return bandwidth info for function
 */
- (Status)load:(NSString *)name
    encoder:(id<MTLComputeCommandEncoder>)encoder
    bandwidth:(TNN_NS::MetalBandwidth &)bandwidth;

/**
 * @brief create compute encoder on default command buffer
 * @return created encoder
 */
- (id<MTLComputeCommandEncoder>)encoder;

/**
 * @brief dispatch encoder with default settings
 * @param encoder   command encoder
 * @param threads   threads size
 * @param bandwidth bandwidth
 */
- (Status)dispatchEncoder:(id<MTLComputeCommandEncoder>)encoder
                threads:(MTLSize)threads
                bandwidth:(TNN_NS::MetalBandwidth)bandwidth;

/**
 * @brief dispatch encoder with specified settings
 * @param encoder           command encoder
 * @param threadsPerGroup   threadsPerGroup size
 * @param groups            threadGroups
 * @param bandwidth         bandwidth
 */
- (Status)dispatchEncoder:(id<MTLComputeCommandEncoder>)encoder
                  threadsPerGroup:(MTLSize)threadsPerGroup
                  groups:(MTLSize)groups
                  bandwidth:(TNN_NS::MetalBandwidth)bandwidth;

/**
 * @brief before instance forward
 */
- (Status)onInstanceForwardBegin;
/**
 * @brief after instance forward
 */
- (Status)onInstanceForwardEnd;

/**
 * @brief commit commands
 */
- (void)commit;
/**
 * @brief commit commands
 */
- (void)commit:(BOOL)force_commit;

/**
 * @brief wait for completion
 */
- (void)waitUntilCompleted:(std::shared_ptr<ProfilingData>)pdata;

#if TNN_METAL_DEBUG || TNN_PROFILE
/**
 * @brief print encoder
 */
- (void)printEncoder:(id<MTLCommandEncoder>)encoder;
#endif
@end
#endif // TNN_SOURCE_TNN_DEVICE_METAL_METAL_CONTEXT_H_
