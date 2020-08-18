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
#import <Metal/Metal.h>

#import "tnn/utils/mat_converter.h"
#import "tnn/utils/mat_utils.h"
#import "tnn/utils/mat_converter_internal.h"
#import "tnn/device//metal/metal_context.h"
#import "tnn/device//metal/metal_command_queue.h"
#import "tnn/device//metal/acc/metal_common.h"
#import "tnn/core/abstract_device.h"
#import "tnn/utils/dims_vector_utils.h"


namespace TNN_NS {

class MetalMatConverterAcc : public MatConverterAcc {
public:
    virtual Status Copy(Mat& src, Mat& dst, void* command_queue = NULL);
    virtual Status Resize(Mat& src, Mat& dst, ResizeParam param, void* command_queue = NULL);
    virtual Status Crop(Mat& src, Mat& dst, CropParam param, void* command_queue = NULL);
    virtual Status WarpAffine(Mat& src, Mat& dst, WarpAffineParam param, void* command_queue = NULL);
    
protected:
    MetalResizeParams resize_param_;
    MetalCropParams crop_param_;
    MetalWarpAffineParams warpaffine_param_;
    MetalCopyParams copy_param_;
    
    id<MTLDevice> device_                           = nil;
    //metal params
    id<MTLBuffer> buffer_resize_param_              = nil;
    id<MTLBuffer> buffer_crop_param_                = nil;
    id<MTLBuffer> buffer_warpaffine_param_          = nil;
    id<MTLBuffer> buffer_copy_param_                = nil;
    
    id<MTLComputePipelineState> pipeline_process_   = nil;
    //Allocate metal kernel param
    Status AllocateBufferResizeParam(ResizeParam param, Mat& src, Mat& dst);
    Status AllocateBufferCropParam(CropParam param, Mat& src, Mat& dst);
    Status AllocateBufferWarpAffineParam(WarpAffineParam param, Mat& src, Mat& dst);
    Status AllocateBufferCopyParam(Mat& src, Mat& dst);
    //Find corresponding metal kernel
    Status AllocateResizeComputePipeline(ResizeParam param, Mat& src, Mat& dst, void *command_queue);
    Status AllocateCropComputePipeline(CropParam param, Mat& src, Mat& dst, void *command_queue);
    Status AllocateWarpAffineComputePipeline(WarpAffineParam param, Mat& src, Mat& dst, void *command_queue);
    Status AllocateCopyComputePipeline(Mat& src, Mat& dst, void *command_queue);
    
};

Status MetalMatConverterAcc::AllocateBufferResizeParam(ResizeParam param, Mat& src, Mat& dst) {
    LOGE("allocate buffer resize param\n");
    //MetalResizeParams metal_resize_param;
    resize_param_.batch    = src.GetBatch();
    resize_param_.width    = src.GetWidth();
    resize_param_.height   = src.GetHeight();
    resize_param_.size     = resize_param_.width * resize_param_.height;
    resize_param_.channel  = src.GetChannel();
    resize_param_.slice    = UP_DIV(resize_param_.channel, 4);
    //resize specific parameters
    resize_param_.scale_w  = param.scale_w;
    resize_param_.scale_h  = param.scale_h;
    //TODO: align with opencv, how to perform rounding
    resize_param_.resized_width  = static_cast<int>(resize_param_.scale_w * resize_param_.width);
    resize_param_.resized_height = static_cast<int>(resize_param_.scale_h * resize_param_.height);
    resize_param_.type           = int(param.type);
    
    buffer_resize_param_ = [device_ newBufferWithBytes:&resize_param_
                                                length:sizeof(MetalResizeParams)
                                               options:MTLResourceCPUCacheModeWriteCombined];
    
    if (!buffer_resize_param_) {
        return Status(TNNERR_INVALID_INPUT, "buffer resize param is nil!");
    }
    //resize_param_ = metal_resize_param;
    return TNN_OK;
}

Status MetalMatConverterAcc::AllocateBufferCropParam(CropParam param, Mat& src, Mat& dst) {
    LOGE("allocate buffer crop param\n");
    MetalCropParams metal_crop_param;
    metal_crop_param.batch          = src.GetBatch();
    metal_crop_param.width          = src.GetWidth();
    metal_crop_param.height         = src.GetHeight();
    metal_crop_param.size           = metal_crop_param.width * metal_crop_param.height;
    metal_crop_param.channel        = src.GetChannel();
    metal_crop_param.slice          = UP_DIV(metal_crop_param.channel, 4);
    //crop specific parameters
    metal_crop_param.crop_width     = param.width;
    metal_crop_param.crop_height    = param.height;
    metal_crop_param.top_left_x     = param.top_left_x;
    metal_crop_param.top_left_y     = param.top_left_y;
    
    buffer_crop_param_ = [device_ newBufferWithBytes:&metal_crop_param
                                              length:sizeof(MetalResizeParams)
                                             options:MTLResourceCPUCacheModeWriteCombined];
    
    if (!buffer_crop_param_) {
        return Status(TNNERR_INVALID_INPUT, "buffer crop param is nil!");
    }
    crop_param_ = metal_crop_param;
    return TNN_OK;
}

Status MetalMatConverterAcc::AllocateBufferWarpAffineParam(WarpAffineParam param, Mat& src, Mat& dst) {
    MetalWarpAffineParams metal_warpaffine_param;
    metal_warpaffine_param.batch    = src.GetBatch();
    metal_warpaffine_param.width    = src.GetWidth();
    metal_warpaffine_param.height   = src.GetHeight();
    metal_warpaffine_param.size     = metal_warpaffine_param.width * metal_warpaffine_param.height;
    metal_warpaffine_param.channel  = src.GetChannel();
    metal_warpaffine_param.slice    = UP_DIV(metal_warpaffine_param.channel, 4);
    
    buffer_warpaffine_param_ = [device_ newBufferWithBytes:&metal_warpaffine_param
                                                    length:sizeof(MetalWarpAffineParams)
                                                   options:MTLResourceCPUCacheModeWriteCombined];
    
    if (!buffer_warpaffine_param_) {
        return Status(TNNERR_INVALID_INPUT, "buffer warpaffine param is nil!");
    }
    return TNN_OK;
}

Status MetalMatConverterAcc::AllocateBufferCopyParam(Mat& src, Mat& dst) {
    LOGE("allocate buffer copy param\n");
    MetalCopyParams metal_copy_param;
    metal_copy_param.batch    = src.GetBatch();
    metal_copy_param.width    = src.GetWidth();
    metal_copy_param.height   = src.GetHeight();
    metal_copy_param.size     = metal_copy_param.width * metal_copy_param.height;
    metal_copy_param.channel  = src.GetChannel();
    metal_copy_param.slice    = UP_DIV(metal_copy_param.channel, 4);
    
    buffer_copy_param_ = [device_ newBufferWithBytes:&metal_copy_param
                                              length:sizeof(MetalCopyParams)
                                             options:MTLResourceCPUCacheModeWriteCombined];
    
    if (!buffer_copy_param_) {
        return Status(TNNERR_INVALID_INPUT, "buffer copy param is nil!");
    }
    copy_param_ = metal_copy_param;
    return TNN_OK;
}

Status MetalMatConverterAcc::AllocateCropComputePipeline(CropParam param, Mat& src, Mat& dst, void *command_queue) {
    LOGE("allocate crop computepipeline\n");
    auto command_queue_impl = (__bridge TNNMetalCommandQueueImpl *)(command_queue);
    if (!command_queue_impl) {
        return Status(TNNERR_INST_ERR, "command queue is nil");
    }
    
    auto library = command_queue_impl.metalContextImpl.library;
    if (!library) {
        return Status(TNNERR_INVALID_INPUT, "metal library is nil");
    }
    
    auto src_mat_type = src.GetMatType();
    auto dst_mat_type = dst.GetMatType();
    
    id<MTLFunction> func_process = nil;
    
    if (src_mat_type == dst_mat_type) {
        if (N8UC4 == src_mat_type) {
            // metal N8UC4 image crop kernel
            func_process = [library newFunctionWithName:@"mat_converter_texture_n8uc4_crop"];
        } else if(N8UC3 == src_mat_type) {
            // matal N8UC3 image crop kernel
            // metal_device only support memory allocation for N8UC4/NCHW_FLOAT.
            return Status(TNNERR_PARAM_ERR, "not support yet");
        } else if(NGRAY == src_mat_type) {
            // metal gray image crop kernel(N8UC1)
            // metal_device only support memory allocation for N8UC4/NCHW_FLOAT.
            return Status(TNNERR_PARAM_ERR, "not support yet");
        } else if(NNV21 == src_mat_type || NNV12 == src_mat_type) {
            // metal NNV image crop kernel
            // metal_device only support memory allocation for N8UC4/NCHW_FLOAT.
            return Status(TNNERR_PARAM_ERR, "not support yet");
        } else if(NCHW_FLOAT == src_mat_type) {
            // metak NCHW_FLOAT image crop kernel
            // arm converter doesnot support NCHW now.
            return Status(TNNERR_PARAM_ERR, "not support yet");
        } else {
            return Status(TNNERR_PARAM_ERR, "not support yet");
        }
    } else {
        return Status(TNNERR_PARAM_ERR, "not support yet");
    }
    if (!func_process) {
        return Status(TNNERR_INVALID_INPUT, "mat converter func not found");
    }
    auto pipeline_process = [device_ newComputePipelineStateWithFunction:func_process error:nil];
    if (!pipeline_process) {
        return Status(TNNERR_INVALID_INPUT, "mat converter pipeline is nil");
    }
    pipeline_process_ = pipeline_process;
    
    return TNN_OK;
}

Status MetalMatConverterAcc::AllocateResizeComputePipeline(ResizeParam param, Mat& src, Mat& dst, void *command_queue) {
    LOGE("allocate resize computepipeline\n");
    auto command_queue_impl = (__bridge TNNMetalCommandQueueImpl *)(command_queue);
    if (!command_queue_impl) {
        return Status(TNNERR_INST_ERR, "command queue is nil");
    }
    
    auto library = command_queue_impl.metalContextImpl.library;
    if (!library) {
        return Status(TNNERR_INVALID_INPUT, "metal library is nil");
    }
    
    auto src_mat_type = src.GetMatType();
    auto dst_mat_type = dst.GetMatType();
    
    id<MTLFunction> func_process = nil;
    
    if (src_mat_type == dst_mat_type) {
        if (N8UC4 == src_mat_type) {
            // metal N8UC4 image crop kernel
            if (0x00 == param.type) {
                //INTERP_TYPE_NEAREST
                func_process = [library newFunctionWithName:@"mat_converter_texture_n8uc4_resize_nearest"];
            } else if (0x01 == param.type) {
                //INTERP_TYPE_LINEAR
                func_process = [library newFunctionWithName:@"mat_converter_texture_n8uc4_resize_bilinear_gather"];
            }
        } else if (N8UC3 == src_mat_type) {
            // matal N8UC3 image crop kernel
            // metal_device only support memory allocation for N8UC4/NCHW_FLOAT.
            return Status(TNNERR_PARAM_ERR, "not support yet");
        } else if (NGRAY == src_mat_type) {
            // metal gray image crop kernel(N8UC1)
            // metal_device only support memory allocation for N8UC4/NCHW_FLOAT.
            return Status(TNNERR_PARAM_ERR, "not support yet");
        } else if (NNV21 == src_mat_type || NNV12 == src_mat_type) {
            // metal NNV image crop kernel
            // metal_device only support memory allocation for N8UC4/NCHW_FLOAT.
            return Status(TNNERR_PARAM_ERR, "not support yet");
        } else if (NCHW_FLOAT == src_mat_type) {
            // metak NCHW_FLOAT image crop kernel
            // arm converter doesnot support NCHW now.
            return Status(TNNERR_PARAM_ERR, "not support yet");
        } else {
            return Status(TNNERR_PARAM_ERR, "not support yet");
        }
    } else {
        return Status(TNNERR_PARAM_ERR, "not support yet");
    }
    if (!func_process) {
        return Status(TNNERR_INVALID_INPUT, "mat converter func not found");
    }
    auto pipeline_process = [device_ newComputePipelineStateWithFunction:func_process error:nil];
    if (!pipeline_process) {
        return Status(TNNERR_INVALID_INPUT, "mat converter pipeline is nil");
    }
    pipeline_process_ = pipeline_process;
    
    return TNN_OK;
}

Status MetalMatConverterAcc::AllocateCopyComputePipeline(Mat& src, Mat& dst, void *command_queue) {
    LOGE("allocate copy computepipeline\n");
    auto command_queue_impl = (__bridge TNNMetalCommandQueueImpl *)(command_queue);
    if (!command_queue_impl) {
        return Status(TNNERR_INST_ERR, "command queue is nil");
    }
    
    auto library = command_queue_impl.metalContextImpl.library;
    if (!library) {
        return Status(TNNERR_INVALID_INPUT, "metal library is nil");
    }
    
    auto src_mat_type = src.GetMatType();
    auto dst_mat_type = dst.GetMatType();
    
    id<MTLFunction> func_process = nil;
    
    if (src_mat_type == dst_mat_type) {
        if (N8UC4 == src_mat_type) {
            func_process = [library newFunctionWithName:@"copy_n8uc4_to_cpu"];
        } else if (NCHW_FLOAT == src_mat_type) {
            func_process = [library newFunctionWithName:@"copy_nchw_to_cpu"];
        } else if (N8UC3 == src_mat_type) {
            // matal N8UC3 image crop kernel
            // metal_device only support memory allocation for N8UC4/NCHW_FLOAT.
            return Status(TNNERR_PARAM_ERR, "not support yet");
        } else if (NGRAY == src_mat_type) {
            // metal gray image crop kernel(N8UC1)
            // metal_device only support memory allocation for N8UC4/NCHW_FLOAT.
            return Status(TNNERR_PARAM_ERR, "not support yet");
        } else if (NNV21 == src_mat_type || NNV12 == src_mat_type) {
            // metal NNV image crop kernel
            // metal_device only support memory allocation for N8UC4/NCHW_FLOAT.
            return Status(TNNERR_PARAM_ERR, "not support yet");
        } else {
            return Status(TNNERR_PARAM_ERR, "not support yet");
        }
    } else {
        return Status(TNNERR_PARAM_ERR, "not support yet");
    }
    if (!func_process) {
        return Status(TNNERR_INVALID_INPUT, "mat converter func not found");
    }
    auto pipeline_process = [device_ newComputePipelineStateWithFunction:func_process error:nil];
    if (!pipeline_process) {
        return Status(TNNERR_INVALID_INPUT, "mat converter pipeline is nil");
    }
    pipeline_process_ = pipeline_process;
    
    return TNN_OK;
}

Status MetalMatConverterAcc::Copy(Mat& src, Mat& dst, void* command_queue) {
    LOGE("start copy\n");
    auto src_device_type = src.GetDeviceType();
    auto dst_device_type = dst.GetDeviceType();
    auto src_mat_type    = src.GetMatType();
    auto dst_mat_type    = dst.GetMatType();
    if (src_device_type == dst_device_type) {
        return TNN_OK;
    }
    // dst must be on Metal
    if (dst_device_type != DEVICE_METAL && src_device_type!=DEVICE_METAL) {
        return Status(TNNERR_INVALID_INPUT, "both src and dst are not Metal Mat!");
    }
    // support NAIVE and ARM
    if (!(src_device_type == DEVICE_NAIVE || src_device_type == DEVICE_ARM || dst_device_type == DEVICE_NAIVE || dst_device_type == DEVICE_ARM)) {
        return Status(TNNERR_INVALID_INPUT, "unsupported device!");
    }
    // should we handle differnt data_type?
    if (src_mat_type != dst_mat_type) {
        return Status(TNNERR_INVALID_INPUT, "src and dst have different data type!");
    }
    // check if dims compatible
    if (!DimsVectorUtils::Equal(src.GetDims(), dst.GetDims())) {
        return Status(TNNERR_INVALID_INPUT, "src and dst have different dims!");
    }
    if (device_ == nil) {
        LOGE("device_ is nil, try to set device\n");
        if (src_mat_type == N8UC4) {
            id<MTLTexture> texture = nil;
            if(src_device_type == DEVICE_METAL)
                texture = (__bridge id<MTLTexture>)(src.GetData());
            else
                texture = (__bridge id<MTLTexture>)(dst.GetData());
            device_     = texture.device;
        } else if(src_mat_type == NCHW_FLOAT) {
            id<MTLBuffer> buffer = nil;
            if(src_device_type == DEVICE_METAL)
                buffer = (__bridge id<MTLBuffer>)(src.GetData());
            else
                buffer = (__bridge id<MTLBuffer>)(dst.GetData());
            device_     = buffer.device;
        }
    }
    if (src_mat_type != N8UC4 && src_mat_type != NCHW_FLOAT) {
        return Status(TNNERR_PARAM_ERR, "not support yet");
    }
    
    auto command_queue_impl = (__bridge TNNMetalCommandQueueImpl *)(command_queue);
    if (!command_queue_impl) {
        return Status(TNNERR_INST_ERR, "command queue is nil");
    }
    
    auto context_impl = command_queue_impl.metalContextImpl;
    
    auto status = AllocateBufferCopyParam(src, dst);
    if (status != TNN_OK) {
        return status;
    }
    
    status = AllocateCopyComputePipeline(src, dst, command_queue);
    if (status != TNN_OK) {
        return status;
    }
    //check copy direction
    auto dims = src.GetDims();
    auto count = DimsVectorUtils::Count(dims);
    auto bytesPerElement = -1;
    //auto mat_type = src.GetMatType();
    if (src_mat_type == N8UC4)
        bytesPerElement = 1;
    else if (src_mat_type == NCHW_FLOAT)
        bytesPerElement = 4;
    if (src_device_type == DEVICE_METAL) {
        // Metal => cpu
        // 1) metal => buffer
        id<MTLBuffer> tmp_buffer = [device_ newBufferWithLength: dims[2]*dims[3]*4
                                                        options:MTLResourceOptionCPUCacheModeDefault];
        if (tmp_buffer == nil) {
            return Status(TNNERR_INST_ERR, "tmp_buffer is nil");
        }
        do {
            auto slice = UP_DIV(dims[1], 4);
            MTLSize group_threads = {(NSUInteger)pipeline_process_.threadExecutionWidth, (NSUInteger)1, (NSUInteger)1};
            MTLSize groups = {(NSUInteger)((dims[3] + group_threads.width - 1) / group_threads.width), (NSUInteger)dims[2], 1};
            if(src_mat_type == NCHW_FLOAT) {
                groups = {(NSUInteger)((dims[3] + group_threads.width - 1) / group_threads.width), (NSUInteger)dims[2], (NSUInteger)dims[1]};
            }
            auto command_buffer = [command_queue_impl commandBuffer];
            [command_buffer enqueue];
            auto encoder = [command_buffer computeCommandEncoder];
            [encoder setComputePipelineState: pipeline_process_];
            if (src_mat_type == N8UC4) {
                id<MTLTexture> input_texture = (__bridge id<MTLTexture>)(src.GetData());
                
                [encoder setTexture:input_texture atIndex:0];
                [encoder setBuffer:tmp_buffer offset:0 atIndex:0];
                [encoder setBuffer:buffer_copy_param_ offset:0 atIndex:1];
            } else if(src_mat_type == NCHW_FLOAT) {
                id<MTLBuffer> input_buffer = (__bridge id<MTLBuffer>)(src.GetData());
                
                [encoder setBuffer:input_buffer offset:0 atIndex:0];
                [encoder setBuffer:tmp_buffer offset:0 atIndex:1];
                [encoder setBuffer:buffer_copy_param_ offset:0 atIndex:2];
            }
            [encoder dispatchThreadgroups:groups threadsPerThreadgroup:group_threads];
            [encoder endEncoding];
            
            [command_buffer commit];
            //wait to complete
            [command_buffer waitUntilCompleted];
        } while(0);
        // 2) buffer => dst
        memcpy(dst.GetData(), [tmp_buffer contents], tmp_buffer.length);
    } else {
        // cpu => Metal
        if (src_mat_type == N8UC4) {
            id<MTLTexture> texture = (__bridge id<MTLTexture>)(dst.GetData());
            if (!texture) {
                return Status(TNNERR_INST_ERR, "dst GetTexture return nil");
            }
            //This method does not synchronize against any GPU accesses to the texture
            [texture replaceRegion:MTLRegionMake2D(0, 0, dims[3], dims[2])
                       mipmapLevel:0
                         withBytes:src.GetData()
                       bytesPerRow:dims[3]*4];
        } else if(src_mat_type == NCHW_FLOAT) {
            // 1) cpu => buffer
            //TODO: will this cause memory leak?
            id<MTLBuffer> tmp_buffer = [device_ newBufferWithLength: count*bytesPerElement
                                                            options:MTLResourceOptionCPUCacheModeDefault];
            memcpy([tmp_buffer contents], src.GetData(), tmp_buffer.length);
            // 2) buffer => metal
            auto slice = UP_DIV(dims[1], 4);
            MTLSize group_threads = {(NSUInteger)pipeline_process_.threadExecutionWidth, (NSUInteger)1, (NSUInteger)1};
            MTLSize groups = {(NSUInteger)((dims[3] + group_threads.width - 1) / group_threads.width), (NSUInteger)dims[2], (NSUInteger)dims[1]};
            auto command_buffer = [command_queue_impl commandBuffer];
            [command_buffer enqueue];
            auto encoder = [command_buffer computeCommandEncoder];
            [encoder setComputePipelineState: pipeline_process_];
            id<MTLBuffer> dst_buffer = (__bridge id<MTLBuffer>)(dst.GetData());
            [encoder setBuffer:tmp_buffer offset:0 atIndex:0];
            [encoder setBuffer:dst_buffer offset:0 atIndex:1];
            [encoder setBuffer:buffer_copy_param_ offset:0 atIndex:2];
            
            [encoder dispatchThreadgroups:groups threadsPerThreadgroup:group_threads];
            [encoder endEncoding];
            
            [command_buffer commit];
            //wait to complete
            [command_buffer waitUntilCompleted];
            
        }
    }
    LOGE("complete copy\n");
    return TNN_OK;
}

Status MetalMatConverterAcc::Resize(Mat& src, Mat& dst, ResizeParam param, void* command_queue) {
    LOGE("start resize\n");
    auto src_mat_type = src.GetMatType();
    auto dst_mat_type = dst.GetMatType();
    if (dst_mat_type != src_mat_type) {
        return Status(TNNERR_PARAM_ERR, "not support yet");
    }
    //Get device
    if (device_ == nil) {
        LOGE("device_ is nil, try to set device\n");
        
        auto texture = (__bridge id<MTLTexture>)(src.GetData());
        device_     = texture.device;
    }
    if (src_mat_type != N8UC4 && src_mat_type != NCHW_FLOAT) {
        return Status(TNNERR_PARAM_ERR, "not support yet");
    }
    auto command_queue_impl = (__bridge TNNMetalCommandQueueImpl *)(command_queue);
    if (!command_queue_impl) {
        return Status(TNNERR_INST_ERR, "command queue is nil");
    }
    
    auto context_impl = command_queue_impl.metalContextImpl;
    
    auto status = AllocateBufferResizeParam(param, src, dst);
    if (status != TNN_OK) {
        return status;
    }
    
    status = AllocateResizeComputePipeline(param, src, dst, command_queue);
    if (status != TNN_OK) {
        return status;
    }
    do {
        MTLSize group_threads = {(NSUInteger)pipeline_process_.threadExecutionWidth, (NSUInteger)1, (NSUInteger)1};
        MTLSize groups = {(NSUInteger)((resize_param_.resized_width + group_threads.width - 1) / group_threads.width), (NSUInteger)resize_param_.resized_height, (NSUInteger)1};
        
        id<MTLTexture> input_texture = (__bridge id<MTLTexture>)(src.GetData());
        id<MTLTexture> output_texture = (__bridge id<MTLTexture>)(dst.GetData());
        
        auto command_buffer = [command_queue_impl commandBuffer];
        [command_buffer enqueue];
        auto encoder = [command_buffer computeCommandEncoder];
        [encoder setComputePipelineState: pipeline_process_];
        
        [encoder setTexture:input_texture atIndex:0];
        [encoder setTexture:output_texture atIndex:1];
        [encoder setBuffer:buffer_resize_param_ offset:0 atIndex:0];
        
        [encoder dispatchThreadgroups:groups threadsPerThreadgroup:group_threads];
        [encoder endEncoding];
        
        [command_buffer commit];
        //wait to complete
        [command_buffer waitUntilCompleted];
    } while(0);
    LOGE("complete resize\n");
    
    return TNN_OK;
}

Status MetalMatConverterAcc::Crop(Mat& src, Mat& dst, CropParam param, void* command_queue) {
    LOGE("start crop\n");
    auto src_mat_type = src.GetMatType();
    auto dst_mat_type = dst.GetMatType();
    if (dst_mat_type != src_mat_type) {
        return Status(TNNERR_PARAM_ERR, "not support yet");
    }
    //Get device
    if (device_ == nil) {
        LOGE("device_ is nil, try to set device\n");
        
        auto texture = (__bridge id<MTLTexture>)(src.GetData());
        device_     = texture.device;
    }
    if (src_mat_type != N8UC4 && src_mat_type != NCHW_FLOAT) {
        return Status(TNNERR_PARAM_ERR, "not support yet");
    }
    auto command_queue_impl = (__bridge TNNMetalCommandQueueImpl *)(command_queue);
    if (!command_queue_impl) {
        return Status(TNNERR_INST_ERR, "command queue is nil");
    }
    
    auto context_impl = command_queue_impl.metalContextImpl;
    
    auto status = AllocateBufferCropParam(param, src, dst);
    if (status != TNN_OK) {
        return status;
    }
    
    status = AllocateCropComputePipeline(param, src, dst, command_queue);
    if (status != TNN_OK) {
        return status;
    }
    do {
        MTLSize group_threads = {(NSUInteger)pipeline_process_.threadExecutionWidth, (NSUInteger)1, (NSUInteger)1};
        MTLSize groups = {(NSUInteger)((param.width + group_threads.width - 1) / group_threads.width), (NSUInteger)param.height, (NSUInteger)1};
        id<MTLTexture> input_texture = (__bridge id<MTLTexture>)(src.GetData());
        id<MTLTexture> output_texture = (__bridge id<MTLTexture>)(dst.GetData());
        
        auto command_buffer = [command_queue_impl commandBuffer];
        [command_buffer enqueue];
        auto encoder = [command_buffer computeCommandEncoder];
        [encoder setComputePipelineState: pipeline_process_];
        
        [encoder setTexture:input_texture atIndex:0];
        [encoder setTexture:output_texture atIndex:1];
        [encoder setBuffer:buffer_crop_param_ offset:0 atIndex:0];
        
        [encoder dispatchThreadgroups:groups threadsPerThreadgroup:group_threads];
        [encoder endEncoding];
        
        [command_buffer commit];
        //wait to complete
        [command_buffer waitUntilCompleted];
    } while(0);
    LOGE("complete crop\n");
    return TNN_OK;
}

Status MetalMatConverterAcc::WarpAffine(Mat& src, Mat& dst, WarpAffineParam param, void* command_queue) {
    return TNN_OK;
}

DECLARE_MAT_CONVERTER_CREATER(Metal);
REGISTER_MAT_CONVERTER(Metal, DEVICE_METAL);

}  // namespace TNN_NS
