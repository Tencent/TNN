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
    
    ~MetalMatConverterAcc();
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

MetalMatConverterAcc::~MetalMatConverterAcc() {
    ;
}

Status MetalMatConverterAcc::AllocateBufferResizeParam(ResizeParam param, Mat& src, Mat& dst) {
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
    //resize_param_.resized_width  = static_cast<int>(resize_param_.scale_w * resize_param_.width);
    //resize_param_.resized_height = static_cast<int>(resize_param_.scale_h * resize_param_.height);
    // align with arm
    resize_param_.resized_width = dst.GetWidth();
    resize_param_.resized_height = dst.GetHeight();
    
    resize_param_.type           = int(param.type);
    
    buffer_resize_param_ = [device_ newBufferWithBytes:&resize_param_
                                                length:sizeof(MetalResizeParams)
                                               options:MTLResourceCPUCacheModeWriteCombined];
    
    if (!buffer_resize_param_) {
        return Status(TNNERR_INVALID_INPUT, "buffer resize param is nil!");
    }
    return TNN_OK;
}

Status MetalMatConverterAcc::AllocateBufferCropParam(CropParam param, Mat& src, Mat& dst) {
    crop_param_.batch          = src.GetBatch();
    crop_param_.width          = src.GetWidth();
    crop_param_.height         = src.GetHeight();
    crop_param_.size           = crop_param_.width * crop_param_.height;
    crop_param_.channel        = src.GetChannel();
    crop_param_.slice          = UP_DIV(crop_param_.channel, 4);
    //crop specific parameters
    crop_param_.crop_width     = param.width;
    crop_param_.crop_height    = param.height;
    crop_param_.top_left_x     = param.top_left_x;
    crop_param_.top_left_y     = param.top_left_y;
    
    buffer_crop_param_ = [device_ newBufferWithBytes:&crop_param_
                                              length:sizeof(MetalResizeParams)
                                             options:MTLResourceCPUCacheModeWriteCombined];
    
    if (!buffer_crop_param_) {
        return Status(TNNERR_INVALID_INPUT, "buffer crop param is nil!");
    }
    return TNN_OK;
}

Status MetalMatConverterAcc::AllocateBufferWarpAffineParam(WarpAffineParam param, Mat& src, Mat& dst) {
    warpaffine_param_.batch    = src.GetBatch();
    warpaffine_param_.width    = src.GetWidth();
    warpaffine_param_.height   = src.GetHeight();
    warpaffine_param_.size     = warpaffine_param_.width * warpaffine_param_.height;
    warpaffine_param_.channel  = src.GetChannel();
    warpaffine_param_.slice    = UP_DIV(warpaffine_param_.channel, 4);
    warpaffine_param_.resized_height = dst.GetHeight();
    warpaffine_param_.resized_width = dst.GetWidth();
    
    warpaffine_param_.interp_type = int(param.interp_type);
    warpaffine_param_.border_type = int(param.border_type);
    warpaffine_param_.border_val = param.border_val;
    
    buffer_warpaffine_param_ = [device_ newBufferWithBytes:&warpaffine_param_
                                                    length:sizeof(MetalWarpAffineParams)
                                                   options:MTLResourceCPUCacheModeWriteCombined];
    // compute the inverse transformation matrix
    /*
     double D   = M[0] * M[4] - M[1] * M[3];
     D          = D != 0 ? 1. / D : 0;
     double A11 = M[4] * D, A22 = M[0] * D;
     m[0]      = A11;
     m[1]      = M[1] * (-D);
     m[3]      = M[3] * (-D);
     m[4]      = A22;
     double b1 = -A11 * M[2] - m[1] * M[5];
     double b2 = -m[3] * M[2] - A22 * M[5];
     m[2]      = b1;
     m[5]      = b2;
     */
    float D   = param.transform[0][0] * param.transform[1][1] - param.transform[0][1] * param.transform[1][0];
    D          = D != 0 ? 1. / D : 0;
    float A11 = param.transform[1][1] * D, A22 = param.transform[0][0] * D;
    warpaffine_param_.transform_inv[0][0]      = A11;
    warpaffine_param_.transform_inv[0][1]      = param.transform[0][1] * (-D);
    warpaffine_param_.transform_inv[1][0]      = param.transform[1][0] * (-D);
    warpaffine_param_.transform_inv[1][1]      = A22;
    float b1 = -A11 * param.transform[0][2] - warpaffine_param_.transform_inv[0][1] * param.transform[1][2];
    float b2 = -warpaffine_param_.transform_inv[1][0] * param.transform[0][2] - A22 * param.transform[1][2];
    warpaffine_param_.transform_inv[0][2]      = b1;
    warpaffine_param_.transform_inv[1][2]      = b2;
    
    buffer_warpaffine_param_ = [device_ newBufferWithBytes:&warpaffine_param_
                                                    length:sizeof(MetalWarpAffineParams)
                                                   options:MTLResourceCPUCacheModeWriteCombined];
    
    if (!buffer_warpaffine_param_) {
        return Status(TNNERR_INVALID_INPUT, "buffer warpaffine param is nil!");
    }
    
    return TNN_OK;
}

Status MetalMatConverterAcc::AllocateBufferCopyParam(Mat& src, Mat& dst) {
    copy_param_.batch    = src.GetBatch();
    copy_param_.width    = src.GetWidth();
    copy_param_.height   = src.GetHeight();
    copy_param_.size     = copy_param_.width * copy_param_.height;
    copy_param_.channel  = src.GetChannel();
    copy_param_.slice    = UP_DIV(copy_param_.channel, 4);
    
    buffer_copy_param_ = [device_ newBufferWithBytes:&copy_param_
                                              length:sizeof(MetalCopyParams)
                                             options:MTLResourceCPUCacheModeWriteCombined];
    
    if (!buffer_copy_param_) {
        return Status(TNNERR_INVALID_INPUT, "buffer copy param is nil!");
    }
    return TNN_OK;
}

Status MetalMatConverterAcc::AllocateCropComputePipeline(CropParam param, Mat& src, Mat& dst, void *command_queue) {
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
                func_process = [library newFunctionWithName:@"mat_converter_texture_n8uc4_resize_linear"];
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
    } else if(src_mat_type == N8UC3 || dst_mat_type == N8UC3) {
        auto src_device_type = src.GetDeviceType();
        auto dst_device_type = dst.GetDeviceType();
        if (src_device_type == DEVICE_METAL) {
            // metal N8UC4 => arm N8UC3
            func_process = [library newFunctionWithName:@"copy_n8uc4_metal_to_n8uc3_cpu"];
            //LOGE("metal N8UC4 to arm N8UC3\n");
        } else {
            // arm N8UC3 => metal N8UC4
            func_process = [library newFunctionWithName:@"copy_n8uc3_cpu_to_n8uc4_metal"];
            //LOGE("arm N8UC3 to metal N8UC4\n");
        }
    }else {
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

Status  MetalMatConverterAcc::AllocateWarpAffineComputePipeline(WarpAffineParam param, Mat& src, Mat& dst, void *command_queue) {
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
    
    auto interp_type = param.interp_type;
    auto border_type = param.border_type;
    
    id<MTLFunction> func_process = nil;
    
    if (src_mat_type == dst_mat_type) {
        if (N8UC4 == src_mat_type) {
            // metal N8UC4 image crop kernel
            if (0x00 == interp_type) {
                //INTERP_TYPE_NEAREST
                return Status(TNNERR_PARAM_ERR, "not support yet");
            } else if (0x01 == interp_type && 0x00 == border_type) {
                //INTERP_TYPE_LINEAR and border type const
                func_process = [library newFunctionWithName:@"mat_converter_texture_n8uc4_warpaffine_linear_const"];
            } else {
                //INTERP_TYPE_LINEAR and other border type
                return Status(TNNERR_PARAM_ERR, "not support yet");
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

Status MetalMatConverterAcc::Copy(Mat& src, Mat& dst, void* command_queue) {
    auto src_device_type = src.GetDeviceType();
    auto dst_device_type = dst.GetDeviceType();
    auto src_mat_type    = src.GetMatType();
    auto dst_mat_type    = dst.GetMatType();
    if (src_device_type == dst_device_type) {
        return TNN_OK;
    }
    // src or dst must be on Metal
    if (dst_device_type != DEVICE_METAL && src_device_type!=DEVICE_METAL) {
        return Status(TNNERR_INVALID_INPUT, "both src and dst are not Metal Mat!");
    }
    // support NAIVE and ARM
    if (!(src_device_type == DEVICE_NAIVE || src_device_type == DEVICE_ARM || dst_device_type == DEVICE_NAIVE || dst_device_type == DEVICE_ARM)) {
        return Status(TNNERR_INVALID_INPUT, "unsupported device!");
    }
    // devan: support N8UC3 <=> N8UC4 copy
    if (src_mat_type != dst_mat_type && !(src_mat_type == N8UC3 && dst_mat_type == N8UC4) && !(src_mat_type == N8UC4 && dst_mat_type == N8UC3)) {
        return Status(TNNERR_INVALID_INPUT, "src and dst have different data type!");
    }
    // check if dims compatible
    if (! ((src_mat_type == dst_mat_type && DimsVectorUtils::Equal(src.GetDims(), dst.GetDims())) || ((src_mat_type == N8UC3 || dst_mat_type == N8UC3) && (src.GetHeight()==dst.GetHeight() && src.GetWidth()==dst.GetWidth()))) ) {
        return Status(TNNERR_INVALID_INPUT, "src and dst have different dims!");
    }
    if (device_ == nil) {
        if (src_device_type == DEVICE_METAL) {
            if(src_mat_type == N8UC4) {
                id<MTLTexture> texture = (__bridge id<MTLTexture>)(src.GetData());
                device_ = texture.device;
            } else {
                id<MTLBuffer> buffer = (__bridge id<MTLBuffer>)(src.GetData());
                device_     = buffer.device;
            }
        } else if(dst_device_type == DEVICE_METAL) {
            if(dst_mat_type == N8UC4) {
                id<MTLTexture> texture = (__bridge id<MTLTexture>)(dst.GetData());
                device_ = texture.device;
            } else {
                id<MTLBuffer> buffer = (__bridge id<MTLBuffer>)(dst.GetData());
                device_     = buffer.device;
            }
        }
    }
    /*
    if (device_ == nil) {
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
     */
    if (src_mat_type != N8UC4 && src_mat_type != NCHW_FLOAT && src_mat_type != N8UC3) {
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
    
    if (src_device_type == DEVICE_METAL) {
        // Metal => cpu
        // 1) metal => buffer
        id<MTLBuffer> tmp_buffer = nil;
        if (dst_mat_type == N8UC3) {
            tmp_buffer = [device_ newBufferWithLength: dims[2]*dims[3]*3
                                              options:MTLResourceOptionCPUCacheModeDefault];
        } else if(dst_mat_type == N8UC4){
            // N8UC4
            tmp_buffer = [device_ newBufferWithLength: dims[2]*dims[3]*4
                                              options:MTLResourceOptionCPUCacheModeDefault];
        } else {
            // NCHW_FLOAT
            auto count = DimsVectorUtils::Count(dims);
            tmp_buffer = [device_ newBufferWithLength: count*4
                                              options:MTLResourceOptionCPUCacheModeDefault];
        }
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
            if (dst_mat_type == N8UC4) {
                id<MTLTexture> input_texture = (__bridge id<MTLTexture>)(src.GetData());
                
                [encoder setTexture:input_texture atIndex:0];
                [encoder setBuffer:tmp_buffer offset:0 atIndex:0];
                [encoder setBuffer:buffer_copy_param_ offset:0 atIndex:1];
            } else if(dst_mat_type == NCHW_FLOAT) {
                id<MTLBuffer> input_buffer = (__bridge id<MTLBuffer>)(src.GetData());
                
                [encoder setBuffer:input_buffer offset:0 atIndex:0];
                [encoder setBuffer:tmp_buffer offset:0 atIndex:1];
                [encoder setBuffer:buffer_copy_param_ offset:0 atIndex:2];
            } else if(dst_mat_type == N8UC3) {
                id<MTLTexture> input_texture = (__bridge id<MTLTexture>)(src.GetData());
                
                [encoder setTexture:input_texture atIndex:0];
                [encoder setBuffer:tmp_buffer offset:0 atIndex:0];
                [encoder setBuffer:buffer_copy_param_ offset:0 atIndex:1];
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
            auto count = DimsVectorUtils::Count(dims);
            id<MTLBuffer> tmp_buffer = [device_ newBufferWithLength: count*4
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
        } else if(src_mat_type == N8UC3) {
            // 1) cpu => buffer
            //TODO: will this cause memory leak?
            id<MTLBuffer> tmp_buffer = [device_ newBufferWithLength: dims[2]*dims[3]*3
                                                            options:MTLResourceOptionCPUCacheModeDefault];
            memcpy([tmp_buffer contents], src.GetData(), tmp_buffer.length);
            // 2) buffer => metal
            auto slice = UP_DIV(dims[1], 4);
            MTLSize group_threads = {(NSUInteger)pipeline_process_.threadExecutionWidth, (NSUInteger)1, (NSUInteger)1};
            MTLSize groups = {(NSUInteger)((dims[3] + group_threads.width - 1) / group_threads.width), (NSUInteger)dims[2], (NSUInteger)1};
            auto command_buffer = [command_queue_impl commandBuffer];
            [command_buffer enqueue];
            auto encoder = [command_buffer computeCommandEncoder];
            [encoder setComputePipelineState: pipeline_process_];
            
            id<MTLTexture> dst_texture = (__bridge id<MTLTexture>)(dst.GetData());
            
            [encoder setBuffer:tmp_buffer offset:0 atIndex:0];
            [encoder setTexture:dst_texture atIndex:0];
            [encoder setBuffer:buffer_copy_param_ offset:0 atIndex:1];
            
            [encoder dispatchThreadgroups:groups threadsPerThreadgroup:group_threads];
            [encoder endEncoding];
            
            [command_buffer commit];
            //wait to complete
            [command_buffer waitUntilCompleted];
        }
    }
    return TNN_OK;
}

Status MetalMatConverterAcc::Resize(Mat& src, Mat& dst, ResizeParam param, void* command_queue) {
    auto src_device_type = src.GetDeviceType();
    auto dst_device_type = dst.GetDeviceType();
    auto src_mat_type = src.GetMatType();
    auto dst_mat_type = dst.GetMatType();
    if (dst_mat_type != src_mat_type) {
        return Status(TNNERR_PARAM_ERR, "not support yet");
    }
    if (src_mat_type != N8UC4 && src_mat_type != NCHW_FLOAT) {
        return Status(TNNERR_PARAM_ERR, "not support yet");
    }
    //Get device
    if (device_ == nil) {
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
#ifdef DUMP_BILINEAR_COOR
    id<MTLBuffer> tmp_buffer = [device_ newBufferWithLength: resize_param_.resized_width*resize_param_.resized_height*2*sizeof(int)
                                                    options:MTLResourceOptionCPUCacheModeDefault];
#endif
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
#ifdef DUMP_BILINEAR_COOR
        //prepare a buffer for sample coordinate
        [encoder setBuffer:tmp_buffer offset:0 atIndex:1];
#endif
        
        [encoder dispatchThreadgroups:groups threadsPerThreadgroup:group_threads];
        [encoder endEncoding];
        
        [command_buffer commit];
        //wait to complete
        [command_buffer waitUntilCompleted];
    } while(0);
#ifdef DUMP_BILINEAR_COOR
    metal_coords.reset(new int[resize_param_.resized_width*resize_param_.resized_height*2]);
    memcpy(metal_coords.get(), [tmp_buffer contents], sizeof(int)*resize_param_.resized_width*resize_param_.resized_height*2);
    int offset = 0;
    int* metal_coord_ptr = metal_coords.get();
    for(int h=0; h<resize_param_.resized_height; ++h){
        for(int w=0; w<resize_param_.resized_width; ++w){
            printf("(%d,%d):(%d,%d)\n", w, h, metal_coord_ptr[offset], metal_coord_ptr[offset+1]);
            offset += 2;
        }
    }
#endif
    return TNN_OK;
}

Status MetalMatConverterAcc::Crop(Mat& src, Mat& dst, CropParam param, void* command_queue) {
    auto src_device_type = src.GetDeviceType();
    auto dst_device_type = dst.GetDeviceType();
    auto src_mat_type = src.GetMatType();
    auto dst_mat_type = dst.GetMatType();
    if (dst_mat_type != src_mat_type) {
        return Status(TNNERR_PARAM_ERR, "not support yet");
    }
    if (src_mat_type != N8UC4 && src_mat_type != NCHW_FLOAT) {
        return Status(TNNERR_PARAM_ERR, "not support yet");
    }
    //Get device
    if (device_ == nil) {
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
    return TNN_OK;
}

Status MetalMatConverterAcc::WarpAffine(Mat& src, Mat& dst, WarpAffineParam param, void* command_queue) {
    auto src_device_type = src.GetDeviceType();
    auto dst_device_type = dst.GetDeviceType();
    auto src_mat_type = src.GetMatType();
    auto dst_mat_type = dst.GetMatType();
    if (dst_mat_type != src_mat_type) {
        return Status(TNNERR_PARAM_ERR, "not support yet");
    }
    if (src_mat_type != N8UC4 && src_mat_type != NCHW_FLOAT) {
        return Status(TNNERR_PARAM_ERR, "not support yet");
    }
    //Get device
    if (device_ == nil) {
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
    auto command_queue_impl = (__bridge TNNMetalCommandQueueImpl *)(command_queue);
    if (!command_queue_impl) {
        return Status(TNNERR_INST_ERR, "command queue is nil");
    }
    
    auto context_impl = command_queue_impl.metalContextImpl;
    
    auto status = AllocateBufferWarpAffineParam(param, src, dst);
    if (status != TNN_OK) {
        return status;
    }
    
    status = AllocateWarpAffineComputePipeline(param, src, dst, command_queue);
    if (status != TNN_OK) {
        return status;
    }
    
    do {
        MTLSize group_threads = {(NSUInteger)pipeline_process_.threadExecutionWidth, (NSUInteger)1, (NSUInteger)1};
        MTLSize groups = {(NSUInteger)((warpaffine_param_.resized_width + group_threads.width - 1) / group_threads.width), (NSUInteger)warpaffine_param_.resized_height, (NSUInteger)1};
        
        id<MTLTexture> input_texture = (__bridge id<MTLTexture>)(src.GetData());
        id<MTLTexture> output_texture = (__bridge id<MTLTexture>)(dst.GetData());
        
        auto command_buffer = [command_queue_impl commandBuffer];
        [command_buffer enqueue];
        auto encoder = [command_buffer computeCommandEncoder];
        [encoder setComputePipelineState: pipeline_process_];
        
        [encoder setTexture:input_texture atIndex:0];
        [encoder setTexture:output_texture atIndex:1];
        [encoder setBuffer:buffer_warpaffine_param_ offset:0 atIndex:0];
        
        [encoder dispatchThreadgroups:groups threadsPerThreadgroup:group_threads];
        [encoder endEncoding];
        
        [command_buffer commit];
        //wait to complete
        [command_buffer waitUntilCompleted];
    } while(0);
    
    return TNN_OK;
}

DECLARE_MAT_CONVERTER_CREATER(Metal);
REGISTER_MAT_CONVERTER(Metal, DEVICE_METAL);

}  // namespace TNN_NS
