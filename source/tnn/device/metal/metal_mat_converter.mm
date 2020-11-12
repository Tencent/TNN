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

#import "tnn/utils/mat_utils.h"
#import "tnn/utils/mat_converter_acc.h"
#import "tnn/device//metal/metal_context.h"
#import "tnn/device//metal/metal_command_queue.h"
#import "tnn/device//metal/acc/metal_common.h"
#import "tnn/core/abstract_device.h"
#import "tnn/utils/dims_vector_utils.h"

#define ENABLE_PIPELINE_CACHE 1
#define KERNEL_SYNC 0

namespace TNN_NS {

class MetalMatConverterAcc : public MatConverterAcc {
public:
    virtual Status Copy(Mat& src, Mat& dst, void* command_queue = NULL);
    virtual Status Resize(Mat& src, Mat& dst, ResizeParam param, void* command_queue = NULL);
    virtual Status Crop(Mat& src, Mat& dst, CropParam param, void* command_queue = NULL);
    virtual Status WarpAffine(Mat& src, Mat& dst, WarpAffineParam param, void* command_queue = NULL);
    virtual Status CvtColor(Mat& src, Mat& dst, ColorConversionType type, void* command_queue = NULL);
    virtual Status CopyMakeBorder(Mat& src, Mat& dst, CopyMakeBorderParam param, void* command_queue = NULL);

    ~MetalMatConverterAcc() {};
protected:
    MetalResizeParams resize_param_;
    MetalCropParams crop_param_;
    MetalWarpAffineParams warpaffine_param_;
    MetalCopyParams copy_param_;
    MetalBGR2GrayParams bgr2gray_param_;
    MetalCopyMakeBorderParam copy_make_border_param_;
    
    id<MTLDevice> device_                           = nil;
    //metal params
    id<MTLBuffer> buffer_resize_param_              = nil;
    id<MTLBuffer> buffer_crop_param_                = nil;
    id<MTLBuffer> buffer_warpaffine_param_          = nil;
    id<MTLBuffer> buffer_copy_param_                = nil;
    id<MTLBuffer> buffer_bgr2gray_param_            = nil;
    id<MTLBuffer> buffer_copymakeborder_param_      = nil;
    
    id<MTLComputePipelineState> pipeline_process_   = nil;
    //Allocate metal kernel param
    Status AllocateBufferResizeParam(ResizeParam param, Mat& src, Mat& dst);
    Status AllocateBufferCropParam(CropParam param, Mat& src, Mat& dst);
    Status AllocateBufferWarpAffineParam(WarpAffineParam param, Mat& src, Mat& dst);
    Status AllocateBufferCopyParam(Mat& src, Mat& dst);
    Status AllocateBufferBGR2GrayParam(Mat& src, Mat& dst);
    Status AllocateBufferCopyMakeBorderParam(CopyMakeBorderParam param, Mat& src, Mat& dst);
    //Find corresponding metal kernel
    Status AllocateResizeComputePipeline(ResizeParam param, Mat& src, Mat& dst, void *command_queue);
    Status AllocateCropComputePipeline(CropParam param, Mat& src, Mat& dst, void *command_queue);
    Status AllocateWarpAffineComputePipeline(WarpAffineParam param, Mat& src, Mat& dst, void *command_queue);
    Status AllocateCopyComputePipeline(Mat& src, Mat& dst, void *command_queue);
    Status AllocateBGR2GrayComputePipeline(Mat& src, Mat& dst, void *command_queue);
    Status AllocateCopyMakeBorderComputePipeline(CopyMakeBorderParam param, Mat& src, Mat& dst, void *command_queue);

    Status BGR2Gray(Mat& src, Mat& dst, void* command_queue = NULL);
    Status CopyInputCheck(Mat& src, Mat& dst,
                          const DeviceType& src_device_type, const DeviceType& dst_device_type,
                          const MatType& src_mat_type, const MatType& dst_mat_type);
    bool DeviceTypeCheck(const DeviceType& src_device_type,
                         const DeviceType& dst_device_type);
    Status SetDevice(Mat& src, Mat& dst,
                     const DeviceType& src_device_type, const DeviceType& dst_device_type,
                     const MatType& src_mat_type, const MatType& dst_mat_type);
    Status MetalCopyToCPU(Mat& src, Mat& dst,
                          const DimsVector& dims,
                          const MatType& src_mat_type, const MatType& dst_mat_type,
                          TNNMetalCommandQueueImpl *command_queue_impl);
    Status CPUCopyToMetal(Mat& src, Mat& dst,
                          const DimsVector& dims,
                          const MatType& src_mat_type, const MatType& dst_mat_type,
                          TNNMetalCommandQueueImpl *command_queue_impl);
};

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
                                              length:sizeof(MetalCropParams)
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
    
    // compute the inverse transformation matrix
    float d   = param.transform[0][0] * param.transform[1][1] - param.transform[0][1] * param.transform[1][0];
    d          = d != 0 ? 1. / d : 0;

    float a11 = param.transform[1][1] * d, a22 = param.transform[0][0] * d;
    warpaffine_param_.transform_inv[0][0]      = a11;
    warpaffine_param_.transform_inv[0][1]      = param.transform[0][1] * (-d);
    warpaffine_param_.transform_inv[1][0]      = param.transform[1][0] * (-d);
    warpaffine_param_.transform_inv[1][1]      = a22;

    float b1 = -a11 * param.transform[0][2] - warpaffine_param_.transform_inv[0][1] * param.transform[1][2];
    float b2 = -warpaffine_param_.transform_inv[1][0] * param.transform[0][2] - a22 * param.transform[1][2];
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

Status MetalMatConverterAcc::AllocateBufferBGR2GrayParam(Mat& src, Mat& dst) {
    bgr2gray_param_.batch    = src.GetBatch();
    bgr2gray_param_.width    = src.GetWidth();
    bgr2gray_param_.height   = src.GetHeight();
    bgr2gray_param_.size     = bgr2gray_param_.width * bgr2gray_param_.height;
    bgr2gray_param_.channel  = src.GetChannel();
    bgr2gray_param_.slice    = UP_DIV(bgr2gray_param_.channel, 4);
    
    buffer_bgr2gray_param_ = [device_ newBufferWithBytes:&bgr2gray_param_
                                              length:sizeof(MetalBGR2GrayParams)
                                             options:MTLResourceCPUCacheModeWriteCombined];
    
    if (!buffer_bgr2gray_param_) {
        return Status(TNNERR_INVALID_INPUT, "buffer bgr2gray param is nil!");
    }
    return TNN_OK;
}

Status MetalMatConverterAcc::AllocateBufferCopyMakeBorderParam(CopyMakeBorderParam param, Mat &src, Mat &dst) {
    copy_make_border_param_.batch = src.GetBatch();
    copy_make_border_param_.channel = src.GetChannel();
    copy_make_border_param_.height = src.GetHeight();
    copy_make_border_param_.width = src.GetWidth();
    copy_make_border_param_.top = param.top;
    copy_make_border_param_.bottom = param.bottom;
    copy_make_border_param_.left = param.left;
    copy_make_border_param_.right = param.right;

    copy_make_border_param_.border_type = int(param.border_type);
    copy_make_border_param_.border_val = param.border_val;

    buffer_copymakeborder_param_= [device_ newBufferWithBytes:&copy_make_border_param_
                                                    length:sizeof(MetalCopyMakeBorderParam)
                                                      options:MTLResourceCPUCacheModeWriteCombined];

    if (!buffer_copymakeborder_param_) {
        return Status(TNNERR_INVALID_INPUT, "buffer copymakeborder param is nil!");
    }
    return TNN_OK;
}

Status MetalMatConverterAcc::AllocateCropComputePipeline(CropParam param, Mat& src, Mat& dst, void *command_queue) {
#if ENABLE_PIPELINE_CACHE
    static std::map<std::string,  id <MTLComputePipelineState> > library_cache;
#endif
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

    std::string kernel_name("");
    if (src_mat_type == dst_mat_type) {
        if (N8UC4 == src_mat_type) {
#if ENABLE_PIPELINE_CACHE
            kernel_name = std::string("mat_converter_texture_n8uc4_crop");
            if (library_cache.count(kernel_name) > 0){
                pipeline_process_ = library_cache[kernel_name];
                return TNN_OK;
            }
#endif
            // metal N8UC4 image crop kernel
            func_process = [library newFunctionWithName:@"mat_converter_texture_n8uc4_crop"];
        } else if(N8UC3 == src_mat_type) {
            return Status(TNNERR_PARAM_ERR, "mat type not support yet");
        } else if(NGRAY == src_mat_type) {
            return Status(TNNERR_PARAM_ERR, "mat type not support yet");
        } else if(NNV21 == src_mat_type || NNV12 == src_mat_type) {
            return Status(TNNERR_PARAM_ERR, "mat type not support yet");
        } else if(NCHW_FLOAT == src_mat_type) {
            return Status(TNNERR_PARAM_ERR, "mat type not support yet");
        } else {
            return Status(TNNERR_PARAM_ERR, "mat type not support yet");
        }
    } else {
        return Status(TNNERR_PARAM_ERR, "src and dst mat type must be same");
    }
    if (!func_process) {
        return Status(TNNERR_INVALID_INPUT, "mat converter func not found");
    }
    auto pipeline_process = [device_ newComputePipelineStateWithFunction:func_process error:nil];
    if (!pipeline_process) {
        return Status(TNNERR_INVALID_INPUT, "crop pipeline is nil");
    }
    pipeline_process_ = pipeline_process;
    
#if ENABLE_PIPELINE_CACHE
    library_cache[kernel_name] = pipeline_process;
#endif
    return TNN_OK;
}

Status MetalMatConverterAcc::AllocateResizeComputePipeline(ResizeParam param, Mat& src, Mat& dst, void *command_queue) {
    auto src_mat_type = src.GetMatType();
    auto dst_mat_type = dst.GetMatType();

#if ENABLE_PIPELINE_CACHE
    static std::map<std::string,  id <MTLComputePipelineState> > library_cache;
#endif
    auto command_queue_impl = (__bridge TNNMetalCommandQueueImpl *)(command_queue);
    if (!command_queue_impl) {
        return Status(TNNERR_INST_ERR, "command queue is nil");
    }
    
    auto library = command_queue_impl.metalContextImpl.library;
    if (!library) {
        return Status(TNNERR_INVALID_INPUT, "metal library is nil");
    }

    id<MTLFunction> func_process = nil;
    
    std::string kernel_name("");
    if (src_mat_type == dst_mat_type) {
        if (N8UC4 == src_mat_type) {
            if (INTERP_TYPE_NEAREST == param.type) {
#if ENABLE_PIPELINE_CACHE
                kernel_name = string("mat_converter_texture_n8uc4_resize_nearest");
                if (library_cache.count(kernel_name) != 0) {
                    // cache hit
                    pipeline_process_ = library_cache[kernel_name];
                    return TNN_OK;
                }
                // cache miss
#endif
                func_process = [library newFunctionWithName:@"mat_converter_texture_n8uc4_resize_nearest"];
            } else if (INTERP_TYPE_LINEAR == param.type) {
#if ENABLE_PIPELINE_CACHE
                kernel_name = string("mat_converter_texture_n8uc4_resize_linear");
                if (library_cache.count(kernel_name) != 0) {
                    // cache hit
                    pipeline_process_ = library_cache[kernel_name];
                    return TNN_OK;
                }
                // cache miss
#endif
                func_process = [library newFunctionWithName:@"mat_converter_texture_n8uc4_resize_linear"];
            }
        } else if (N8UC3 == src_mat_type) {
            return Status(TNNERR_PARAM_ERR, "mat type not support yet");
        } else if (NGRAY == src_mat_type) {
            return Status(TNNERR_PARAM_ERR, "mat type not support yet");
        } else if (NNV21 == src_mat_type || NNV12 == src_mat_type) {
            return Status(TNNERR_PARAM_ERR, "mat type not support yet");
        } else if (NCHW_FLOAT == src_mat_type) {
            return Status(TNNERR_PARAM_ERR, "mat type not support yet");
        } else {
            return Status(TNNERR_PARAM_ERR, "mat type not support yet");
        }
    } else {
        return Status(TNNERR_PARAM_ERR, "src and dst mat type must be same");
    }
    if (!func_process) {
        return Status(TNNERR_INVALID_INPUT, "mat converter func not found");
    }
    auto pipeline_process = [device_ newComputePipelineStateWithFunction:func_process error:nil];
    if (!pipeline_process) {
        return Status(TNNERR_INVALID_INPUT, "resize pipeline is nil");
    }
    pipeline_process_ = pipeline_process;
#if ENABLE_PIPELINE_CACHE
    library_cache[kernel_name] = pipeline_process;
#endif
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
            return Status(TNNERR_PARAM_ERR, "mat type not support yet");
        } else if (NGRAY == src_mat_type) {
            return Status(TNNERR_PARAM_ERR, "mat type not support yet");
        } else if (NNV21 == src_mat_type || NNV12 == src_mat_type) {
            return Status(TNNERR_PARAM_ERR, "mat type not support yet");
        } else {
            return Status(TNNERR_PARAM_ERR, "mat type not support yet");
        }
    } else if(src_mat_type == N8UC3 || dst_mat_type == N8UC3) {
        auto src_device_type = src.GetDeviceType();
        auto dst_device_type = dst.GetDeviceType();
        if (src_device_type == DEVICE_METAL) {
            func_process = [library newFunctionWithName:@"copy_n8uc4_metal_to_n8uc3_cpu"];
        } else {
            func_process = [library newFunctionWithName:@"copy_n8uc3_cpu_to_n8uc4_metal"];
        }
    }else {
        return Status(TNNERR_PARAM_ERR, "src and dst mat type must be same");
    }
    if (!func_process) {
        return Status(TNNERR_INVALID_INPUT, "mat converter func not found");
    }
    auto pipeline_process = [device_ newComputePipelineStateWithFunction:func_process error:nil];
    if (!pipeline_process) {
        return Status(TNNERR_INVALID_INPUT, "copy pipeline is nil");
    }
    pipeline_process_ = pipeline_process;
    
    return TNN_OK;
}

Status MetalMatConverterAcc::AllocateCopyMakeBorderComputePipeline(CopyMakeBorderParam param, Mat &src, Mat &dst, void *command_queue) {
    auto src_mat_type = src.GetMatType();
    auto dst_mat_type = dst.GetMatType();

#if ENABLE_PIPELINE_CACHE
    static std::map<std::string,  id <MTLComputePipelineState> > library_cache;
#endif

    auto command_queue_impl = (__bridge TNNMetalCommandQueueImpl *)(command_queue);
    if (!command_queue_impl) {
        return Status(TNNERR_INST_ERR, "command queue is nil");
    }
    auto library = command_queue_impl.metalContextImpl.library;
    if (!library) {
        return Status(TNNERR_INVALID_INPUT, "metal library is nil");
    }

    if (src_mat_type != dst_mat_type) {
        return Status(TNNERR_PARAM_ERR, "src and dst mat type must be same");
    }
    if(BORDER_TYPE_CONSTANT != param.border_type) {
        return Status(TNNERR_INVALID_INPUT, "border type not support yet");
    }

    id<MTLFunction> func_process = nil;
    std::string kernel_name("");
    if (N8UC4 == src_mat_type) {
#if ENABLE_PIPELINE_CACHE
        kernel_name = string("copymakeborder_n8uc4_constant");
        if (library_cache.count(kernel_name) != 0) {
            // cache hit
            pipeline_process_ = library_cache[kernel_name];
            return TNN_OK;
        }
        // cache miss
#endif
        func_process = [library newFunctionWithName:@"copymakeborder_n8uc4_constant"];
    } else if (NCHW_FLOAT == src_mat_type) {
#if ENABLE_PIPELINE_CACHE
        kernel_name = string("copymakeborder_nchw_constant");
        if (library_cache.count(kernel_name) != 0) {
            // cache hit
            pipeline_process_ = library_cache[kernel_name];
            return TNN_OK;
        }
        // cache miss
#endif
        func_process = [library newFunctionWithName:@"copymakeborder_nchw_constant"];
    } else {
        return Status(TNNERR_PARAM_ERR, "mat type not support yet");
    }
    if (!func_process) {
        return Status(TNNERR_INVALID_INPUT, "mat converter func not found");
    }
    auto pipeline_process = [device_ newComputePipelineStateWithFunction:func_process error:nil];
    if (!pipeline_process) {
        return Status(TNNERR_INVALID_INPUT, "copymakeborder pipeline is nil");
    }
    pipeline_process_ = pipeline_process;
#if ENABLE_PIPELINE_CACHE
    library_cache[kernel_name] = pipeline_process;
#endif
    return TNN_OK;
}

Status  MetalMatConverterAcc::AllocateWarpAffineComputePipeline(WarpAffineParam param, Mat& src, Mat& dst, void *command_queue) {
#if ENABLE_PIPELINE_CACHE
    static std::map<std::string, id<MTLComputePipelineState>> library_cache;
#endif
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
    
    std::string kernel_name("");
    if (src_mat_type == dst_mat_type) {
        if (N8UC4 == src_mat_type) {
            if (INTERP_TYPE_NEAREST == interp_type) {
#if ENABLE_PIPELINE_CACHE
                kernel_name = "mat_converter_texture_n8uc4_warpaffine_nearest_const";
                if (library_cache.count(kernel_name) > 0) {
                    pipeline_process_ = library_cache[kernel_name];
                    return TNN_OK;
                }
#endif
                func_process = [library newFunctionWithName:@"mat_converter_texture_n8uc4_warpaffine_nearest_const"];
            } else if (INTERP_TYPE_LINEAR == interp_type && BORDER_TYPE_CONSTANT == border_type) {
#if ENABLE_PIPELINE_CACHE
                kernel_name = "mat_converter_texture_n8uc4_warpaffine_linear_const";
                if (library_cache.count(kernel_name) > 0) {
                    pipeline_process_ = library_cache[kernel_name];
                    return TNN_OK;
                }
#endif
                func_process = [library newFunctionWithName:@"mat_converter_texture_n8uc4_warpaffine_linear_const"];
            } else {
                return Status(TNNERR_PARAM_ERR, "not support yet");
            }
        } else if (N8UC3 == src_mat_type) {
            return Status(TNNERR_PARAM_ERR, "mat type not support yet");
        } else if (NGRAY == src_mat_type) {
            return Status(TNNERR_PARAM_ERR, "mat type not support yet");
        } else if (NNV21 == src_mat_type || NNV12 == src_mat_type) {
            return Status(TNNERR_PARAM_ERR, "mat type not support yet");
        } else if (NCHW_FLOAT == src_mat_type) {
            return Status(TNNERR_PARAM_ERR, "mat type not support yet");
        } else {
            return Status(TNNERR_PARAM_ERR, "mat type not support yet");
        }
    } else {
        return Status(TNNERR_PARAM_ERR, "src and dst mat type must be same");
    }
    if (!func_process) {
        return Status(TNNERR_INVALID_INPUT, "mat converter func not found");
    }
    auto pipeline_process = [device_ newComputePipelineStateWithFunction:func_process error:nil];
    if (!pipeline_process) {
        return Status(TNNERR_INVALID_INPUT, "warpaffine pipeline is nil");
    }
    pipeline_process_ = pipeline_process;
#if ENABLE_PIPELINE_CACHE
    library_cache[kernel_name] = pipeline_process;
#endif
    
    return TNN_OK;
}

Status MetalMatConverterAcc::AllocateBGR2GrayComputePipeline(Mat& src, Mat& dst, void *command_queue) {
#if ENABLE_PIPELINE_CACHE
        static std::map<std::string, id<MTLComputePipelineState>> library_cache;
#endif
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
    
    std::string kernel_name("");
    if (src_mat_type == N8UC4) {
        if (NCHW_FLOAT == dst_mat_type) {
#if ENABLE_PIPELINE_CACHE
            kernel_name = "bgr2gray_n8uc4_nchw_float";
            if (library_cache.count(kernel_name) > 0) {
                pipeline_process_ = library_cache[kernel_name];
                return TNN_OK;
            }
#endif
            func_process = [library newFunctionWithName:@"bgr2gray_n8uc4_nchw_float"];
        } else {
            return Status(TNNERR_PARAM_ERR, "dst mat type not support yet");
        }
    }else {
        return Status(TNNERR_PARAM_ERR, "mat type not support yet");
    }
    if (!func_process) {
        return Status(TNNERR_INVALID_INPUT, "mat converter func not found");
    }
    auto pipeline_process = [device_ newComputePipelineStateWithFunction:func_process error:nil];
    if (!pipeline_process) {
        return Status(TNNERR_INVALID_INPUT, "bgr2gray pipeline is nil");
    }
    pipeline_process_ = pipeline_process;
#if ENABLE_PIPELINE_CACHE
    library_cache[kernel_name] = pipeline_process;
#endif
    return TNN_OK;
}

bool MetalMatConverterAcc::DeviceTypeCheck(const DeviceType& src_device_type,
                                           const DeviceType& dst_device_type) {
    return dst_device_type != DEVICE_METAL && src_device_type != DEVICE_METAL;
}

Status MetalMatConverterAcc::CopyInputCheck(Mat& src,
                                            Mat& dst,
                                            const DeviceType& src_device_type,
                                            const DeviceType& dst_device_type,
                                            const MatType& src_mat_type,
                                            const MatType& dst_mat_type) {
    if (DeviceTypeCheck(src_device_type, dst_device_type)) {
        return Status(TNNERR_INVALID_INPUT, "neither src nor dst is not Metal Mat");
    }

    if (!(src_device_type == DEVICE_NAIVE || src_device_type == DEVICE_ARM || dst_device_type == DEVICE_NAIVE || dst_device_type == DEVICE_ARM)) {
        return Status(TNNERR_INVALID_INPUT, "device type not support yet");
    }

    if (src_mat_type != N8UC4 && src_mat_type != NCHW_FLOAT && src_mat_type != N8UC3) {
        return Status(TNNERR_PARAM_ERR, "mat type not support yet");
    }

    if (src_mat_type != dst_mat_type && !(src_mat_type == N8UC3 && dst_mat_type == N8UC4) && !(src_mat_type == N8UC4 && dst_mat_type == N8UC3)) {
        return Status(TNNERR_INVALID_INPUT, "src and dst mat type must be same");
    }

    if (! ((src_mat_type == dst_mat_type && DimsVectorUtils::Equal(src.GetDims(), dst.GetDims())) || ((src_mat_type == N8UC3 || dst_mat_type == N8UC3) && (src.GetHeight()==dst.GetHeight() && src.GetWidth()==dst.GetWidth()))) ) {
        return Status(TNNERR_INVALID_INPUT, "src and dst shape not match");
    }

    return TNN_OK;
}

Status MetalMatConverterAcc::SetDevice(Mat& src,
                                       Mat& dst,
                                       const DeviceType& src_device_type,
                                       const DeviceType& dst_device_type,
                                       const MatType& src_mat_type,
                                       const MatType& dst_mat_type) {
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

    return TNN_OK;
}

Status MetalMatConverterAcc::MetalCopyToCPU(Mat& src, Mat& dst,
                                            const DimsVector& dims,
                                            const MatType& src_mat_type, const MatType& dst_mat_type,
                                            TNNMetalCommandQueueImpl *command_queue_impl) {
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
        MTLSize group_threads = {(NSUInteger)pipeline_process_.threadExecutionWidth, (NSUInteger)1, (NSUInteger)1};
        MTLSize groups = {(NSUInteger)((dims[3] + group_threads.width - 1) / group_threads.width), (NSUInteger)dims[2], (NSUInteger)1};
        if (src_mat_type == NCHW_FLOAT) {
            groups = {(NSUInteger)((dims[3] + group_threads.width - 1) / group_threads.width), (NSUInteger)dims[2], (NSUInteger)dims[1]*dims[0]};
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

    return TNN_OK;
}

Status MetalMatConverterAcc::CPUCopyToMetal(Mat& src, Mat& dst,
                                            const DimsVector& dims,
                                            const MatType& src_mat_type, const MatType& dst_mat_type,
                                            TNNMetalCommandQueueImpl *command_queue_impl) {
    if (src_mat_type == N8UC4) {
        id<MTLTexture> texture = (__bridge id<MTLTexture>)(dst.GetData());
        if (!texture) {
            return Status(TNNERR_INST_ERR, "dst GetTexture return nil");
        }
        [texture replaceRegion:MTLRegionMake2D(0, 0, dims[3], dims[2])
                    mipmapLevel:0
                        withBytes:src.GetData()
                    bytesPerRow:dims[3]*4];
    } else if(src_mat_type == NCHW_FLOAT) {
        // 1) cpu => buffer
        auto count = DimsVectorUtils::Count(dims);
        id<MTLBuffer> tmp_buffer = [device_ newBufferWithLength: count*4
                                                        options:MTLResourceOptionCPUCacheModeDefault];
        memcpy([tmp_buffer contents], src.GetData(), tmp_buffer.length);
        // 2) buffer => metal
        MTLSize group_threads = {(NSUInteger)pipeline_process_.threadExecutionWidth, (NSUInteger)1, (NSUInteger)1};
        MTLSize groups = {(NSUInteger)((dims[3] + group_threads.width - 1) / group_threads.width), (NSUInteger)dims[2], (NSUInteger)dims[1]*dims[0]};

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
        id<MTLBuffer> tmp_buffer = [device_ newBufferWithLength: dims[2]*dims[3]*3
                                                        options:MTLResourceOptionCPUCacheModeDefault];
        memcpy([tmp_buffer contents], src.GetData(), tmp_buffer.length);

        // 2) buffer => metal
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

    auto status = CopyInputCheck(src, dst, src_device_type, dst_device_type, src_mat_type, dst_mat_type);
    if (status != TNN_OK) {
        return status;
    }

    status = SetDevice(src, dst, src_device_type, dst_device_type, src_mat_type, dst_mat_type);
    if (status != TNN_OK) {
        return status;
    }
    
    auto command_queue_impl = (__bridge TNNMetalCommandQueueImpl *)(command_queue);
    if (!command_queue_impl) {
        return Status(TNNERR_INST_ERR, "command queue is nil");
    }
    
    auto context_impl = command_queue_impl.metalContextImpl;
    
    status = AllocateBufferCopyParam(src, dst);
    if (status != TNN_OK) {
        return status;
    }
    
    status = AllocateCopyComputePipeline(src, dst, command_queue);
    if (status != TNN_OK) {
        return status;
    }

    auto dims = src.GetDims();
    //check copy direction
    if (src_device_type == DEVICE_METAL) {
        // Metal => cpu
        status = MetalCopyToCPU(src, dst, dims, src_mat_type, dst_mat_type, command_queue_impl);
        if (status != TNN_OK) {
            return status;
        }
    } else {
        // cpu => Metal
        status = CPUCopyToMetal(src, dst, dims, src_mat_type, dst_mat_type, command_queue_impl);
        if (status != TNN_OK) {
            return status;
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
        return Status(TNNERR_PARAM_ERR, "src and dst mat type must be same");
    }

    if (src_mat_type != N8UC4) {
        return Status(TNNERR_PARAM_ERR, "mat type not support yet");
    }
    //Get device
    if (device_ == nil) {
        id<MTLTexture> texture = (__bridge id<MTLTexture>)(src.GetData());
        device_     = texture.device;
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
#if KERNEL_SYNC
        [command_buffer waitUntilCompleted];
#endif
    } while(0);
    return TNN_OK;
}

Status MetalMatConverterAcc::Crop(Mat& src, Mat& dst, CropParam param, void* command_queue) {
    auto src_device_type = src.GetDeviceType();
    auto dst_device_type = dst.GetDeviceType();

    auto src_mat_type = src.GetMatType();
    auto dst_mat_type = dst.GetMatType();

    if (dst_mat_type != src_mat_type) {
        return Status(TNNERR_PARAM_ERR, "src and dst mat type must be same");
    }

    if (src_device_type != dst_device_type) {
        return Status(TNNERR_PARAM_ERR, "src and dst device type must be same");
    }

    if (src_mat_type != N8UC4) {
        return Status(TNNERR_PARAM_ERR, "mat type not support yet");
    }
    //Get device
    if (device_ == nil) {
        id<MTLTexture> texture = (__bridge id<MTLTexture>)(src.GetData());
        device_     = texture.device;
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
#if KERNEL_SYNC
        //wait to complete
        [command_buffer waitUntilCompleted];
#endif
    } while(0);
    return TNN_OK;
}

Status MetalMatConverterAcc::WarpAffine(Mat& src, Mat& dst, WarpAffineParam param, void* command_queue) {
    auto src_device_type = src.GetDeviceType();
    auto dst_device_type = dst.GetDeviceType();

    auto src_mat_type = src.GetMatType();
    auto dst_mat_type = dst.GetMatType();

    if (dst_mat_type != src_mat_type) {
        return Status(TNNERR_PARAM_ERR, "src and dst mat type must be same");
    }

    if (src_device_type != dst_device_type) {
        return Status(TNNERR_PARAM_ERR, "src and dst device type must be same");
    }

    if (src_mat_type != N8UC4) {
        return Status(TNNERR_PARAM_ERR, "mat type not support yet");
    }
    //Get device
    if (device_ == nil) {
        id<MTLTexture> texture = (__bridge id<MTLTexture>)(src.GetData());
        device_     = texture.device;
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
#if KERNEL_SYNC
        //wait to complete
        [command_buffer waitUntilCompleted];
#endif
    } while(0);
    
    return TNN_OK;
}

Status MetalMatConverterAcc::CopyMakeBorder(Mat &src, Mat &dst, CopyMakeBorderParam param, void* command_queue) {
    auto src_device_type = src.GetDeviceType();
    auto dst_device_type = dst.GetDeviceType();

    auto src_mat_type = src.GetMatType();
    auto dst_mat_type = dst.GetMatType();

    if (dst_mat_type != src_mat_type) {
        return Status(TNNERR_PARAM_ERR, "src and dst mat type must be same");
    }

    if (src_device_type != dst_device_type) {
        return Status(TNNERR_PARAM_ERR, "src and dst device type must be same");
    }

    if (src_mat_type != N8UC4 && src_mat_type != NCHW_FLOAT) {
        return Status(TNNERR_PARAM_ERR, "mat type not support yet");
    }
    //Get device
    if (device_ == nil) {
        id<MTLTexture> texture = (__bridge id<MTLTexture>)(src.GetData());
        device_     = texture.device;
    }

    auto command_queue_impl = (__bridge TNNMetalCommandQueueImpl *)(command_queue);
    if (!command_queue_impl) {
        return Status(TNNERR_INST_ERR, "command queue is nil");
    }

    auto context_impl = command_queue_impl.metalContextImpl;

    auto status = AllocateBufferCopyMakeBorderParam(param, src, dst);
    if (status != TNN_OK) {
        return status;
    }

    status = AllocateCopyMakeBorderComputePipeline(param, src, dst, command_queue);
    if (status != TNN_OK) {
        return status;
    }

    do {
        auto dst_dims = dst.GetDims();
        const int dst_height = dst_dims[2];
        const int dst_width  = dst_dims[3];
        MTLSize group_threads = {(NSUInteger)pipeline_process_.threadExecutionWidth, (NSUInteger)1, (NSUInteger)1};
        MTLSize groups = {(NSUInteger)((dst_width + group_threads.width - 1) / group_threads.width), (NSUInteger)dst_height, (NSUInteger)1};
        if (src_mat_type == NCHW_FLOAT) {
            groups.depth = dst_dims[1] * dst_dims[0];
        }

        auto command_buffer = [command_queue_impl commandBuffer];
        [command_buffer enqueue];
        auto encoder = [command_buffer computeCommandEncoder];
        [encoder setComputePipelineState: pipeline_process_];

        if (src_mat_type == N8UC4) {
            id<MTLTexture> input_texture = (__bridge id<MTLTexture>)(src.GetData());
            id<MTLTexture> output_texture = (__bridge id<MTLTexture>)(dst.GetData());

            [encoder setTexture:input_texture atIndex:0];
            [encoder setTexture:output_texture atIndex:1];
            [encoder setBuffer:buffer_copymakeborder_param_ offset:0 atIndex:0];
        } else if (src_mat_type == NCHW_FLOAT) {
            id<MTLBuffer> input_buffer = (__bridge id<MTLBuffer>)(src.GetData());
            id<MTLBuffer> output_buffer = (__bridge id<MTLBuffer>)(dst.GetData());

            [encoder setBuffer:input_buffer offset:0 atIndex:0];
            [encoder setBuffer:output_buffer offset:0 atIndex:1];
            [encoder setBuffer:buffer_copymakeborder_param_ offset:0 atIndex:2];
        }

        [encoder dispatchThreadgroups:groups threadsPerThreadgroup:group_threads];
        [encoder endEncoding];

        [command_buffer commit];
#if KERNEL_SYNC
        //wait to complete
        [command_buffer waitUntilCompleted];
#endif
    } while(0);
    return TNN_OK;
}

Status MetalMatConverterAcc::BGR2Gray(Mat& src, Mat& dst, void* command_queue) {
    auto src_device_type = src.GetDeviceType();
    auto dst_device_type = dst.GetDeviceType();
    
    auto src_mat_type = src.GetMatType();
    auto dst_mat_type = dst.GetMatType();
    
    if (src_mat_type != N8UC4 || dst_mat_type != NCHW_FLOAT) {
        return Status(TNNERR_PARAM_ERR, "mat type not support yet");
    }

    if (src_device_type != dst_device_type) {
        return Status(TNNERR_PARAM_ERR, "src and dst device type must be same");
    }

    //Get device
    if (device_ == nil) {
        id<MTLTexture> texture = (__bridge id<MTLTexture>)(src.GetData());
        device_     = texture.device;
    }

    auto command_queue_impl = (__bridge TNNMetalCommandQueueImpl *)(command_queue);
    if (!command_queue_impl) {
        return Status(TNNERR_INST_ERR, "command queue is nil");
    }
    
    auto context_impl = command_queue_impl.metalContextImpl;
    
    auto status = AllocateBufferBGR2GrayParam(src, dst);
    if (status != TNN_OK) {
        return status;
    }
    
    status = AllocateBGR2GrayComputePipeline(src, dst, command_queue);
    if (status != TNN_OK) {
        return status;
    }
    
    do {
        MTLSize group_threads = {(NSUInteger)pipeline_process_.threadExecutionWidth, (NSUInteger)1, (NSUInteger)1};
        MTLSize groups = {(NSUInteger)((bgr2gray_param_.width + group_threads.width - 1) / group_threads.width), (NSUInteger)bgr2gray_param_.height, (NSUInteger)1};
        id<MTLTexture> input_texture = (__bridge id<MTLTexture>)(src.GetData());
        id<MTLBuffer> output = (__bridge id<MTLBuffer>)(dst.GetData());
        
        auto command_buffer = [command_queue_impl commandBuffer];
        [command_buffer enqueue];
        auto encoder = [command_buffer computeCommandEncoder];
        [encoder setComputePipelineState: pipeline_process_];
        
        [encoder setTexture:input_texture atIndex:0];
        [encoder setBuffer:output offset:0 atIndex:0];
        [encoder setBuffer:buffer_bgr2gray_param_ offset:0 atIndex:1];
        
        [encoder dispatchThreadgroups:groups threadsPerThreadgroup:group_threads];
        [encoder endEncoding];
        
        [command_buffer commit];
#if KERNEL_SYNC
        //wait to complete
        [command_buffer waitUntilCompleted];
#endif
    } while(0);
    return TNN_OK;
}

Status MetalMatConverterAcc::CvtColor(Mat& src, Mat& dst, ColorConversionType type, void* command_queue) {
    return Status(TNNERR_PARAM_ERR, "metal not support color conversion");
}

DECLARE_MAT_CONVERTER_CREATER(Metal);
REGISTER_MAT_CONVERTER(Metal, DEVICE_METAL);

}  // namespace TNN_NS
