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

#import "tnn/core/macro.h"
#import "tnn/device/metal/acc/metal_common.h"
#import "tnn/device/metal/metal_command_queue.h"
#import "tnn/device/metal/metal_context.h"
#import "tnn/utils/blob_converter_internal.h"
#import "tnn/utils/data_type_utils.h"
#import "tnn/utils/dims_utils.h"
#include "tnn/utils/half_utils_inner.h"

namespace TNN_NS {
class MetalBlobConverterAcc : public BlobConverterAcc {
public:
    MetalBlobConverterAcc(Blob *blob);
    virtual ~MetalBlobConverterAcc(){};
    virtual Status ConvertToMat(Mat &image, MatConvertParam param, void *command_queue = NULL);
    virtual Status ConvertToMatAsync(Mat &image, MatConvertParam param, void *command_queue = NULL);

    virtual Status ConvertFromMat(Mat &image, MatConvertParam param, void *command_queue = NULL);
    virtual Status ConvertFromMatAsync(Mat &image, MatConvertParam param, void *command_queue = NULL);

protected:
    MatConvertParam param_;
    id<MTLDevice> device_                         = nil;
    id<MTLBuffer> buffer_param_                = nil;
    NSString *pipeline_func_name_             = nil;
    
    // buffer for scale and bias used in nchw_float mat transformation
    id<MTLBuffer> buffer_scale_;
    id<MTLBuffer> buffer_bias_;
    // @param waitState: 0: no wait, 1: wait gpu completed, 2: wait gpu scheduled.
    Status ConvertToMatCommon(Mat &input_mat, Blob *output_blob, void *command_queue, int waitState = 1);
    Status ConvertFromMatCommon(Mat &input_mat, Blob *output_blob, void *command_queue, int waitState = 1);

    Status AllocateBufferParam(MatConvertParam param, Mat *mat, Blob *blob, bool is_mat_to_blob);
    Status AllocateComputePipeline(MatConvertParam param, Mat *mat, Blob *blob, bool is_mat_to_blob,
                                   void *command_queue);
    bool CheckDeviceAndMat(DeviceType device_type, MatType mat_type);
    std::shared_ptr<Mat> buffer_mat_ = nullptr;
};

MetalBlobConverterAcc::MetalBlobConverterAcc(Blob *blob) : BlobConverterAcc(blob) {
    auto buffer = (__bridge id<MTLBuffer>)(blob->GetHandle().base);
    device_     = buffer.device;
}

bool MetalBlobConverterAcc::CheckDeviceAndMat(DeviceType device_type, MatType mat_type) {
    bool device_supported = (device_type == DEVICE_METAL || device_type == DEVICE_ARM || 
            device_type == DEVICE_X86 || device_type == DEVICE_NAIVE);

    bool mat_supported = (mat_type == N8UC4 || mat_type == N8UC3 || mat_type == NGRAY||
            mat_type == NCHW_FLOAT || mat_type == RESERVED_BFP16_TEST || mat_type == NC_INT32);

    return device_supported && mat_supported;
}

Status MetalBlobConverterAcc::AllocateBufferParam(MatConvertParam param, Mat *mat, Blob *blob, bool is_mat_to_blob) {
    auto dims = blob->GetBlobDesc().dims;
    MetalImageConverterParams metal_param;
    metal_param.width        = DimsFunctionUtils::GetDimProduct(dims, 3);
    metal_param.height       = DimsFunctionUtils::GetDim(dims, 2);
    metal_param.size         = metal_param.height * metal_param.width;
    metal_param.channel      = DimsFunctionUtils::GetDim(dims, 1);
    metal_param.slice        = UP_DIV(metal_param.channel, 4);
    metal_param.batch        = dims[0];
    metal_param.bgra_to_rgba = param.reverse_channel;

    LOGD("metal_param size: %d %d %d %d %d\n", metal_param.batch, metal_param.channel, metal_param.height,
         metal_param.width, metal_param.size);

    float scale_texture_buffer = 1.0f;
    float bias_texture_buffer  = 1.0f;
    const auto mat_type = mat->GetMatType();
    // Metal does not support N8UC3 mat, only N8UC4 metal mat uses mtltexture
    bool need_rescale = (mat_type == N8UC4) && (is_mat_to_blob || mat->GetDeviceType() == DEVICE_METAL);
    if (need_rescale) {
        scale_texture_buffer = is_mat_to_blob ? 255.0f : 1.0 / 255.0f;
        bias_texture_buffer  = is_mat_to_blob ? 1.0    : 1.0 / 255.0f;
    }

    if (mat_type == NCHW_FLOAT || mat_type == NGRAY || mat_type == RESERVED_BFP16_TEST || mat_type == NC_INT32) {
        // scale and bias should at least have channel elements, so we use another buffer instead of metal_param
        if (param.scale.size() < metal_param.channel || param.bias.size() < metal_param.channel) {
            // invalid scale and bias
            return Status(TNNERR_INVALID_INPUT, "invalid scale or bias shape!");
        }
        if (buffer_scale_ == nil) {
            buffer_scale_ = [device_ newBufferWithBytes:&(param.scale[0])
                                                 length:sizeof(float)*metal_param.channel
                                                options:MTLResourceCPUCacheModeWriteCombined];
        }
        if (buffer_bias_ == nil) {
            buffer_bias_ = [device_ newBufferWithBytes:&(param.bias[0])
                                                 length:sizeof(float)*metal_param.channel
                                                options:MTLResourceCPUCacheModeWriteCombined];
        }
        if (buffer_scale_ == nil) {
            return Status(TNNERR_INVALID_INPUT, "buffer scale is nil");
        }
        if (buffer_bias_ == nil) {
            return Status(TNNERR_INVALID_INPUT, "buffer bias is nil");
        }
    } else {
        if (param.scale.size() >= 3) {
            metal_param.scale_x = scale_texture_buffer * param.scale[0];
            metal_param.scale_y = scale_texture_buffer * param.scale[1];
            metal_param.scale_z = scale_texture_buffer * param.scale[2];
            if (param.scale.size() > 3)
                metal_param.scale_w = scale_texture_buffer * param.scale[3];
        } else {
            metal_param.scale_x = 1.0f;
            metal_param.scale_y = 1.0f;
            metal_param.scale_z = 1.0f;
            metal_param.scale_w = 1.0f;
        }
        if (param.bias.size() >= 3) {
            metal_param.bias_x = bias_texture_buffer * param.bias[0];
            metal_param.bias_y = bias_texture_buffer * param.bias[1];
            metal_param.bias_z = bias_texture_buffer * param.bias[2];
            if (param.bias.size() > 3)
                metal_param.bias_w = bias_texture_buffer * param.bias[3];
        } else {
            metal_param.bias_x = 0.0f;
            metal_param.bias_y = 0.0f;
            metal_param.bias_z = 0.0f;
            metal_param.bias_w = 0.0f;
        }
        LOGD("metal_param scale: %.6f %.6f %.6f\n", metal_param.scale_x, metal_param.scale_y, metal_param.scale_z);
        LOGD("metal_param size: %d %d %d\n", metal_param.size, metal_param.slice, metal_param.batch);
    }


    buffer_param_ = [device_ newBufferWithBytes:&metal_param
                                         length:sizeof(MetalImageConverterParams)
                                        options:MTLResourceCPUCacheModeWriteCombined];
    if (!buffer_param_) {
        return Status(TNNERR_INVALID_INPUT, "buffer param is nil");
    }
    return TNN_OK;
}

Status MetalBlobConverterAcc::AllocateComputePipeline(MatConvertParam param, Mat *mat, Blob *blob, bool is_mat_to_blob,
                                                      void *command_queue) {
    auto command_queue_impl = (__bridge TNNMetalCommandQueueImpl *)(command_queue);
    if (!command_queue_impl) {
        return Status(TNNERR_INST_ERR, "command queue is nil");
    }

    auto library = command_queue_impl.metalContextImpl.library;
    if (!library) {
        return Status(TNNERR_INVALID_INPUT, "metal library is nil");
    }

    auto mat_device_type  = mat->GetDeviceType();
    auto mat_type         = mat->GetMatType();
    auto blob_device_type = blob->GetBlobDesc().device_type;
    auto blob_data_type   = blob->GetBlobDesc().data_type;
    auto blob_data_format = blob->GetBlobDesc().data_format;

    NSString *func_name = nil;

    // texture <-> blob
    if (mat_type == N8UC4) {
        if (is_mat_to_blob) {
            if (blob_data_format == DATA_FORMAT_NC4HW4) {
                func_name = @"image_converter_texture_bgra8888_2_buffer_nc4hw4";
                LOGD("image_converter_texture_bgra8888_2_buffer_nc4hw4\n");
            } else if (blob_data_format == DATA_FORMAT_NCHW) {
                if (blob_data_type == DATA_TYPE_FLOAT) {
                    func_name = @"image_converter_texture_bgra8888_2_buffer_nchw_f";
                    LOGD("image_converter_texture_bgra8888_2_buffer_nchw_f\n");
                }
            }
        } else {
            if (mat_device_type != DEVICE_METAL && blob_data_format == DATA_FORMAT_NC4HW4) {
                func_name = @"image_converter_buffer_nc4hw4_2_buffer_bgra";
                LOGD("image_converter_buffer_nc4hw4_2_buffer_bgra\n");
            } else if (blob_data_format == DATA_FORMAT_NC4HW4) {
                func_name = @"image_converter_buffer_nc4hw4_2_texture_bgra8888";
                LOGD("image_converter_buffer_nc4hw4_2_texture_bgra8888\n");
            } else if (blob_data_format == DATA_FORMAT_NCHW) {
                if (blob_data_type == DATA_TYPE_FLOAT) {
                    func_name = @"image_converter_buffer_nchw_f_2_texture_bgra8888";
                    LOGD("image_converter_buffer_nchw_f_2_texture_bgra8888\n");
                }
            }
        }
    } else if (mat_type == N8UC3) {
        if (is_mat_to_blob) {
            if (blob_data_format == DATA_FORMAT_NC4HW4) {
                func_name = @"image_converter_buffer_bgr_2_buffer_nc4hw4";
                LOGD("image_converter_buffer_bgr_2_buffer_nc4hw4\n");

            }
        } else {
            if (blob_data_format == DATA_FORMAT_NC4HW4) {
                func_name = @"image_converter_buffer_nc4hw4_2_buffer_bgr";
                LOGD("image_converter_buffer_nc4hw4_2_buffer_bgr\n");
            }
        }
    } else if (mat_type == NGRAY) {
            if (is_mat_to_blob) {
                if (blob_data_format == DATA_FORMAT_NC4HW4) {
                    func_name = @"data_converter_ngray_2_nc4hw4_float_v2";
                    LOGD("data_converter_ngray_2_nc4hw4_float_v2\n");
                }
            } else {
                if (blob_data_format == DATA_FORMAT_NC4HW4) {
                    func_name = @"data_converter_nc4hw4_2_ngray_v2";
                    LOGD("data_converter_nc4hw4_2_ngray_v2\n");
                }
            }
    } else if (mat_type == NCHW_FLOAT) {
        if (is_mat_to_blob) {
            if (blob_data_format == DATA_FORMAT_NCHW) {
                func_name = @"data_converter_nchw_float2ftype";
                LOGD("data_converter_nchw_float2ftype\n");
            } else if (blob_data_format == DATA_FORMAT_NC4HW4) {
                func_name = @"data_converter_nchw_2_nc4hw4_float_v2";
                LOGD("data_converter_nchw_2_nc4hw4_float_v2\n");
            }
        } else {
            if (blob_data_type == DATA_TYPE_INT32) {
                // int32 blob to float mat
                func_name = @"data_converter_nc4hw4_2_nchw_int322float_v2";
                LOGD("data_converter_nc4hw4_2_nchw_int32_v2\n");
            } else if (blob_data_format == DATA_FORMAT_NCHW) {
                func_name = @"data_converter_nchw_ftype2float";
                LOGD("data_converter_nchw_ftype2float\n");
            } else if (blob_data_format == DATA_FORMAT_NC4HW4) {
                func_name = @"data_converter_nc4hw4_2_nchw_float_v2";
                LOGD("data_converter_nc4hw4_2_nchw_float_v2\n");
            }
        }
    } else if (mat_type == RESERVED_BFP16_TEST) {
        if (is_mat_to_blob) {
            if (blob_data_format == DATA_FORMAT_NCHW) {
                func_name = @"data_converter_nchw_half2ftype";
                LOGD("data_converter_nchw_half2ftype\n");
            } else if (blob_data_format == DATA_FORMAT_NC4HW4) {
                func_name = @"data_converter_nchw_2_nc4hw4_half_v2";
                LOGD("data_converter_nchw_2_nc4hw4_float_v2\n");
            }
        } else {
            if (blob_data_format == DATA_FORMAT_NCHW) {
                func_name = @"data_converter_nchw_ftype2half";
                LOGD("data_converter_nchw_ftype2half\n");
            } else if (blob_data_format == DATA_FORMAT_NC4HW4) {
                func_name = @"data_converter_nc4hw4_2_nchw_half_v2";
                LOGD("data_converter_nc4hw4_2_nchw_float_v2\n");
            }
        }
    } else if (mat_type == NC_INT32) {
        if (blob_data_type == DATA_TYPE_INT32) {
            if (is_mat_to_blob) {
                if (blob_data_format == DATA_FORMAT_NC4HW4) {
                    func_name = @"data_converter_nchw_2_nc4hw4_int32_v2";
                    LOGD("data_converter_nchw_2_nc4hw4_int32_v2\n");
                } else if (blob_data_format == DATA_FORMAT_NCHW) {
                    func_name = @"data_converter_nchw_int";
                    LOGD("data_converter_nchw_int\n");
                }
            } else {
                if (blob_data_format == DATA_FORMAT_NC4HW4) {
                    func_name = @"data_converter_nc4hw4_2_nchw_int32_v2";
                    LOGD("data_converter_nc4hw4_2_nchw_int32_v2\n");
                } else if (blob_data_format == DATA_FORMAT_NCHW) {
                    func_name = @"data_converter_nchw_int";
                    LOGD("data_converter_nchw_int\n");
                }
            }
        }
    }

    if (!func_name) {
        return Status(TNNERR_INVALID_INPUT, "mat converter func not found");
    }
    pipeline_func_name_ = func_name;

    return TNN_OK;
}

Status MetalBlobConverterAcc::ConvertToMat(Mat &image, MatConvertParam param, void *command_queue) {
    param_ = param;
    return ConvertToMatCommon(image, blob_, command_queue, 1);
}

Status MetalBlobConverterAcc::ConvertToMatAsync(Mat &image, MatConvertParam param, void *command_queue) {
    param_ = param;
    return ConvertToMatCommon(image, blob_, command_queue, 2);
}

// #lizard forgives
Status MetalBlobConverterAcc::ConvertToMatCommon(Mat &output_mat, Blob *input_blob, void *command_queue,
                                                 int waitState) {
    auto mat_device_type = output_mat.GetDeviceType();
    auto mat_type        = output_mat.GetMatType();

    if (!CheckDeviceAndMat(mat_device_type, mat_type)) {
        return Status(TNNERR_COMMON_ERROR, "input_mat.GetDeviceType() or.GetMatType() is invalid");
    }

    auto dims               = blob_->GetBlobDesc().dims;
    auto command_queue_impl = (__bridge TNNMetalCommandQueueImpl *)(command_queue);
    if (!command_queue_impl) {
        return Status(TNNERR_INST_ERR, "command queue is nil");
    }
    
    auto context_impl = command_queue_impl.metalContextImpl;
    if (waitState == 1) {
        [context_impl commit:YES];
    }

    // check class type
    if (!input_blob || typeid(*input_blob) != typeid(Blob)) {
        LOGE("Error: input_blob is nil or not instance of Blob*\n");
        return Status(TNNERR_INST_ERR, "input_blob is nil or not instance of Blob*");
    }

    auto status = AllocateBufferParam(param_, &output_mat, input_blob, false);
    if (status != TNN_OK) {
        return status;
    }

    status = AllocateComputePipeline(param_, &output_mat, input_blob, false, command_queue);
    if (status != TNN_OK) {
        return status;
    }

    auto output_mat_device              = output_mat.GetDeviceType();
    id<MTLCommandBuffer> command_buffer = nil;
    if (mat_type == N8UC4 && output_mat_device == DEVICE_METAL) {
        auto output_texture       = (__bridge id<MTLTexture>)(output_mat.GetData());
        Blob *input_buffer_blob = (Blob *)(input_blob);
        if (output_texture.height != DimsFunctionUtils::GetDim(dims, 2) || output_texture.width != DimsFunctionUtils::GetDim(dims, 3) ||
            (output_texture.pixelFormat != MTLPixelFormatBGRA8Unorm &&
             output_texture.pixelFormat != MTLPixelFormatRGBA8Unorm)) {
            return Status(TNNERR_INST_ERR, "output mat's texture is invalid, wrong size or pixel format");
        }

        auto encoder = [context_impl encoder];
        if (!encoder) {
            LOGE("ERROR: ConvertToMatCommon cannot allocate new encoder\n");
            return Status(TNNERR_PARAM_ERR, "ConvertToMatCommon cannot allocate new encoder");
        }
        
        MetalBandwidth bandwidth;
        status = [context_impl load:pipeline_func_name_
                            encoder:encoder
                            bandwidth:bandwidth];
        RETURN_ON_NEQ(status, TNN_OK);
        
        MTLSize group_threads = {(NSUInteger)bandwidth.thread_execution_width, (NSUInteger)1, (NSUInteger)1};
        MTLSize groups = {(NSUInteger)((DimsFunctionUtils::GetDim(dims, 3) + group_threads.width - 1) / group_threads.width),
                          (NSUInteger)DimsFunctionUtils::GetDim(dims, 2),
                          (NSUInteger)1};

        [encoder setTexture:output_texture atIndex:0];
        [encoder setBuffer:(__bridge id<MTLBuffer>)(void *)input_buffer_blob->GetHandle().base
                    offset:(NSUInteger)input_buffer_blob->GetHandle().bytes_offset
                   atIndex:0];
        [encoder setBuffer:buffer_param_ offset:0 atIndex:1];

        [encoder dispatchThreadgroups:groups threadsPerThreadgroup:group_threads];
        [encoder endEncoding];

        command_buffer = context_impl.commandBuffer;

        [context_impl commit:YES];
        if (waitState == 1) {
            [command_buffer waitUntilCompleted];
        } else if (waitState == 2) {
            [command_buffer waitUntilScheduled];
        }
    } else if ((mat_type == N8UC4 || mat_type == N8UC3) && output_mat_device != DEVICE_METAL) {
        auto safe_dims = dims;
        safe_dims[1]   = mat_type == N8UC4 ? 4 : 3;
        auto count = DimsVectorUtils::Count(safe_dims);
        auto bytes_size = sizeof(unsigned char);
        id<MTLBuffer> output_mtl_buffer = [command_queue_impl.device newBufferWithLength:count * bytes_size
                                                                       options:MTLResourceCPUCacheModeDefaultCache];

        auto encoder = [context_impl encoder];
        if (!encoder) {
            LOGE("ERROR: ConvertToMatCommon cannot allocate new encoder\n");
            return Status(TNNERR_PARAM_ERR, "ConvertToMatCommon cannot allocate new encoder");
        }
        
        MetalBandwidth bandwidth;
        status = [context_impl load:pipeline_func_name_
                            encoder:encoder
                            bandwidth:bandwidth];
        RETURN_ON_NEQ(status, TNN_OK);
        
        MTLSize group_threads = {(NSUInteger)bandwidth.thread_execution_width, (NSUInteger)1, (NSUInteger)1};
        MTLSize groups = {(NSUInteger)((DimsFunctionUtils::GetDim(dims, 3) + group_threads.width - 1) / group_threads.width),
                          (NSUInteger)DimsFunctionUtils::GetDim(dims, 2),
                          (NSUInteger)1};

        [encoder setBuffer:output_mtl_buffer offset:0 atIndex:0];
        [encoder setBuffer:(__bridge id<MTLBuffer>)(void *)input_blob->GetHandle().base
                    offset:(NSUInteger)input_blob->GetHandle().bytes_offset
                   atIndex:1];
        [encoder setBuffer:buffer_param_ offset:0 atIndex:2];

        [encoder dispatchThreadgroups:groups threadsPerThreadgroup:group_threads];
        [encoder endEncoding];

        command_buffer = context_impl.commandBuffer;

        [context_impl commit:YES];

        [command_buffer waitUntilCompleted];
        memcpy(output_mat.GetData(), output_mtl_buffer.contents, count * bytes_size);
    } else if (mat_type == NGRAY ||mat_type == NCHW_FLOAT || mat_type == RESERVED_BFP16_TEST) {
        auto input_buffer_blob          = dynamic_cast<Blob *>(input_blob);
        id<MTLBuffer> output_mtl_buffer = nil;

        int count = DimsVectorUtils::Count(dims);
        const auto bytes_size = (mat_type == NCHW_FLOAT) ? sizeof(float) : ((mat_type == NGRAY) ? sizeof(unsigned char) : sizeof(fp16_t));
        
        if (output_mat_device == DEVICE_METAL) {
            output_mtl_buffer = (__bridge id<MTLBuffer>)(output_mat.GetData());
        } else if (output_mat_device == DEVICE_ARM || output_mat_device == DEVICE_NAIVE || mat_device_type == DEVICE_X86) {
            output_mtl_buffer = [command_queue_impl.device newBufferWithLength:count * bytes_size
                                                                       options:MTLResourceCPUCacheModeDefaultCache];
        }

        NSUInteger image_size  = DimsFunctionUtils::GetDimProduct(dims, 2);
        NSUInteger image_slice = UP_DIV(dims[1], 4);
        bool is_blob_nchw = input_buffer_blob->GetBlobDesc().data_format == DATA_FORMAT_NCHW;

        auto encoder = [context_impl encoder];
        if (!encoder) {
            LOGE("ERROR: ConvertToMatCommon cannot allocate new encoder\n");
            return Status(TNNERR_PARAM_ERR, "ConvertToMatCommon cannot allocate new encoder");
        }
        
        MetalBandwidth bandwidth;
        status = [context_impl load:pipeline_func_name_
                            encoder:encoder
                            bandwidth:bandwidth];
        RETURN_ON_NEQ(status, TNN_OK);
        
        auto group_threads = MTLSizeMake(bandwidth.thread_execution_width, 1, 1);
        auto groups = MTLSizeMake((image_size + group_threads.width - 1) / group_threads.width,
                                  image_slice, dims[0]);
        if (is_blob_nchw) {
            groups = MTLSizeMake((image_size + group_threads.width - 1) / group_threads.width,
                                 dims[1], dims[0]);
        }

        if (image_size <= image_slice) {
            group_threads = MTLSizeMake(1, bandwidth.thread_execution_width, 1);
            groups = MTLSizeMake(image_size,
                                 (image_slice + group_threads.height - 1) / group_threads.height,
                                 dims[0]);
            if (is_blob_nchw) {
                groups = MTLSizeMake(image_size,
                                     (dims[1] + group_threads.height - 1) / group_threads.height,
                                     dims[0]);
            }
        }

        [encoder setBuffer:output_mtl_buffer offset:0 atIndex:0];
        [encoder setBuffer:(__bridge id<MTLBuffer>)(void *)input_buffer_blob->GetHandle().base
                    offset:(NSUInteger)input_buffer_blob->GetHandle().bytes_offset
                   atIndex:1];
        [encoder setBuffer:buffer_param_ offset:0 atIndex:2];
        // scale and bias
        [encoder setBuffer:buffer_scale_ offset:0 atIndex:3];
        [encoder setBuffer:buffer_bias_  offset:0 atIndex:4];

        [encoder dispatchThreadgroups:groups threadsPerThreadgroup:group_threads];
        [encoder endEncoding];

        command_buffer = context_impl.commandBuffer;
        [context_impl commit:YES];

        if (output_mat_device == DEVICE_METAL) {
            if (waitState == 1) {
                [command_buffer waitUntilCompleted];
            } else if (waitState == 2) {
                [command_buffer waitUntilScheduled];
            }
        } else {
            [command_buffer waitUntilCompleted];
            memcpy(output_mat.GetData(), output_mtl_buffer.contents, count * bytes_size);
        }
    } else if (mat_type == NC_INT32) {
        auto input_buffer_blob          = dynamic_cast<Blob *>(input_blob);
        id<MTLBuffer> output_mtl_buffer = nil;

        int count = DimsVectorUtils::Count(dims);
        const auto bytes_size = sizeof(int);
        if (output_mat_device == DEVICE_METAL) {
            output_mtl_buffer = (__bridge id<MTLBuffer>)(output_mat.GetData());
        } else if (output_mat_device == DEVICE_ARM || output_mat_device == DEVICE_NAIVE || mat_device_type == DEVICE_X86) {
            output_mtl_buffer = [command_queue_impl.device newBufferWithLength:count * bytes_size
                                                                       options:MTLResourceCPUCacheModeDefaultCache];
        }

        NSUInteger image_size  = DimsFunctionUtils::GetDimProduct(dims, 2);
        NSUInteger image_slice = UP_DIV(dims[1], 4);
        
        auto encoder = [context_impl encoder];
        if (!encoder) {
            LOGE("ERROR: ConvertToMatCommon cannot allocate new encoder\n");
            return Status(TNNERR_PARAM_ERR, "ConvertToMatCommon cannot allocate new encoder");
        }
        
        MetalBandwidth bandwidth;
        status = [context_impl load:pipeline_func_name_
                            encoder:encoder
                            bandwidth:bandwidth];
        RETURN_ON_NEQ(status, TNN_OK);
        
        auto group_threads = MTLSizeMake(bandwidth.thread_execution_width, 1, 1);
        auto groups = MTLSizeMake((image_size + group_threads.width - 1) / group_threads.width,
                                  image_slice, dims[0]);

        if (image_size <= image_slice) {
            group_threads = MTLSizeMake(1, bandwidth.thread_execution_width, 1);
            groups = MTLSizeMake(image_size,
                                 (image_slice + group_threads.height - 1) / group_threads.height,
                                 dims[0]);
        }

        [encoder setBuffer:output_mtl_buffer offset:0 atIndex:0];
        [encoder setBuffer:(__bridge id<MTLBuffer>)(void *)input_buffer_blob->GetHandle().base
                    offset:(NSUInteger)input_buffer_blob->GetHandle().bytes_offset
                   atIndex:1];
        [encoder setBuffer:buffer_param_ offset:0 atIndex:2];
        // scale and bias
        [encoder setBuffer:buffer_scale_ offset:0 atIndex:3];
        [encoder setBuffer:buffer_bias_  offset:0 atIndex:4];

        [encoder dispatchThreadgroups:groups threadsPerThreadgroup:group_threads];
        [encoder endEncoding];

        command_buffer = context_impl.commandBuffer;
        [context_impl commit:YES];

        if (output_mat_device == DEVICE_METAL) {
            if (waitState == 1) {
                [command_buffer waitUntilCompleted];
            } else if (waitState == 2) {
                [command_buffer waitUntilScheduled];
            }
        } else {
            [command_buffer waitUntilCompleted];
            memcpy(output_mat.GetData(), output_mtl_buffer.contents, count * bytes_size);
        }
    }
    return TNN_OK;
}

Status MetalBlobConverterAcc::ConvertFromMat(Mat &image, MatConvertParam param, void *command_queue) {
    param_ = param;
    return ConvertFromMatCommon(image, blob_, command_queue, 1);
}

Status MetalBlobConverterAcc::ConvertFromMatAsync(Mat &image, MatConvertParam param, void *command_queue) {
    param_ = param;
    return ConvertFromMatCommon(image, blob_, command_queue, 2);
}

// #lizard forgives
Status MetalBlobConverterAcc::ConvertFromMatCommon(Mat &input_mat, Blob *output_blob, void *command_queue,
                                                   int waitState) {
    auto mat_device_type = input_mat.GetDeviceType();
    auto mat_type        = input_mat.GetMatType();

    if (!CheckDeviceAndMat(mat_device_type, mat_type)) {
        LOGE("GetDeviceType: %d GetMatType: %d\n", input_mat.GetDeviceType(), input_mat.GetMatType());
        return Status(TNNERR_COMMON_ERROR, "input_mat.GetDeviceType() or.GetMatType() is invalid");
    }

    auto dims               = blob_->GetBlobDesc().dims;
    auto command_queue_impl = (__bridge TNNMetalCommandQueueImpl *)(command_queue);
    if (!command_queue_impl) {
        return Status(TNNERR_INST_ERR, "command queue is nil");
    }

    auto context_impl = command_queue_impl.metalContextImpl;
    if (waitState == 1) {
        [context_impl commit:YES];
    }
    
    // check class type
    if (!output_blob || typeid(*output_blob) != typeid(Blob)) {
        LOGE("Error: output_blob is nil or not instance of Blob*\n");
        return Status(TNNERR_INST_ERR, "output_blob is nil or not instance of Blob*");
    }

    auto status = AllocateBufferParam(param_, &input_mat, output_blob, true);
    if (status != TNN_OK) {
        return status;
    }

    status = AllocateComputePipeline(param_, &input_mat, output_blob, true, command_queue);
    if (status != TNN_OK) {
        return status;
    }

    do {
        if (mat_type == N8UC4) {
            // For Texture input
            
            id<MTLTexture> input_texture = nil;
            if (mat_device_type == DEVICE_METAL) {
                input_texture = (__bridge id<MTLTexture>)(input_mat.GetData());
            } else if (mat_device_type == DEVICE_NAIVE || mat_device_type == DEVICE_ARM || mat_device_type == DEVICE_X86) {
                buffer_mat_ = std::make_shared<TNN_NS::Mat>(DEVICE_METAL, TNN_NS::N8UC4, dims);
                input_texture = (__bridge id<MTLTexture>)buffer_mat_->GetData();
                if (!input_texture) {
                    LOGE("Error: newTextureWithDescriptor return nil\n");
                    return Status(TNNERR_INST_ERR, "newTextureWithDescriptor return nil");
                }

                [input_texture replaceRegion:MTLRegionMake2D(0, 0, DimsFunctionUtils::GetDim(dims, 3), DimsFunctionUtils::GetDim(dims, 2))
                                 mipmapLevel:0
                                   withBytes:input_mat.GetData()
                                 bytesPerRow:DimsFunctionUtils::GetDim(dims, 3) * 4];
            } else {
                break;
            }

            Blob *output_buffer_blob = (Blob *)(output_blob);
            if (input_texture.height != DimsFunctionUtils::GetDim(dims, 2) || input_texture.width != DimsFunctionUtils::GetDim(dims, 3) ||
                (input_texture.pixelFormat != MTLPixelFormatBGRA8Unorm &&
                 input_texture.pixelFormat != MTLPixelFormatRGBA8Unorm)) {
                return Status(TNNERR_INST_ERR, "input mat's texture is invalid, wrong size or pixel format");
            }

            auto encoder = [context_impl encoder];
            if (!encoder) {
                LOGE("ERROR: ConvertFromMatCommon cannot allocate new encoder\n");
                return Status(TNNERR_PARAM_ERR, "ConvertFromMatCommon cannot allocate new encoder");
            }
            
            MetalBandwidth bandwidth;
            status = [context_impl load:pipeline_func_name_
                                encoder:encoder
                                bandwidth:bandwidth];
            RETURN_ON_NEQ(status, TNN_OK);
            
            MTLSize group_threads = {(NSUInteger)bandwidth.thread_execution_width, (NSUInteger)1, (NSUInteger)1};
            MTLSize groups        = {(NSUInteger)((DimsFunctionUtils::GetDim(dims, 3) + group_threads.width - 1) / group_threads.width),
                              (NSUInteger)DimsFunctionUtils::GetDim(dims, 2), (NSUInteger)1};
            
            [encoder setTexture:input_texture atIndex:0];
            [encoder setBuffer:(__bridge id<MTLBuffer>)(void *)output_buffer_blob->GetHandle().base
                        offset:(NSUInteger)output_buffer_blob->GetHandle().bytes_offset
                       atIndex:0];
            [encoder setBuffer:buffer_param_ offset:0 atIndex:1];

            [encoder dispatchThreadgroups:groups threadsPerThreadgroup:group_threads];
            [encoder endEncoding];
            
            auto command_buffer = context_impl.commandBuffer;
            [context_impl commit:YES];
            
            if (waitState == 1) {
                [command_buffer waitUntilCompleted];
            } else if (waitState == 2) {
                [command_buffer waitUntilScheduled];
            }
            return TNN_OK;
        } else if (mat_type == N8UC3 && mat_device_type != DEVICE_METAL) {
            const auto bytes_size = sizeof(unsigned char);
            auto count = DimsVectorUtils::Count(dims);
            id<MTLBuffer> input_tmp_buffer = [command_queue_impl.device newBufferWithBytes:input_mat.GetData()
                                                                      length:count * bytes_size
                                                                     options:MTLCPUCacheModeDefaultCache];

            auto encoder = [context_impl encoder];
            if (!encoder) {
                LOGE("ERROR: ConvertFromMatCommon cannot allocate new encoder\n");
                return Status(TNNERR_PARAM_ERR, "ConvertFromMatCommon cannot allocate new encoder");
            }
            
            MetalBandwidth bandwidth;
            status = [context_impl load:pipeline_func_name_
                                encoder:encoder
                                bandwidth:bandwidth];
            RETURN_ON_NEQ(status, TNN_OK);
            
            MTLSize group_threads = {(NSUInteger)bandwidth.thread_execution_width, (NSUInteger)1, (NSUInteger)1};
            MTLSize groups        = {(NSUInteger)((DimsFunctionUtils::GetDim(dims, 3) + group_threads.width - 1) / group_threads.width),
                              (NSUInteger)DimsFunctionUtils::GetDim(dims, 2), (NSUInteger)1};

            [encoder setBuffer:(__bridge id<MTLBuffer>)(void *)output_blob->GetHandle().base
                        offset:(NSUInteger)output_blob->GetHandle().bytes_offset
                       atIndex:0];
            [encoder setBuffer:input_tmp_buffer offset:0 atIndex:1];
            [encoder setBuffer:buffer_param_ offset:0 atIndex:2];

            [encoder dispatchThreadgroups:groups threadsPerThreadgroup:group_threads];
            [encoder endEncoding];

            auto command_buffer = context_impl.commandBuffer;
            [context_impl commit:YES];
            
            if (waitState == 1) {
                [command_buffer waitUntilCompleted];
            } else if (waitState == 2) {
                [command_buffer waitUntilScheduled];
            }
            return TNN_OK;
        } else if (mat_type == NGRAY || mat_type == NCHW_FLOAT || mat_type == RESERVED_BFP16_TEST) {
            // For Buffer input

            id<MTLBuffer> input_buffer = nil;
            const auto bytes_size = (mat_type == NCHW_FLOAT) ? sizeof(float) : ((mat_type == NGRAY) ? sizeof(unsigned char) : sizeof(fp16_t));

            if (mat_device_type == DEVICE_METAL) {
                input_buffer = (__bridge id<MTLBuffer>)(input_mat.GetData());
            } else if (mat_device_type == DEVICE_NAIVE || mat_device_type == DEVICE_ARM || mat_device_type == DEVICE_X86) {
                int count    = DimsVectorUtils::Count(dims);
                input_buffer = [command_queue_impl.device newBufferWithBytes:input_mat.GetData()
                                                                      length:count * bytes_size
                                                                     options:MTLCPUCacheModeDefaultCache];
            } else {
                break;
            }
            
            NSUInteger image_size  = DimsFunctionUtils::GetDimProduct(dims, 2);
            NSUInteger image_slice = UP_DIV(dims[1], 4);
            bool is_blob_nchw = output_blob->GetBlobDesc().data_format == DATA_FORMAT_NCHW;
            Blob *output_buffer_blob = (Blob *)(output_blob);

            auto encoder = [context_impl encoder];
            if (!encoder) {
                LOGE("ERROR: ConvertFromMatCommon cannot allocate new encoder\n");
                return Status(TNNERR_PARAM_ERR, "ConvertFromMatCommon cannot allocate new encoder");
            }
            
            MetalBandwidth bandwidth;
            status = [context_impl load:pipeline_func_name_
                                encoder:encoder
                                bandwidth:bandwidth];
            BREAK_IF(status != TNN_OK);
            
            auto group_threads = MTLSizeMake(bandwidth.thread_execution_width, 1, 1);
            auto groups = MTLSizeMake((image_size + group_threads.width - 1) / group_threads.width,
                                      image_slice, dims[0]);
            if (is_blob_nchw)
                groups.height = dims[1];

            if (image_size <= image_slice) {
                group_threads = MTLSizeMake(1, bandwidth.thread_execution_width, 1);
                groups = MTLSizeMake(image_size,
                                     (image_slice + group_threads.height - 1) / group_threads.height,
                                     dims[0]);
                if (is_blob_nchw)
                    groups.height = (dims[1] + group_threads.height - 1) / group_threads.height;
            }

            [encoder setBuffer:(__bridge id<MTLBuffer>)(void *)output_buffer_blob->GetHandle().base
                        offset:(NSUInteger)output_buffer_blob->GetHandle().bytes_offset
                       atIndex:0];
            [encoder setBuffer:input_buffer offset:0 atIndex:1];
            [encoder setBuffer:buffer_param_ offset:0 atIndex:2];
            //scale and bias
            [encoder setBuffer:buffer_scale_ offset:0 atIndex:3];
            [encoder setBuffer:buffer_bias_  offset:0 atIndex:4];

            [encoder dispatchThreadgroups:groups threadsPerThreadgroup:group_threads];
            [encoder endEncoding];
            
            auto command_buffer = context_impl.commandBuffer;
            [context_impl commit:YES];
            if (waitState == 1) {
                [command_buffer waitUntilCompleted];
            } else if (waitState == 2) {
                [command_buffer waitUntilScheduled];
            }
            return TNN_OK;
        } else if (mat_type == NC_INT32) {
            id<MTLBuffer> input_buffer = nil;
            const auto bytes_size = sizeof(int);
            if (mat_device_type == DEVICE_METAL) {
                input_buffer = (__bridge id<MTLBuffer>)(input_mat.GetData());
            } else if (mat_device_type == DEVICE_NAIVE || mat_device_type == DEVICE_ARM || mat_device_type == DEVICE_X86) {
                int count    = DimsVectorUtils::Count(dims);
                input_buffer = [command_queue_impl.device newBufferWithBytes:input_mat.GetData()
                                                                      length:count * bytes_size
                                                                     options:MTLCPUCacheModeDefaultCache];
            }
            
            NSUInteger image_size  = DimsFunctionUtils::GetDimProduct(dims, 2);
            NSUInteger image_slice = UP_DIV(dims[1], 4);
            bool is_blob_nchw = output_blob->GetBlobDesc().data_format == DATA_FORMAT_NCHW;
            Blob *output_buffer_blob = (Blob *)(output_blob);
            
            auto encoder = [context_impl encoder];
            if (!encoder) {
                LOGE("ERROR: ConvertFromMatCommon cannot allocate new encoder\n");
                return Status(TNNERR_PARAM_ERR, "ConvertFromMatCommon cannot allocate new encoder");
            }
            
            MetalBandwidth bandwidth;
            status = [context_impl load:pipeline_func_name_
                                encoder:encoder
                                bandwidth:bandwidth];
            BREAK_IF(status != TNN_OK);
            
            auto group_threads = MTLSizeMake(bandwidth.thread_execution_width, 1, 1);
            auto groups = MTLSizeMake((image_size + group_threads.width - 1) / group_threads.width,
                                      image_slice, dims[0]);
            if (is_blob_nchw) {
                groups.height = dims[1];
            }
            
            if (image_size <= image_slice) {
                group_threads = MTLSizeMake(1, bandwidth.thread_execution_width, 1);
                groups = MTLSizeMake(image_size,
                                     (image_slice + group_threads.height - 1) / group_threads.height,
                                     dims[0]);
                if (is_blob_nchw)
                    groups.height = (dims[1] + group_threads.height - 1) / group_threads.height;
            }
            
            [encoder setBuffer:(__bridge id<MTLBuffer>)(void *)output_buffer_blob->GetHandle().base
                        offset:(NSUInteger)output_buffer_blob->GetHandle().bytes_offset
                       atIndex:0];
            [encoder setBuffer:input_buffer offset:0 atIndex:1];
            [encoder setBuffer:buffer_param_ offset:0 atIndex:2];
            //scale and bias
            [encoder setBuffer:buffer_scale_ offset:0 atIndex:3];
            [encoder setBuffer:buffer_bias_  offset:0 atIndex:4];
            
            [encoder dispatchThreadgroups:groups threadsPerThreadgroup:group_threads];
            [encoder endEncoding];
            
            auto command_buffer = context_impl.commandBuffer;
            [context_impl commit:YES];
            if (waitState == 1) {
                [command_buffer waitUntilCompleted];
            } else if (waitState == 2) {
                [command_buffer waitUntilScheduled];
            }
            return TNN_OK;
        } else {
            break;
        }
    } while (0);
    return Status(TNNERR_COMMON_ERROR, "input_mat.GetDeviceType() or.GetMatType() is invalid");
}

DECLARE_BLOB_CONVERTER_CREATER(Metal);
REGISTER_BLOB_CONVERTER(Metal, DEVICE_METAL);

} // namespace TNN_NS
