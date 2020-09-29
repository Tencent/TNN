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
#import "tnn/device//metal/acc/metal_common.h"
#import "tnn/device//metal/metal_command_queue.h"
#import "tnn/device//metal/metal_context.h"
#import "tnn/utils/blob_converter_internal.h"
#import "tnn/utils/data_type_utils.h"
#include "tnn/utils/dims_vector_utils.h"

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
    id<MTLBuffer> buffer_param_                   = nil;
    id<MTLComputePipelineState> pipeline_process_ = nil;
    // buffer for scale and bias used in nchw_float mat transformation
    id<MTLBuffer> buffer_scale_;
    id<MTLBuffer> buffer_bias_;
    // @param waitState: 0: no wait, 1: wait gpu completed, 2: wait gpu scheduled.
    Status ConvertToMatCommon(Mat &input_mat, Blob *output_blob, void *command_queue, int waitState = 1);
    Status ConvertFromMatCommon(Mat &input_mat, Blob *output_blob, void *command_queue, int waitState = 1);

    Status AllocateBufferParam(MatConvertParam param, Mat *mat, Blob *blob, bool is_mat_to_blob);
    Status AllocateComputePipeline(MatConvertParam param, Mat *mat, Blob *blob, bool is_mat_to_blob,
                                   void *command_queue);
};

MetalBlobConverterAcc::MetalBlobConverterAcc(Blob *blob) : BlobConverterAcc(blob) {
    auto buffer = (__bridge id<MTLBuffer>)(blob->GetHandle().base);
    device_     = buffer.device;
}

Status MetalBlobConverterAcc::AllocateBufferParam(MatConvertParam param, Mat *mat, Blob *blob, bool is_mat_to_blob) {
    auto dims = blob->GetBlobDesc().dims;
    MetalImageConverterParams metal_param;
    metal_param.width        = dims[3];
    metal_param.height       = dims[2];
    metal_param.size         = metal_param.height * metal_param.width;
    metal_param.channel      = dims[1];
    metal_param.slice        = UP_DIV(metal_param.channel, 4);
    metal_param.batch        = dims[0];
    metal_param.bgra_to_rgba = param.reverse_channel;

    LOGD("metal_param size: %d %d %d %d %d\n", metal_param.batch, metal_param.channel, metal_param.height,
         metal_param.width, metal_param.size);

    float scale_texture_buffer = 1.0f;
    float bias_texture_buffer  = 1.0f;
    if (mat->GetDeviceType() == DEVICE_METAL) {
        if (mat->GetMatType() == N8UC4) {
            scale_texture_buffer = is_mat_to_blob ? 255.0f : 1.0 / 255.0f;
            bias_texture_buffer  = is_mat_to_blob ? 1.0    : 1.0 / 255.0f;
        }
    }

    if (mat->GetMatType() == NCHW_FLOAT) {
        // scale and bias should at least have channel elements, so we use another buffer instead of metal_param
        if (param.scale.size() < dims[1] || param.bias.size() < dims[1]) {
            // invalid scale and bias
            return Status(TNNERR_INVALID_INPUT, "invalid scale or bias shape!");
        }
        if (buffer_scale_ == nil) {
            buffer_scale_ = [device_ newBufferWithBytes:&(param.scale[0])
                                                 length:sizeof(float)*dims[1]
                                                options:MTLResourceCPUCacheModeWriteCombined];
        }
        if (buffer_bias_ == nil) {
            buffer_bias_ = [device_ newBufferWithBytes:&(param.bias[0])
                                                 length:sizeof(float)*dims[1]
                                                options:MTLResourceCPUCacheModeWriteCombined];
        }
        if (buffer_scale_ == nil) {
            return Status(TNNERR_INVALID_INPUT, "buffer scale is nil");
        }
        if (buffer_bias_ == nil) {
            return Status(TNNERR_INVALID_INPUT, "buffer bias is nil");
        }
    } else {
        if (param.scale.size() >= 4) {
            metal_param.scale_x = scale_texture_buffer * param.scale[0];
            metal_param.scale_y = scale_texture_buffer * param.scale[1];
            metal_param.scale_z = scale_texture_buffer * param.scale[2];
            metal_param.scale_w = scale_texture_buffer * param.scale[3];
        } else {
            metal_param.scale_x = 1.0f;
            metal_param.scale_y = 1.0f;
            metal_param.scale_z = 1.0f;
            metal_param.scale_w = 1.0f;
        }
        if (param.bias.size() >= 4) {
            metal_param.bias_x = bias_texture_buffer * param.bias[0];
            metal_param.bias_y = bias_texture_buffer * param.bias[1];
            metal_param.bias_z = bias_texture_buffer * param.bias[2];
            metal_param.bias_w = bias_texture_buffer * param.bias[3];
        } else {
            metal_param.bias_x = 0.0f;
            metal_param.bias_y = 0.0f;
            metal_param.bias_z = 0.0f;
            metal_param.bias_w = 0.0f;
        }
    }

    LOGD("metal_param scale: %.6f %.6f %.6f\n", metal_param.scale_x, metal_param.scale_y, metal_param.scale_z);
    LOGD("metal_param size: %d %d %d\n", metal_param.size, metal_param.slice, metal_param.batch);

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

    id<MTLFunction> func_process = nil;

    // texture <-> blob
    if (mat_type == N8UC4) {
        if (is_mat_to_blob) {
            if (blob_data_format == DATA_FORMAT_NC4HW4) {
                func_process = [library newFunctionWithName:@"image_converter_texture_bgra8888_2_buffer_nc4hw4"];
                LOGD("image_converter_texture_bgra8888_2_buffer_nc4hw4\n");
            } else if (blob_data_format == DATA_FORMAT_NCHW) {
                if (blob_data_type == DATA_TYPE_FLOAT) {
                    func_process = [library newFunctionWithName:@"image_converter_texture_bgra8888_2_buffer_nchw_f"];
                    LOGD("image_converter_texture_bgra8888_2_buffer_nchw_f\n");
                }
            }
        } else {
            if (blob_data_format == DATA_FORMAT_NC4HW4) {
                func_process = [library newFunctionWithName:@"image_converter_buffer_nc4hw4_2_texture_bgra8888"];
                LOGD("image_converter_buffer_nc4hw4_2_texture_bgra8888\n");
            } else if (blob_data_format == DATA_FORMAT_NCHW) {
                if (blob_data_type == DATA_TYPE_FLOAT) {
                    func_process = [library newFunctionWithName:@"image_converter_buffer_nchw_f_2_texture_bgra8888"];
                    LOGD("image_converter_buffer_nchw_f_2_texture_bgra8888\n");
                }
            }
        }
    } else if (mat_type == NCHW_FLOAT) {
        if (is_mat_to_blob) {
            func_process = [library newFunctionWithName:@"data_converter_nchw_2_nc4hw4_float_v2"];
            LOGD("data_converter_nchw_2_nc4hw4_float_v2\n");
        } else {
            func_process = [library newFunctionWithName:@"data_converter_nc4hw4_2_nchw_float_v2"];
            LOGD("data_converter_nc4hw4_2_nchw_float_v2\n");
        }
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
    if (!((mat_device_type == DEVICE_METAL || mat_device_type == DEVICE_ARM || mat_device_type == DEVICE_NAIVE) &&
          (mat_type == N8UC4 || mat_type == NCHW_FLOAT))) {
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
        MTLSize group_threads = {(NSUInteger)pipeline_process_.threadExecutionWidth, (NSUInteger)1, (NSUInteger)1};
        MTLSize groups = {(NSUInteger)((dims[3] + group_threads.width - 1) / group_threads.width), (NSUInteger)dims[2],
                          (NSUInteger)1};

        auto output_texture       = (__bridge id<MTLTexture>)(output_mat.GetData());
        Blob *input_buffer_blob = (Blob *)(input_blob);
        if (output_texture.height != dims[2] || output_texture.width != dims[3] ||
            (output_texture.pixelFormat != MTLPixelFormatBGRA8Unorm &&
             output_texture.pixelFormat != MTLPixelFormatRGBA8Unorm)) {
            return Status(TNNERR_INST_ERR, "output mat's texture is invalid, wrong size or pixel format");
        }

        command_buffer = [command_queue_impl commandBuffer];
        [command_buffer enqueue];
        auto encoder = [command_buffer computeCommandEncoder];
        [encoder setComputePipelineState:pipeline_process_];

        [encoder setTexture:output_texture atIndex:0];
        [encoder setBuffer:(__bridge id<MTLBuffer>)(void *)input_buffer_blob->GetHandle().base
                    offset:(NSUInteger)input_buffer_blob->GetHandle().bytes_offset
                   atIndex:0];
        [encoder setBuffer:buffer_param_ offset:0 atIndex:1];

        [encoder dispatchThreadgroups:groups threadsPerThreadgroup:group_threads];
        [encoder endEncoding];

        [command_buffer commit];
        if (waitState == 1) {
            [command_buffer waitUntilCompleted];
        } else if (waitState == 2) {
            [command_buffer waitUntilScheduled];
        }
    } else if (mat_type == NCHW_FLOAT) {
        auto input_buffer_blob          = dynamic_cast<Blob *>(input_blob);
        id<MTLBuffer> output_mtl_buffer = nil;

        int count = DimsVectorUtils::Count(dims);
        if (output_mat_device == DEVICE_METAL) {
            output_mtl_buffer = (__bridge id<MTLBuffer>)(output_mat.GetData());
        } else if (output_mat_device == DEVICE_ARM || output_mat_device == DEVICE_NAIVE) {
            output_mtl_buffer = [command_queue_impl.device newBufferWithLength:count * sizeof(float)
                                                                       options:MTLResourceCPUCacheModeDefaultCache];
        }

        NSUInteger image_size  = dims[2] * dims[3];
        NSUInteger image_slice = UP_DIV(dims[1], 4);

        auto group_threads = MTLSizeMake(pipeline_process_.threadExecutionWidth, 1, 1);
        auto groups = MTLSizeMake((image_size + group_threads.width - 1) / group_threads.width,
                                  image_slice, dims[0]);

        if (image_size <= image_slice) {
            group_threads = MTLSizeMake(1, pipeline_process_.threadExecutionWidth, 1);
            groups = MTLSizeMake(image_size,
                                 (image_slice + group_threads.height - 1) / group_threads.height,
                                 dims[0]);
        }

        command_buffer = [command_queue_impl commandBuffer];
        [command_buffer enqueue];
        auto encoder = [command_buffer computeCommandEncoder];
        [encoder setComputePipelineState:pipeline_process_];

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

        [command_buffer commit];

        if (output_mat_device == DEVICE_METAL) {
            if (waitState == 1) {
                [command_buffer waitUntilCompleted];
            } else if (waitState == 2) {
                [command_buffer waitUntilScheduled];
            }
        } else {
            [command_buffer waitUntilCompleted];
            memcpy(output_mat.GetData(), output_mtl_buffer.contents, count * sizeof(float));
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
    if (!((mat_device_type == DEVICE_METAL || mat_device_type == DEVICE_ARM || mat_device_type == DEVICE_NAIVE) &&
          (mat_type == N8UC4 || mat_type == NCHW_FLOAT))) {
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
            MTLSize group_threads = {(NSUInteger)pipeline_process_.threadExecutionWidth, (NSUInteger)1, (NSUInteger)1};
            MTLSize groups        = {(NSUInteger)((dims[3] + group_threads.width - 1) / group_threads.width),
                              (NSUInteger)dims[2], (NSUInteger)1};

            id<MTLTexture> input_texture = nil;
            if (mat_device_type == DEVICE_METAL) {
                input_texture = (__bridge id<MTLTexture>)(input_mat.GetData());
            } else if (mat_device_type == DEVICE_NAIVE || mat_device_type == DEVICE_ARM) {
                return Status(TNNERR_COMMON_ERROR, "input_mat.GetDeviceType() or.GetMatType() is invalid");
                // now this will not work, disable first
                TNN_NS::Mat image_mat_gpu(DEVICE_METAL, TNN_NS::N8UC4, dims);
                input_texture = (__bridge id<MTLTexture>)image_mat_gpu.GetData();
                if (!input_texture) {
                    LOGE("Error: newTextureWithDescriptor return nil\n");
                    return Status(TNNERR_INST_ERR, "newTextureWithDescriptor return nil");
                }

                [input_texture replaceRegion:MTLRegionMake2D(0, 0, dims[3], dims[2])
                                 mipmapLevel:0
                                   withBytes:input_mat.GetData()
                                 bytesPerRow:dims[3] * 4];
            } else {
                break;
            }

            Blob *output_buffer_blob = (Blob *)(output_blob);
            if (input_texture.height != dims[2] || input_texture.width != dims[3] ||
                (input_texture.pixelFormat != MTLPixelFormatBGRA8Unorm &&
                 input_texture.pixelFormat != MTLPixelFormatRGBA8Unorm)) {
                return Status(TNNERR_INST_ERR, "input mat's texture is invalid, wrong size or pixel format");
            }

            auto command_buffer = [command_queue_impl commandBuffer];
            [command_buffer enqueue];
            auto encoder = [command_buffer computeCommandEncoder];
            [encoder setComputePipelineState:pipeline_process_];

            [encoder setTexture:input_texture atIndex:0];
            [encoder setBuffer:(__bridge id<MTLBuffer>)(void *)output_buffer_blob->GetHandle().base
                        offset:(NSUInteger)output_buffer_blob->GetHandle().bytes_offset
                       atIndex:0];
            [encoder setBuffer:buffer_param_ offset:0 atIndex:1];

            [encoder dispatchThreadgroups:groups threadsPerThreadgroup:group_threads];
            [encoder endEncoding];

            [command_buffer commit];
            if (waitState == 1) {
                [command_buffer waitUntilCompleted];
            } else if (waitState == 2) {
                [command_buffer waitUntilScheduled];
            }
            return TNN_OK;
        } else if (mat_type == NCHW_FLOAT) {
            // For Buffer input

            id<MTLBuffer> input_buffer = nil;
            if (mat_device_type == DEVICE_METAL) {
                input_buffer = (__bridge id<MTLBuffer>)(input_mat.GetData());
            } else if (mat_device_type == DEVICE_NAIVE || mat_device_type == DEVICE_ARM) {
                int count    = DimsVectorUtils::Count(dims);
                input_buffer = [command_queue_impl.device newBufferWithBytes:input_mat.GetData()
                                                                      length:count * sizeof(float)
                                                                     options:MTLCPUCacheModeDefaultCache];
            } else {
                break;
            }

            NSUInteger image_size  = dims[2] * dims[3];
            NSUInteger image_slice = UP_DIV(dims[1], 4);

            auto group_threads = MTLSizeMake(pipeline_process_.threadExecutionWidth, 1, 1);
            auto groups = MTLSizeMake((image_size + group_threads.width - 1) / group_threads.width,
                                      image_slice, dims[0]);

            if (image_size <= image_slice) {
                group_threads = MTLSizeMake(1, pipeline_process_.threadExecutionWidth, 1);
                groups = MTLSizeMake(image_size,
                                     (image_slice + group_threads.height - 1) / group_threads.height,
                                     dims[0]);
            }

            Blob *output_buffer_blob = (Blob *)(output_blob);

            auto command_buffer = [command_queue_impl commandBuffer];
            [command_buffer enqueue];
            auto encoder = [command_buffer computeCommandEncoder];
            [encoder setComputePipelineState:pipeline_process_];

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

            [command_buffer commit];
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
