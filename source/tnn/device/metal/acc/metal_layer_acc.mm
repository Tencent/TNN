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

#include "tnn/device/metal/acc/metal_layer_acc.h"
#include "tnn/device/metal/acc/metal_common.h"
#include "tnn/device/metal/metal_context.h"
#include "tnn/utils/data_format_converter.h"
#include "tnn/utils/data_type_utils.h"
#include "tnn/utils/half_utils_inner.h"
#include "tnn/utils/dims_utils.h"

namespace TNN_NS {

Status MetalLayerAcc::Init(Context *context, LayerParam *param, LayerResource *resource,
                           const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto status = AbstractLayerAcc::Init(context, param, resource, inputs, outputs);
    RETURN_ON_NEQ(status, TNN_OK);

    context_ = dynamic_cast<MetalContext *>(context);

    param_    = param;
    resource_ = resource;

    status = UpdateBlobDataType(inputs, outputs);
    RETURN_ON_NEQ(status, TNN_OK);

    status = ReloadConstantBlobs(inputs, false);
    RETURN_ON_NEQ(status, TNN_OK);

    return Reshape(inputs, outputs);
    //    return Reshape(inputs, outputs);
}

MetalLayerAcc::~MetalLayerAcc() {
    buffer_param_ = nil;
}

Status MetalLayerAcc::UpdateBlobDataType(const std::vector<Blob *> &inputs,
                                   const std::vector<Blob *> &outputs) {
    //修正BlobManager::Init中设置的data_type
    // metal 运行时只支持half，debug模式支持fp32
#if TNN_METAL_FULL_PRECISION
    inputs[0]->GetBlobDesc().data_type  = DATA_TYPE_FLOAT;
    outputs[0]->GetBlobDesc().data_type = DATA_TYPE_FLOAT;
#else
    inputs[0]->GetBlobDesc().data_type  = DATA_TYPE_HALF;
    outputs[0]->GetBlobDesc().data_type = DATA_TYPE_HALF;
#endif
    return TNN_OK;
}

Status MetalLayerAcc::Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    return AllocateBufferParam(inputs, outputs);
}

Status MetalLayerAcc::ReloadConstantBlobs(const std::vector<Blob *> &inputs, bool only_reload_shape_differ_blob) {
    auto const_resource = const_resource_;
    auto const_resource_flag = const_resource_flag_;
    auto const_blob_map = const_blob_map_;
    for (auto iter : inputs) {
        auto name = iter->GetBlobDesc().name;
        if (const_resource == nullptr || const_resource->find(name) == const_resource->end()) {
            continue;
        }
        if (only_reload_shape_differ_blob && const_resource_flag &&
            const_resource_flag->find(name) == const_resource_flag->end()) {
            continue;
        }

        auto buffer = (*const_resource)[name];
        std::shared_ptr<Blob> blob = nullptr;
        if (const_blob_map.find(name) != const_blob_map.end()) {
            blob = const_blob_map[name];
        }

        Status status = TNN_OK;
        if (UseNaiveConstantBlobs()) {
            status = RawBuffer2Blob(buffer.get(), blob);
        } else {
            status = RawBuffer2MetalBlob(buffer.get(), blob, iter->GetBlobDesc());
        }
        RETURN_ON_NEQ(status, TNN_OK);

        blob->SetFlag(DATA_FLAG_CHANGE_NEVER);
        const_blob_map[name] = blob;
        iter->SetHandle(blob->GetHandle());
        iter->GetBlobDesc() = blob->GetBlobDesc();
        iter->SetFlag(blob->GetFlag());
        LOGD("Reload constant blob: %s\n", name.c_str());
    }
    const_blob_map_ = const_blob_map;
    return TNN_OK;
}

Status MetalLayerAcc::RawBuffer2MetalBlob(RawBuffer *buffer, std::shared_ptr<Blob> &blob, BlobDesc &desc) {
    if (!buffer || buffer->GetBytesSize() <= 0) {
        LOGE("RawBuffer2MetalBlob: buffer for blob:%s is null\n", desc.name.c_str());
        return Status(TNNERR_PARAM_ERR, "RawBuffer2MetalBlob: buffer is null");
    }

    if (buffer->GetDataType() != DATA_TYPE_FLOAT && buffer->GetDataType() != DATA_TYPE_HALF) {
        LOGE("RawBuffer2MetalBlob: unsupported buffer data type: %d\n", buffer->GetDataType());
        return Status(TNNERR_PARAM_ERR, "RawBuffer2MetalBlob: unsupported buffer data type");
    }

    const int count = blob ? DimsVectorUtils::Count(blob->GetBlobDesc().dims) : 0;
    const int ele_size = blob ? DataTypeUtils::GetBytesSize(blob->GetBlobDesc().data_type) : 0;
    if (!blob || buffer->GetBytesSize() != count*ele_size) {
        desc.device_type = DEVICE_METAL;
#if TNN_METAL_FULL_PRECISION
        desc.data_type  = DATA_TYPE_FLOAT;
#else
        desc.data_type  = DATA_TYPE_HALF;
#endif

        desc.data_format = DATA_FORMAT_NC4HW4;
        ConfigBuffer2MetalBlobDesc(desc);

        if (buffer->GetBytesSize() > 0) {
            blob = std::make_shared<Blob>(desc, true);
        } else {
            blob = std::make_shared<Blob>(desc, false);
        }
    }

    if (!blob->GetHandle().base) {
        LOGE("RawBuffer2MetalBlob: Allocate blob failed!\n");
        return Status(TNNERR_INST_ERR, "RawBuffer2MetalBlob: Allocate blob failed!");
    }

    // convert data type if necessary
    void *buffer_data = buffer->force_to<void *>();
    auto buffer_size  = buffer->GetBytesSize();
    float *buffer_float = nullptr;
    uint16_t *buffer_half = nullptr;
    LOGD("ReloadConstantBlob for layer:%s\n", param_->name.c_str());
#if TNN_METAL_FULL_PRECISION
    if (buffer->GetDataType() == DATA_TYPE_HALF) {
        auto buffer_data_count = buffer->GetDataCount();
        buffer_float = new float[buffer_data_count];
        if (ConvertFromHalfToFloat(buffer_data, buffer_float, buffer_data_count) != 0) {
            LOGE("RawBuffer2MetalBlob: Convert buffer from half to float failed!");
            return Status(TNNERR_INST_ERR, "RawBuffer2MetalBlob: Convert buffer from half to float failed!");
        }
        buffer_data = buffer_float;
        buffer_size = sizeof(float) * buffer_data_count;
        LOGE("ReloadConstantBlob for layer:%s, convert from half to float!\n", param_->name.c_str());
    }
#else
    if (buffer->GetDataType() == DATA_TYPE_FLOAT) {
        auto buffer_data_count = buffer->GetDataCount();
        buffer_half = new uint16_t[buffer_data_count];
        if (ConvertFromFloatToHalf((float *)buffer_data, buffer_half, buffer_data_count) != 0) {
            LOGE("RawBuffer2MetalBlob: Convert buffer from float to half failed!");
            return Status(TNNERR_INST_ERR, "RawBuffer2MetalBlob: Convert buffer from float to half failed!");
        }
        buffer_data = buffer_half;
        buffer_size = sizeof(uint16_t) * buffer_data_count;
    }
#endif

    Status status = TNN_OK;
    // copy to metal blob
    id<MTLDevice> device = [TNNMetalDeviceImpl sharedDevice];
    id<MTLBuffer> tmp_buffer = nil;
    tmp_buffer = [device newBufferWithBytes:buffer_data
                                     length:buffer_size
                                    options:MTLResourceCPUCacheModeWriteCombined];
    if (buffer_float) delete[] buffer_float;
    if (buffer_half) delete[] buffer_half;

    auto context_impl = context_->getMetalContextImpl();
    // _commandBuffer may be nil, call 'commit' to initialize a commandBuffer
    [context_impl commit:YES];

    auto encoder = [context_impl encoder];
    if (!encoder) {
        LOGE("ERROR: RawBuffer2MetalBlob cannot allocate new encoder for blob with name (%s) \n", desc.name.c_str());
        return Status(TNNERR_PARAM_ERR, "RawBuffer2MetalBlob cannot allocate new encoder for blob");
    }

    MetalImageConverterParams params;
    const auto blob_dims = blob->GetBlobDesc().dims;
    params.size    = DimsFunctionUtils::GetDimProduct(blob_dims, 2);
    params.channel = DimsFunctionUtils::GetDim(blob_dims, 1);
    params.batch   = DimsFunctionUtils::GetDim(blob_dims, 0);
    params.slice   = UP_DIV(params.channel, 4);

    id<MTLBuffer> buffer_param = [device newBufferWithBytes:(const void *)(&params)
                                        length:sizeof(params)
                                       options:MTLResourceCPUCacheModeWriteCombined];

    MTLSize threads;
    std::string kernel_name;
    if (desc.data_format == DATA_FORMAT_NCHW) {
        threads = MTLSizeMake(params.size, params.channel, params.batch);
        kernel_name = "data_converter_nchw";
    } else if (desc.data_format == DATA_FORMAT_NC4HW4) {
        threads = MTLSizeMake(params.size, params.slice, params.batch);
        kernel_name = "data_converter_nchw_2_nc4hw4_ftype_identity";
    } else {
        LOGE("RawBuffer2MetalBlob: unsupported data_format: %d\n", desc.data_format);
        return Status(TNNERR_PARAM_ERR, "RawBuffer2MetalBlob: unsupported data_format");
    }
    
    MetalBandwidth bandwidth;
    status = [context_impl load:[NSString stringWithUTF8String:kernel_name.c_str()]
                        encoder:encoder
                        bandwidth:bandwidth];
    RETURN_ON_NEQ(status, TNN_OK);

    [encoder setBuffer:(__bridge id<MTLBuffer>)(void *)blob->GetHandle().base
                offset:(NSUInteger)blob->GetHandle().bytes_offset
                atIndex:0];
    [encoder setBuffer:tmp_buffer offset:0 atIndex:1];
    [encoder setBuffer:buffer_param offset:0 atIndex:2];

    status = [context_impl dispatchEncoder:encoder threads:threads bandwidth:bandwidth];
    RETURN_ON_NEQ(status, TNN_OK);
    [encoder endEncoding];
    [context_impl commit];

    return status;
}

Status MetalLayerAcc::ConfigBuffer2MetalBlobDesc(BlobDesc &desc) {
    return TNN_OK;
}

bool MetalLayerAcc::UseNaiveConstantBlobs() {
    return false;
}

Status MetalLayerAcc::AllocateBufferParam(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    id<MTLDevice> device = [TNNMetalDeviceImpl sharedDevice];
    auto dims_input      = inputs[0]->GetBlobDesc().dims;
    auto dims_output     = outputs[0]->GetBlobDesc().dims;

    // buffer_param_
    {
        auto metal_params = GetDefaultMetalParams(dims_input, dims_output);
        buffer_param_     = [device newBufferWithBytes:(const void *)(&metal_params)
                                            length:sizeof(metal_params)
                                           options:MTLResourceCPUCacheModeWriteCombined];
    }
    return TNN_OK;
}

std::string MetalLayerAcc::KernelName(const std::vector<Blob *> &inputs,
                                      const std::vector<Blob *> &outputs) {
    LOGE("Error: subclass must implement the interface KernelName\n");
    return "";
}

Status MetalLayerAcc::ComputeThreadSize(const std::vector<Blob *> &inputs,
                                        const std::vector<Blob *> &outputs,
                                        MTLSize &size) {
    auto output = outputs[0];
    auto dims_output  = output->GetBlobDesc().dims;
    size = GetDefaultThreadSize(dims_output, true);
    return TNN_OK;
}

/*
  If an acc prefers to dispatch kernel with threadsPerGroup and threadGroups specified,
  it should override this method to give how many threadGroups to use, and it should also
  override the @ComputeThreadSize method to give threadsPerGroup.
  Use this implementaion means dispatching kernels without caring about the threadGroup config.
*/
Status MetalLayerAcc::ComputeThreadgroupSize(const std::vector<Blob *> &inputs,
                                     const std::vector<Blob *> &outputs,
                                     MTLSize &size) {
    size = MTLSizeMake(0, 0, 0);
    return TNN_OK;
}

Status MetalLayerAcc::SetKernelEncoderParam(
                                            id<MTLComputeCommandEncoder> encoder,
                                            const std::vector<Blob *> &inputs,
                                            const std::vector<Blob *> &outputs) {
    auto input  = inputs[0];
    auto output = outputs[0];
    
    [encoder setBuffer:(__bridge id<MTLBuffer>)(void *)input->GetHandle().base
                offset:(NSUInteger)input->GetHandle().bytes_offset
               atIndex:0];
    [encoder setBuffer:(__bridge id<MTLBuffer>)(void *)output->GetHandle().base
                offset:(NSUInteger)output->GetHandle().bytes_offset
               atIndex:1];
    [encoder setBuffer:buffer_param_ offset:0 atIndex:2];
    
    return TNN_OK;
}

NSString * MetalLayerAcc::GetKernelLabel() {
    if (kernel_label_.length > 0) {
        return kernel_label_;
    } else if ((!kernel_label_ || kernel_label_.length <= 0) && param_) {
        kernel_label_ = [NSString stringWithUTF8String:param_->name.c_str()];
    }
    return kernel_label_;
}

Status MetalLayerAcc::Forward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto data_type = outputs[0]->GetBlobDesc().data_type;
    auto data_type_str = DataTypeUtils::GetDataTypeString(data_type);
    if (data_type != DATA_TYPE_FLOAT && data_type != DATA_TYPE_HALF && data_type != DATA_TYPE_INT32) {
        LOGE("MetalLayerAcc: DataType must be float or half or int32\n");
        return Status(TNNERR_LAYER_ERR, "MetalLayerAcc: DataType must be float or half or int32");
    }

    
    //
    auto context_impl = context_->getMetalContextImpl();
    auto encoder = [context_impl encoder];
    encoder.label = GetKernelLabel();
    
    MTLSize threads;
    auto status = ComputeThreadSize(inputs, outputs, threads);
    if (status != TNN_OK) {
        return status;
    }
    // check if perferring to launch kernel with threadGroups specified
    MTLSize groups;
    status = ComputeThreadgroupSize(inputs, outputs, groups);
    bool preferDispatchingWithGroups = (groups.width!=0 && groups.height!=0 && groups.depth!=0);
    if (status != TNN_OK) {
        return status;
    }
    
    do {
        auto kernel_name = KernelName(inputs, outputs);
        if (kernel_name.length() <= 0) {
            status = Status(TNNERR_LAYER_ERR, "empty kernel name");
            break;
        }
        
        MetalBandwidth bandwidth;
        status = [context_impl load:[NSString stringWithUTF8String:kernel_name.c_str()]
                            encoder:encoder
                          bandwidth:bandwidth];
        BREAK_IF(status != TNN_OK);
        
        status = SetKernelEncoderParam(encoder, inputs, outputs);
        BREAK_IF(status != TNN_OK);
        if (preferDispatchingWithGroups) {
            status = [context_impl dispatchEncoder:encoder threadsPerGroup:threads groups: groups bandwidth:bandwidth];
        } else {
            status = [context_impl dispatchEncoder:encoder threads:threads bandwidth:bandwidth];
        }
        BREAK_IF(status != TNN_OK);
    } while (0);

    [encoder endEncoding];
    
    if (status == TNN_OK) {
        [context_impl commit];
        TNN_PRINT_ENCODER(context_, encoder, this);
    }
    return status;
}

std::vector<DataFormat> MetalLayerAcc::SupportDataFormat(DataType data_type, int dims_size, BlobType blob_type) {
    std::vector<DataFormat> support_list;
    if (dims_size == 4) {
        support_list.push_back(DATA_FORMAT_NC4HW4);
    } else {
        support_list.push_back(DATA_FORMAT_NC4HW4);
    }
    return support_list;
}

MTLSize GetDefaultThreadSize(DimsVector dims, bool combineHeightWidth) {
    auto output_height  = DimsFunctionUtils::GetDim(dims, 2);
    auto output_width   = DimsFunctionUtils::GetDimProduct(dims, 3);
    auto output_size  = output_width * output_height;
    auto output_slice = UP_DIV(dims[1], 4);
    auto output_batch = dims[0];
    
    if (combineHeightWidth) {
        return MTLSizeMake(output_size, output_slice, output_batch);
    } else {
        return MTLSizeMake(output_width, output_height, output_batch*output_slice);
    }
}

MTLSize GetDefaultThreadSizeFusedLast(DimsVector dims, bool combineHeightWidth) {
    auto output_height  = DimsFunctionUtils::GetDim(dims, 2);
    auto output_width   = DimsFunctionUtils::GetDimProduct(dims, 3);
    auto output_size  = output_width * output_height;
    auto output_slice = UP_DIV(dims[1], 4);
    auto output_batch = dims[0];
    
    if (combineHeightWidth) {
        return MTLSizeMake(output_size, output_slice, output_batch);
    } else {
        return MTLSizeMake(output_width, output_height, output_batch*output_slice);
    }
}

void GetSingleAxisSplitSize(const DimsVector& dims, int axis, MTLSize& size, bool reduce_on_axis) {
    auto axis_size = DimsFunctionUtils::GetDim(dims, axis);
    auto dims_copy = dims;
    dims_copy[1] = UP_DIV(dims[1], 4);
    size = MTLSizeMake(DimsVectorUtils::Count(dims_copy, axis+1),
                        reduce_on_axis? 1 : (axis == 1? UP_DIV(axis_size, 4) : axis_size),
                        DimsVectorUtils::Count(dims_copy, 0, axis));
}

struct MetalParams GetDefaultMetalParams(DimsVector dims_input, DimsVector dims_output) {
    MetalParams metal_params;
    SetDefaultMetalParams(metal_params, dims_input, dims_output);
    return metal_params;
}

id<MTLBuffer> AllocateMetalBufferFormRawBuffer1D(RawBuffer buffer, int count, Status &status) {
    id<MTLDevice> device     = [TNNMetalDeviceImpl sharedDevice];
    id<MTLBuffer> mtl_buffer = nil;

    // ensure count >= 16/2
    count = std::max(count, 16 / 2);

    const float *b_handle_data = buffer.force_to<float *>();
    if (b_handle_data == nullptr) {
        LOGE("ERROR: Data is nil \n");
        return nil;
    }
    const int b_handle_size   = buffer.GetBytesSize();
    const DataType data_type  = buffer.GetDataType();
    const int data_count_4    = ROUND_UP(count, 4);
    const int total_byte_size = data_count_4 * DataTypeUtils::GetBytesSize(data_type);

    if (data_type != DATA_TYPE_FLOAT && data_type != DATA_TYPE_HALF) {
        LOGE("Error: DataType %d not support\n", data_type);
        status = Status(TNNERR_MODEL_ERR, "bias_handle DataType is not supported");
        return mtl_buffer;
    }
    if (total_byte_size < b_handle_size) {
        LOGE("Error: Invalid model, buffer has wrong byte size\n");
        status = Status(TNNERR_MODEL_ERR,  "Error: Invalid model, buffer has wrong byte size");
        return mtl_buffer;
    }

#if TNN_METAL_FULL_PRECISION
    if (data_type == DATA_TYPE_FLOAT) {
        //补齐
        float *data_fill_4 = (float *)b_handle_data;
        if (total_byte_size != b_handle_size) {
            data_fill_4 = (float *)new char[total_byte_size];
            memset((void *)data_fill_4, 0, total_byte_size);
            memcpy(data_fill_4, b_handle_data, b_handle_size);
        }

        mtl_buffer = [device newBufferWithBytes:(const void *)data_fill_4
                                         length:total_byte_size
                                        options:MTLResourceCPUCacheModeWriteCombined];

        if (total_byte_size != b_handle_size) {
            delete[] data_fill_4;
        }
    } else if (data_type == DATA_TYPE_HALF) {
        //补齐
        uint16_t *data_fill_4 = (uint16_t *)b_handle_data;
        if (total_byte_size != b_handle_size) {
            data_fill_4 = (uint16_t *)new char[total_byte_size];
            memset((void *)data_fill_4, 0, total_byte_size);
            memcpy(data_fill_4, b_handle_data, b_handle_size);
        }

        // convert to float
        float *data_fp32_data = new float[data_count_4];
        if (ConvertFromHalfToFloat((void *)data_fill_4, data_fp32_data, data_count_4) != 0) {
            LOGE("Error: DataType %d not support\n", data_type);
            status = Status(TNNERR_MODEL_ERR, "Convert LayerRerouece from half to float failed!");
            return nil;
        }

        mtl_buffer = [device newBufferWithBytes:(const void *)data_fp32_data
                                         length:data_count_4*sizeof(float)
                                        options:MTLResourceCPUCacheModeWriteCombined];
        delete[] data_fp32_data;

        if (total_byte_size != b_handle_size) {
            delete[] data_fill_4;
        }
    }
#else
    if (data_type == DATA_TYPE_FLOAT) {
        //补齐
        float *data_fill_4 = (float *)b_handle_data;
        if (total_byte_size != b_handle_size) {
            data_fill_4 = (float *)new char[total_byte_size];
            memset((void *)data_fill_4, 0, total_byte_size);
            memcpy(data_fill_4, b_handle_data, b_handle_size);
        }

        // convert to half
        uint16_t *data_fp16_data = new uint16_t[data_count_4];
        ConvertFromFloatToHalf((float *)data_fill_4, (void *)data_fp16_data, data_count_4);

        mtl_buffer = [device newBufferWithBytes:(const void *)data_fp16_data
                                         length:data_count_4*sizeof(uint16_t)
                                        options:MTLResourceCPUCacheModeWriteCombined];
        delete[] data_fp16_data;

        if (total_byte_size != b_handle_size) {
            delete[] data_fill_4;
        }
    } else if (data_type == DATA_TYPE_HALF) {
        //补齐
        uint16_t *data_fill_4 = (uint16_t *)b_handle_data;
        if (total_byte_size != b_handle_size) {
            data_fill_4 = (uint16_t *)new char[total_byte_size];
            memset((void *)data_fill_4, 0, total_byte_size);
            memcpy(data_fill_4, b_handle_data, b_handle_size);
        }

        mtl_buffer = [device newBufferWithBytes:(const void *)data_fill_4
                                         length:total_byte_size
                                        options:MTLResourceCPUCacheModeWriteCombined];

        if (total_byte_size != b_handle_size) {
            delete[] data_fill_4;
        }
    }
#endif
    return mtl_buffer;
}

id<MTLBuffer> AllocatePackedGOIHW4MetalBufferFormRawBuffer(RawBuffer buffer, DimsVector buffer_shape, int group,
                                                            Status &status, bool transpose) {
    id<MTLDevice> device     = [TNNMetalDeviceImpl sharedDevice];
    id<MTLBuffer> mtl_buffer = nil;

    const int output_channel = buffer_shape[0];
    const int input_channel  = buffer_shape[1];
    const int kh             = buffer_shape[2];
    const int kw             = buffer_shape[3];

    const int goc   = output_channel / group;
    const int gic   = input_channel / group;
    const int goc_4 = UP_DIV(goc, 4);

    int weight_count_nopack  = group * goc * gic * kh * kw;
    int weight_count_pack    = group * goc_4 * gic * kh * kw * 4;
    const DataType data_type = buffer.GetDataType();

    if (data_type != DATA_TYPE_FLOAT && data_type != DATA_TYPE_HALF) {
        LOGE("Error: DataType %d not support\n", data_type);
        status = Status(TNNERR_MODEL_ERR, "conv_res DataType is not supported");
        return nil;
    }

#if TNN_METAL_FULL_PRECISION
    if (data_type == DATA_TYPE_FLOAT) {
        // convert to float
        float *weight_fp32_data = buffer.force_to<float *>();

        // pack
        float *weight_pack_fp32_data = new float[weight_count_pack];
        memset((void *)weight_pack_fp32_data, 0, weight_count_pack * sizeof(float));

        DataFormatConverter::ConvertFromNCHWToNCHW4Float(weight_fp32_data, weight_pack_fp32_data, group,
                                                            goc, gic, kh*kw, transpose);

        mtl_buffer = [device newBufferWithBytes:(const void *)weight_pack_fp32_data
                                         length:weight_count_pack * sizeof(float)
                                        options:MTLResourceCPUCacheModeWriteCombined];
        delete[] weight_pack_fp32_data;
    } else if (data_type == DATA_TYPE_HALF) {
        uint16_t *weight_fp16_data = buffer.force_to<uint16_t *>();

        // convert to float
        float *weight_fp32_data = new float[weight_count_nopack];
        if (ConvertFromHalfToFloat((void *)weight_fp16_data, (float *)weight_fp32_data, weight_count_nopack) != 0) {
            LOGE("Error: DataType %d not support\n", data_type);
            status = Status(TNNERR_MODEL_ERR, "Convert LayerRerouece from half to float failed!");
            return nil;
        }

        // pack
        float *weight_pack_fp32_data = new float[weight_count_pack];
        memset((void *)weight_pack_fp32_data, 0, weight_count_pack * sizeof(float));

        DataFormatConverter::ConvertFromNCHWToNCHW4Float(weight_fp32_data, weight_pack_fp32_data, group,
                                                            goc, gic, kh*kw, transpose);

        mtl_buffer = [device newBufferWithBytes:(const void *)weight_pack_fp32_data
                                         length:weight_count_pack * sizeof(float)
                                        options:MTLResourceCPUCacheModeWriteCombined];
        delete[] weight_fp32_data;
        delete[] weight_pack_fp32_data;
    }
#else
    if (data_type == DATA_TYPE_FLOAT) {
        float *weight_fp32_data = buffer.force_to<float *>();

        // convert to half
        uint16_t *weight_fp16_data = new uint16_t[weight_count_nopack];
        ConvertFromFloatToHalf((float *)weight_fp32_data, (void *)weight_fp16_data, weight_count_nopack);

        // pack
        uint16_t *weight_pack_fp16_data = new uint16_t[weight_count_pack];
        memset((void *)weight_pack_fp16_data, 0, weight_count_pack * sizeof(uint16_t));

        DataFormatConverter::ConvertFromNCHWToNCHW4Half((short *)weight_fp16_data, (short *)weight_pack_fp16_data,
                                                           group, goc, gic, kh*kw, transpose);

        mtl_buffer = [device newBufferWithBytes:(const void *)weight_pack_fp16_data
                                         length:weight_count_pack * sizeof(uint16_t)
                                        options:MTLResourceCPUCacheModeWriteCombined];
        delete[] weight_fp16_data;
        delete[] weight_pack_fp16_data;
    } else if (data_type == DATA_TYPE_HALF) {
        // convert to half
        uint16_t *weight_fp16_data = buffer.force_to<uint16_t *>();

        // pack
        uint16_t *weight_pack_fp16_data = new uint16_t[weight_count_pack];
        memset((void *)weight_pack_fp16_data, 0, weight_count_pack * sizeof(uint16_t));

        DataFormatConverter::ConvertFromNCHWToNCHW4Half((short *)weight_fp16_data, (short *)weight_pack_fp16_data,
                                                           group, goc, gic, kh*kw, transpose);

        mtl_buffer = [device newBufferWithBytes:(const void *)weight_pack_fp16_data
                                         length:weight_count_pack * sizeof(uint16_t)
                                        options:MTLResourceCPUCacheModeWriteCombined];
        delete[] weight_pack_fp16_data;
    }
#endif
    return mtl_buffer;
}

id<MTLBuffer> AllocatePackedGOIHW16MetalBufferFormRawBuffer(RawBuffer buffer, DimsVector buffer_shape, int group,
                                                            Status &status, bool transpose) {
    id<MTLDevice> device     = [TNNMetalDeviceImpl sharedDevice];
    id<MTLBuffer> mtl_buffer = nil;

    const int output_channel = buffer_shape[0];
    const int input_channel  = buffer_shape[1];
    const int kh             = buffer_shape[2];
    const int kw             = buffer_shape[3];

    const int goc   = output_channel / group;
    const int gic   = input_channel / group;
    const int goc_4 = UP_DIV(goc, 4);
    const int gic_4 = UP_DIV(gic, 4);

    int weight_count_nopack  = group * goc * gic * kh * kw;
    int weight_count_pack    = group * goc_4 * gic_4 * kh * kw * 16;
    const DataType data_type = buffer.GetDataType();

    if (data_type != DATA_TYPE_FLOAT && data_type != DATA_TYPE_HALF) {
        LOGE("Error: DataType %d not support\n", data_type);
        status = Status(TNNERR_MODEL_ERR, "conv_res DataType is not supported");
        return nil;
    }

#if TNN_METAL_FULL_PRECISION
    if (data_type == DATA_TYPE_FLOAT) {
        // convert to float
        float *weight_fp32_data = buffer.force_to<float *>();

        // pack
        float *weight_pack_fp32_data = new float[weight_count_pack];
        memset((void *)weight_pack_fp32_data, 0, weight_count_pack * sizeof(float));

        DataFormatConverter::ConvertFromGOIHWToGOIHW16Float(weight_fp32_data, weight_pack_fp32_data, group,
                                                            input_channel, output_channel, kh, kw, transpose);

        mtl_buffer = [device newBufferWithBytes:(const void *)weight_pack_fp32_data
                                         length:weight_count_pack * sizeof(float)
                                        options:MTLResourceCPUCacheModeWriteCombined];
        delete[] weight_pack_fp32_data;
    } else if (data_type == DATA_TYPE_HALF) {
        uint16_t *weight_fp16_data = buffer.force_to<uint16_t *>();

        // convert to float
        float *weight_fp32_data = new float[weight_count_pack];
        ConvertFromHalfToFloat((void *)weight_fp16_data, (float *)weight_fp32_data, weight_count_nopack);

        // pack
        float *weight_pack_fp32_data = new float[weight_count_pack];
        memset((void *)weight_pack_fp32_data, 0, weight_count_pack * sizeof(float));

        DataFormatConverter::ConvertFromGOIHWToGOIHW16Float(weight_fp32_data, weight_pack_fp32_data, group,
                                                            input_channel, output_channel, kh, kw, transpose);

        mtl_buffer = [device newBufferWithBytes:(const void *)weight_pack_fp32_data
                                         length:weight_count_pack * sizeof(float)
                                        options:MTLResourceCPUCacheModeWriteCombined];
        delete[] weight_fp32_data;
        delete[] weight_pack_fp32_data;
    }
#else
    if (data_type == DATA_TYPE_FLOAT) {
        float *weight_fp32_data = buffer.force_to<float *>();

        // convert to half
        uint16_t *weight_fp16_data = new uint16_t[weight_count_pack];
        ConvertFromFloatToHalf((float *)weight_fp32_data, (void *)weight_fp16_data, weight_count_nopack);

        // pack
        uint16_t *weight_pack_fp16_data = new uint16_t[weight_count_pack];
        memset((void *)weight_pack_fp16_data, 0, weight_count_pack * sizeof(uint16_t));

        DataFormatConverter::ConvertFromGOIHWToGOIHW16Half((short *)weight_fp16_data, (short *)weight_pack_fp16_data,
                                                           group, input_channel, output_channel, kh, kw, transpose);

        mtl_buffer = [device newBufferWithBytes:(const void *)weight_pack_fp16_data
                                         length:weight_count_pack * sizeof(uint16_t)
                                        options:MTLResourceCPUCacheModeWriteCombined];
        delete[] weight_fp16_data;
        delete[] weight_pack_fp16_data;
    } else if (data_type == DATA_TYPE_HALF) {
        // convert to half
        uint16_t *weight_fp16_data = buffer.force_to<uint16_t *>();

        // pack
        uint16_t *weight_pack_fp16_data = new uint16_t[weight_count_pack];
        memset((void *)weight_pack_fp16_data, 0, weight_count_pack * sizeof(uint16_t));

        DataFormatConverter::ConvertFromGOIHWToGOIHW16Half((short *)weight_fp16_data, (short *)weight_pack_fp16_data,
                                                           group, input_channel, output_channel, kh, kw, transpose);

        mtl_buffer = [device newBufferWithBytes:(const void *)weight_pack_fp16_data
                                         length:weight_count_pack * sizeof(uint16_t)
                                        options:MTLResourceCPUCacheModeWriteCombined];
        delete[] weight_pack_fp16_data;
    }
#endif
    return mtl_buffer;
}

id<MTLBuffer> AllocatePackedNC4HW4MetalBufferFormRawBuffer(RawBuffer buffer, DimsVector buffer_shape, int group,
                                                           Status &status) {
    id<MTLDevice> device     = [TNNMetalDeviceImpl sharedDevice];
    id<MTLBuffer> mtl_buffer = nil;

    int channel = DimsFunctionUtils::GetDim(buffer_shape, 1);
    int kh      = DimsFunctionUtils::GetDim(buffer_shape, 2);
    int kw      = DimsFunctionUtils::GetDimProduct(buffer_shape, 3);

    const int channel4 = UP_DIV(channel, 4) * 4;

    int data_count_nopack  = channel * kh * kw;
    int data_count_pack    = channel4 * kh * kw;
    const DataType data_type = buffer.GetDataType();

    if (data_type != DATA_TYPE_FLOAT && data_type != DATA_TYPE_HALF) {
        LOGE("Error: DataType %d not support\n", data_type);
        status = Status(TNNERR_MODEL_ERR, "conv_res DataType is not supported");
        return nil;
    }

#if TNN_METAL_FULL_PRECISION
    if (data_type == DATA_TYPE_FLOAT) {
        // convert to float
        float *data_fp32_data = buffer.force_to<float *>();

        // pack
        float *data_pack_fp32_data = new float[data_count_pack];
        memset((void *)data_pack_fp32_data, 0, data_count_pack * sizeof(float));

        DataFormatConverter::ConvertFromNCHWToNCHW4Float(data_fp32_data, data_pack_fp32_data, 1, channel, kh, kw);

        mtl_buffer = [device newBufferWithBytes:(const void *)data_pack_fp32_data
                                         length:data_count_pack * sizeof(float)
                                        options:MTLResourceCPUCacheModeWriteCombined];
        delete[] data_pack_fp32_data;
    } else if (data_type == DATA_TYPE_HALF) {
        uint16_t *data_fp16_data = buffer.force_to<uint16_t *>();

        // convert to float
        float *data_fp32_data = new float[data_count_pack];
        ConvertFromHalfToFloat((void *)data_fp16_data, (float *)data_fp32_data, data_count_nopack);

        // pack
        float *data_pack_fp32_data = new float[data_count_pack];
        memset((void *)data_pack_fp32_data, 0, data_count_pack * sizeof(float));

        DataFormatConverter::ConvertFromNCHWToNCHW4Float(data_fp32_data, data_pack_fp32_data, 1, channel, kh, kw);

        mtl_buffer = [device newBufferWithBytes:(const void *)data_pack_fp32_data
                                         length:data_count_pack * sizeof(float)
                                        options:MTLResourceCPUCacheModeWriteCombined];
        delete[] data_fp32_data;
        delete[] data_pack_fp32_data;
    }
#else
    if (data_type == DATA_TYPE_FLOAT) {
        float *data_fp32_data = buffer.force_to<float *>();

        // convert to half
        uint16_t *data_fp16_data = new uint16_t[data_count_pack];
        ConvertFromFloatToHalf((float *)data_fp32_data, (void *)data_fp16_data, data_count_nopack);

        // pack
        uint16_t *data_pack_fp16_data = new uint16_t[data_count_pack];
        memset((void *)data_pack_fp16_data, 0, data_count_pack * sizeof(uint16_t));

        DataFormatConverter::ConvertFromNCHWToNCHW4Half((short *)data_fp16_data, (short *)data_pack_fp16_data, 1,
                                                        channel, kh, kw);

        mtl_buffer = [device newBufferWithBytes:(const void *)data_pack_fp16_data
                                         length:data_count_pack * sizeof(uint16_t)
                                        options:MTLResourceCPUCacheModeWriteCombined];
        delete[] data_fp16_data;
        delete[] data_pack_fp16_data;
    } else if (data_type == DATA_TYPE_HALF) {
        // convert to half
        uint16_t *data_fp16_data = buffer.force_to<uint16_t *>();

        // pack
        uint16_t *data_pack_fp16_data = new uint16_t[data_count_pack];
        memset((void *)data_pack_fp16_data, 0, data_count_pack * sizeof(uint16_t));

        DataFormatConverter::ConvertFromNCHWToNCHW4Half((short *)data_fp16_data, (short *)data_pack_fp16_data, 1,
                                                        channel, kh, kw);

        mtl_buffer = [device newBufferWithBytes:(const void *)data_pack_fp16_data
                                         length:data_count_pack * sizeof(uint16_t)
                                        options:MTLResourceCPUCacheModeWriteCombined];
        delete[] data_pack_fp16_data;
    }
#endif
    return mtl_buffer;
}

} // namespace TNN_NS
