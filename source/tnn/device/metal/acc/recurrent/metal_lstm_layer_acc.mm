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

#include "tnn/device/metal/acc/recurrent/metal_lstm_layer_acc.h"
#include "tnn/device/metal/metal_context.h"
#include "tnn/utils/dims_utils.h"
#include "tnn/utils/data_type_utils.h"
#include "tnn/utils/half_utils_inner.h"

// [outer, axis, inner] => [outer, inner, axis]
template <typename T>
static void TransposeWeight(const T *src, T *dst, int outer, int inner, int axis_size) {
    int dst_idx = 0;
    for(int o=0; o<outer; ++o) {
        for(int i=0; i<inner; ++i) {
            for(int v=0; v<axis_size; ++v) {
                int src_idx = (o * axis_size + v) * inner + i;
                dst[dst_idx++] = src[src_idx];
            }
        }
    }
}

static void TransposeWeightBlob(Blob* blob, void *buffer, int outer, int inner, int axis_size) {
    auto data_type = blob->GetBlobDesc().data_type;
    void *ptr = (static_cast<char *>(blob->GetHandle().base) + blob->GetHandle().bytes_offset);
    if (data_type == DATA_TYPE_HALF) {
        using T = uint16_t;
        TransposeWeight<T>((T *)ptr, (T *)buffer, outer, inner, axis_size);
    } else if (data_type == DATA_TYPE_FLOAT) {
        using T = float;
        TransposeWeight<T>((T *)ptr, (T *)buffer, outer, inner, axis_size);
    }
}

static id<MTLBuffer> AllocateBufferForWeightBlob(Blob* weight_blob, int weight_count, int outer, int inner, int axis_size,
                                          id<MTLDevice> device) {
    auto data_type = weight_blob->GetBlobDesc().data_type;
    auto buffer_bytes = weight_count * DataTypeUtils::GetBytesSize(data_type);
    std::shared_ptr<char> weight_cpu_buffer(new char[buffer_bytes], [](char* p){delete [] p;});
    std::shared_ptr<char> weight_cpu_buffer_type = nullptr;
    TransposeWeightBlob(weight_blob, weight_cpu_buffer.get(), outer, inner, axis_size);
#if TNN_METAL_FULL_PRECISION
    auto metal_buffer_bytes = weight_count * sizeof(float);
    if (data_type == DATA_TYPE_HALF) {
        // half to float
        weight_cpu_buffer_type.reset(new char[metal_buffer_bytes], [](char* p){delete [] p;});
        if (ConvertFromHalfToFloat(weight_cpu_buffer.get(), (float *)weight_cpu_buffer_type.get(), weight_count) != 0) {
            LOGE("lstm weight DataType is not supported");
            return nil;
        }
        weight_cpu_buffer = weight_cpu_buffer_type;
    }
#else
    auto metal_buffer_bytes = weight_count * sizeof(uint16_t);
    if (data_type == DATA_TYPE_FLOAT) {
        // float to half
        weight_cpu_buffer_type.reset(new char [metal_buffer_bytes], [](char* p){ delete [] p;});
        if (ConvertFromFloatToHalf((float *)(weight_cpu_buffer.get()), weight_cpu_buffer_type.get(), weight_count) != 0) {
            LOGE("lstm weight DataType is not supported");
            return nil;
        };
        weight_cpu_buffer = weight_cpu_buffer_type;
    }
#endif
    id<MTLBuffer> buffer = [device newBufferWithBytes:weight_cpu_buffer.get()
                                     length:metal_buffer_bytes
                                    options:MTLResourceCPUCacheModeWriteCombined];
    return buffer;
}

namespace TNN_NS {
Status MetalLSTMLayerAcc::Init(Context *context, LayerParam *param, LayerResource *resource, const std::vector<Blob *> &inputs,
                                   const std::vector<Blob *> &outputs) {
    return MetalLayerAcc::Init(context, param, resource, inputs, outputs);
}

MetalLSTMLayerAcc::~MetalLSTMLayerAcc() {}

Status MetalLSTMLayerAcc::AllocateBufferParam(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    
    id<MTLDevice> device       = [TNNMetalDeviceImpl sharedDevice];
    auto lstm_param = dynamic_cast<LSTMONNXLayerParam *>(param_);

    int num_direction = lstm_param->direction >= 2 ? 2 : 1;
    int hidden_size = lstm_param->hidden_size;
    auto dims_input  = inputs[0]->GetBlobDesc().dims;
    auto dims_output = outputs[0]->GetBlobDesc().dims;
    // buffer_param_
    {
        MetalRecurrentParams metal_params;
        
        metal_params.seq_len = dims_input[0];
        metal_params.batch   = dims_input[1];
        metal_params.hidden_size = hidden_size;
        metal_params.input_width = DimsVectorUtils::Count(dims_input, 2);
        metal_params.reverse = lstm_param->direction==1;
        metal_params.direction = num_direction;
        metal_params.has_init_h = inputs.size() > 4 && (!!(inputs[4]->GetHandle().base));
        metal_params.has_init_c = inputs.size() > 5 && (!!(inputs[5]->GetHandle().base));
        
        buffer_param_ = [device newBufferWithBytes:(const void *)(&metal_params)
                                            length:sizeof(MetalRecurrentParams)
                                           options:MTLResourceCPUCacheModeWriteCombined];
    }
    return TNN_OK;
}

Status MetalLSTMLayerAcc::AllocateBufferWeights(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    Status status = TNN_OK;
    
    auto layer_param = dynamic_cast<LSTMONNXLayerParam *>(param_);
    id<MTLDevice> device = [TNNMetalDeviceImpl sharedDevice];
    // get input shape
    int num_directions = layer_param->direction >=2 ? 2 : 1;
    const auto input_dims = inputs[0]->GetBlobDesc().dims;
    const auto input_size  = DimsVectorUtils::Count(input_dims, 2);
    const auto hidden_size = layer_param->hidden_size;
    
    {
        // buffer weights_input
        auto weight_count = num_directions * hidden_size * input_size * 4;
        auto inner = hidden_size * input_size;
        auto outer = num_directions;
        // TODO: transpose to [dir, input, output, 4]
        // transpose from: [dir, 4, hidden, input] to [dir, hidden, input, 4]
        if (!buffer_wi_) {
            buffer_wi_ = AllocateBufferForWeightBlob(inputs[1], weight_count, outer, inner, 4, device);
            RETURN_VALUE_ON_NEQ(!buffer_wi_, false,
                                Status(TNNERR_MODEL_ERR, "allocating buffer for lstm weight_input failed!"));
        }
    }
    
    {
        // buffer weight_hidden
        auto weight_count = num_directions * hidden_size * hidden_size * 4;
        auto inner = hidden_size * hidden_size;
        auto outer = num_directions;
        // TODO: transpose to [dir, hidden_in, hidden_out, 4]
        // transpose from: [dir, 4, hidden_out, hidden_in] to [dir, hidden_out, hidden_in, 4]
        if (!buffer_wh_) {
            buffer_wh_ = AllocateBufferForWeightBlob(inputs[2], weight_count, outer, inner, 4, device);
            RETURN_VALUE_ON_NEQ(!buffer_wh_, false,
                                Status(TNNERR_MODEL_ERR, "allocating buffer for lstm weight_hidden failed!"));
        }
    }
    
    return status;
}

Status MetalLSTMLayerAcc::AllocateBufferBias(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    Status status = TNN_OK;
    
    auto layer_param = dynamic_cast<LSTMONNXLayerParam *>(param_);
    id<MTLDevice> device = [TNNMetalDeviceImpl sharedDevice];
    // get input shape
    int num_directions = layer_param->direction >=2 ? 2 : 1;
    const auto hidden_size = layer_param->hidden_size;
    {
        // buffer bias
        auto weight_count = num_directions * hidden_size * 8;
        auto inner = hidden_size;
        auto outer = num_directions;
        // transpose from: [dir, 8, hidden_size] to [dir, hidden_size, 8]
        buffer_bias_ = AllocateBufferForWeightBlob(inputs[3], weight_count, outer, inner, 8, device);
        RETURN_VALUE_ON_NEQ(!buffer_bias_, false, Status(TNNERR_MODEL_ERR, "allocating buffer for lstm bias failed!"));
    }

    return status;
}

Status MetalLSTMLayerAcc::AllocateBufferStates(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    Status status = TNN_OK;
    auto layer_param = dynamic_cast<LSTMONNXLayerParam *>(param_);
    id<MTLDevice> device = [TNNMetalDeviceImpl sharedDevice];
    // get input shape
    int num_directions = layer_param->direction >= 2 ? 2 : 1;
    const auto input_dims = inputs[0]->GetBlobDesc().dims;
    const auto seq_len = input_dims[0];
    const auto batch = input_dims[1];
    const auto input_size  = DimsVectorUtils::Count(input_dims, 2);
    const auto hidden_size = layer_param->hidden_size;
    
    auto state_buffer_bytes = 0;
    
    // gates buffer
#if TNN_METAL_FULL_PRECISION
    state_buffer_bytes = num_directions * seq_len * batch * hidden_size * 4 * sizeof(float);
#else
    state_buffer_bytes = num_directions * seq_len * batch * hidden_size * 4 * sizeof(uint16_t);
#endif
    if (!buffer_gates_ || buffer_gates_.length != state_buffer_bytes) {
        buffer_gates_ = [device newBufferWithLength:state_buffer_bytes
                                            options:MTLResourceStorageModePrivate];  // only metal kernel writes to this
    }
    
    // initial states buffer
#if TNN_METAL_FULL_PRECISION
    auto metal_state_buffer_bytes = num_directions * batch * hidden_size * sizeof(float);
#else
    auto metal_state_buffer_bytes = num_directions * batch * hidden_size * sizeof(uint16_t);
#endif
    if (inputs.size() > 5 && (!buffer_c0_ || buffer_c0_.length != metal_state_buffer_bytes)) {
        Blob *c0 = inputs[5];
        auto data_type = c0->GetBlobDesc().data_type;
        void *ptr = static_cast<char *>(c0->GetHandle().base) + c0->GetHandle().bytes_offset;
        std::shared_ptr<char> buffer_type = nullptr;
        if (ptr) {
#if TNN_METAL_FULL_PRECISION
        if (data_type == DATA_TYPE_HALF) {
            buffer_type.reset(new char [metal_state_buffer_bytes], [](char *p){delete [] p;});
            if (ConvertFromHalfToFloat(ptr, (float *)buffer_type.get(), num_directions * batch * hidden_size) < 0) {
                return Status(TNNERR_MODEL_ERR, "lstm initial state DataType is not supported");
            }
            ptr = buffer_type.get();
        }
#else
        if (data_type == DATA_TYPE_FLOAT) {
            buffer_type.reset(new char [metal_state_buffer_bytes], [](char *p){delete [] p;});
            if (ConvertFromFloatToHalf((float *)ptr, buffer_type.get(), num_directions * batch * hidden_size) < 0) {
                return Status(TNNERR_MODEL_ERR, "lstm initial state DataType is not supported");
            }
            ptr = buffer_type.get();
        }
#endif
        buffer_c0_ = [device newBufferWithBytes:ptr
                                         length:metal_state_buffer_bytes
                                        options:MTLResourceOptionCPUCacheModeWriteCombined];
        }
    }
    
    if (inputs.size() > 4 && (!buffer_h0_ || buffer_h0_.length != metal_state_buffer_bytes)) {
        Blob *h0 = inputs[4];
        auto data_type = h0->GetBlobDesc().data_type;
        void *ptr = static_cast<char *>(h0->GetHandle().base) + h0->GetHandle().bytes_offset;
        std::shared_ptr<char> buffer_type = nullptr;
        if (ptr) {
#if TNN_METAL_FULL_PRECISION
        if (data_type == DATA_TYPE_HALF) {
            buffer_type.reset( new char [metal_state_buffer_bytes], [](char *p){delete [] p;});
            if (ConvertFromHalfToFloat(ptr, (float *)buffer_type.get(), num_directions * batch * hidden_size) < 0) {
                return Status(TNNERR_MODEL_ERR, "lstm initial state DataType is not supported");
            }
            ptr = buffer_type.get();
        }
#else
        if (data_type == DATA_TYPE_FLOAT) {
            buffer_type.reset( new char [metal_state_buffer_bytes], [](char *p){delete [] p;});
            if (ConvertFromFloatToHalf((float *)ptr, buffer_type.get(), num_directions * batch * hidden_size) < 0) {
                return Status(TNNERR_MODEL_ERR, "lstm initial state DataType is not supported");
            }
            ptr = buffer_type.get();
        }
#endif
        buffer_h0_ = [device newBufferWithBytes:ptr
                                         length:metal_state_buffer_bytes
                                        options:MTLResourceOptionCPUCacheModeWriteCombined];
        }

    }
    
    if (!buffer_h0_) {
        // no initial states, set them to a valid potision to avoid error when binded with kernels
        buffer_h0_ = buffer_gates_;
    }
    
    if (!buffer_c0_) {
        // no initial states, set them to a valid potision to avoid error when binded with kernels
        buffer_c0_ = buffer_gates_;
    }
    
    return status;
}

Status MetalLSTMLayerAcc::Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto status = MetalLayerAcc::Reshape(inputs, outputs);
    RETURN_ON_NEQ(status, TNN_OK);
    
    status = AllocateBufferStates(inputs, outputs);
    RETURN_ON_NEQ(status, TNN_OK);

    if (!buffer_wh_ || !buffer_wi_) {
        status = AllocateBufferWeights(inputs, outputs);
        RETURN_ON_NEQ(status, TNN_OK);
    }
    
    if (!buffer_bias_) {
        status = AllocateBufferBias(inputs, outputs);
        RETURN_ON_NEQ(status, TNN_OK);
    }
    
    return status;
}

std::vector<DataFormat> MetalLSTMLayerAcc::SupportDataFormat(DataType data_type, int dims_size, BlobType blob_type) {
    std::vector<DataFormat> support_list;
    if (dims_size >= 2) {
        // inputs to lstm layer should at least has two dimensions
        support_list.push_back(DATA_FORMAT_NCHW);
    }
    return support_list;
}

Status MetalLSTMLayerAcc::Forward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    // check if inputs valid
    if (inputs.size() < 4) {
        return Status(TNNERR_LAYER_ERR, "LSTM has invalid inputs");
    }
    
    Blob *blob_x = inputs[0];
    Blob *blob_y = outputs[0];
    Blob *blob_h = outputs[1];
    Blob *blob_c = outputs[2];
    
    const auto input_dims = blob_x->GetBlobDesc().dims;
    const auto seq_len = input_dims[0]; // length of sequence
    const auto batch   = input_dims[1];  // batch_size
    const auto input_size  = DimsVectorUtils::Count(input_dims, 2); // input dimension
    const auto output_dims = blob_y->GetBlobDesc().dims;
    auto layer_param = dynamic_cast<LSTMONNXLayerParam *>(param_);
    const int num_directions = layer_param->direction >= 2 ? 2 : 1;
    const auto hidden_size = layer_param->hidden_size; // output dimension
    
    auto context_impl = context_->getMetalContextImpl();
    auto encoder      = [context_impl encoder];

    Status status = TNN_OK;
    MetalBandwidth bandwidth;
    do {
        {
            // lstm_gates
            status = [context_impl load:@"lstm_gates" encoder:encoder bandwidth:bandwidth];
            BREAK_IF(status != TNN_OK);
            auto threads = MTLSizeMake(hidden_size, batch, seq_len*num_directions);
            
            [encoder setBuffer:(__bridge id<MTLBuffer>)(void *)blob_x->GetHandle().base
                        offset:(NSUInteger)blob_x->GetHandle().bytes_offset
                       atIndex:0];
            [encoder setBuffer:buffer_wi_    offset:0 atIndex:1];
            [encoder setBuffer:buffer_gates_ offset:0 atIndex:2];
            [encoder setBuffer:buffer_param_ offset:0 atIndex:3];
            
            status = [context_impl dispatchEncoder:encoder threads:threads bandwidth:bandwidth];
            BREAK_IF(status != TNN_OK);
        }
        {
            // lstm_forward
            status = [context_impl load:@"lstm_forward" encoder:encoder bandwidth:bandwidth];
            BREAK_IF(status != TNN_OK);
            auto threads_per_group = MTLSizeMake(hidden_size, 1, 1);
            auto groups = MTLSizeMake(1, batch, num_directions);
#if TNN_METAL_FULL_PRECISION
            const auto hidden_bytes = hidden_size * sizeof(float);
#else
            const auto hidden_bytes = hidden_size * sizeof(uint16_t);
#endif
            
            [encoder setBuffer:buffer_gates_ offset:0 atIndex:0];
            [encoder setBuffer:buffer_c0_ offset:0 atIndex:1];
            [encoder setBuffer:buffer_h0_ offset:0 atIndex:2];
            [encoder setBuffer:buffer_wh_ offset:0 atIndex:3];
            [encoder setBuffer:buffer_bias_ offset:0 atIndex:4];
            [encoder setBuffer:(__bridge id<MTLBuffer>)(void *)blob_c->GetHandle().base
                        offset:(NSUInteger)blob_c->GetHandle().bytes_offset
                       atIndex:5];
            [encoder setBuffer:(__bridge id<MTLBuffer>)(void *)blob_h->GetHandle().base
                        offset:(NSUInteger)blob_h->GetHandle().bytes_offset
                       atIndex:6];
            [encoder setBuffer:(__bridge id<MTLBuffer>)(void *)blob_y->GetHandle().base
                        offset:(NSUInteger)blob_y->GetHandle().bytes_offset
                       atIndex:7];
            [encoder setBuffer:buffer_param_ offset:0 atIndex:8];
            [encoder setThreadgroupMemoryLength:hidden_bytes atIndex:0];
            
            status = [context_impl dispatchEncoder:encoder threadsPerGroup:threads_per_group groups:groups bandwidth:bandwidth];
            BREAK_IF(status != TNN_OK);
        }
    } while (0);
    if (status != TNN_OK) {
        [encoder endEncoding];
        return status;
    }
    
    [encoder endEncoding];
    [context_impl commit];
    TNN_PRINT_ENCODER(context_, encoder, this);
    return status;
}

REGISTER_METAL_ACC(LSTM, LAYER_LSTMONNX);
REGISTER_METAL_LAYOUT(LAYER_LSTMONNX, DATA_FORMAT_NCHW);

} // namespace TNN_NS
