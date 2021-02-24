#include "tnn/device/cuda/acc/cuda_layer_acc.h"
#include "tnn/utils/dims_vector_utils.h"
#include <limits>
#include <cuda.h>
#include <numeric>
#include <cuda_fp16.h>
#include <iostream>

namespace TNN_NS {

DECLARE_CUDA_ACC(CbamFusedReduce, LAYER_CBAM_FUSED_REDUCE);

typedef uint32_t uint32;

template <typename T>
__device__ __forceinline__ T getMin() {
    return T(-FLT_MAX);
}   

__global__ void cbam_fused_reduce_kernel(float *in, float *out, int n, int c, int h, int w) {
    // each thread processes one output element
    int stride = h * w;
    int tidx = blockDim.x * blockIdx.x + threadIdx.x;
    int sample_idx = tidx / (h * w);

    if (sample_idx >= n) return;

    int sample_offset = tidx  % (h * w);
    int src_offset = sample_idx * (c * h * w) + sample_offset;
    int dst_offset = sample_idx * (2 * h * w) + sample_offset;

    float accumulate = 0;
    float max = getMin<float>();

    for (int i = 0; i < c; i ++) {
        float in_value = in[src_offset + i * stride];
        accumulate += in_value;
        max = fmaxf(max, in_value);
    }

    out[dst_offset] = accumulate / c;
    out[dst_offset + stride] = max;
}

template<typename T>
__global__ void cbam_fused_reduce_half_kernel(__half *in, T *out, int n, int c, int h, int w) {
    // each thread processes one output element
    int stride = h * w;
    int tidx = blockDim.x * blockIdx.x + threadIdx.x;
    int sample_idx = tidx / (h * w);

    if (sample_idx >= n) return;

    int sample_offset = tidx  % (h * w);
    int src_offset = sample_idx * (c * h * w) + sample_offset;
    int dst_offset = sample_idx * (2 * h * w) + sample_offset;

    float accumulate = 0;
    float max = getMin<float>();

    for(int i = 0; i < c; i ++) {
        float in_value = __half2float(in[src_offset + i * stride]);
        accumulate += in_value;
        max = fmaxf(max, in_value);
    }

    out[dst_offset] = convert_float_value<T>(accumulate / c);
    out[dst_offset + stride] = convert_float_value<T>(max);
}


// for fp16, N(c/x)HWx  formats
// only process cases where 2 <= x <= 16, and c <= 1024
template<typename T>
__global__ void cbam_fused_reduce_packed_kernel(__half *in, T *out,
        int n, int c, int h, int w, int pack_num) {

    // Each block process blockDim.x fp16 spatial elements of the input map (include all the channels)
    int data_stride_c = h * w * pack_num;
    int data_num_c = (c + pack_num - 1) / pack_num;

    // make sure a block is not across a sample's boundary
    int blocks_per_sample = (data_stride_c + blockDim.x - 1) / blockDim.x;
    
    int sample_idx = blockIdx.x / blocks_per_sample;
    int element_idx = blockIdx.x % blocks_per_sample * blockDim.x + threadIdx.x;
    int element_offset = sample_idx * data_stride_c * data_num_c + element_idx;
    
    int pixel_idx = element_idx / pack_num;
    int channel_idx = element_idx % pack_num;
    
    // the actual number of bytes will be used is:
    // blockDim.x / pack_num * sizeof(float)
    // make sure blockDim.x / pack_num <= 64
    // half of the array used for saving mean
    // the other half is for saving max
#define MAX_NUM_PIXELS_PER_BLOCK 64
    __shared__ float reduction_results[MAX_NUM_PIXELS_PER_BLOCK * 2];
    
    float accumulate = 0;
    float max = getMin<float>();

    if (pixel_idx <  h * w && channel_idx < c) {
        int count = 0;
        while (channel_idx < c) {
            float in_value = __half2float(in[element_offset]);
            accumulate += in_value;
            max = fmaxf(max, in_value);
            channel_idx += pack_num;
            element_offset += data_stride_c;            
        }

        reduction_results[threadIdx.x] = accumulate / count;
        reduction_results[threadIdx.x + MAX_NUM_PIXELS_PER_BLOCK] = max;
    }
    
    __syncthreads();
    
    // use the first (blockDim.x / pack_num) threads to do the final reduction
    // each thread processes one pixel in the output image/feature
    if (threadIdx.x < blockDim.x / pack_num) {
        float accumulate = 0;
        float count = 0;
        int mean_value_offset = threadIdx.x * pack_num;
        int max_value_offset = threadIdx.x * pack_num + MAX_NUM_PIXELS_PER_BLOCK;
        float max = reduction_results[max_value_offset];

        for(int idx = threadIdx.x; idx < pack_num && idx < c; idx ++){
            accumulate =+ reduction_results[mean_value_offset + idx];
            max = fmaxf(max, reduction_results[max_value_offset + idx]);
            count += 1;                        
        }

        accumulate /= count;
        
        // write to output buffer
        int output_sample_stride = pack_num * h * w;
        int output_pixel_idx = (element_idx - threadIdx.x + threadIdx.x * pack_num) / pack_num;
        int offset = sample_idx * output_sample_stride + pack_num * output_pixel_idx;
        out[offset] = convert_float_value<T>(accumulate);//__float2half(accumulate);
        out[offset+1] = convert_float_value<T>(max);//__float2half(max);
    }
}


Status CudaCbamFusedReduceLayerAcc::Init(Context *context, LayerParam *param, LayerResource *resource,
        const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    return CudaLayerAcc::Init(context, param, resource, inputs, outputs);
}

Status CudaCbamFusedReduceLayerAcc::Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    return TNN_OK;
}

Status CudaCbamFusedReduceLayerAcc::Forward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    Blob *input_blob  = inputs[0];
    Blob *output_blob = outputs[0];

    DataType type = input_blob->GetBlobDesc().data_type;
    DataFormat format = input_blob->GetBlobDesc().data_format;

    int batch_size = input_blob->GetBlobDesc().dims[0];
    int nchannels = input_blob->GetBlobDesc().dims[1];
    int inp_H = input_blob->GetBlobDesc().dims[2];
    int inp_W = input_blob->GetBlobDesc().dims[3];

    if (type == DataType::DATA_TYPE_FLOAT) {
        float* input_ptr = static_cast<float*>(input_blob->GetHandle().base);
        float* output_ptr = static_cast<float*>(output_blob->GetHandle().base);

        int thread_num = 32;
        int block_num = (inp_H * inp_W * batch_size + thread_num - 1) / thread_num;
        cbam_fused_reduce_kernel<<<block_num, thread_num, 0, context_->GetStream()>>>(static_cast<float *>(input_ptr), 
            static_cast<float *>(output_ptr), 
            batch_size, nchannels, inp_H, inp_W);
    } else if (type == DataType::DATA_TYPE_HALF) {
        void* input_ptr = input_blob->GetHandle().base;
        void* output_ptr = output_blob->GetHandle().base;
        int thread_num = 64;
        if (format == DataFormat::DATA_FORMAT_NCHW) {
            int block_num = (inp_H * inp_W * batch_size + thread_num - 1) / thread_num;
            if (output_blob->GetBlobDesc().data_type == DATA_TYPE_FLOAT)
                cbam_fused_reduce_half_kernel<float><<<block_num, thread_num, 0, context_->GetStream()>>>(
                    static_cast<__half *>(input_ptr), static_cast<float *>(output_ptr), batch_size, nchannels, inp_H, inp_W);
            else
                cbam_fused_reduce_half_kernel<__half><<<block_num, thread_num, 0, context_->GetStream()>>>(
                    static_cast<__half *>(input_ptr), static_cast<__half *>(output_ptr), batch_size, nchannels, inp_H, inp_W);
        } else if (format == DataFormat::DATA_FORMAT_NC2HW2 || format == DataFormat::DATA_FORMAT_NC4HW4 ||
                format == DataFormat::DATA_FORMAT_NC16HW16) {
            int pack_num = 2;
            switch (format) {
                case DataFormat::DATA_FORMAT_NC2HW2: pack_num = 2; break;
                case DataFormat::DATA_FORMAT_NC4HW4: pack_num = 4; break;
                case DataFormat::DATA_FORMAT_NC16HW16: pack_num = 16; break;
                default: pack_num = 2;
            }
            int blocks_per_sample = (inp_H * inp_W * pack_num + thread_num - 1) / thread_num;
            int blocks_num = blocks_per_sample * batch_size;
            cbam_fused_reduce_packed_kernel<__half><<<blocks_num, thread_num, 0, context_->GetStream()>>>(static_cast<__half *>(input_ptr), 
                static_cast<__half *>(output_ptr), batch_size, nchannels, inp_H, inp_W, pack_num);
        } else {
            LOGE("Error: layer acc dont support data format: %d\n", input_blob->GetBlobDesc().data_format);
            return Status(TNNERR_MODEL_ERR, "Error: layer acc don't support data format");
        }
    } else {
        LOGE("Error: layer acc dont support datatype: %d\n", input_blob->GetBlobDesc().data_type);
        return Status(TNNERR_MODEL_ERR, "Error: layer acc don't support datatype");
    }
    return TNN_OK;
}

REGISTER_CUDA_ACC(CbamFusedReduce, LAYER_CBAM_FUSED_REDUCE);

}   // namespace TNN_NS
