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

#include "tnn/device/cuda/acc/cuda_deconv_layer_acc.h"
#include "tnn/utils/dims_utils.h"

#include <cuda_fp16.h>
namespace TNN_NS {

// DECLARE_CUDA_ACC(Deconvolution, LAYER_DECONVOLUTION);

template <typename Dtype, typename AccT, bool BIAS>
__global__ void ConvBackward(const int nthreads, const Dtype *const input, const int num, const int channels,
                             const int height, const int width, const int conved_height, const int conved_width,
                             const int kernel_h, const int kernel_w, const int stride_h, const int stride_w,
                             const int pad_h, const int pad_w, Dtype *const output, const Dtype *const weight,
                             const Dtype *const bias) {
    CUDA_KERNEL_LOOP(index, nthreads) {
        const int w = index % width + pad_w;
        const int h = (index / width) % height + pad_h;
        const int c = (index / width / height) % channels;
        const int n = index / width / height / channels;

        const int phstart = (h < kernel_h) ? 0 : (h - kernel_h) / stride_h + 1;
        const int phend   = min(h / stride_h + 1, conved_height);
        const int pwstart = (w < kernel_w) ? 0 : (w - kernel_w) / stride_w + 1;
        const int pwend   = min(w / stride_w + 1, conved_width);

        const int khstart = (h >= kernel_h) ? ((h - kernel_h) % stride_h) + (kernel_h - stride_h) : h;
        const int kwstart = (w >= kernel_w) ? ((w - kernel_w) % stride_w) + (kernel_w - stride_w) : w;

        AccT res                       = 0;
        const Dtype *const input_slice = input + (n * channels + c) * conved_height * conved_width;

        const Dtype *const weight_slice = weight + c * kernel_h * kernel_w;

        for (int ph = phstart; ph < phend; ++ph) {
            for (int pw = pwstart; pw < pwend; ++pw) {
                int kh = khstart - (ph - phstart) * stride_h;
                int kw = kwstart - (pw - pwstart) * stride_w;
                res += (AccT)input_slice[ph * conved_width + pw] * (AccT)weight_slice[kh * kernel_w + kw];
            }
        }
        if (BIAS)
            output[index] = (Dtype)(res + (AccT)bias[c]);
        else
            output[index] = (Dtype)res;
    }
}

Status CudaDeconvolutionLayerAcc::Init(Context *context, LayerParam *param, LayerResource *resource,
                                       const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    DimsVector input_dims  = inputs[0]->GetBlobDesc().dims;
    DimsVector output_dims = outputs[0]->GetBlobDesc().dims;

    ConvLayerParam *conv_param = dynamic_cast<ConvLayerParam *>(param);

    ConvLayerResource *conv_resource = dynamic_cast<ConvLayerResource *>(resource);

    CUDA_CHECK(cudaMalloc((void **)&weights_, conv_resource->filter_handle.GetBytesSize()));
    CUDA_CHECK(cudaMemcpy(weights_, conv_resource->filter_handle.force_to<float *>(),
                          conv_resource->filter_handle.GetBytesSize(), cudaMemcpyHostToDevice));

    auto weight_fp16_buf = ConvertFloatToHalf(conv_resource->filter_handle);
    CUDA_CHECK(cudaMalloc((void **)&weights_fp16_, weight_fp16_buf.GetBytesSize()));
    CUDA_CHECK(cudaMemcpy(weights_fp16_, weight_fp16_buf.force_to<void *>(), weight_fp16_buf.GetBytesSize(),
                          cudaMemcpyHostToDevice));

    if (conv_param->bias) {
        CUDA_CHECK(cudaMalloc((void **)&bias_, conv_resource->bias_handle.GetBytesSize()));
        CUDA_CHECK(cudaMemcpy(bias_, conv_resource->bias_handle.force_to<float *>(),
                              conv_resource->bias_handle.GetBytesSize(), cudaMemcpyHostToDevice));
        auto bias_fp16_buf = ConvertFloatToHalf(conv_resource->bias_handle);
        CUDA_CHECK(cudaMalloc((void **)&bias_fp16_, bias_fp16_buf.GetBytesSize()));
        CUDA_CHECK(
            cudaMemcpy(bias_fp16_, bias_fp16_buf.force_to<void *>(), bias_fp16_buf.GetBytesSize(), cudaMemcpyHostToDevice));
    }

    return CudaLayerAcc::Init(context, param, resource, inputs, outputs);
}

CudaDeconvolutionLayerAcc::~CudaDeconvolutionLayerAcc() {}

Status CudaDeconvolutionLayerAcc::Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    return TNN_OK;
}

Status CudaDeconvolutionLayerAcc::Forward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto input = inputs[0];
    auto output = outputs[0];
    auto dtype = input->GetBlobDesc().data_type;
    ConvLayerParam *conv_param = dynamic_cast<ConvLayerParam *>(param_);

    auto input_data = reinterpret_cast<void *>(input->GetHandle().base);
    auto output_data = reinterpret_cast<void *>(output->GetHandle().base);

    auto input_dims = input->GetBlobDesc().dims;
    auto output_dims = output->GetBlobDesc().dims;

    auto count = DimsVectorUtils::Count(output_dims);
    if (dtype == DATA_TYPE_FLOAT) {
        if (conv_param->bias) {
            ConvBackward<float, float, true><<<TNN_CUDA_GET_BLOCKS(count), TNN_CUDA_NUM_THREADS, 0, context_->GetStream()>>>(
                count, (float*)input_data, output_dims[0], output_dims[1], output_dims[2], output_dims[3], input_dims[2],
                input_dims[3], conv_param->kernels[1], conv_param->kernels[0], conv_param->strides[1], conv_param->strides[0],
                conv_param->pads[2], conv_param->pads[0], (float*)output_data, (float*)weights_, (float*)bias_);
        } else {
            ConvBackward<float, float, false><<<TNN_CUDA_GET_BLOCKS(count), TNN_CUDA_NUM_THREADS, 0, context_->GetStream()>>>(
                count, (float*)input_data, output_dims[0], output_dims[1], output_dims[2], output_dims[3], input_dims[2],
                input_dims[3], conv_param->kernels[1], conv_param->kernels[0], conv_param->strides[1], conv_param->strides[0],
                conv_param->pads[2], conv_param->pads[0], (float*)output_data, (float*)weights_, (float*)bias_);            
        }
    } else if (dtype == DATA_TYPE_HALF) {
        if (conv_param->bias) {
            ConvBackward<__half, float, true><<<TNN_CUDA_GET_BLOCKS(count), TNN_CUDA_NUM_THREADS, 0, context_->GetStream()>>>(
                count, (__half*)input_data, output_dims[0], output_dims[1], output_dims[2], output_dims[3], input_dims[2],
                input_dims[3], conv_param->kernels[1], conv_param->kernels[0], conv_param->strides[1], conv_param->strides[0],
                conv_param->pads[2], conv_param->pads[0], (__half*)output_data, (__half*)weights_fp16_, (__half*)bias_fp16_);
        } else {
            ConvBackward<__half, float, false><<<TNN_CUDA_GET_BLOCKS(count), TNN_CUDA_NUM_THREADS, 0, context_->GetStream()>>>(
                count, (__half*)input_data, output_dims[0], output_dims[1], output_dims[2], output_dims[3], input_dims[2],
                input_dims[3], conv_param->kernels[1], conv_param->kernels[0], conv_param->strides[1], conv_param->strides[0],
                conv_param->pads[2], conv_param->pads[0], (__half*)output_data, (__half*)weights_fp16_, (__half*)bias_fp16_);
        }
    }

    return TNN_OK;
}

REGISTER_CUDA_ACC(Deconvolution, LAYER_DECONVOLUTION);

}  // namespace TNN_NS