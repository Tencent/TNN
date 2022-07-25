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

#include "tnn/device/cuda/acc/cuda_layer_acc.h"
#include "tnn/utils/dims_utils.h"

namespace TNN_NS {

DECLARE_CUDA_ACC(LogSoftMax, LAYER_LOGSOFTMAX);

__global__ void log_softmax_channel_max_kernel(const int num, const int channels,
    const int spatial_dim, const float* data, float* out) {
    CUDA_KERNEL_LOOP(index, num*spatial_dim) {
        int n = index / spatial_dim;
        int s = index % spatial_dim;
        float maxval = -FLT_MAX;
        for (int c = 0; c < channels; ++c) {
            maxval = max(data[(n * channels + c) * spatial_dim + s], maxval);
        }
        out[index] = maxval;
    }
}

__global__ void log_softmax_channel_subtract_exp_kernel(const int count,
    const int num, const int channels,
    const int spatial_dim, const float* bottom_data, const float* channel_max, float* top_data) {
    CUDA_KERNEL_LOOP(index, count) {
        int n = index / channels / spatial_dim;
        int s = index % spatial_dim;
        top_data[index] = exp(bottom_data[index] - channel_max[n * spatial_dim + s]);
    }
}

__global__ void log_softmax_channel_sum_kernel(const int num, const int channels,
    const int spatial_dim, const float* data, float* channel_sum) {
    CUDA_KERNEL_LOOP(index, num * spatial_dim) {
        int n = index / spatial_dim;
        int s = index % spatial_dim;
        float sum = 0;
        for (int c = 0; c < channels; ++c) {
          sum += data[(n * channels + c) * spatial_dim + s];
        }
        channel_sum[index] = sum;
    }
}

__global__ void log_softmax_channel_div_log_kernel(const int count,
    const int num, const int channels,
    const int spatial_dim, const float* channel_sum, float* data) {
    CUDA_KERNEL_LOOP(index, count) {
        int n = index / channels / spatial_dim;
        int s = index % spatial_dim;
        data[index] = log(data[index] / channel_sum[n * spatial_dim + s]);
    }
}

Status CudaLogSoftMaxLayerAcc::Init(Context *context, LayerParam *param, LayerResource *resource,
        const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    Status ret = CudaLayerAcc::Init(context, param, resource, inputs, outputs);
    if (ret != TNN_OK) {
        return ret;
    }
    CreateTempBuf(DimsVectorUtils::Count(inputs[0]->GetBlobDesc().dims) * sizeof(float));
    return TNN_OK;
}

Status CudaLogSoftMaxLayerAcc::Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    return TNN_OK;
}

Status CudaLogSoftMaxLayerAcc::Forward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto params = dynamic_cast<LogSoftmaxLayerParam *>(param_);
    if (!params) {
        LOGE("Error: LogSoftMaxLayerParam is nil\n");
        return Status(TNNERR_MODEL_ERR, "Error: LogSoftMaxLayerParam is nil");
    }

    Blob *input_blob  = inputs[0];
    Blob *output_blob = outputs[0];
    auto dims = input_blob->GetBlobDesc().dims;
    int count = DimsVectorUtils::Count(dims);
    float* input_data = static_cast<float*>(input_blob->GetHandle().base);
    float* output_data = static_cast<float*>(output_blob->GetHandle().base);
    int axis = params->axis;
    axis = static_cast<int>((axis + dims.size()) % dims.size());
    int channel = dims[axis];
    int outer_num = DimsVectorUtils::Count(input_blob->GetBlobDesc().dims, 0, axis);
    int inner_num = DimsVectorUtils::Count(input_blob->GetBlobDesc().dims, axis + 1);

    log_softmax_channel_max_kernel<<<TNN_CUDA_GET_BLOCKS(outer_num * inner_num), TNN_CUDA_NUM_THREADS, 0, context_->GetStream()>>>
      (outer_num, channel, inner_num, input_data, (float*)tempbufs_[0].ptr);
    log_softmax_channel_subtract_exp_kernel<<<TNN_CUDA_GET_BLOCKS(count), TNN_CUDA_NUM_THREADS, 0, context_->GetStream()>>>
      (count, outer_num, channel, inner_num, input_data, (float*)tempbufs_[0].ptr, output_data);
    log_softmax_channel_sum_kernel<<<TNN_CUDA_GET_BLOCKS(outer_num * inner_num), TNN_CUDA_NUM_THREADS, 0, context_->GetStream()>>>
      (outer_num, channel, inner_num, output_data, (float*)tempbufs_[0].ptr);
    log_softmax_channel_div_log_kernel<<<TNN_CUDA_GET_BLOCKS(count), TNN_CUDA_NUM_THREADS, 0, context_->GetStream()>>>
      (count, outer_num, channel, inner_num, (float*)tempbufs_[0].ptr, output_data);

    return TNN_OK;
}

REGISTER_CUDA_ACC(LogSoftMax, LAYER_LOGSOFTMAX);

}  // namespace TNN_NS
