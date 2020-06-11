// Copyright 2019 Tencent. All Rights Reserved

#include "device/cuda/acc/cuda_roi_pooling_layer_acc.h"
#include <iostream>
#include "device/cuda/cuda_utils.h"
#include "utils/dims_vector_utils.h"

namespace TNN_NS {

template <typename T>
__global__ void RoiPoolingForward2D(
        const int nthreads,
        const T* bottom_data,
        const T spatial_scale,
        const int channels,
        const int height,
        const int width,
        const int pooled_height,
        const int pooled_width,
        const T* bottom_rois,
        T* top_data,
        int* argmax_data) {
    CUDA_KERNEL_LOOP(index, nthreads) {
        // (n, c, ph, pw) is an element in the pooled output
        int pw = index % pooled_width;
        int ph = (index / pooled_width) % pooled_height;
        int c = (index / pooled_width / pooled_height) % channels;
        int n = index / pooled_width / pooled_height / channels;

        const T* offset_bottom_rois = bottom_rois + n * 5;
        int roi_batch_ind = offset_bottom_rois[0];
        int roi_start_w = roundf(offset_bottom_rois[1] * spatial_scale);
        int roi_start_h = roundf(offset_bottom_rois[2] * spatial_scale);
        int roi_end_w = roundf(offset_bottom_rois[3] * spatial_scale);
        int roi_end_h = roundf(offset_bottom_rois[4] * spatial_scale);

        // Force malformed ROIs to be 1x1
        int roi_width = max(roi_end_w - roi_start_w + 1, 1);
        int roi_height = max(roi_end_h - roi_start_h + 1, 1);
        T bin_size_h = static_cast<T>(roi_height) / static_cast<T>(pooled_height);
        T bin_size_w = static_cast<T>(roi_width) / static_cast<T>(pooled_width);

        int hstart = static_cast<int>(floor(static_cast<T>(ph) * bin_size_h));
        int wstart = static_cast<int>(floor(static_cast<T>(pw) * bin_size_w));
        int hend = static_cast<int>(ceil(static_cast<T>(ph + 1) * bin_size_h));
        int wend = static_cast<int>(ceil(static_cast<T>(pw + 1) * bin_size_w));

        // Add roi offsets and clip to input boundaries
        hstart = min(max(hstart + roi_start_h, 0), height);
        hend = min(max(hend + roi_start_h, 0), height);
        wstart = min(max(wstart + roi_start_w, 0), width);
        wend = min(max(wend + roi_start_w, 0), width);
        bool is_empty = (hend <= hstart) || (wend <= wstart);

        // Define an empty pooling region to be zero
        T maxval = is_empty ? 0 : -FLT_MAX;
        // If nothing is pooled, argmax = -1 causes nothing to be backprop'd
        int maxidx = -1;
        const T* offset_bottom_data =
                bottom_data + (roi_batch_ind * channels + c) * height * width;
        for (int h = hstart; h < hend; ++h) {
            for (int w = wstart; w < wend; ++w) {
                int bottom_index = h * width + w;
                if (offset_bottom_data[bottom_index] > maxval) {
                    maxval = offset_bottom_data[bottom_index];
                    maxidx = bottom_index;
                }
            }
        }
        top_data[index] = maxval;
        if (argmax_data) {
            argmax_data[index] = maxidx;
        }
    }
}


template <typename T>
__global__ void RoiPoolingForward3D(
        const int nthreads,
        const T* bottom_data,
        const T spatial_scale,
        const int channels,
        const int depth,
        const int height,
        const int width,
        const int pooled_depth,
        const int pooled_height,
        const int pooled_width,
        const T* bottom_rois,
        T* top_data,
        int* argmax_data) {

        int phw  = pooled_height * pooled_width;
        int pdhw = pooled_height * pooled_width * pooled_depth;

        CUDA_KERNEL_LOOP(index, nthreads) {
                // (n, c, pd, ph, pw) is an element in the pooled output
                int pw = index % pooled_width;
                int ph = (index / pooled_width) % pooled_height;
                int pd = (index / phw ) % pooled_depth;
                int c = (index / pdhw ) % channels;
                int n = index / pdhw / channels;

                const T* offset_bottom_rois = bottom_rois + n * 7;
                int roi_batch_ind = offset_bottom_rois[0];
                int roi_start_w = roundf(offset_bottom_rois[1] * spatial_scale);
                int roi_start_h = roundf(offset_bottom_rois[2] * spatial_scale);
                int roi_start_d = roundf(offset_bottom_rois[3] * spatial_scale);
                int roi_end_w = roundf(offset_bottom_rois[4] * spatial_scale);
                int roi_end_h = roundf(offset_bottom_rois[5] * spatial_scale);
                int roi_end_d = roundf(offset_bottom_rois[6] * spatial_scale);

                // Force malformed ROIs to be 1x1
                int roi_width = max(roi_end_w - roi_start_w + 1, 1);
                int roi_height = max(roi_end_h - roi_start_h + 1, 1);
                int roi_depth = max(roi_end_d - roi_start_d + 1, 1);
                T bin_size_h = static_cast<T>(roi_height) / static_cast<T>(pooled_height);
                T bin_size_w = static_cast<T>(roi_width) / static_cast<T>(pooled_width);
                T bin_size_d = static_cast<T>(roi_depth) / static_cast<T>(pooled_depth);

                int dstart = static_cast<int>(floor(static_cast<T>(pd) * bin_size_d));
                int hstart = static_cast<int>(floor(static_cast<T>(ph) * bin_size_h));
                int wstart = static_cast<int>(floor(static_cast<T>(pw) * bin_size_w));
                int dend = static_cast<int>(ceil(static_cast<T>(pd + 1) * bin_size_d));
                int hend = static_cast<int>(ceil(static_cast<T>(ph + 1) * bin_size_h));
                int wend = static_cast<int>(ceil(static_cast<T>(pw + 1) * bin_size_w));

                // Add roi offsets and clip to input boundaries
                dstart = min(max(dstart + roi_start_d, 0), depth);
                hstart = min(max(hstart + roi_start_h, 0), height);
                wstart = min(max(wstart + roi_start_w, 0), width);
                dend = min(max(dend + roi_start_d, 0), depth);
                hend = min(max(hend + roi_start_h, 0), height);
                wend = min(max(wend + roi_start_w, 0), width);
                bool is_empty = (hend <= hstart) || (wend <= wstart) || (dend <= dstart);

                // Define an empty pooling region to be zero
                T maxval = is_empty ? 0 : -FLT_MAX;
                // If nothing is pooled, argmax = -1 causes nothing to be backprop'd
                int maxidx = -1;
                const T* offset_bottom_data =
                                bottom_data + (roi_batch_ind * channels + c) * depth * height * width;
                for (int d = dstart; d < dend; ++d) {
                        for (int h = hstart; h < hend; ++h) {
                                for (int w = wstart; w < wend; ++w) {
                                        int bottom_index = d* height * width + h * width + w;
                                        if (offset_bottom_data[bottom_index] > maxval) {
                                                maxval = offset_bottom_data[bottom_index];
                                                maxidx = bottom_index;
                                        }
                                }
                        }
                }
                top_data[index] = maxval;
                if (argmax_data) {
                        argmax_data[index] = maxidx;
                }
        }
}


Status CudaRoiPoolingLayerAcc::Forward2D(const std::vector<Blob *> &inputs,
                                         const std::vector<Blob *> &outputs) {

    RoiPoolingLayerParam *pooling_param = dynamic_cast<RoiPoolingLayerParam *>(param_);

    float* bottom_data  = (float*) inputs[0]->GetHandle().base;
    float* roi_data     = (float*) inputs[1]->GetHandle().base;
    float* top_data     = (float*) outputs[0]->GetHandle().base;
    size_t out_count    = DimsVectorUtils::Count(outputs[ 0 ]->GetBlobDesc().dims); 

    float spatial_scale = pooling_param->spatial_scale;
    float pooled_width  = pooling_param->pooled_dims[0];
    float pooled_height = pooling_param->pooled_dims[1];

    RoiPoolingForward2D<float><<<RPD_GET_BLOCKS(out_count), RPD_CUDA_NUM_THREADS, 0, context_->stream_>>>
            (out_count, bottom_data, 
                    spatial_scale, 
                    blob_info_.input_c,
                    blob_info_.input_h,
                    blob_info_.input_w,
                    pooled_height, 
                    pooled_width,
                    roi_data,
                    top_data,
                    nullptr);

    return TNN_OK; 
}


Status CudaRoiPoolingLayerAcc::Forward3D(const std::vector<Blob *> &inputs,
                                         const std::vector<Blob *> &outputs) {

    RoiPoolingLayerParam *pooling_param = dynamic_cast<RoiPoolingLayerParam *>(param_);

    float* bottom_data  = (float*) inputs[0]->GetHandle().base;
    float* roi_data     = (float*) inputs[1]->GetHandle().base;
    float* top_data     = (float*) outputs[0]->GetHandle().base;
    size_t out_count    = DimsVectorUtils::Count(outputs[ 0 ]->GetBlobDesc().dims); 

    float spatial_scale = pooling_param->spatial_scale;
    float pooled_width  = pooling_param->pooled_dims[0];
    float pooled_height = pooling_param->pooled_dims[1];
    float pooled_depth  = pooling_param->pooled_dims[2];

    RoiPoolingForward3D<float><<<RPD_GET_BLOCKS(out_count), RPD_CUDA_NUM_THREADS, 0, context_->stream_>>>
            (out_count, bottom_data, 
                    spatial_scale, 
                    blob_info_.input_c,
                    blob_info_.input_d,
                    blob_info_.input_h,
                    blob_info_.input_w,
                    pooled_depth, 
                    pooled_height, 
                    pooled_width,
                    roi_data,
                    top_data,
                    nullptr);

    return TNN_OK; 
}

Status CudaRoiPoolingLayerAcc::Forward(const std::vector<Blob *> &inputs,
                                       const std::vector<Blob *> &outputs) {

    if (inputs[0]->GetBlobDesc().dims.size() == 5) {
            return this->Forward3D(inputs, outputs);
    } else {
            return this->Forward2D(inputs, outputs);
    }
}

}


