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

#include <z3.h>

#include "tnn/device/cpu/acc/cpu_layer_acc.h"
#include "tnn/utils/dims_vector_utils.h"

namespace TNN_NS {

DECLARE_CPU_ACC(RoiAlign, LAYER_ROI_ALIGN);

Status CpuRoiAlignLayerAcc::Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    return TNN_OK;
}

Status CpuRoiAlignLayerAcc::Forward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto layer_param = dynamic_cast<RoiAlignLayerParam *>(param_);
    if (!layer_param) {
        LOGE("Error: RoiAlignLayerParam is nil\n");
        return Status(TNNERR_MODEL_ERR, "Error: RoiAlignLayerParam is nil");
    }

    Blob *input_blob       = inputs[0];
    Blob *rois             = inputs[1];
    Blob *output_blob      = outputs[0];
    int channel            = input_blob->GetBlobDesc().dims[1];
    int height             = input_blob->GetBlobDesc().dims[2];
    int width              = input_blob->GetBlobDesc().dims[3];
    int count              = DimsVectorUtils::Count(output_blob->GetBlobDesc().dims);
    const int image_size   = DimsVectorUtils::Count(output_blob->GetBlobDesc().dims, 1);
    const int channel_size = DimsVectorUtils::Count(output_blob->GetBlobDesc().dims, 2);
    if (0 == channel_size) {
        LOGE("Error: blob count is zero\n");
        return Status(TNNERR_COMMON_ERROR, "Error: blob count is zero");
    }

    auto num_roi        = rois->GetBlobDesc().dims[0];
    auto batch_size     = input_blob->GetBlobDesc().dims[0];
    auto sampling_ratio = layer_param->sampling_ratio;
    auto spatial_scale  = layer_param->spatial_scale;
    auto output_height  = layer_param->output_height;
    auto output_width   = layer_param->output_width;

    float *input_data  = static_cast<float *>(input_blob->GetHandle().base);
    float *output_data = static_cast<float *>(output_blob->GetHandle().base);
    float *rois_data   = static_cast<float *>(rois->GetHandle().base);

    for (int n = 0; n < num_roi; ++n) {
        int index         = n * 5;
        int roi_batch_ind = (int)rois_data[index + 0];

        auto x1 = rois_data[index + 1];
        auto y1 = rois_data[index + 2];
        auto x2 = rois_data[index + 3];
        auto y2 = rois_data[index + 4];

        // padding
        auto pad_w       = (x2 - x1 + 1.) * sampling_ratio;
        auto pad_h       = (y2 - y1 + 1.) * sampling_ratio;
        auto roi_start_w = (x1 - pad_w) * spatial_scale;
        auto roi_start_h = (y1 - pad_h) * spatial_scale;
        auto roi_end_w   = (x2 + pad_w) * spatial_scale;
        auto roi_end_h   = (y2 + pad_h) * spatial_scale;

        // clipping
        roi_start_w    = fmax(roi_start_w, 0.);
        roi_start_h    = fmax(roi_start_h, 0);
        int img_width  = round(width / spatial_scale);
        int img_height = round(height / spatial_scale);
        roi_end_w      = fmin((img_width - 1), roi_end_w);
        roi_end_h      = fmin((img_height - 1), roi_end_h);

        auto roi_height       = fmax(roi_end_h - roi_start_h + 1, 1);
        auto roi_width        = fmax(roi_end_w - roi_start_w + 1, 1);
        const auto bin_size_h = roi_height / output_height;
        const auto bin_size_w = roi_width / output_width;

        auto *batch_data = input_data + roi_batch_ind * image_size;

        {
            float fX0;
            float fX1;
            float fY0;
            float fY1;
            float fFactorA;
            float fFactorB;
            float fFactorC;
            float fFactorD;

            for (int c = 0; c < channel; ++c) {
                for (int ph = 0; ph < output_height; ++ph) {
                    for (int pw = 0; pw < output_width; ++pw) {
                        // Compute pooling region for this output unit:
                        //  start (included) = floor(ph * roi_height / pooled_height_)
                        //  end (excluded) = ceil((ph + 1) * roi_height / pooled_height_)
                        auto hcenter = (ph + 0.5) * bin_size_h;
                        auto wcenter = (pw + 0.5) * bin_size_w;

                        hcenter = fmin(fmax(hcenter + roi_start_h, 0), (height - 1));
                        wcenter = fmin(fmax(wcenter + roi_start_w, 0), (width - 1));

                        int hstart = fmin(fmax(hcenter, 0), (height - 1));
                        int wstart = fmin(fmax(wcenter, 0), (width - 1));
                        int hend   = fmin(fmax(hstart + 1, 0), height - 1);
                        int wend   = fmin(fmax(wstart + 1, 0), width - 1);

                        const int pool_index = ph * output_width + pw;

                        fX0      = wcenter - wstart;
                        fX1      = wend - wcenter;
                        fY0      = hcenter - hstart;
                        fY1      = hend - hcenter;
                        fFactorA = fY1 * fX1;
                        fFactorB = fY1 * fX0;
                        fFactorC = fY0 * fX1;
                        fFactorD = fY0 * fX0;

                        auto result = batch_data[hstart * width + wstart] * fFactorA +
                                      batch_data[hstart * width + wend] * fFactorB +
                                      batch_data[hend * width + wstart] * fFactorC +
                                      batch_data[hend * width + wend] * fFactorD;
                        output_data[pool_index] = result;
                        //[index_lb, index_rb, index_lt, index_rt, , w_lb, w_rb, w_lt, w_rt] for each top pixel
                        //                        argmax_data[4 * pool_index + 0] = hstart * width + wstart;
                        //                        argmax_data[4 * pool_index + 1] = hstart * width + wend;
                        //                        argmax_data[4 * pool_index + 2] = hend * width + wstart;
                        //                        argmax_data[4 * pool_index + 3] = hend * width + wend;
                        //                        w_data[4 * pool_index + 0] = fFactorA;
                        //                        w_data[4 * pool_index + 1] = fFactorB;
                        //                        w_data[4 * pool_index + 2] = fFactorC;
                        //                        w_data[4 * pool_index + 3] = fFactorD;
                    }
                }
                // Increment all data pointers by one channel
                batch_data += image_size;
                //                top_data += top[0]->offset(0, 1);
                //                argmax_data += bili_idx.offset(0, 1);
                //                w_data += bili_w.offset(0, 1);
            }
        }
    }
}

REGISTER_CPU_ACC(RoiAlign, LAYER_ROI_ALIGN);

}  // namespace TNN_NS
