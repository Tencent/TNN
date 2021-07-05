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

#ifndef TNN_SOURCE_TNN_DEVICE_CUDA_ACC_CUDA_DETECTION_OUTPUT_LAYER_ACC_KERNEL_CUH_
#define TNN_SOURCE_TNN_DEVICE_CUDA_ACC_CUDA_DETECTION_OUTPUT_LAYER_ACC_KERNEL_CUH_

#include "tnn/device/cuda/acc/cuda_layer_acc.h"

namespace TNN_NS {

void decode_arm_loc_launcher(const float* arm_loc_data, const float* prior_data, const int num,
    const int m_num_priors, const int m_code_type, const bool m_variance_encoded_in_target,
    const bool clip_bbox, float * prior_decoded, cudaStream_t stream);

void decode_bboxes_all_launcher(const float* loc_data, const float* prior_data, const int num,
    const int m_num_priors, const int m_num_loc_classes, const int m_background_label_id,
    const int m_code_type, const bool m_share_location, const bool m_variance_encoded_in_target,
    const bool clip_bbox, const bool with_arm_loc, const float * prior_decoded, float* decode_bboxes,
    cudaStream_t stream);

int NMSFast(const float * decode_bboxes_d, const float * conf_data_d, const int num,
    const int m_num_classes, const int m_num_loc_classes, const int m_num_priors,
    const int m_background_label_id, const bool m_share_location, const int m_keep_top_k,
    const int m_top_k, const float m_confidence_threshold, const float m_nms_threshold,
    const float m_class_wise_nms_threshold, const float m_eta, const bool with_arm_conf,
    const float * arm_conf_data, const float m_objectness_score, void * m_d_temp_storage,
    const size_t m_temp_storage_bytes, std::vector<CudaTempBufUnit> tempbufs,
    float * all_out_d, int * all_out_size, int * num_kept, cudaStream_t stream);

}  //  namespace TNN_NS

#endif  //  TNN_SOURCE_TNN_DEVICE_CUDA_ACC_CUDA_DETECTION_OUTPUT_LAYER_ACC_KERNEL_CUH_