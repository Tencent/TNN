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

#include <cub/cub.cuh>
#include <cub/block/block_load.cuh>
#include <cub/block/block_store.cuh>
#include <cub/block/block_radix_sort.cuh>

#include "tnn/device/cuda/acc/cuda_detection_output_layer_acc_kernel.cuh"
#include "tnn/utils/bbox_util.h"

namespace TNN_NS {

__device__ void decode_bbox_one(const float* loc_data, const float* prior_bbox, const float* prior_variance,
        const int code_type, const bool variance_encoded_in_target, const bool clip_bbox, float* decode_bbox) {
    if (code_type == PriorBoxParameter_CodeType_CORNER) {
        if (variance_encoded_in_target) {
            // variance is encoded in target, we simply need to add the offset
            // predictions.
            #pragma unroll
            for(int i = 0; i < 4; i++) {
                decode_bbox[i] = prior_bbox[i] + loc_data[i];
            }
        } else {
            // variance is encoded in bbox, we need to scale the offset accordingly.
            #pragma unroll
            for(int i = 0; i < 4; i++) {
                decode_bbox[i] = prior_bbox[i] + prior_variance[i] * loc_data[i];
            }
        }
    } else if (code_type == PriorBoxParameter_CodeType_CENTER_SIZE) {
        float prior_width = prior_bbox[2] - prior_bbox[0];
        float prior_height = prior_bbox[3] - prior_bbox[1];
        float prior_center_x = (prior_bbox[2] + prior_bbox[0]) / 2.;
        float prior_center_y = (prior_bbox[3] + prior_bbox[1]) / 2.;

        float decode_bbox_center_x, decode_bbox_center_y;
        float decode_bbox_width, decode_bbox_height;
        if (variance_encoded_in_target) {
            // variance is encoded in target, we simply need to retore the offset
            // predictions.
            decode_bbox_center_x = loc_data[0] * prior_width + prior_center_x;
            decode_bbox_center_y = loc_data[1] * prior_height + prior_center_y;
            decode_bbox_width    = exp(loc_data[2]) * prior_width;
            decode_bbox_height   = exp(loc_data[3]) * prior_height;
        } else {
            // variance is encoded in bbox, we need to scale the offset accordingly.
            decode_bbox_center_x =
                    prior_variance[0] * loc_data[0] * prior_width + prior_center_x;
            decode_bbox_center_y =
                    prior_variance[1] * loc_data[1] * prior_height + prior_center_y;
            decode_bbox_width =
                    exp(prior_variance[2] * loc_data[2]) * prior_width;
            decode_bbox_height =
                    exp(prior_variance[3] * loc_data[3]) * prior_height;
        }

        decode_bbox[0] = (decode_bbox_center_x - decode_bbox_width / 2.);
        decode_bbox[1] = (decode_bbox_center_y - decode_bbox_height / 2.);
        decode_bbox[2] = (decode_bbox_center_x + decode_bbox_width / 2.);
        decode_bbox[3] = (decode_bbox_center_y + decode_bbox_height / 2.);
    } else if (code_type == PriorBoxParameter_CodeType_CORNER_SIZE) {
        float prior_width = prior_bbox[2] - prior_bbox[0];
        float prior_height = prior_bbox[3] - prior_bbox[1];
        if (variance_encoded_in_target) {
            // variance is encoded in target, we simply need to add the offset
            // predictions.
            decode_bbox[0] = (prior_bbox[0] + loc_data[0] * prior_width);
            decode_bbox[1] = (prior_bbox[1] + loc_data[1] * prior_height);
            decode_bbox[2] = (prior_bbox[2] + loc_data[2] * prior_width);
            decode_bbox[3] = (prior_bbox[3] + loc_data[3] * prior_height);
        } else {
            // variance is encoded in bbox, we need to scale the offset accordingly.
            decode_bbox[0] =
                        prior_bbox[0] + prior_variance[0] * loc_data[0] * prior_width;
            decode_bbox[1] = 
                        prior_bbox[1] + prior_variance[1] * loc_data[1] * prior_height;
            decode_bbox[2] = 
                        prior_bbox[2] + prior_variance[2] * loc_data[2] * prior_width;
            decode_bbox[3] = 
                        prior_bbox[3] + prior_variance[3] * loc_data[3] * prior_height;
        }
    } else {
        //LOG(FATAL) << "Unknown LocLossType.";
    }
    if (clip_bbox) {
        #pragma unroll
        for(int i = 0; i < 4; i++) {
            decode_bbox[i] = fmaxf(fminf(decode_bbox[i], 1.f), 0.f);
        }
    } 
}

template<int THREAD_PER_BLOCK, int ELE_PER_THREAD>
__global__ void decode_arm_loc(const float* arm_loc_data, const float* prior_data, const int num,
        const int m_num_priors, const int m_code_type, const bool m_variance_encoded_in_target, 
        const bool clip_bbox, float * prior_decoded) {
    for (int i = 0; i < ELE_PER_THREAD; i++) {
        int global_id = ELE_PER_THREAD * THREAD_PER_BLOCK * blockIdx.x + i * THREAD_PER_BLOCK + threadIdx.x;
        if (global_id >= num * m_num_priors ) continue;
        int idx_num = global_id / m_num_priors;
        int p = global_id % m_num_priors;

        const float* loc_inner      = &arm_loc_data[idx_num * m_num_priors  * 4 + p * 4];
        const float* prior_bbox     = &prior_data[p * 4];
        const float* prior_variance = &prior_data[m_num_priors * 4 + p * 4];
        float* decode_bbox  = &prior_decoded[idx_num * m_num_priors * 4 + p * 4];

        decode_bbox_one(loc_inner, prior_bbox, prior_variance, m_code_type, m_variance_encoded_in_target,
            clip_bbox, decode_bbox);
    }
}

template<int THREAD_PER_BLOCK, int ELE_PER_THREAD>
__global__ void decode_bboxes_all_kernel(const float* loc_data, const float* prior_data, const int num,
        const int m_num_priors, const int m_num_loc_classes, const int m_background_label_id,
        const int m_code_type, const bool m_share_location, const bool m_variance_encoded_in_target,
        const bool clip_bbox, const bool with_arm_loc, const float * prior_decoded, float* decode_bboxes) {
    int idx_num = blockIdx.y;
    for(int i = 0; i < ELE_PER_THREAD; i++) {
        int global_id = ELE_PER_THREAD * THREAD_PER_BLOCK * blockIdx.x + i * THREAD_PER_BLOCK + threadIdx.x;
        if (global_id >= m_num_loc_classes * m_num_priors) continue;
        int c = global_id / m_num_priors;
        int p = global_id % m_num_priors;

        int label = m_share_location ? -1 : c;
        if (label == m_background_label_id) {
            // Ignore background class.
            continue;
        }

        const float* loc_inner = &loc_data[idx_num * (m_num_priors * m_num_loc_classes * 4) +
            p * (m_num_loc_classes * 4) + c * 4];
        const float* arm_prior_bbox  = &prior_decoded[idx_num * m_num_priors * 4 + p * 4];
        const float* prior_bbox  = &prior_data[p * 4];
        const float* prior_variance = &prior_data[m_num_priors * 4 + p * 4];
        float* decode_bbox  = &decode_bboxes[idx_num * (m_num_loc_classes * m_num_priors * 4) +
            c * (m_num_priors * 4) + p * 4];

        if (with_arm_loc) {
            decode_bbox_one(loc_inner, arm_prior_bbox, prior_variance,
                m_code_type, m_variance_encoded_in_target, clip_bbox,
                decode_bbox);
        } else {
            decode_bbox_one(loc_inner, prior_bbox, prior_variance,
                m_code_type, m_variance_encoded_in_target, clip_bbox,
                decode_bbox);
        }
    }
}

template<int THREAD_PER_BLOCK, int ELE_PER_THREAD>
__global__ void nms_topk_init_kernel(const int num, const int m_num_classes, const int m_num_priors,
        const float *conf_data, const bool with_arm_conf, const float *arm_conf_data,
        const float m_objectness_score, float *key, int *value, int *sort_offset_start,
        int *sort_offset_end) {
    int i = blockIdx.y;
    int block_offset = blockIdx.x * THREAD_PER_BLOCK * ELE_PER_THREAD;
    int key_offset = i * m_num_classes * m_num_priors;
    for (int j = 0; j < ELE_PER_THREAD; j++) {
        int gid = block_offset + j * THREAD_PER_BLOCK + threadIdx.x; 
        // gid = c * m_num_priors + p;
        if (gid < m_num_classes * m_num_priors) {
            int c = gid / m_num_priors;
            int p = gid % m_num_priors;
            if (with_arm_conf && arm_conf_data[i * m_num_priors * 2 + p * 2 +1] < m_objectness_score ) {
                key[key_offset + gid] = (c == 0 ? 1.0 : 0);
            } else {
                key[key_offset + gid] = conf_data[key_offset + p * m_num_classes + c];
            }
            value[key_offset + gid] = gid;
            // segmented sort for nms [per class sort]
            if (p==0) {
                sort_offset_start[i * m_num_classes + c] = (i * m_num_classes + c ) * m_num_priors;
                sort_offset_end[i * m_num_classes + c]   = (i * m_num_classes + c + 1) * m_num_priors;
            }
        }
    }
}

__device__ inline float JaccardOverlap(const float xmin1, const float xmin2, const float ymin1, const float ymin2, 
    const float xmax1, const float xmax2, const float ymax1, const float ymax2, const bool normalized) {

    float norm_add = normalized ? 0.f : 1.f;
    float left  = max(xmin1, xmin2), right  = min(xmax1, xmax2);
    float top   = max(ymin1, ymin2), bottom = min(ymax1, ymax2);
    float width = max(right - left + norm_add, 0.f), height = max(bottom - top + norm_add, 0.f);
    float interS = width * height;
    float Sa = (xmax1 - xmin1 + norm_add) * (ymax1 - ymin1 + norm_add);
    float Sb = (xmax2 - xmin2 + norm_add) * (ymax2 - ymin2 + norm_add);
    return interS / (Sa + Sb - interS);
}

__device__ inline float JaccardOverlap(const int a, const int b, const float* decode_bboxes, const bool normalized) {
    float xmin1, xmin2;
    float ymin1, ymin2;
    float xmax1, xmax2;
    float ymax1, ymax2;

    xmin1 = decode_bboxes[a * 4 + 0];
    ymin1 = decode_bboxes[a * 4 + 1];
    xmax1 = decode_bboxes[a * 4 + 2];
    ymax1 = decode_bboxes[a * 4 + 3];

    xmin2 = decode_bboxes[b * 4 + 0];
    ymin2 = decode_bboxes[b * 4 + 1];
    xmax2 = decode_bboxes[b * 4 + 2];
    ymax2 = decode_bboxes[b * 4 + 3];

    return JaccardOverlap(xmin1, xmin2, ymin1, ymin2, xmax1, xmax2, ymax1, ymax2, normalized);
}

template<int THREAD_PER_BLOCK, int ELE_PER_THREAD, int TILE_X, int TILE_Y>
__global__ void jaccardoverlap_kernel(const int top_k, const int num_overlaps, const int m_num_classes,
        const int m_num_loc_classes, const bool m_share_location, const int m_background_label_id,
        const int *indices, const int m_num_priors, const float *decode_bboxes, float *overlaps) {
    int tile_idx = blockIdx.x;
    int tile_idy = blockIdx.y;
    int idx_num = blockIdx.z / m_num_classes;
    int c = blockIdx.z % m_num_classes;
    if (c == m_background_label_id) {
        return ;
    }

    int start_x = tile_idx * TILE_X;
    int start_y = tile_idy * TILE_Y;

    if (start_y >= start_x + TILE_X) {
        return ;
    }

    __shared__ float bbox[(TILE_X + TILE_Y) * 4];

    int lablel = m_share_location ? 0 : c;
    int bbox_offset = idx_num * (m_num_loc_classes * m_num_priors * 4) + lablel * (m_num_priors * 4);
    int index_offset = idx_num * m_num_classes * m_num_priors + c * m_num_priors;
    int overlap_offset = idx_num * m_num_classes * num_overlaps + c * num_overlaps;

    if (threadIdx.x  < TILE_X * 4) {
        int a = start_x + threadIdx.x / 4;
        int index = indices[index_offset + a] % m_num_priors;
        bbox[threadIdx.x ] = decode_bboxes[bbox_offset + index * 4 + threadIdx.x % 4];
    }
    if (threadIdx.x  < (TILE_X + TILE_Y) * 4 && threadIdx.x >= TILE_X *4 ) {
        int b = start_y + (threadIdx.x - TILE_X * 4) / 4;
        int index = indices[index_offset + b] % m_num_priors;
        bbox[threadIdx.x ] = decode_bboxes[bbox_offset + index * 4 + threadIdx.x % 4];
    }

    __syncthreads();

    #pragma unroll  
    for(int i = 0; i < ELE_PER_THREAD; i++) {
        int inner_tile_id = i * THREAD_PER_BLOCK + threadIdx.x;
        int tile_a = inner_tile_id % TILE_X ;
        int tile_b = inner_tile_id / TILE_X;
        int a = tile_a + start_x;
        int b = tile_b + start_y;
        int gid = a * (a - 1) / 2 + b;
        if (b < a && gid < num_overlaps) {
            float xmin1 = bbox[tile_a * 4 + 0];
            float ymin1 = bbox[tile_a * 4 + 1];
            float xmax1 = bbox[tile_a * 4 + 2];
            float ymax1 = bbox[tile_a * 4 + 3];
            float xmin2 = bbox[TILE_X * 4 + tile_b * 4 + 0];
            float ymin2 = bbox[TILE_X * 4 + tile_b * 4 + 1];
            float xmax2 = bbox[TILE_X * 4 + tile_b * 4 + 2];
            float ymax2 = bbox[TILE_X * 4 + tile_b * 4 + 3];
            overlaps[overlap_offset + gid] = JaccardOverlap(xmin1, xmin2, ymin1, ymin2, xmax1, xmax2, ymax1, ymax2, true);
        }
    }
}

template<int THREAD_PER_BLOCK, int ELE_PER_THREAD>
__global__ void jaccardoverlap_batch_kernel(const int top_k, const int m_keep_top_k, const int m_num_priors,
        const int m_num_classes, const int m_num_loc_classes, const bool m_share_location, const int *indices,
        const int *num_select_d, const float *decode_bboxes, float *overlaps, int *indices_out) {
    int idx_batch = blockIdx.y;
    int keep_top_k = min(m_keep_top_k, num_select_d[idx_batch]);
    int num_overlaps = keep_top_k * (keep_top_k - 1) / 2;

    int block_offset = blockIdx.x * THREAD_PER_BLOCK * ELE_PER_THREAD;
    int index_offset = idx_batch * m_num_classes * top_k;
    int bbox_offset  = idx_batch* (m_num_loc_classes * m_num_priors * 4);
    int overlap_offset =  idx_batch* m_keep_top_k * (m_keep_top_k- 1) / 2;

    for(int i = 0; i < ELE_PER_THREAD; i++) {
        int gid =  block_offset + i * THREAD_PER_BLOCK + threadIdx.x;
        if (gid < num_overlaps) {
            int a = floorf(sqrtf(gid * 2));
            if (gid * 2 < a * (a + 1)) {
                a -= 1;
            }
            a += 1;
            int b = gid - a * (a - 1) / 2;

            int c_a  = indices[index_offset + a] / m_num_priors;
            int c_b  = indices[index_offset + b] / m_num_priors;

            int label_a  = m_share_location ? 0 : c_a;
            int label_b  = m_share_location ? 0 : c_b;

            int idxa = indices[index_offset + a] % m_num_priors + label_a * m_num_priors;
            int idxb = indices[index_offset + b] % m_num_priors + label_b * m_num_priors;

            overlaps[overlap_offset + gid] = JaccardOverlap(idxa, idxb, &decode_bboxes[bbox_offset], true);
        }
        if (gid < m_num_classes * top_k) {
            indices_out[index_offset + gid] = idx_batch * m_num_classes * m_num_priors + indices[index_offset + gid];
        }
    }
}

template<int THREAD_PER_BLOCK>
__global__ void adaptive_nms_kernel(const int top_k, const float m_nms_threshold, const float m_eta,
        const int m_background_label_id, const int m_num_classes, const int num_overlaps, const int m_num_priors,
        const float *overlaps, const float *conf_sorted, const float m_confidence_threshold, bool *keep, int *num) {
    int overlap_offset  = blockIdx.x * m_num_classes * num_overlaps + blockIdx.y * num_overlaps;
    int keep_offset     = blockIdx.x * m_num_classes * m_num_priors  + blockIdx.y * m_num_priors;
    int num_offset = blockIdx.x * m_num_classes + blockIdx.y;
    int conf_offset = blockIdx.x * m_num_classes * m_num_priors + blockIdx.y * m_num_priors;
    if (blockIdx.y == m_background_label_id) {
        return;
    }

    __shared__ int keep_flag;
    __shared__ float adaptive_threshold;
    __shared__ bool first_rank_fail;
    int keep_count = 1;

    int tid = threadIdx.x;
    if (tid == 0) {
        keep[keep_offset + 0]=true;
        keep_flag = 1;
        adaptive_threshold = m_nms_threshold;

        first_rank_fail = (conf_sorted[conf_offset + 0] > m_confidence_threshold ? false : true);
        if (first_rank_fail) {
            num[num_offset] = 0;
        }
    }
    __syncthreads();  
    if (first_rank_fail) {
        for(int i = 0; i < top_k; i += THREAD_PER_BLOCK) {
            int pr = i + threadIdx.x;
            if (pr < top_k){
                keep[keep_offset + pr] = false;
            }
        }
        return;
    }

    for(int r = 1; r < top_k; r++) {
        __syncthreads();  
        if (conf_sorted[conf_offset + r] > m_confidence_threshold) {
            for(int i = 0; i < r; i += THREAD_PER_BLOCK) {
                int pr = i + threadIdx.x;
                if (pr < r) {
                    if (keep[keep_offset + pr] && overlaps[overlap_offset + r * (r - 1) / 2 + pr] > adaptive_threshold) {
                        atomicCAS(&keep_flag, 1, 0);
                    }
                }
            }
        } else {
            if (tid==0) {
                keep_flag = 0;
            }
        }
        __syncthreads();  
        if (tid ==0 ) {
            if (keep_flag == 1) {
                keep[keep_offset + r] = true;
                keep_count += 1;
                if (m_eta < 1 && adaptive_threshold > 0.5) {
                    adaptive_threshold *= m_eta;
                }
            } else {
                keep[keep_offset + r] = false;
            }
            keep_flag = 1;
        }
        __syncthreads();  
    }
    if (tid == 0) {
        num[num_offset] = keep_count;
    }
    __syncthreads();  

    return ;
}

template<int THREAD_PER_BLOCK>
__global__ void adaptive_nms_batch_kernel(const int m_keep_top_k, const int batch_offset, const float m_nms_threshold,
        const float m_eta, const float *overlaps, const float *conf_sorted, const float m_confidence_threshold,
        const int *num_select_d, bool *keep, int *num) {
    int overlap_offset  = blockIdx.x * m_keep_top_k * (m_keep_top_k - 1) / 2;
    int keep_offset     = blockIdx.x * batch_offset;
    int conf_offset     = blockIdx.x * batch_offset;
    int num_offset      = blockIdx.x ;
    int keep_top_k      = min(m_keep_top_k, num_select_d[blockIdx.x]);

    __shared__ int keep_flag;
    __shared__ float adaptive_threshold;
    __shared__ bool first_rank_fail;
    int keep_count = 1;

    int tid = threadIdx.x;
    if (tid == 0) {
        keep[keep_offset + 0] = true;
        keep_flag = 1;
        adaptive_threshold = m_nms_threshold;

        first_rank_fail = ( keep_top_k == 0 ) ||  (conf_sorted[conf_offset + 0] > m_confidence_threshold ? false : true);
        if (first_rank_fail) {
            num[num_offset] = 0;
        }
    }
    __syncthreads();  
    if (first_rank_fail){
        for(int i=0;i<m_keep_top_k;i+=THREAD_PER_BLOCK){
            int pr = i + threadIdx.x;
            if (pr < m_keep_top_k){
                keep[keep_offset + pr] = false;
            }
        }
        return;
    }

    for(int r = 1; r < keep_top_k; r++) {
        __syncthreads();  
        if (conf_sorted[conf_offset + r] > m_confidence_threshold) {
            for(int i = 0; i < r; i += THREAD_PER_BLOCK) {
                int pr = i + threadIdx.x;
                if (pr < r) {
                    if (keep[keep_offset + pr] && overlaps[overlap_offset + r * (r - 1) / 2 + pr] > adaptive_threshold) {
                        atomicCAS(&keep_flag, 1, 0);
                    }
                }
            }
        } else {
            if (tid == 0) {
                keep_flag = 0;
            }
        }
        __syncthreads();  
        if (tid ==0) {
            if (keep_flag == 1) {
                keep[keep_offset + r] = true;
                keep_count += 1;
                if (m_eta < 1 && adaptive_threshold > 0.5) {
                    adaptive_threshold *= m_eta;
                }
            } else {
                keep[keep_offset + r] = false;
            }
            keep_flag = 1;
        }
        __syncthreads();  
    }
    if (tid == 0) {
        num[num_offset] = keep_count;
    }
    __syncthreads();  

    return ;
}

template<int THREAD_PER_BLOCK, int ELE_PER_THREAD>
__global__ void write_output_batch_nms_kernel(const int num_kept, const int m_num_classes, const int m_num_loc_classes,
        const int m_num_priors, const bool m_share_location, const float * conf_kept_out, const int * index_kept_out,
        const float * decode_bboxes, float* all_out) {
    int block_offset = blockIdx.x * THREAD_PER_BLOCK * ELE_PER_THREAD;
    for (int iele = 0; iele < ELE_PER_THREAD; iele++) {
        int out_offset =  block_offset + iele * THREAD_PER_BLOCK + threadIdx.x;
        if (out_offset < num_kept){
            int i       = index_kept_out[out_offset] / (m_num_classes * m_num_priors); 
            int index   = index_kept_out[out_offset] % (m_num_classes * m_num_priors);
            float score = conf_kept_out[out_offset];

            int p = index % m_num_priors;
            int c = index / m_num_priors;

            int lablel = m_share_location ? 0 : c;
            int bbox_offset = i * (m_num_loc_classes * m_num_priors * 4) + lablel * (m_num_priors * 4) + p * 4;
            float xmin = decode_bboxes[bbox_offset + 0];
            float ymin = decode_bboxes[bbox_offset + 1];
            float xmax = decode_bboxes[bbox_offset + 2];
            float ymax = decode_bboxes[bbox_offset + 3];

            // int all_out_offset = i * m_keep_top_k * 7;
            all_out[out_offset * 7 + 0] = i;
            all_out[out_offset * 7 + 1] = c;
            all_out[out_offset * 7 + 2] = score;
            all_out[out_offset * 7 + 3] = xmin;
            all_out[out_offset * 7 + 4] = ymin;
            all_out[out_offset * 7 + 5] = xmax;
            all_out[out_offset * 7 + 6] = ymax;
        }
    }
}

__global__ void set_sort_offset(const int num, const int m_num_classes, const int top_k, 
        const int *num_select_d, int *sort_offset_start,  int *sort_offset_end) {
    int i = threadIdx.x;
    if (i < num) {
        sort_offset_start[i] = i * m_num_classes * top_k;
        sort_offset_end[i] = i * m_num_classes * top_k + num_select_d[i];
    }
}

void decode_arm_loc_launcher(const float* arm_loc_data, const float* prior_data, const int num, const int m_num_priors, 
        const int m_code_type, const bool m_variance_encoded_in_target, const bool clip_bbox, float * prior_decoded,
        cudaStream_t stream) {
    const int THREAD_PER_BLOCK = 128;
    const int ELE_PER_THREAD = 16;
    dim3 griddim;
    griddim.x = (num * m_num_priors + ELE_PER_THREAD * THREAD_PER_BLOCK - 1) /(ELE_PER_THREAD * THREAD_PER_BLOCK);
    griddim.y = 1;
    decode_arm_loc<THREAD_PER_BLOCK, ELE_PER_THREAD><<<griddim, THREAD_PER_BLOCK, 0, stream>>>(arm_loc_data,
        prior_data, num, m_num_priors, m_code_type, m_variance_encoded_in_target, false, prior_decoded);
}

void decode_bboxes_all_launcher(const float* loc_data, const float* prior_data, const int num, const int m_num_priors, 
        const int m_num_loc_classes, const int m_background_label_id, const int m_code_type, const bool m_share_location,
        const bool m_variance_encoded_in_target, const bool clip_bbox, const bool with_arm_loc, const float * prior_decoded,
        float* decode_bboxes, cudaStream_t stream) {
    const int THREAD_PER_BLOCK = 128;
    const int ELE_PER_THREAD = 16;

    dim3 griddim;
    griddim.x = (m_num_loc_classes * m_num_priors + ELE_PER_THREAD * THREAD_PER_BLOCK - 1) /(ELE_PER_THREAD * THREAD_PER_BLOCK);
    griddim.y = num;
    
    decode_bboxes_all_kernel<THREAD_PER_BLOCK, ELE_PER_THREAD><<<griddim, 128, 0, stream>>>(loc_data, prior_data, num,
        m_num_priors, m_num_loc_classes, m_background_label_id, m_code_type, m_share_location, m_variance_encoded_in_target,
        clip_bbox, with_arm_loc, prior_decoded, decode_bboxes);
}

int NMSFast(const float * decode_bboxes_d, const float * conf_data_d, const int num, const int m_num_classes,
        const int m_num_loc_classes, const int m_num_priors, const int m_background_label_id, const bool m_share_location,
        const int m_keep_top_k, const int m_top_k, const float m_confidence_threshold, const float m_nms_threshold,
        const float m_class_wise_nms_threshold, const float m_eta, const bool with_arm_conf, const float * arm_conf_data,
        const float m_objectness_score, void * m_d_temp_storage, const size_t m_temp_storage_bytes, std::vector<CudaTempBufUnit> tempbufs, 
        float * all_out_d, int * all_out_size, int * num_kept, cudaStream_t stream) {
    float *key_d = (float*)tempbufs[2].ptr;
    float *key_out_d = (float*)tempbufs[3].ptr;
    int *value_d = (int*)tempbufs[4].ptr;
    int *value_out_d = (int*)tempbufs[5].ptr;

    float *overlaps_d = (float*)tempbufs[6].ptr;
    bool *keep_d = (bool*)tempbufs[7].ptr;
    int *num_kept_per_class_d = (int*)tempbufs[8].ptr;

    CUDA_CHECK(cudaMemset(keep_d, 0, num * m_num_classes * m_num_priors * sizeof(bool)));

    const int top_k = min(m_top_k, m_num_priors);
    int num_overlaps = top_k * (top_k-1) / 2;

    int *sort_offset_start = (int*)tempbufs[11].ptr;
    int *sort_offset_end = (int*)tempbufs[12].ptr;

    const int THREAD_PER_BLOCK = 128;
    const int ELE_PER_THREAD = 16;
    dim3 initdim;
    initdim.x = (m_num_priors * m_num_classes + ELE_PER_THREAD * THREAD_PER_BLOCK - 1) / (ELE_PER_THREAD * THREAD_PER_BLOCK);
    initdim.y = num;

    nms_topk_init_kernel<THREAD_PER_BLOCK, ELE_PER_THREAD><<<initdim, THREAD_PER_BLOCK, 0, stream>>>(num, m_num_classes, m_num_priors,
        conf_data_d, with_arm_conf, arm_conf_data, m_objectness_score, key_d, value_d, sort_offset_start, sort_offset_end);

    void * d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;

    // Sort by confidence in each class
    CubDebug(cub::DeviceSegmentedRadixSort::SortPairsDescending(d_temp_storage, temp_storage_bytes, key_d, key_out_d, 
        value_d, value_out_d, num * m_num_classes * m_num_priors, num *m_num_classes, sort_offset_start, sort_offset_end));

    bool malloc_flag = false;
    if (temp_storage_bytes > m_temp_storage_bytes) {
        malloc_flag = true;
        CUDA_CHECK(cudaMalloc((void**)&d_temp_storage, temp_storage_bytes));
    } else {
        d_temp_storage = m_d_temp_storage;
        temp_storage_bytes = m_temp_storage_bytes;
    }

    CubDebug(cub::DeviceSegmentedRadixSort::SortPairsDescending(d_temp_storage, temp_storage_bytes, key_d, key_out_d, 
        value_d, value_out_d, num * m_num_classes * m_num_priors, num * m_num_classes, sort_offset_start, sort_offset_end));

    dim3 griddim;
    {
        const int tile_x = 16;
        const int tile_y = 16;
        // THREAD_PER_BLOCK > (tile_x + tile_y) * 4
        const int THREAD_PER_BLOCK = 128;
        const int ELE_PER_THREAD = tile_x * tile_y / THREAD_PER_BLOCK;

        griddim.x = (top_k + tile_x-1) / tile_x;
        griddim.y = (top_k + tile_y-1) / tile_y;
        griddim.z = num * m_num_classes;

        // calculate jaccardoverlap in each class for the top_k predictions 
        jaccardoverlap_kernel<THREAD_PER_BLOCK, ELE_PER_THREAD, tile_x, tile_y><<<griddim, THREAD_PER_BLOCK, 0, stream>>>(top_k,
            num_overlaps, m_num_classes, m_num_loc_classes, m_share_location, m_background_label_id, value_out_d, m_num_priors,
            decode_bboxes_d, overlaps_d);
        
    }

    dim3 nms_dim;
    nms_dim.x = num;
    nms_dim.y = m_num_classes;

    // apply nms in each class
    if (top_k <= 256) {
        adaptive_nms_kernel<256><<<nms_dim, 256, 0, stream>>>(top_k, m_nms_threshold, m_eta, 
            m_background_label_id, m_num_classes, num_overlaps, m_num_priors,
            overlaps_d , key_out_d, m_confidence_threshold,
            keep_d , num_kept_per_class_d);
    } else if (top_k <=512) {
        adaptive_nms_kernel<512><<<nms_dim, 512, 0, stream>>>(top_k, m_nms_threshold, m_eta, 
            m_background_label_id, m_num_classes, num_overlaps, m_num_priors,
            overlaps_d , key_out_d, m_confidence_threshold,
            keep_d , num_kept_per_class_d);
    } else {
        adaptive_nms_kernel<1024><<<nms_dim, 1024, 0, stream>>>(top_k, m_nms_threshold, m_eta, 
            m_background_label_id, m_num_classes, num_overlaps, m_num_priors,
            overlaps_d , key_out_d, m_confidence_threshold,
            keep_d , num_kept_per_class_d);
    }

    // variables for select kept values
    float *conf_kept_out_d = (float*)tempbufs[9].ptr;
    int *index_kept_out_d = (int*)tempbufs[10].ptr;
    int *num_select_d = (int*) tempbufs[13].ptr;

    // select those kept predictions
    for (int i = 0; i < num; i++) { 
        float *key_per_n_d = key_out_d + i * m_num_classes * m_num_priors;
        int *value_per_n_d = value_out_d + i * m_num_classes * m_num_priors;
        bool *keep_per_n_d = keep_d + i * m_num_classes * m_num_priors;

        float *conf_per_n_out_d = conf_kept_out_d + i * m_num_classes * top_k;
        int *index_per_n_out_d = index_kept_out_d + i * m_num_classes * top_k;

        cub::DeviceSelect::Flagged(d_temp_storage, temp_storage_bytes, key_per_n_d, keep_per_n_d, conf_per_n_out_d,
            num_select_d + i, m_num_classes * m_num_priors);
        cub::DeviceSelect::Flagged(d_temp_storage, temp_storage_bytes, value_per_n_d, keep_per_n_d, index_per_n_out_d,
            num_select_d + i, m_num_classes * m_num_priors);
    }

    // sort by confidence in each batch 
    set_sort_offset<<<1, 1024, 0, stream>>>(num, m_num_classes, top_k, num_select_d, sort_offset_start, sort_offset_end);

    CubDebug(cub::DeviceSegmentedRadixSort::SortPairsDescending(d_temp_storage, temp_storage_bytes,
        conf_kept_out_d, key_out_d, index_kept_out_d, value_out_d, num * m_num_classes * top_k, 
        num, sort_offset_start, sort_offset_end));

    num_overlaps = m_keep_top_k * (m_keep_top_k - 1) / 2;
    griddim.x = (num_overlaps + ELE_PER_THREAD * THREAD_PER_BLOCK - 1) / (ELE_PER_THREAD * THREAD_PER_BLOCK);
    griddim.y = num;
    griddim.z = 1;

    // calculate jaccardoverlap in each batch for the top keep_top_k predictions
    jaccardoverlap_batch_kernel<THREAD_PER_BLOCK, ELE_PER_THREAD><<<griddim, THREAD_PER_BLOCK, 0, stream>>>(top_k,
        m_keep_top_k, m_num_priors,m_num_classes, m_num_loc_classes, m_share_location, value_out_d, num_select_d,
        decode_bboxes_d, overlaps_d, value_d);

    CUDA_CHECK(cudaMemset(keep_d, 0, num * m_num_classes * top_k * sizeof(bool)));

    nms_dim.x = num;
    nms_dim.y = 1;

    // applay nms in each batch
    adaptive_nms_batch_kernel<1024><<<nms_dim, 1024, 0, stream>>>(m_keep_top_k, m_num_classes * top_k,
        m_class_wise_nms_threshold, m_eta, overlaps_d, key_out_d, m_confidence_threshold, num_select_d,
        keep_d , num_kept_per_class_d);

    // select those kept predictions for each batch
    cub::DeviceSelect::Flagged(d_temp_storage, temp_storage_bytes, value_d, keep_d, index_kept_out_d,
        num_kept_per_class_d+num, num * m_num_classes * top_k);
    cub::DeviceSelect::Flagged(d_temp_storage, temp_storage_bytes, key_out_d, keep_d, conf_kept_out_d,
        num_kept_per_class_d+num, num * m_num_classes * top_k);

    CUDA_CHECK(cudaMemcpy(num_kept, num_kept_per_class_d+ num, sizeof(int), cudaMemcpyDeviceToHost));

    if (*num_kept > 0) {
        dim3 writedim;
        writedim.x = (*num_kept + ELE_PER_THREAD * THREAD_PER_BLOCK - 1) / (ELE_PER_THREAD * THREAD_PER_BLOCK);
        write_output_batch_nms_kernel<THREAD_PER_BLOCK, ELE_PER_THREAD><<<writedim, THREAD_PER_BLOCK, 0, stream>>>(*num_kept,
            m_num_classes, m_num_loc_classes,m_num_priors, m_share_location, conf_kept_out_d, index_kept_out_d, decode_bboxes_d,
            all_out_d);
    }

    if (malloc_flag) {
        CUDA_CHECK(cudaFree(d_temp_storage));
    }

    return 0; 
} 

}  //  namespace TNN_NS