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

#include "tnn/device/cuda/acc/cuda_detection_output_layer_acc.h"
#include "tnn/device/cuda/acc/cuda_detection_output_layer_acc_kernel.cuh"
#include "tnn/utils/dims_utils.h"
#include "tnn/utils/bbox_util.h"

namespace TNN_NS {

inline CodeType GetCodeType(const int number) {
    ASSERT(number > 0 && number < 4);

    switch (number) {
        case 1: {
            return PriorBoxParameter_CodeType_CORNER;
        }
        case 2: {
            return PriorBoxParameter_CodeType_CENTER_SIZE;
        }
        default: {
            return PriorBoxParameter_CodeType_CORNER_SIZE;
        }
    }
}

void CudaDetectionOutputLayerAcc::AllocTempBuf() {
    DetectionOutputLayerParam *params = dynamic_cast<DetectionOutputLayerParam *>(param_);
    int num_overlaps = top_k * (top_k - 1) / 2;
    if (params->keep_top_k > 0) {
        max_top_k = max_num * params->keep_top_k;
    } else {
        max_top_k = max_num * 256;
    }

    CreateTempBuf(max_num * num_loc_classes * num_priors * 4 * sizeof(float));
    CreateTempBuf(max_num * params->keep_top_k * 7 * sizeof(float));
    CreateTempBuf(max_num * params->num_classes * num_priors * sizeof(float));
    CreateTempBuf(max_num * params->num_classes * num_priors * sizeof(float));
    CreateTempBuf(max_num * params->num_classes * num_priors * sizeof(int));
    CreateTempBuf(max_num * params->num_classes * num_priors * sizeof(int));
    CreateTempBuf(max_num * params->num_classes * num_overlaps * sizeof(float));
    CreateTempBuf(max_num * params->num_classes * num_priors * sizeof(bool));
    CreateTempBuf(max_num * params->num_classes * sizeof(int));
    CreateTempBuf(max_num * params->num_classes * top_k * sizeof(float));
    CreateTempBuf(max_num * params->num_classes * top_k * sizeof(int));
    CreateTempBuf(max_num * params->keep_top_k * sizeof(float));
    CreateTempBuf(max_num * params->keep_top_k * sizeof(float));
    CreateTempBuf(max_num * sizeof(int));
    temp_storage_bytes = 32 * 1024 * 1024 + 256;
    CreateTempBuf(temp_storage_bytes);
}

Status CudaDetectionOutputLayerAcc::Init(Context *context, LayerParam *param, LayerResource *resource,
        const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    return CudaLayerAcc::Init(context, param, resource, inputs, outputs);
}

Status CudaDetectionOutputLayerAcc::Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    return TNN_OK;
}

Status CudaDetectionOutputLayerAcc::Forward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    Blob *input_blob1 = inputs[0];
    Blob *input_blob2 = inputs[1];
    Blob *input_blob3 = inputs[2];

    Blob *output_blob = outputs[0];
    float* loc_data_d = static_cast<float*>(input_blob1->GetHandle().base);
    float* conf_data_d = static_cast<float*>(input_blob2->GetHandle().base);
    float* prior_data_d = static_cast<float*>(input_blob3->GetHandle().base);

    int num = input_blob1->GetBlobDesc().dims[0];
    DetectionOutputLayerParam *params = dynamic_cast<DetectionOutputLayerParam *>(param_);
    CHECK_PARAM_NULL(params);

    if (tempbufs_.size() == 0) {
        max_num = num;
        num_priors = inputs[2]->GetBlobDesc().dims[2] / 4;
        num_loc_classes = params->share_location ? 1 : params->num_classes;
        top_k = std::min(params->nms_param.top_k, num_priors);
        AllocTempBuf();
    }

    if (num > max_num) {
        for (int i = 0; i < tempbufs_.size(); i++) {
            Status ret = device_->Free(tempbufs_[i].ptr);
            if (ret != TNN_OK) {
                LOGE("Error cuda free acc temp buf failed\n");
            }
        }
        tempbufs_.clear();
        max_num = num;
        AllocTempBuf();
    }

    CodeType code_type = GetCodeType(params->code_type);
    decode_bboxes_all_launcher(loc_data_d, prior_data_d, num, num_priors, num_loc_classes,
        params->background_label_id, code_type, params->share_location, params->variance_encoded_in_target,
        false, false, nullptr, (float*)tempbufs_[0].ptr, context_->GetStream());

    int *all_out_size = new int[num];
    int num_kept = 0;
    NMSFast((float*)tempbufs_[0].ptr, conf_data_d, num, params->num_classes, num_loc_classes, num_priors,
        params->background_label_id, params->share_location, params->keep_top_k, top_k, params->confidence_threshold,
        params->nms_param.nms_threshold, 1.001f, params->eta, false, nullptr, 0, (float*)tempbufs_[14].ptr,
        temp_storage_bytes, tempbufs_, (float*)tempbufs_[1].ptr, all_out_size, &num_kept, context_->GetStream());

    std::vector<int> top_shape(2, 1);
    top_shape.push_back(num_kept);
    top_shape.push_back(7);

    if (num_kept == 0) {
        top_shape[2] = num;
    }

    output_blob->GetBlobDesc().dims[2] = top_shape[2];
    float* top_data_d = static_cast<float*>(output_blob->GetHandle().base);
    if (num_kept == 0) {
        int out_size = DimsVectorUtils::Count(output_blob->GetBlobDesc().dims);
        float *top_data = new float[out_size];
        for (int vi = 0; vi < out_size; vi++) {
            top_data[vi] = -1;
        }
        for (int i = 0; i < num; ++i) {
            top_data[i * 7 + 0] = i;
        }
        CUDA_CHECK(cudaMemcpyAsync(top_data_d, top_data, out_size * sizeof(float), cudaMemcpyHostToDevice, context_->GetStream()));
        //TODO (johnzlli) need refactor
        CUDA_CHECK(cudaStreamSynchronize(context_->GetStream()));
        delete [] top_data;
    } else {
        CUDA_CHECK(cudaMemcpyAsync(top_data_d, tempbufs_[1].ptr, num_kept * 7 * sizeof(float), cudaMemcpyDeviceToDevice,
            context_->GetStream()));
    }

    delete [] all_out_size;
    return TNN_OK;
}

REGISTER_CUDA_ACC(DetectionOutput, LAYER_DETECTION_OUTPUT);

}  // namespace TNN_NS
