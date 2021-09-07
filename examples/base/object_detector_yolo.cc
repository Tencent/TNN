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

#include "object_detector_yolo.h"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <string>
#include <unordered_set>


namespace TNN_NS {
ObjectDetectorYoloOutput::~ObjectDetectorYoloOutput() {}

MatConvertParam ObjectDetectorYolo::GetConvertParamForInput(std::string name) {
    MatConvertParam input_convert_param;
    input_convert_param.scale = {1.0 / 255, 1.0 / 255, 1.0 / 255, 0.0};
    input_convert_param.bias  = {0.0, 0.0, 0.0, 0.0};
    return input_convert_param;
}

std::shared_ptr<Mat> ObjectDetectorYolo::ProcessSDKInputMat(std::shared_ptr<Mat> input_mat,
                                                                   std::string name) {
    return TNNSDKSample::ResizeToInputShape(input_mat, name);
}

std::shared_ptr<TNNSDKOutput> ObjectDetectorYolo::CreateSDKOutput() {
    return std::make_shared<ObjectDetectorYoloOutput>();
}

Status ObjectDetectorYolo::ProcessSDKOutput(std::shared_ptr<TNNSDKOutput> output_) {
    Status status = TNN_OK;
    
    auto output = dynamic_cast<ObjectDetectorYoloOutput *>(output_.get());
    RETURN_VALUE_ON_NEQ(!output, false,
                        Status(TNNERR_PARAM_ERR, "TNNSDKOutput is invalid"));
    
    auto output_mat_0 = output->GetMat("output");
    RETURN_VALUE_ON_NEQ(!output_mat_0, false,
                        Status(TNNERR_PARAM_ERR, "GetMat is invalid"));
    auto output_mat_1 = output->GetMat("463");
    RETURN_VALUE_ON_NEQ(!output_mat_1, false,
                        Status(TNNERR_PARAM_ERR, "GetMat is invalid"));
    auto output_mat_2 = output->GetMat("482");
    RETURN_VALUE_ON_NEQ(!output_mat_2, false,
                        Status(TNNERR_PARAM_ERR, "GetMat is invalid"));
    
    auto input_shape = GetInputShape();
    RETURN_VALUE_ON_NEQ(input_shape.size() ==4, true,
                        Status(TNNERR_PARAM_ERR, "GetInputShape is invalid"));
    
    std::vector<ObjectInfo> object_list;
    GenerateDetectResult({output_mat_0, output_mat_1, output_mat_2}, object_list, input_shape[3], input_shape[2]);
    output->object_list = object_list;
    return status;
}

void ObjectDetectorYolo::NMS(std::vector<ObjectInfo>& objs, std::vector<ObjectInfo>& results) {
    ::TNN_NS::NMS(objs, results, iou_threshold_, TNNHardNMS);
}

ObjectDetectorYolo::~ObjectDetectorYolo() {}

void ObjectDetectorYolo::PostProcessMat(std::vector<std::shared_ptr<Mat> >outputs, std::vector<std::shared_ptr<Mat> >& post_mats) {
    for (auto &output : outputs) {
        auto dims = output->GetDims();
        auto h_stride = DimsVectorUtils::Count(dims, 2);
        auto w_stride = DimsVectorUtils::Count(dims, 3);
        DimsVector permute_dims = {dims[0], dims[2], dims[3], dims[1] * dims[4]}; // batch, height, width, anchor*detect_dim
        auto mat = std::make_shared<Mat>(output->GetDeviceType(), output->GetMatType(), permute_dims);
        float *src_data = reinterpret_cast<float *>(output->GetData());
        float *dst_data = reinterpret_cast<float *>(mat->GetData());
        int out_idx = 0;
        for (int h = 0; h < permute_dims[1]; h++) {
            for (int w = 0; w < permute_dims[2]; w++) {
                for (int s = 0; s < permute_dims[3]; s++) {
                    int anchor_idx = s / dims[4];
                    int detect_idx = s % dims[4];
                    int in_idx = anchor_idx * h_stride + h * w_stride + w * dims[4] + detect_idx;
                    dst_data[out_idx++] = 1.0f / (1.0f + exp(-src_data[in_idx]));
                }
            }
        }

        post_mats.emplace_back(mat);
    }
}

void ObjectDetectorYolo::GenerateDetectResult(std::vector<std::shared_ptr<Mat> >outputs,
                                              std::vector<ObjectInfo>& detecs, int image_width, int image_height) {
    std::vector<ObjectInfo> extracted_objs;
    int blob_index = 0;

    std::vector<std::shared_ptr<Mat>> post_mats;
    PostProcessMat(outputs, post_mats);
    auto output_mats = post_mats;

    for(auto& output:output_mats){
        auto dim = output->GetDims();
  
        if(dim[3] != num_anchor_ * detect_dim_) {
            LOGE("Invalid detect output, the size of last dimension is: %d\n", dim[3]);
            return;
        }
        float* data = static_cast<float*>(output->GetData());
        
        int num_potential_detecs = dim[1] * dim[2] * num_anchor_;
        for(int i=0; i<num_potential_detecs; ++i){
            float x = data[i * detect_dim_ + 0];
            float y = data[i * detect_dim_ + 1];
            float width = data[i * detect_dim_ + 2];
            float height = data[i * detect_dim_ + 3];
            
            float objectness = data[i * detect_dim_ + 4];
            if(objectness < conf_thres)
                continue;
            //center point coord
            x = (x * 2 - 0.5 + ((i / num_anchor_) % dim[2])) * strides_[blob_index];
            y = (y * 2 - 0.5 + ((i / num_anchor_) / dim[2]) % dim[1]) * strides_[blob_index];
            width  = pow((width  * 2), 2) * anchor_grids_[blob_index * grid_per_input_ + (i%num_anchor_) * 2 + 0];
            height = pow((height * 2), 2) * anchor_grids_[blob_index * grid_per_input_ + (i%num_anchor_) * 2 + 1];
            // compute coords
            float x1 = x - width  / 2;
            float y1 = y - height / 2;
            float x2 = x + width  / 2;
            float y2 = y + height / 2;
            // compute confidence
            auto conf_start = data + i * detect_dim_ + 5;
            auto conf_end   = data + (i+1) * detect_dim_;
            auto max_conf_iter = std::max_element(conf_start, conf_end);
            int conf_idx = static_cast<int>(std::distance(conf_start, max_conf_iter));
            float score = (*max_conf_iter) * objectness;
            
            ObjectInfo obj_info;
            obj_info.image_width = image_width;
            obj_info.image_height = image_height;
            obj_info.x1 = x1;
            obj_info.y1 = y1;
            obj_info.x2 = x2;
            obj_info.y2 = y2;
            obj_info.score = score;
            obj_info.class_id = conf_idx;
            extracted_objs.push_back(obj_info);
        }
        blob_index += 1;
    }
    NMS(extracted_objs, detecs);
}

}
