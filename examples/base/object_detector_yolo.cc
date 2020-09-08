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
#include <sys/time.h>


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
    auto target_dims = GetInputShape(name);
    auto input_height = input_mat->GetHeight();
    auto input_width = input_mat->GetWidth();
    if (target_dims.size() >= 4 &&
        (input_height != target_dims[2] || input_width != target_dims[3])) {
        auto target_mat = std::make_shared<TNN_NS::Mat>(input_mat->GetDeviceType(),
                                                        input_mat->GetMatType(), target_dims);
        auto status = Resize(input_mat, target_mat, TNNInterpLinear);
        if (status == TNN_OK) {
            return target_mat;
        } else {
            LOGE("%s\n", status.description().c_str());
            return nullptr;
        }
    }
    return input_mat;
}

std::shared_ptr<TNNSDKOutput> ObjectDetectorYolo::CreateSDKOutput() {
    return std::make_shared<ObjectDetectorYoloOutput>();
}

Status ObjectDetectorYolo::ProcessSDKOutput(std::shared_ptr<TNNSDKOutput> output_) {
    Status status = TNN_OK;
    
    auto output = dynamic_cast<ObjectDetectorYoloOutput *>(output_.get());
    RETURN_VALUE_ON_NEQ(!output, false,
                        Status(TNNERR_PARAM_ERR, "TNNSDKOutput is invalid"));
    
    auto output_mat_0 = output->GetMat("428");
    RETURN_VALUE_ON_NEQ(!output_mat_0, false,
                        Status(TNNERR_PARAM_ERR, "GetMat is invalid"));
    auto output_mat_1 = output->GetMat("427");
    RETURN_VALUE_ON_NEQ(!output_mat_1, false,
                        Status(TNNERR_PARAM_ERR, "GetMat is invalid"));
    auto output_mat_2 = output->GetMat("426");
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
    std::sort(objs.begin(), objs.end(), [](const ObjectInfo &a, const ObjectInfo &b) { return a.score > b.score; });
    
    results.clear();
    auto box_num = objs.size();
    std::vector<int> merged(box_num, 0);

    for (int i = 0; i < box_num; i++) {
        if (merged[i])
            continue;

        merged[i] = 1;
        float h0 = objs[i].y2 - objs[i].y1 + 1;
        float w0 = objs[i].x2 - objs[i].x1 + 1;
        float area0 = h0 * w0;

        for (int j = i + 1; j < box_num; j++) {
            if (merged[j])
                continue;

            float inner_x0 = objs[i].x1 > objs[j].x1 ? objs[i].x1 : objs[j].x1;
            float inner_y0 = objs[i].y1 > objs[j].y1 ? objs[i].y1 : objs[j].y1;

            float inner_x1 = objs[i].x2 < objs[j].x2 ? objs[i].x2 : objs[j].x2;
            float inner_y1 = objs[i].y2 < objs[j].y2 ? objs[i].y2 : objs[j].y2;

            float inner_h = inner_y1 - inner_y0 + 1;
            float inner_w = inner_x1 - inner_x0 + 1;

            if (inner_h <= 0 || inner_w <= 0)
                continue;

            float inner_area = inner_h * inner_w;
            float h1 = objs[j].y2 - objs[j].y1 + 1;
            float w1 = objs[j].x2 - objs[j].x1 + 1;
            float area1 = h1 * w1;
            float iou = inner_area / (area0 + area1 - inner_area);

            if (iou > iou_threshold_) {
                merged[j] = 1;
            }
        }
        results.push_back(objs[i]);
    }
}

ObjectDetectorYolo::~ObjectDetectorYolo() {}

void ObjectDetectorYolo::GenerateDetectResult(std::vector<std::shared_ptr<Mat> >outputs,
                                              std::vector<ObjectInfo>& detecs, int image_width, int image_height) {
    std::vector<ObjectInfo> extracted_objs;
    int blob_index = 0;
    
    for(auto& output:outputs){
        auto dim = output->GetDims();
  
        if(dim[3] != num_anchor_ * detect_dim_) {
            LOGE("Invalid detect output, the size of last dimension is: %d\n", dim[3]);
            return;
        }
        float* data = static_cast<float*>(output->GetData());
//        unsigned int count = dim[0]*dim[1]*dim[2]*dim[3];
        
//        Sigmoid(data, count);
        
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
