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

#include "ObjectDetectorYolo.h"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <string>
#include <unordered_set>
#include <sys/time.h>

void ObjectDetectorYolo::Sigmoid(float* v, const unsigned int count) {
    for(int i=0; i<count; ++i){
        float in = v[i];
        float rst = 1.0f / (1.0f + exp(-in));
        v[i] = rst;
    }
}

void ObjectDetectorYolo::NMS(std::vector<ObjInfo>& objs, std::vector<ObjInfo>& results) {
    std::sort(objs.begin(), objs.end(), [](const ObjInfo &a, const ObjInfo &b) { return a.score > b.score; });
    
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

ObjectDetectorYolo::ObjectDetectorYolo(int input_width, int input_length, int num_thread_) {
    num_thread = num_thread_;
    in_w  = input_width;
    in_h = input_length;
}

ObjectDetectorYolo::~ObjectDetectorYolo() {}

int ObjectDetectorYolo::Detect(std::shared_ptr<tnn::Mat> image, int image_height, int image_width, std::vector<ObjInfo> &obj_list) {
    if(!image || !image->GetData()) {
        std::cout << "image is empty, please check!" << std::endl;
        return -1;
    }
#if TNN_SDK_ENABLE_BENCHMARK
    bench_result_.Reset();
    for(int fcount = 0; fcount < bench_option_.forward_count; fcount++) {
        timeval tv_begin, tv_end;
        gettimeofday(&tv_begin, NULL);
#endif
        obj_list.clear();
        
        TNN_NS::MatConvertParam input_convert_param;
        input_convert_param.scale = {1.0 / 255, 1.0 / 255, 1.0 / 255, 0.0};
        input_convert_param.bias  = {0.0, 0.0, 0.0, 0.0};
        auto status = instance_->SetInputMat(image, input_convert_param);
        if (status != TNN_NS::TNN_OK) {
            LOGE("input_convert.ConvertFromMatAsync Error: %s\n", status.description().c_str());
            return status;
        }
        
        status = instance_->ForwardAsync(nullptr);
        if (status != TNN_NS::TNN_OK) {
            LOGE("instance.Forward Error: %s\n", status.description().c_str());
            return status;
        }
        
        for(auto& name:output_blob_names_) {
            TNN_NS::MatConvertParam output_convert_param;
            std::shared_ptr<TNN_NS::Mat> output_mat = nullptr;
            status = instance_->GetOutputMat(output_mat, output_convert_param, name);
            if (status != TNN_NS::TNN_OK) {
                LOGE("GetOutputMat:%s Error: %s\n", name.c_str(), status.description().c_str());
                return status;
            }
            outputs_.push_back(output_mat);
        }
        
#if TNN_SDK_ENABLE_BENCHMARK
        gettimeofday(&tv_end, NULL);
        double time_elapsed = (tv_end.tv_sec - tv_begin.tv_sec) * 1000.0 + (tv_end.tv_usec - tv_begin.tv_usec) / 1000.0;
        bench_result_.AddTime(time_elapsed);
#endif
        GenerateDetectResult(obj_list);
#if TNN_SDK_ENABLE_BENCHMARK
    }
#endif
    return 0;
}

void ObjectDetectorYolo::GenerateDetectResult(std::vector<ObjInfo>& detecs) {
    std::vector<ObjInfo> extracted_objs;
    int blob_index = 0;
    
    for(auto& output:outputs_){
        auto dim = output->GetDims();
  
        if(dim[3] != num_anchor_ * detect_dim_) {
            LOGE("Invalid detect output, the size of last dimension is: %d\n", dim[3]);
            return;
        }
        float* data = static_cast<float*>(output->GetData());
        unsigned int count = dim[0]*dim[1]*dim[2]*dim[3];
        
        Sigmoid(data, count);
        
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
            
            extracted_objs.push_back(ObjInfo({x1, y1, x2, y2, score, conf_idx}));
        }
        blob_index += 1;
    }
    NMS(extracted_objs, detecs);
}
