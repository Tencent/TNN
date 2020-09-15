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

#include "ultra_face_detector.h"
#include "sample_timer.h"
#include <cmath>
#include <cstring>

#define clip(x, y) (x < 0 ? 0 : (x > y ? y : x))

/*
 * Adjust face info to original size
 */
std::vector<FaceInfo> AdjustFaceInfoToOriginalSize(std::vector<FaceInfo> face_info,
                                                int detect_image_height, int detect_image_width,
                                                int orig_image_height, int orig_image_width)
{
    float scale_x = orig_image_width/(float)detect_image_width;
    float scale_y = orig_image_height/(float)detect_image_height;
    
    std::vector<FaceInfo>  face_info_orig;
    for (int i = 0; i < face_info.size(); i++) {
        auto face = face_info[i];
        int x_min = std::min(face.x1, face.x2)*scale_x;
        int x_max = std::max(face.x1, face.x2)*scale_x;
        int y_min = std::min(face.y1, face.y2)*scale_y;
        int y_max = std::max(face.y1, face.y2)*scale_y;
        
        x_min = std::min(std::max(x_min, 0), orig_image_width-1);
        x_max = std::min(std::max(x_max, 0), orig_image_width-1);
        y_min = std::min(std::max(y_min, 0), orig_image_height-1);
        y_max = std::min(std::max(y_max, 0), orig_image_height-1);
        
        face.x1 = x_min;
        face.x2 = x_max;
        face.y1 = y_min;
        face.y2 = y_max;
        
        face_info_orig.push_back(face);
    }
    return face_info_orig;
}

/*
 * Initializing the FaceDetector.
 */
UltraFaceDetector::UltraFaceDetector(int input_width, int input_length, int num_thread_, float score_threshold_,
                                     float iou_threshold_, int topk_) {
    num_thread      = num_thread_;
    topk            = topk_;
    score_threshold = score_threshold_;
    iou_threshold   = iou_threshold_;
    in_w            = input_width;
    in_h            = input_length;
    w_h_list        = {in_w, in_h};

    for (auto size : w_h_list) {
        std::vector<float> fm_item;
        for (float stride : strides) {
            fm_item.push_back(ceil(size / stride));
        }
        featuremap_size.push_back(fm_item);
    }

    for (auto size : w_h_list) {
        shrinkage_size.push_back(strides);
    }

    /* generate prior anchors */
    for (int index = 0; index < num_featuremap; index++) {
        float scale_w = in_w / shrinkage_size[0][index];
        float scale_h = in_h / shrinkage_size[1][index];
        for (int j = 0; j < featuremap_size[1][index]; j++) {
            for (int i = 0; i < featuremap_size[0][index]; i++) {
                float x_center = (i + 0.5) / scale_w;
                float y_center = (j + 0.5) / scale_h;

                for (float k : min_boxes[index]) {
                    float w = k / in_w;
                    float h = k / in_h;
                    priors.push_back({clip(x_center, 1), clip(y_center, 1), clip(w, 1), clip(h, 1)});
                }
            }
        }
    }
    num_anchors = priors.size();
    /* generate prior anchors finished */
}

/*
 * Destruct the FaceDetector.
 */
UltraFaceDetector::~UltraFaceDetector() {}

/*
 * Do detection
 */
int UltraFaceDetector::Detect(std::shared_ptr<TNN_NS::Mat> image_mat, int image_height, int image_width,
                              std::vector<FaceInfo> &face_list) {
    if (!image_mat || !image_mat->GetData()) {
        std::cout << "image is empty ,please check!" << std::endl;
        return -1;
    }

    image_h = image_height;
    image_w = image_width;

#if TNN_SDK_ENABLE_BENCHMARK
    bench_result_.Reset();
    tnn::SampleTimer timer;
    for (int fcount = 0; fcount < bench_option_.forward_count; fcount++) {    
        timer.Start();
#endif
        // step 1. set input mat
        TNN_NS::MatConvertParam input_convert_param;
        input_convert_param.scale = {1.0 / 128, 1.0 / 128, 1.0 / 128, 0.0};
        input_convert_param.bias  = {-127.0 / 128, -127.0 / 128, -127.0 / 128, 0.0};
        auto status = instance_->SetInputMat(image_mat, input_convert_param);
        if (status != TNN_NS::TNN_OK) {
            LOGE("input_blob_convert.ConvertFromMatAsync Error: %s\n", status.description().c_str());
            return status;
        }

        // step 2. Forward
        status = instance_->ForwardAsync(nullptr);
        if (status != TNN_NS::TNN_OK) {
            LOGE("instance.Forward Error: %s\n", status.description().c_str());
            return status;
        }

        // step 3. get output mat
        TNN_NS::MatConvertParam output_convert_param;
        std::shared_ptr<TNN_NS::Mat> output_mat_scores = nullptr;
        status = instance_->GetOutputMat(output_mat_scores, output_convert_param, "scores");
        if (status != TNN_NS::TNN_OK) {
            LOGE("GetOutputMat Error: %s\n", status.description().c_str());
            return status;
        }
        
        std::shared_ptr<TNN_NS::Mat> output_mat_boxes = nullptr;
        status = instance_->GetOutputMat(output_mat_boxes, output_convert_param, "boxes");
        if (status != TNN_NS::TNN_OK) {
            LOGE("GetOutputMat Error: %s\n", status.description().c_str());
            return status;
        }
        
#if TNN_SDK_ENABLE_BENCHMARK
        timer.Stop();
        bench_result_.AddTime(timer.GetTime());
        timer.Reset();
#endif

        std::vector<FaceInfo> bbox_collection;
        std::vector<FaceInfo> valid_input;

        GenerateBBox(bbox_collection, *(output_mat_scores.get()), *(output_mat_boxes.get()), score_threshold,
                     num_anchors);
        NMS(bbox_collection, face_list);

#if TNN_SDK_ENABLE_BENCHMARK
    }
#endif
    // Detection done
    return 0;
}

/*
 * Generating Bbox from output blobs
 */
void UltraFaceDetector::GenerateBBox(std::vector<FaceInfo> &bbox_collection, TNN_NS::Mat &scores, TNN_NS::Mat &boxes,
                                     float score_threshold, int num_anchors) {
    float *scores_data = (float *)scores.GetData();
    float *boxes_data  = (float *)boxes.GetData();

    for (int i = 0; i < num_anchors; i++) {
        if (scores_data[i * 2 + 1] > score_threshold) {
            FaceInfo rects;
            float x_center = boxes_data[i * 4] * center_variance * priors[i][2] + priors[i][0];
            float y_center = boxes_data[i * 4 + 1] * center_variance * priors[i][3] + priors[i][1];
            float w        = exp(boxes_data[i * 4 + 2] * size_variance) * priors[i][2];
            float h        = exp(boxes_data[i * 4 + 3] * size_variance) * priors[i][3];

            rects.x1    = clip(x_center - w / 2.0, 1) * image_w;
            rects.y1    = clip(y_center - h / 2.0, 1) * image_h;
            rects.x2    = clip(x_center + w / 2.0, 1) * image_w;
            rects.y2    = clip(y_center + h / 2.0, 1) * image_h;
            rects.score = clip(scores_data[i * 2 + 1], 1);
            bbox_collection.push_back(rects);
        }
    }
}

/*
 * NMS
 */
void UltraFaceDetector::NMS(std::vector<FaceInfo> &input, std::vector<FaceInfo> &output, int type) {
    std::sort(input.begin(), input.end(), [](const FaceInfo &a, const FaceInfo &b) { return a.score > b.score; });
    output.clear();

    int box_num = input.size();

    std::vector<int> merged(box_num, 0);

    for (int i = 0; i < box_num; i++) {
        if (merged[i])
            continue;
        std::vector<FaceInfo> buf;

        buf.push_back(input[i]);
        merged[i] = 1;

        float h0 = input[i].y2 - input[i].y1 + 1;
        float w0 = input[i].x2 - input[i].x1 + 1;

        float area0 = h0 * w0;

        for (int j = i + 1; j < box_num; j++) {
            if (merged[j])
                continue;

            float inner_x0 = input[i].x1 > input[j].x1 ? input[i].x1 : input[j].x1;
            float inner_y0 = input[i].y1 > input[j].y1 ? input[i].y1 : input[j].y1;

            float inner_x1 = input[i].x2 < input[j].x2 ? input[i].x2 : input[j].x2;
            float inner_y1 = input[i].y2 < input[j].y2 ? input[i].y2 : input[j].y2;

            float inner_h = inner_y1 - inner_y0 + 1;
            float inner_w = inner_x1 - inner_x0 + 1;

            if (inner_h <= 0 || inner_w <= 0)
                continue;

            float inner_area = inner_h * inner_w;

            float h1 = input[j].y2 - input[j].y1 + 1;
            float w1 = input[j].x2 - input[j].x1 + 1;

            float area1 = h1 * w1;

            float score;

            score = inner_area / (area0 + area1 - inner_area);

            if (score > iou_threshold) {
                merged[j] = 1;
                buf.push_back(input[j]);
            }
        }
        switch (type) {
            case hard_nms: {
                output.push_back(buf[0]);
                break;
            }
            case blending_nms: {
                float total = 0;
                for (int i = 0; i < buf.size(); i++) {
                    total += exp(buf[i].score);
                }
                FaceInfo rects;
                memset(&rects, 0, sizeof(rects));
                for (int i = 0; i < buf.size(); i++) {
                    float rate = exp(buf[i].score) / total;
                    rects.x1 += buf[i].x1 * rate;
                    rects.y1 += buf[i].y1 * rate;
                    rects.x2 += buf[i].x2 * rate;
                    rects.y2 += buf[i].y2 * rate;
                    rects.score += buf[i].score * rate;
                }
                output.push_back(rects);
                break;
            }
            default: {
                printf("wrong type of nms.");
                exit(-1);
            }
        }
    }
}
