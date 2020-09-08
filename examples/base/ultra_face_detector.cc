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

#define num_featuremap 4
#define hard_nms 1
#define blending_nms 2 /* mix nms was been proposaled in paper blaze face, aims to minimize the temporal jitter*/
#define clip(x, y) (x < 0 ? 0 : (x > y ? y : x))

namespace TNN_NS {
UltraFaceDetectorOption::UltraFaceDetectorOption() {}

UltraFaceDetectorOption::~UltraFaceDetectorOption() {}

UltraFaceDetectorInput::~UltraFaceDetectorInput() {}

UltraFaceDetectorOutput::~UltraFaceDetectorOutput() {}

Status UltraFaceDetector::Init(std::shared_ptr<TNNSDKOption> option_i) {
    Status status = TNN_OK;
    auto option = dynamic_cast<UltraFaceDetectorOption *>(option_i.get());
    RETURN_VALUE_ON_NEQ(!option, false,
                        Status(TNNERR_PARAM_ERR, "TNNSDKOption is invalid"));
    
    status = TNNSDKSample::Init(option_i);
    RETURN_ON_NEQ(status, TNN_OK);
    
    auto input_dims = GetInputShape();
    int in_h = input_dims[2];
    int in_w = input_dims[3];
    option->input_height = in_h;
    option->input_width  = in_w;

    auto w_h_list = {in_w, in_h};

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
    
    return status;
}

/*
 * Destruct the FaceDetector.
 */
UltraFaceDetector::~UltraFaceDetector() {}

MatConvertParam UltraFaceDetector::GetConvertParamForInput(std::string tag) {
    MatConvertParam input_convert_param;
    input_convert_param.scale = {1.0 / 128, 1.0 / 128, 1.0 / 128, 0.0};
    input_convert_param.bias  = {-127.0 / 128, -127.0 / 128, -127.0 / 128, 0.0};
    return input_convert_param;
}

std::shared_ptr<TNNSDKOutput> UltraFaceDetector::CreateSDKOutput() {
    return std::make_shared<UltraFaceDetectorOutput>();
}

std::shared_ptr<Mat> UltraFaceDetector::ProcessSDKInputMat(std::shared_ptr<Mat> input_mat,
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

Status UltraFaceDetector::ProcessSDKOutput(std::shared_ptr<TNNSDKOutput> output_) {
    Status status = TNN_OK;
    auto option = dynamic_cast<UltraFaceDetectorOption *>(option_.get());
    RETURN_VALUE_ON_NEQ(!option, false,
                        Status(TNNERR_PARAM_ERR, "TNNSDKOption is invalid"));
    auto output = dynamic_cast<UltraFaceDetectorOutput *>(output_.get());
    RETURN_VALUE_ON_NEQ(!output, false,
                        Status(TNNERR_PARAM_ERR, "TNNSDKOutput is invalid"));
    
    std::vector<FaceInfo> bbox_collection;
    std::vector<FaceInfo> valid_input;
    
    auto output_mat_scores = output->GetMat("scores");
    auto output_mat_boxes = output->GetMat("boxes");
    RETURN_VALUE_ON_NEQ(!output_mat_scores, false,
                        Status(TNNERR_PARAM_ERR, "output_mat_scores is invalid"));
    RETURN_VALUE_ON_NEQ(!output_mat_boxes, false,
                        Status(TNNERR_PARAM_ERR, "output_mat_boxes is invalid"));
    
    GenerateBBox(bbox_collection, *(output_mat_scores.get()), *(output_mat_boxes.get()),
                 option->input_width, option->input_height,
                 option->score_threshold, num_anchors);
    
    std::vector<FaceInfo> face_list;
    NMS(bbox_collection, face_list, option->iou_threshold);
    output->face_list = face_list;
    return status;
}

/*
 * Generating Bbox from output blobs
 */
void UltraFaceDetector::GenerateBBox(std::vector<FaceInfo> &bbox_collection,
                                     TNN_NS::Mat &scores, TNN_NS::Mat &boxes,
                                     int image_w, int image_h,
                                     float score_threshold, int num_anchors) {
    float *scores_data = (float *)scores.GetData();
    float *boxes_data  = (float *)boxes.GetData();

    for (int i = 0; i < num_anchors; i++) {
        if (scores_data[i * 2 + 1] > score_threshold) {
            FaceInfo rects;
            rects.image_width = image_w;
            rects.image_height = image_h;
            
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
void UltraFaceDetector::NMS(std::vector<FaceInfo> &input, std::vector<FaceInfo> &output,
                            float iou_threshold, int type) {
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
                    
                    rects.image_height = buf[i].image_height;
                    rects.image_width = buf[i].image_width;
                }
                output.push_back(rects);
                break;
            }
            default: {
            }
        }
    }
}

}
