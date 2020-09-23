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

#include "hand_detector.h"
#include "tnn/utils/dims_vector_utils.h"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <string>
#include <unordered_set>
#include <sys/time.h>


namespace TNN_NS {
HandDetectorOutput::~HandDetectorOutput() {}

MatConvertParam HandDetector::GetConvertParamForInput(std::string name) {
    MatConvertParam input_convert_param;
    input_convert_param.scale = {1.0 / 255.0, 1.0 / 255.0, 1.0 / 255.0, 0.0};
    input_convert_param.bias  = {0.0, 0.0, 0.0, 0.0};
    return input_convert_param;
}

Status HandDetector::Init(std::shared_ptr<TNNSDKOption> option_i) {
    Status status = TNN_OK;
    auto option = dynamic_cast<HandDetectorOption *>(option_i.get());
    RETURN_VALUE_ON_NEQ(!option, false, Status(TNNERR_PARAM_ERR, "TNNSDKOption is invalid"));

    status = TNNSDKSample::Init(option_i);
    RETURN_ON_NEQ(status, TNN_OK);

    auto input_dims = GetInputShape();
    option->input_height = input_dims[2];
    option->input_width  = input_dims[3];

    this->conf_thresh = option->conf_threshold;
    this->nms_thresh  = option->nms_threshold;

    return status;
}

std::shared_ptr<Mat> HandDetector::ProcessSDKInputMat(std::shared_ptr<Mat> input_mat,
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

std::shared_ptr<TNNSDKOutput> HandDetector::CreateSDKOutput() {
    return std::make_shared<HandDetectorOutput>();
}

Status HandDetector::ProcessSDKOutput(std::shared_ptr<TNNSDKOutput> output_) {
    Status status = TNN_OK;

    auto option = dynamic_cast<HandDetectorOption *>(option_.get());
    RETURN_VALUE_ON_NEQ(!option, false, Status(TNNERR_PARAM_ERR, "TNNOption is invalid"));

    auto output = dynamic_cast<HandDetectorOutput *>(output_.get());
    RETURN_VALUE_ON_NEQ(!output, false, Status(TNNERR_PARAM_ERR, "TNNSDKOutput is invalid"));

    auto x = output->GetMat("x");
    auto y = output->GetMat("y");
    auto w = output->GetMat("w");
    auto h = output->GetMat("h");
    auto conf = output->GetMat("conf");
    auto cls  = output->GetMat("cls");

    std::vector<ObjectInfo> bboxes, outputs;
    GenerateBBox(x.get(), y.get(), h.get(), w.get(), conf.get(), bboxes);
    BlendingNMS(bboxes, outputs);

    output->hands = outputs;

    return status;
}

void HandDetector::GenerateBBox(Mat *bbox_x, Mat *bbox_y, Mat *bbox_h, Mat *bbox_w, Mat *conf, std::vector<ObjectInfo>& bboxes) {
    auto dims   = bbox_x->GetDims();
    // process (x, y)
    float *bbox_x_ptr = static_cast<float *>(bbox_x->GetData());
    float *bbox_y_ptr = static_cast<float *>(bbox_y->GetData());
    float *bbox_h_ptr = static_cast<float *>(bbox_h->GetData());
    float *bbox_w_ptr = static_cast<float *>(bbox_w->GetData());

    int offset = 0;
    for(int c=0; c<dims[1]; ++c){
        for(int h=0; h<dims[2]; ++h) {
            for(int w=0; w<dims[3]; ++w) {
                bbox_x_ptr[offset] = (bbox_x_ptr[offset] + GetGridX(h, w)) * this->stride;
                bbox_y_ptr[offset] = (bbox_y_ptr[offset] + GetGridY(h, w)) * this->stride;
                bbox_h_ptr[offset] = exp(bbox_h_ptr[offset]) * this->anchor_h[c] * this->stride;
                bbox_w_ptr[offset] = exp(bbox_w_ptr[offset]) * this->anchor_w[c] * this->stride;
                offset += 1;
            }
        }
    }
    // generate bboxes
    float *conf_ptr = static_cast<float *>(conf->GetData());
    auto option = reinterpret_cast<HandDetectorOption *>(option_.get());
    auto count = DimsVectorUtils::Count(dims);
    for(int i=0; i<count; ++i) {
        float score    = conf_ptr[i];
        if (score < this->conf_thresh)
            continue;
        float center_x = bbox_x_ptr[i];
        float center_y = bbox_y_ptr[i];
        float width    = bbox_w_ptr[i];
        float height   = bbox_h_ptr[i];
        ObjectInfo obj;
        obj.x1 = center_x - width / 2.0;
        obj.y1 = center_y - height / 2.0;
        obj.x2 = center_x + width / 2.0;
        obj.y2 = center_y + height / 2.0;
        obj.score = score;
        obj.image_width  = option->input_width;
        obj.image_height = option->input_height;
        bboxes.push_back(std::move(obj));
    }
}

void HandDetector::BlendingNMS(std::vector<ObjectInfo> &input, std::vector<ObjectInfo> &output) {
    std::sort(input.begin(), input.end(), [](const ObjectInfo &a, const ObjectInfo &b) { return a.score > b.score; });
    output.clear();

    size_t box_num = input.size();
    std::vector<int> merged(box_num, 0);

    for (int i = 0; i < box_num; i++) {
        if (merged[i])
            continue;
        //buffer for blending NMS
        std::vector<ObjectInfo> buf;
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

            if (score > this->nms_thresh) {
                merged[j] = 1;
                buf.push_back(input[j]);
            }
        }
        float total = 0;
        for (int i = 0; i < buf.size(); i++) {
            total += buf[i].score;
        }
        ObjectInfo rects;
        rects.image_width = buf[0].image_width;
        rects.image_height = buf[0].image_height;
        rects.score = buf[0].score;
        for (int i = 0; i < buf.size(); i++) {
            float rate = buf[i].score / total;
            rects.x1 += buf[i].x1 * rate;
            rects.y1 += buf[i].y1 * rate;
            rects.x2 += buf[i].x2 * rate;
            rects.y2 += buf[i].y2 * rate;
        }
        output.push_back(rects);
    }
}


}

