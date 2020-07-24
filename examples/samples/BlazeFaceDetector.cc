//
//  BlazeFaceDetector.cpp
//  TNNExamples
//
//  Created by devandong on 2020/7/23.
//  Copyright © 2020 tencent. All rights reserved.
//

#include "BlazeFaceDetector.h"
#include <sys/time.h>
#include <cmath>
#include <fstream>
#include <cstring>

namespace TNN_NS {

Status BlazeFaceDetector::Init(std::shared_ptr<TNNSDKOption> option_i) {
    Status status = TNN_OK;
    auto option = dynamic_cast<BlazeFaceDetectorOption *>(option_i.get());
    RETURN_VALUE_ON_NEQ(!option, false,
                        Status(TNNERR_PARAM_ERR, "TNNSDKOption is invalid"));
    
    status = TNNSDKSample::Init(option_i);
    RETURN_ON_NEQ(status, TNN_OK);
    
    //TODO: load anchors
    std::ifstream inFile(option->anchor_path);
    RETURN_VALUE_ON_NEQ(inFile.good(), true, Status(TNNERR_PARAM_ERR, "TNNSDKOption.anchor_path is invalid"));
    std::string line;
    anchors.reserve(num_anchors * 4);
    int index = 0;
    while(std::getline(inFile, line, '\n')) {
        float val = std::stof(line);
        anchors[index++] = val;
    }
    RETURN_VALUE_ON_NEQ(index == num_anchors*4, true,
    Status(TNNERR_PARAM_ERR, "TNNSDKOption.anchor_path doesnot contain valid blazeface anchors"));
    
    return status;
}

MatConvertParam BlazeFaceDetector::GetConvertParamForInput(std::string tag) {
    MatConvertParam input_convert_param;
    input_convert_param.scale = {1.0 / 127.5, 1.0 / 127.5, 1.0 / 127.5, 0.0};
    input_convert_param.bias  = {-1.0, -1.0, -1.0, 0.0};
    //input_convert_param.reverse_channel = true;
    return input_convert_param;
}

std::shared_ptr<TNNSDKOutput> BlazeFaceDetector::CreateSDKOutput() {
    return std::make_shared<BlazeFaceDetectorOutput>();
}

Status BlazeFaceDetector::ProcessSDKOutput(std::shared_ptr<TNNSDKOutput> output_) {
    Status status = TNN_OK;
    auto option = dynamic_cast<BlazeFaceDetectorOption *>(option_.get());
    RETURN_VALUE_ON_NEQ(!option, false,
                           Status(TNNERR_PARAM_ERR, "TNNSDKOption is invalid"));
    auto output = dynamic_cast<BlazeFaceDetectorOutput *>(output_.get());
    RETURN_VALUE_ON_NEQ(!output, false,
    Status(TNNERR_PARAM_ERR, "TNNSDKOutput is invalid"));
    
    
    auto scores = output->GetMat("525");
    auto boxes  = output->GetMat("544");
    std::vector<BlazeFaceInfo> bbox_collection;
    //decode bbox
    GenerateBBox(bbox_collection, *(scores.get()), *(boxes.get()), option->input_width, option->input_height, option->min_score_threshold);
    LOGE("\n\n ===== \n find:%d faces\n ===== \n\n", bbox_collection.size());
    std::vector<BlazeFaceInfo> face_list;
    BlendingNMS(bbox_collection, face_list, option->min_suppression_threshold);
    LOGE("\n\n ===== \n find:%d faces\n ===== \n\n", face_list.size());
    output->face_list = face_list;
    
    return status;
}

void BlazeFaceDetector::GenerateBBox(std::vector<BlazeFaceInfo> &detects, TNN_NS::Mat &scores, TNN_NS::Mat &boxes, int image_w, int image_h, float min_score_threshold) {
    float *boxes_data = static_cast<float*>(boxes.GetData());
    float *score_data = static_cast<float*>(scores.GetData());
    
    ClampSigmoid(score_data, num_anchors);
    
    for(int i=0; i<num_anchors; ++i) {
        if(score_data[i] < min_score_threshold)
            continue;
        float x_center = boxes_data[i * detect_dims + 0] / image_w * anchors[i * 4 + 2] + anchors[i * 4 + 0];
        float y_center = boxes_data[i * detect_dims + 1] / image_h * anchors[i * 4 + 3] + anchors[i * 4 + 1];
        float width    = boxes_data[i * detect_dims + 2] / image_w * anchors[i * 4 + 2] ;
        float height   = boxes_data[i * detect_dims + 3] / image_h * anchors[i * 4 + 3] ;
        BlazeFaceInfo info;
        info.score = score_data[i];
        // bbox
        info.x1 = (x_center - width / 2.0) * image_w;
        info.y1 = (y_center - height / 2.0) * image_h;
        info.x2 = (x_center + width / 2.0) * image_w;
        info.y2 = (y_center + height / 2.0) * image_h;
        //info.x1 = x_center - width / 2.0;
        //info.y1 = y_center - height / 2.0;
        //info.x2 = x_center + width / 2.0;
        //info.y2 = y_center + height / 2.0;
        // key points
        for(int j=0; j<num_keypoints; ++j) {
            int offset = j * 2 + 4;
            float kp_x = (boxes_data[i * detect_dims + offset + 0] / image_w * anchors[i * 4 + 2] + anchors[i * 4 + 0]) * image_w;
            float kp_y = (boxes_data[i * detect_dims + offset + 1] / image_h * anchors[i * 4 + 3] + anchors[i * 4 + 1]) * image_h;
            info.key_points.push_back(std::make_pair(kp_x, kp_y));
        }
        detects.push_back(std::move(info));
    }
    
}

void BlazeFaceDetector::BlendingNMS(std::vector<BlazeFaceInfo> &input, std::vector<BlazeFaceInfo> &output, float min_suppression_threshold) {
    std::sort(input.begin(), input.end(), [](const BlazeFaceInfo &a, const BlazeFaceInfo &b) { return a.score > b.score; });
    output.clear();

    size_t box_num = input.size();

    std::vector<int> merged(box_num, 0);

    for (int i = 0; i < box_num; i++) {
        if (merged[i])
            continue;
        //buffer for blending NMS
        std::vector<BlazeFaceInfo> buf;
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

            if (score > min_suppression_threshold) {
                merged[j] = 1;
                buf.push_back(input[j]);
            }
        }
        float total = 0;
        for (int i = 0; i < buf.size(); i++) {
            total += exp(buf[i].score);
        }
        BlazeFaceInfo rects;
        for (int i = 0; i < buf.size(); i++) {
            float rate = exp(buf[i].score) / total;
            rects.x1 += buf[i].x1 * rate;
            rects.y1 += buf[i].y1 * rate;
            rects.x2 += buf[i].x2 * rate;
            rects.y2 += buf[i].y2 * rate;
            rects.score += buf[i].score * rate;
        }
        rects.key_points = buf[0].key_points;
        output.push_back(rects);
    }
    //print output face list
    for(auto& face:output) {
        int x1 = face.x1;
        int y1 = face.y1;
        int x2 = face.x2;
        int y2 = face.y2;
        float score = face.score;
        printf(" ==== face ====\n");
        printf("%d, %d, %d, %d, %f\n", x1, y1, x2, y2, score);
        printf("==== keypoints ====\n");
        for(auto&p:face.key_points) {
            printf("%f, %f\n", p.first, p.second);
        }
    }
}

void BlazeFaceDetector::ClampSigmoid(float* dataPtr, size_t count){
    for(int i=0; i<count; ++i) {
        float val = dataPtr[i];
        val = std::min(std::max(-score_clipping_threshold, val), score_clipping_threshold);
        float rst = 1.0f / (1.0f + exp(-val));
        dataPtr[i] = rst;
    }
}

}
