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

#include "worker.h"

#include <string>
#include <fstream>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include "tnn/core/macro.h"
#include "macro.h"
#include "utils/utils.h"

using TNN_NS::TNN_OK;

Status Worker::Init(std::string model_path) {
    fps_counter_ = std::make_shared<TNNFPSCounter>();

    // Init FaceDetector
    auto proto_content = fdLoadFile(model_path+"/face_detector/version-slim-320_simplified.tnnproto");
    auto model_content = fdLoadFile(model_path+"/face_detector/version-slim-320_simplified.tnnmodel");

    auto option = std::make_shared<TNN_NS::UltraFaceDetectorOption>();
    {
        option->proto_content = proto_content;
        option->model_content = model_content;
        option->library_path = "";
        option->compute_units = TNN_NS::TNNComputeUnitsOpenvino;
        option->score_threshold = 0.95;
        option->iou_threshold = 0.15;
    }
    
    detecotr_ = std::make_shared<TNN_NS::UltraFaceDetector>();
    CHECK_TNN_STATUS(detecotr_->Init(option));

    // Init BlazeFaceDetector
    auto blaze_detector_proto_content = fdLoadFile(model_path+"/blazeface/blazeface.tnnproto");
    auto blaze_detector_model_content = fdLoadFile(model_path+"/blazeface/blazeface.tnnmodel");
    auto blaze_detector_option = std::make_shared<TNN_NS::BlazeFaceDetectorOption>();
    {
        blaze_detector_option->proto_content = blaze_detector_proto_content;
        blaze_detector_option->model_content = blaze_detector_model_content;
        blaze_detector_option->library_path = "";
        blaze_detector_option->compute_units = TNN_NS::TNNComputeUnitsCPU;
        blaze_detector_option->min_suppression_threshold = 0.3;
        blaze_detector_option->anchor_path = model_path + "/blazeface/blazeface_anchors.txt";
    }

    blaze_detecotr_ = std::make_shared<TNN_NS::BlazeFaceDetector>();
    CHECK_TNN_STATUS(blaze_detecotr_->Init(blaze_detector_option));

    
    return TNN_OK;
};


Status Worker::DrawUI(cv::Mat &frame) {
    // FPS
    char fps_char[200];
    snprintf(fps_char, 200, "FPS:%3.0f", fps_counter_->GetFPS("frame"));
    std::string text(fps_char);
    int font_face = cv::FONT_HERSHEY_COMPLEX; 
    double font_scale = 1.2;
    int thickness = 2;
    int baseline;
    cv::Size text_size = cv::getTextSize(text, font_face, font_scale, thickness, &baseline);
 
    cv::Point origin; 
    origin.x = 10;
    origin.y = 10 + text_size.height;
    cv::putText(frame, text, origin, font_face, font_scale, cv::Scalar(0, 255, 255), thickness, 8, 0);

    // CMD
    std::vector<std::string> sentences = {
        "a: toggle facealign",
        "d: toggle facedetect",
        "c: quit",
        "Press:",
    };
    font_scale = 0.5;
    thickness = 1;
    origin.x = 10;
    origin.y = frame.rows;
    for(auto str : sentences) {
        text_size = cv::getTextSize(str, font_face, font_scale, thickness, &baseline);
        origin.y -= text_size.height + 5;
        cv::putText(frame, str, origin, font_face, font_scale, cv::Scalar(0, 0, 255), thickness, 8, 0);
    }
    return TNN_OK;
}

Status Worker::FaceDetectWithDraw(cv::Mat &frame, cv::Mat &frame_paint) {
    //prepare input
    std::vector<int> nchw = {1, frame.channels(), frame.rows, frame.cols};
    auto image_mat = std::make_shared<TNN_NS::Mat>(TNN_NS::DEVICE_NAIVE, TNN_NS::N8UC3, nchw, frame.data);

    //Predict
    std::shared_ptr<TNN_NS::TNNSDKOutput> sdk_output = detecotr_->CreateSDKOutput();
    CHECK_TNN_STATUS(detecotr_->Predict(std::make_shared<TNN_NS::UltraFaceDetectorInput>(image_mat), sdk_output));

    std::vector<TNN_NS::FaceInfo> face_info;
    if (sdk_output && dynamic_cast<TNN_NS::UltraFaceDetectorOutput *>(sdk_output.get())) {
        auto face_output = dynamic_cast<TNN_NS::UltraFaceDetectorOutput *>(sdk_output.get());
        face_info = face_output->face_list;
    }

    const int image_orig_height = nchw[2];
    const int image_orig_width  = nchw[3];
    const int h = detecotr_->GetInputShape()[2];
    const int w = detecotr_->GetInputShape()[3];
    float scale_x               = image_orig_width / (float)w;
    float scale_y               = image_orig_height / (float)h;

    for (int i = 0; i < face_info.size(); i++) {
        auto face = face_info[i];
        cv::Point top_left(face.x1 * scale_x, face.y1 * scale_y);
        cv::Point bottom_right(face.x2 * scale_x, face.y2 * scale_y);
        cv::rectangle(frame_paint, top_left, bottom_right, cv::Scalar(0, 255, 0));
    }

    return TNN_OK;
};

Status Worker::BlazeFaceDetectWithDraw(cv::Mat &frame, cv::Mat &frame_paint) {
    std::vector<int> nchw = {1, frame.channels(), frame.rows, frame.cols};
    auto image_mat = std::make_shared<TNN_NS::Mat>(TNN_NS::DEVICE_NAIVE, TNN_NS::N8UC3, nchw, frame.data);

    //Predict
    std::shared_ptr<TNN_NS::TNNSDKOutput> sdk_output = blaze_detecotr_->CreateSDKOutput();
    CHECK_TNN_STATUS(blaze_detecotr_->Predict(std::make_shared<TNN_NS::BlazeFaceDetectorInput>(image_mat), sdk_output));

    std::vector<TNN_NS::BlazeFaceInfo> face_info;
    if (sdk_output && dynamic_cast<TNN_NS::BlazeFaceDetectorOutput *>(sdk_output.get())) {
        auto face_output = dynamic_cast<TNN_NS::BlazeFaceDetectorOutput *>(sdk_output.get());
        face_info = face_output->face_list;
    }

    const int image_orig_height = nchw[2];
    const int image_orig_width  = nchw[3];
    const int h = blaze_detecotr_->GetInputShape()[2];
    const int w = blaze_detecotr_->GetInputShape()[3];
    float scale_x               = image_orig_width / (float)w;
    float scale_y               = image_orig_height / (float)h;


    for (int i = 0; i < face_info.size(); i++) {
        auto face = face_info[i];
        int width  = (face.x2 - face.x1 ) * scale_x ;
        cv::Point top_left(face.x1 * scale_x, face.y1 * scale_y);
        cv::Point bottom_right(face.x2 * scale_x, face.y1 * scale_y + width);
        cv::rectangle(frame_paint, top_left, bottom_right, cv::Scalar(0, 0, 255));
    }

    return TNN_OK;
};


Status Worker::FrocessFrame(cv::Mat &frame, cv::Mat &frame_paint) {
    fps_counter_->Begin("frame");

    RETURN_ON_NEQ(FaceDetectWithDraw(frame, frame_paint), TNN_OK);
    RETURN_ON_NEQ(BlazeFaceDetectWithDraw(frame, frame_paint), TNN_OK);

    fps_counter_->End("frame");
    RETURN_ON_NEQ(DrawUI(frame_paint), TNN_OK);
    return TNN_OK; 
}
