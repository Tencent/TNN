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

#include <fstream>
#include <string>
#include <vector>

#include "tnn_sdk_sample.h"
#include "macro.h"
#include "utils/utils.h"

#include "ocr_driver.h"
#include "ocr_textbox_detector.h"
#include "ocr_text_recognizer.h"
#include "ocr_angle_predictor.h"

#define STB_IMAGE_IMPLEMENTATION
#include "../../../../third_party/stb/stb_image.h"
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "../../../../third_party/stb/stb_image_resize.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../../../../third_party/stb/stb_image_write.h"

#include <opencv2/opencv.hpp>
using namespace TNN_NS;

Status initOCRDetector(std::shared_ptr<OCRDriver> &predictor, DimsVector &target_dims) {
    predictor = std::make_shared<OCRDriver>();
    auto gOCRTextBoxDetector = std::make_shared<OCRTextboxDetector>();
    auto gOCRAnglePredictor = std::make_shared<OCRAnglePredictor>();
    auto gOCRTextRecognizer = std::make_shared<OCRTextRecognizer>();

    TNNComputeUnits compute_units;
    compute_units = TNN_NS::TNNComputeUnitsCPU;
    // if enable openvino/tensorrt, set option compute_units to openvino/tensorrt
    #ifdef _CUDA_
        compute_units = TNN_NS::TNNComputeUnitsTensorRT;
    #elif _OPENVINO_
        compute_units = TNN_NS::TNNComputeUnitsCPU;
    #endif

    std::string protoContent, modelContent;
    std::string modelPath = "../../../../model/chinese-ocr/";
    
    // text box detector
    protoContent = fdLoadFile(modelPath + "dbnet.tnnproto");
    modelContent = fdLoadFile(modelPath + "dbnet.tnnmodel");
    {
        auto option = std::make_shared<OCRTextboxDetectorOption>();
        option->compute_units = compute_units;
        option->library_path = "";
        option->proto_content = protoContent;
        option->model_content = modelContent;
        option->scale_down_ratio = .75f;
        option->padding = 10;
        auto status = gOCRTextBoxDetector->Init(option);
        if (status != TNN_OK) {
            LOGE("ocr textbox detector init failed %d", (int)status);
            return -1;
        }
        target_dims = gOCRTextBoxDetector->GetInputShape("input0");
    }

    // angle prediction
    protoContent = fdLoadFile(modelPath + "angle_net.tnnproto");
    modelContent = fdLoadFile(modelPath + "angle_net.tnnmodel");
    {
        auto option = std::make_shared<TNNSDKOption>();
        option->compute_units = compute_units;
        option->library_path = "";
        option->proto_content = protoContent;
        option->model_content = modelContent;
        auto status = gOCRAnglePredictor->Init(option);
        if (status != TNN_OK) {
            LOGE("ocr angle predictor init failed %d",(int)status);
            return -1;
        }
    }

    protoContent = fdLoadFile(modelPath + "crnn_lite_lstm.tnnproto");
    modelContent = fdLoadFile(modelPath + "crnn_lite_lstm.tnnmodel");
    {
        auto option = std::make_shared<OCRTextRecognizerOption>();
        option->compute_units = compute_units;
        option->library_path = "";
        option->vocab_path = modelPath + "keys.txt";
        option->proto_content = protoContent;
        option->model_content = modelContent;
        auto status = gOCRTextRecognizer->Init(option);
        if (status != TNN_OK) {
            LOGE("ocr text recognizer init failed %d", (int)status);
            return -1;
        }
    }

    auto status = predictor->Init({gOCRTextBoxDetector, gOCRAnglePredictor, gOCRTextRecognizer});
    if (status != TNN_OK) {
        LOGE("ocr detector init failed %d", (int)status);
        return -1;
    }

    return TNN_OK;
}

int main(int argc, char **argv) {
    auto predictor = std::shared_ptr<OCRDriver>();
    DimsVector target_dims;
    initOCRDetector(predictor, target_dims);

    printf("Please choose the source you want to detect:\n");
    printf("1. picture;\t2. video;\t3. camera.\n");

    int detect_type; 
    scanf("%d", &detect_type);

    char img_buff[256];
    // char* input_imgfn = "../../../assets/test_face.jpg";
    char *input_imgfn = img_buff;
    int image_width, image_height, image_channel;
    cv::VideoCapture cap;
    unsigned char *data;

    if (detect_type == 1) {
        printf("Please enter the image path you want to detect:\n");
        std::cin.getline(img_buff, 256);
        std::cin.getline(img_buff, 256);
        if (strlen(img_buff) == 0) {
            strncpy(input_imgfn, "/Users/wangshenpeng/Desktop/test_ocr.jpg", 256);
        } else {
            strncpy(input_imgfn, img_buff, 256);
        }
        data = stbi_load(input_imgfn, &image_width, &image_height, &image_channel, 4);
        if (!data) {
            std::cerr << "Image open failed.\n";
            return -1;
        }
        printf("%d %d %d\n", image_channel, image_height, image_width);
        // NHWC2NCHW(data, image_height, image_width, 4);
        printf("OCR Detector is about to start, and the picrture is %s\n", input_imgfn);
    } else if (detect_type == 2) {
        printf("Please enter the video path you want to detect:\n");
        std::cin.getline(img_buff, 256);
        std::cin.getline(img_buff, 256);
        cap.open(input_imgfn);
        if (!cap.isOpened()) {
            std::cerr << "ERROR! Unable to open video\n";
            return -1;
        }
    } else {
        int deviceID = 0;             // 0 = open default camera
        int apiID = cv::CAP_ANY;      // 0 = autodetect default API
        cap.open(deviceID, apiID);
        if (!cap.isOpened()) {
            std::cerr << "ERROR! Unable to open camera\n";
            return -1;
        }
        printf("Enter 'q' to quit from capture.\n");
    }

    auto skd_output = predictor->CreateSDKOutput();
    target_dims = predictor->GetInputShape();
    cv::Mat frame;

    while(1) {
        if (detect_type != 1) {
            cap >> frame;
            if (frame.empty()) break;
            MatType mat_type = N8UC3;
            cv::cvtColor(frame, frame, cv::COLOR_RGB2RGBA);
            image_width = frame.cols;
            image_height = frame.rows;
            image_channel = frame.channels();
            cv::Mat img = frame.clone();
            data = img.ptr();
        }

        DimsVector nchw = {1, 4, image_height, image_width};
        
        auto image_mat = std::make_shared<TNN_NS::Mat>(DEVICE_NAIVE, TNN_NS::N8UC4, nchw, data);
        auto resized_mat = std::make_shared<TNN_NS::Mat>(DEVICE_NAIVE, TNN_NS::N8UC3, target_dims);
        predictor->Resize(image_mat, image_mat, TNNInterpLinear);
        auto status = predictor->Predict(std::make_shared<TNNSDKInput>(image_mat), skd_output);
        if (status != TNN_OK) 
            if (detect_type == 1) break;
            else {
                cv::imshow("ocr detector", frame);
                auto key_num = cv::waitKey(30);
                if (key_num == 'q') break;
                continue;
            }

        auto ocr_output = dynamic_cast<TNN_NS::OCROutput *>(skd_output.get());
        const float scale_x = image_width / ocr_output->image_width;
        const float scale_y = image_height / ocr_output->image_height;

        if (detect_type == 1) {
            frame = cv::Mat(image_height, image_width, CV_8UC4, data);
        }

        if (ocr_output && ocr_output->texts.size() > 0) {
            for (int i = 0; i < ocr_output->texts.size(); i++) {
                auto box_ptr = &(ocr_output->box[i * 4]);
                std::vector<cv::Point> pts;
                pts.push_back({(int)(box_ptr[0].first * scale_x), (int)(box_ptr[0].second * scale_y)});
                pts.push_back({(int)(box_ptr[1].first * scale_x), (int)(box_ptr[1].second * scale_y)});
                pts.push_back({(int)(box_ptr[2].first * scale_x), (int)(box_ptr[2].second * scale_y)});
                pts.push_back({(int)(box_ptr[3].first * scale_x), (int)(box_ptr[3].second * scale_y)});
    
                cv::polylines(frame, pts, true, {0, 255, 255, 255});
            }
        }

        if (detect_type != 1) {
            cv::imshow("ocr detector" ,frame);
            auto key_num = cv::waitKey(30);
            if (key_num == 'q') break;
        } else {
            char buff[256];
            sprintf(buff, "%s.png", "predictions");
            int success = stbi_write_bmp(buff, image_width, image_height, 4, frame.ptr());
            if(!success) 
                return -1;
            free(data);
            break;
        }
    }
    return 0;    
}