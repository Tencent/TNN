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
#include <iostream>
#include <string>
#include <vector>

#include "object_detector_yolo.h"
#include "macro.h"
#include "utils/utils.h"
#include "tnn_sdk_sample.h"

#include "../flags.h"

#define STB_IMAGE_IMPLEMENTATION
#include "../../../../third_party/stb/stb_image.h"
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "../../../../third_party/stb/stb_image_resize.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../../../../third_party/stb/stb_image_write.h"

#ifdef _OPENCV_
    #include <opencv2/opencv.hpp>
#endif

const std::string label_list[] = {
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
                "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
                "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
                "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
                "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
                "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
                "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
                "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
                "hair drier", "toothbrush"};

int main(int argc, char **argv) {
    if (!ParseAndCheckCommandLine(argc, argv, false)) {
        ShowUsage(argv[0]);
        printf("The default proto and model file could be found at ../../../../model/yolov5/\n");
        return -1;
    }

    auto proto_content = fdLoadFile(FLAGS_p.c_str());
    auto model_content = fdLoadFile(FLAGS_m.c_str());

    printf("Please choose the source you want to detect:\n");
    printf("1. picture;\t2. video;\t3. camera.\n");

    // detect type: 1.image; 2.video; 3.camera
    int detect_type; 
    scanf("%d", &detect_type);
#ifdef _OPENCV_
    if (detect_type < 1 || detect_type > 3) {
        std::cerr << "ERROR! Invalid source type!\n";
        return -1;
    }
#else
    if (detect_type > 1) {
        std::cerr << "ERROR! OpenCV not installed! this source is invalid\n";
        return -1;
    }
#endif

    char img_buff[256];
    char *input_imgfn = img_buff;
    int image_width, image_height, image_channel;
#ifdef _OPENCV_
    cv::VideoCapture cap;
#endif
    unsigned char *data;

    // build input source
    if (detect_type == 1) {
        printf("Please enter the image path you want to detect:\n");
        std::cin.getline(img_buff, 256);
        std::cin.getline(img_buff, 256);
        if (strlen(img_buff) == 0) {
            strncpy(input_imgfn, "../../../assets/004545.jpg", 256);
        } else {
            strncpy(input_imgfn, img_buff, 256);
        }
        printf("Face-detector is about to start, and the picrture is %s\n", input_imgfn);
        data = stbi_load(input_imgfn, &image_width, &image_height, &image_channel, 3);
        if (!data) {
            std::cerr << "Image open failed.\n";
            return -1;
        }
    } else if (detect_type == 2) {
#ifdef _OPENCV_
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
#endif
    }

    // auto proto_path = "../../../../model/yolov'5/yolov5s-permute.tnnproto";
    // auto model_path = "../../../../model/yolov5/yolov5s.tnnmodel";

    auto option = std::make_shared<TNN_NS::TNNSDKOption>();
    {
        option->proto_content = proto_content;
        option->model_content = model_content;
        option->compute_units = TNN_NS::TNNComputeUnitsCPU;
        // if enable openvino/tensorrt, set option compute_units to openvino/tensorrt
        #ifdef _CUDA_
            option->compute_units = TNN_NS::TNNComputeUnitsTensorRT;
        #elif _OPENVINO_
            option->compute_units = TNN_NS::TNNComputeUnitsOpenvino;
        #endif
    }

    auto predictor = std::make_shared<TNN_NS::ObjectDetectorYolo>();
    auto status = predictor->Init(option);
    if (status != TNN_NS::TNN_OK) {
        std::cout << "Predictor Initing failed, please check the option parameters" << std::endl;
    }
    std::shared_ptr<TNN_NS::TNNSDKOutput> sdk_output = nullptr;

#ifdef _OPENCV_
    cv::Mat frame;
    while(1) {
        if (detect_type != 1) {
            cap >> frame;
            if (frame.empty()) break;
            image_width = frame.cols;
            image_height = frame.rows;
            image_channel = frame.channels();
            // cv::Mat img = frame.clone();
            data = frame.ptr();
        }
#endif

    TNN_NS::DimsVector nchw = {1, image_channel, image_height, image_width};
    auto image_mat = std::make_shared<TNN_NS::Mat>(TNN_NS::DEVICE_NAIVE, TNN_NS::N8UC3, nchw, data);
    auto resize_mat = predictor->ProcessSDKInputMat(image_mat, "images");
    CHECK_TNN_STATUS(predictor->Predict(std::make_shared<TNN_NS::TNNSDKInput>(resize_mat), sdk_output));
    CHECK_TNN_STATUS(predictor->ProcessSDKOutput(sdk_output));

    std::vector<TNN_NS::ObjectInfo> object_list;
    if (sdk_output && dynamic_cast<TNN_NS::ObjectDetectorYoloOutput *>(sdk_output.get())) {
        auto obj_output = dynamic_cast<TNN_NS::ObjectDetectorYoloOutput *>(sdk_output.get());
        object_list = obj_output->object_list;
    }

    const int image_orig_height = int(image_height);
    const int image_orig_width  = int(image_width);
    const auto& target_dims     = predictor->GetInputShape();
    const int target_height     = target_dims[2];
    const int target_width      = target_dims[3];
    float scale_x               = image_orig_width  / (float)target_width;
    float scale_y               = image_orig_height / (float)target_height;

    uint8_t *ifm_buf = new uint8_t[image_orig_width*image_orig_height*4];
    for (int i = 0; i < image_orig_height * image_orig_width; i++) {
        ifm_buf[i * 4] = data[i * 3];
        ifm_buf[i * 4 + 1] = data[i * 3 + 1];
        ifm_buf[i * 4 + 2] = data[i * 3 + 2];
        ifm_buf[i * 4 + 3] = 255;
    }

    for (int i = 0; i < object_list.size(); i++) {
        auto object = object_list[i];
        TNN_NS::Rectangle((void*)ifm_buf, image_orig_height, image_orig_width, object.x1, object.y1,
                           object.x2, object.y2, scale_x, scale_y);
    }

#ifdef _OPENCV_
        if (detect_type != 1) {
            cv::Mat face_frame(image_height, image_width, CV_8UC4, ifm_buf);
            for (auto object : object_list) {
                int x = (int)(std::min)(object.x1, object.x2) * scale_x;
                int y = (int)(std::min)(object.y1, object.y2) * scale_y;
                cv::Point point(x, y);
                cv::Scalar color(0, 0, 255);
                cv::putText(face_frame, label_list[object.class_id], point, cv::FONT_HERSHEY_PLAIN, 2.0, color);
            }
            cv::imshow("object_dectecting", face_frame);

            auto key_num = cv::waitKey(30);
            delete [] ifm_buf;
            if (key_num == 'q') break;
        } else {
#endif
            char buff[256];
            sprintf(buff, "%s.png", "predictions");
            int success = stbi_write_bmp(buff, image_orig_width, image_orig_height, 4, ifm_buf);
            if(!success) 
                return -1;
            delete [] ifm_buf;
            fprintf(stdout, "Object-Detector Done.\nNumber of objects: %d\n", int(object_list.size()));
            printf("The result was saved in %s.png\n", "predictions");
            free(data);
#ifdef _OPENCV_
            break;
        }
    }
#endif

    return 0;
}