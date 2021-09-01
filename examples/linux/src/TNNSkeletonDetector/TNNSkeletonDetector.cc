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

#include "blazepose_detector.h"
#include "skeleton_detector.h"
#include "tnn_sdk_sample.h"
#include "macro.h"
#include "utils/utils.h"

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
using namespace TNN_NS;

int main(int argc, char** argv) {
        if (!ParseAndCheckCommandLine(argc, argv, false)) {
        ShowUsage(argv[0]);
        printf("The default proto and model file could be found at ../../../../model/sklenton/\n");
        return -1;
    }

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
            strncpy(input_imgfn, "../../../assets/skeleton_test.jpg", 256);
        } else {
            strncpy(input_imgfn, img_buff, 256);
        }
        data = stbi_load(input_imgfn, &image_width, &image_height, &image_channel, 3);
        if (!data) {
            std::cerr << "Image open failed.\n";
            return -1;
        }
        printf("Pose-detector is about to start, and the picrture is %s\n", input_imgfn);
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

    // 创建tnn实例
    auto proto_content = fdLoadFile(FLAGS_p.c_str());
    auto model_content = fdLoadFile(FLAGS_m.c_str());
    int h = 128, w = 128;
    if(argc >= 5) {
        h = std::atoi(argv[3]);
        w = std::atoi(argv[4]);
    }
    auto option = std::make_shared<SkeletonDetectorOption>();
    {
        option->proto_content = proto_content;
        option->model_content = model_content;
        option->library_path = "";
        option->compute_units = TNN_NS::TNNComputeUnitsCPU;
        #ifdef _CUDA_
            option->compute_units = TNN_NS::TNNComputeUnitsGPU;
        #endif
    
        option->input_width = w;
        option->input_height = h;
        // option->score_threshold = 0.95;
        // option->iou_threshold = 0.15;
    }
    
    auto predictor = std::make_shared<SkeletonDetector>();
    std::vector<int> nchw = {1, 3, h, w};

    // Init
    std::shared_ptr<TNNSDKOutput> sdk_output = predictor->CreateSDKOutput();
    CHECK_TNN_STATUS(predictor->Init(option));

#ifdef _OPENCV_
    cv::Mat frame;
    while(1) {
        if (detect_type != 1) {
            cap >> frame;
            if (frame.empty()) break;
            MatType mat_type = N8UC3;
            image_width = frame.cols;
            image_height = frame.rows;
            image_channel = frame.channels();
            // cv::Mat img = frame.clone();
            data = frame.ptr();
        }
#endif
        DimsVector orig_dims = {1, image_channel, image_height, image_width};

        //Predict
        auto image_mat = std::make_shared<TNN_NS::Mat>(TNN_NS::DEVICE_NAIVE, TNN_NS::N8UC3, orig_dims, data);
        auto resize_mat = predictor->ProcessSDKInputMat(image_mat, "input");
        CHECK_TNN_STATUS(predictor->Predict(std::make_shared<SkeletonDetectorInput>(resize_mat), sdk_output));
        SkeletonInfo pose_info;
        if (sdk_output && dynamic_cast<SkeletonDetectorOutput *>(sdk_output.get())) {
            auto pose_output = dynamic_cast<SkeletonDetectorOutput *>(sdk_output.get());
            pose_info = pose_output->keypoints;
        }

        const int image_orig_height = int(image_height);
        const int image_orig_width  = int(image_width);
        const DimsVector target_dims = predictor->GetInputShape();
        const int target_height = target_dims[2];
        const int target_width = target_dims[3];
        float scale_x               = image_orig_width / (float)target_width;
        float scale_y               = image_orig_height / (float)target_height;

        //convert rgb to rgb-a
        uint8_t *ifm_buf = new uint8_t[image_width*image_height*4];
        for (int i = 0; i < image_width * image_height; ++i) {
            ifm_buf[i*4]   = data[i*3];
            ifm_buf[i*4+1] = data[i*3+1];
            ifm_buf[i*4+2] = data[i*3+2];
            ifm_buf[i*4+3] = 255;
        }

        pose_info = pose_info.AdjustToImageSize(image_orig_height, image_orig_width);
        for (int i = 0; i < pose_info.lines.size(); i++) {
            auto pose = pose_info.lines[i];
            if (pose.first <= 4 || pose.second <= 4) continue;
            auto x1 = pose_info.key_points[pose.first].first;
            auto y1 = pose_info.key_points[pose.first].second;
            auto x2 = pose_info.key_points[pose.second].first;
            auto y2 = pose_info.key_points[pose.second].second;
            TNN_NS::Line((void *)ifm_buf, image_orig_height, image_orig_width, x1, y1, x2, y2);
        }
        for (auto point : pose_info.key_points) {
            TNN_NS::Point(ifm_buf, image_height, image_width, point.first, point.second, 0.f);
        }

#ifdef _OPENCV_
        if (detect_type != 1) {
            cv::Mat pose_frame(image_height, image_width, CV_8UC4, ifm_buf);
            cv::imshow("pose_dectecting" ,pose_frame);

            delete [] ifm_buf;
            auto key_num = cv::waitKey(30);
            if (key_num == 'q') break;
        } else {
#endif
            char buff[256];
            sprintf(buff, "%s.png", "predictions");
            int success = stbi_write_bmp(buff, image_orig_width, image_orig_height, 4, ifm_buf);
            if(!success) 
                return -1;
            printf("Pose Detect done. The result was saved in %s.png\n", "predictions");
            delete [] ifm_buf;
            free(data);
#ifdef _OPENCV_
            break;
        }
    }
#endif

    return 0;
}
