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

#include "hair_segmentation.h"
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
            strncpy(input_imgfn, "../../../assets/test_blazeface.jpg", 256);
        } else {
            strncpy(input_imgfn, img_buff, 256);
        }
        printf("Hair-segmentation is about to start, and the picrture is %s\n", input_imgfn);
        data = stbi_load(input_imgfn, &image_width, &image_height, &image_channel, 4);
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

    // 创建tnn实例
    auto proto_content = fdLoadFile(FLAGS_p.c_str());
    auto model_content = fdLoadFile(FLAGS_m.c_str());
   // int h = 240, w = 320;

    auto option = std::make_shared<HairSegmentationOption>();
    {
        option->proto_content = proto_content;
        option->model_content = model_content;
        option->library_path = "";
        option->compute_units = TNN_NS::TNNComputeUnitsCPU;
        // if enable openvino/tensorrt, set option compute_units to openvino/tensorrt
        #ifdef _CUDA_
            option->compute_units = TNN_NS::TNNComputeUnitsGPU;
        #endif
    }
    
    auto predictor = std::make_shared<HairSegmentation>();

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
            cv::cvtColor(frame, frame, cv::COLOR_RGB2RGBA);
            image_width = frame.cols;
            image_height = frame.rows;
            image_channel = frame.channels();
            // cv::Mat img = frame.clone();
            data = frame.ptr();
        }
#endif
        DimsVector orig_dims = {1, 4, image_height, image_width};
        //Predict

        auto image_mat = std::make_shared<TNN_NS::Mat>(TNN_NS::DEVICE_NAIVE, TNN_NS::N8UC4, orig_dims, data);
        CHECK_TNN_STATUS(predictor->Predict(std::make_shared<TNNSDKInput>(image_mat), sdk_output));
        // CHECK_TNN_STATUS(predictor->ProcessSDKOutput(sdk_output));
        ImageInfo merged_image;
        if (sdk_output && dynamic_cast<HairSegmentationOutput *>(sdk_output.get())) {
            auto hair_output = dynamic_cast<HairSegmentationOutput *>(sdk_output.get());
            merged_image = hair_output->merged_image;
        }

        auto ifm_buf = (uint8_t*)(merged_image.data.get());

#ifdef _OPENCV_
        if (detect_type != 1) {
            cv::Mat hair_frame(image_height, image_width, CV_8UC4, ifm_buf);
            cv::imshow("hair_segmentation" ,hair_frame);

            // delete [] ifm_buf;
            auto key_num = cv::waitKey(30);
            if (key_num == 'q') break;
        } else {
#endif
            char buff[256];
            sprintf(buff, "%s.png", "predictions");
            int success = stbi_write_bmp(buff, merged_image.image_width, merged_image.image_height, merged_image.image_channel, ifm_buf);
            if(!success) 
                return -1;
            printf("Hair Segmentation Done. The result was saved in %s.png\n", "predictions");
            // delete [] ifm_buf;
            free(data);
#ifdef _OPENCV_
            break;
        }
    }
#endif

    return 0;
}
