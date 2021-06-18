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

#include "face_detect_aligner.h"
#include "blazeface_detector.h"
#include "face_mesh.h"
#include "youtu_face_align.h"
#include "tnn_sdk_sample.h"
#include "macro.h"
#include "utils/utils.h"

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

Status initDetectPredictor(std::shared_ptr<BlazeFaceDetector>& predictor, int argc, char** argv) {
    char detect_path_buff[256];
    char *detect_model_path = detect_path_buff;
    if (argc < 3) {
        strncpy(detect_model_path, "../../../../model/blazeface/", 256);
    } else {
        strncpy(detect_model_path, argv[2], 256);
    }

    std::string detect_proto = std::string(detect_model_path) + "blazeface.tnnproto";
    std::string detect_model = std::string(detect_model_path) + "blazeface.tnnmodel";
    std::string anchor_path = std::string(detect_model_path) + "blazeface_anchors.txt";

    auto detect_proto_content = fdLoadFile(detect_proto);
    auto detect_model_content = fdLoadFile(detect_model);
    auto detect_option = std::make_shared<BlazeFaceDetectorOption>();

    const int targer_height = 128;
    const int targer_width = 128;
    DimsVector target_dims = {1, 3, targer_height, targer_width};
    {
        detect_option->proto_content = detect_proto_content;
        detect_option->model_content = detect_model_content;
        detect_option->library_path = "";
        
        detect_option->compute_units = TNN_NS::TNNComputeUnitsCPU;
        // if enable openvino/tensorrt, set option compute_units to openvino/tensorrt
        #ifdef _CUDA_
            detect_option->compute_units = TNN_NS::TNNComputeUnitsGPU;
        #endif
        
        detect_option->min_score_threshold = 0.75;
        detect_option->min_suppression_threshold = 0.3;
        detect_option->anchor_path = anchor_path;
    }

    predictor = std::make_shared<BlazeFaceDetector>();
    auto status = predictor->Init(detect_option);
    return status;
}

Status initFaceAlignPredictor(std::shared_ptr<YoutuFaceAlign>& predictor, int argc, char **argv, int phase) {
    char align_path_buff[256];
    char *align_model_path = align_path_buff;
    if (argc < 2) {
        strncpy(align_model_path, "../../../../model/youtu_face_alignment/", 256);
    } else {
        strncpy(align_model_path, argv[1], 256);
    }

    std::string align_proto, align_model, align_pts;
    if (phase == 1) {
        align_proto = std::string(align_model_path) + "youtu_face_alignment_phase1.tnnproto";
        align_model = std::string(align_model_path) + "youtu_face_alignment_phase1.tnnmodel";
        align_pts   = std::string(align_model_path) + "youtu_mean_pts_phase1.txt";
    } else {
        align_proto = std::string(align_model_path) + "youtu_face_alignment_phase2.tnnproto";
        align_model = std::string(align_model_path) + "youtu_face_alignment_phase2.tnnmodel";
        align_pts   = std::string(align_model_path) + "youtu_mean_pts_phase2.txt";
    }
    auto align_proto_content = fdLoadFile(align_proto);
    auto align_model_content = fdLoadFile(align_model);

    auto align_option = std::make_shared<YoutuFaceAlignOption>();
    {
        align_option->proto_content = align_proto_content;
        align_option->model_content = align_model_content;
        align_option->library_path = "";
        align_option->compute_units = TNN_NS::TNNComputeUnitsCPU;
        // if enable openvino/tensorrt, set option compute_units to openvino/tensorrt
        #ifdef _CUDA_
            align_option->compute_units = TNN_NS::TNNComputeUnitsGPU;
        #endif

        // set parameters
        const int target_height = 128;
        const int target_width = 128;
        const int target_channel = 1;
        align_option->input_width = target_width;
        align_option->input_height = target_height;
        align_option->face_threshold = 0.5;
        align_option->min_face_size = 20;;
        align_option->phase = phase;
        align_option->net_scale = phase == 1 ? 1.2 : 1.3;
        align_option->mean_pts_path = align_pts;
    }

    predictor = std::make_shared<YoutuFaceAlign>();
    auto status = predictor->Init(align_option);

    return status;
}

int main(int argc, char** argv) {
    std::shared_ptr<BlazeFaceDetector> detect_sdk;
    CHECK_TNN_STATUS(initDetectPredictor(detect_sdk, argc, argv));

    std::shared_ptr<YoutuFaceAlign> align_sdk1;
    CHECK_TNN_STATUS(initFaceAlignPredictor(align_sdk1, argc, argv, 1));

    std::shared_ptr<YoutuFaceAlign> align_sdk2;
    CHECK_TNN_STATUS(initFaceAlignPredictor(align_sdk2, argc, argv, 2));

    auto predictor = std::make_shared<FaceDetectAligner>();
    predictor->Init({detect_sdk, align_sdk1, align_sdk2});

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
    // char* input_imgfn = "../../../assets/test_face.jpg";
    char *input_imgfn = img_buff;
    int image_width, image_height, image_channel;
#ifdef _OPENCV_
    cv::VideoCapture cap;
#endif
    unsigned char *data;

    if (detect_type == 1) {
        printf("Please enter the image path you want to detect:\n");
        std::cin.getline(img_buff, 256);
        std::cin.getline(img_buff, 256);
        if (strlen(img_buff) == 0) {
            strncpy(input_imgfn, "../../../assets/test_blazeface.jpg", 256);
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

    const int target_width = 128, target_height = 128;
    DimsVector target_dims = {1, 3, target_height, target_width};

    std::shared_ptr<TNNSDKOutput> output = nullptr;
    std::vector<BlazeFaceInfo> face_info;

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
        DimsVector nchw = {1, image_channel, image_height, image_width};
        auto image_mat = std::make_shared<Mat>(DEVICE_NAIVE, N8UC3, nchw, data);
        auto resize_mat = std::make_shared<Mat>(DEVICE_NAIVE, N8UC3, target_dims);
        // TNN_NS::ResizeParam param;
        // TNN_NS::MatUtils::Resize(*image_mat, *resize_mat, param, NULL);

        CHECK_TNN_STATUS(detect_sdk->Resize(image_mat, resize_mat, TNNInterpNearest));
        CHECK_TNN_STATUS(predictor->Predict(std::make_shared<TNNSDKInput>(resize_mat), output));
        CHECK_TNN_STATUS(predictor->ProcessSDKOutput(output));

        uint8_t *ifm_buf = new uint8_t[image_width * image_height * 4];
        for (int i = 0; i < image_width * image_height; ++i) {
            ifm_buf[i*4]    = data[i*3];
            ifm_buf[i*4+1]  = data[i*3+1];
            ifm_buf[i*4+2]  = data[i*3+2];
            ifm_buf[i*4+3]  = 255;
        }
        if (output && dynamic_cast<YoutuFaceAlignOutput *>(output.get())) {
            auto face = dynamic_cast<YoutuFaceAlignOutput *>(output.get())->face;
            
            auto face_preview = face.AdjustToImageSize(image_height, image_width);
            for (auto point : face_preview.key_points) {
                Point(ifm_buf, image_height, image_width, point.first, point.second, 0.f);
            }
        }

#ifdef _OPENCV_
        if (detect_type != 1) {
            cv::Mat face_frame(image_height, image_width, CV_8UC4, ifm_buf);
            cv::imshow("face_dectecting" ,face_frame);

            delete [] ifm_buf;
            auto key_num = cv::waitKey(30);
            if (key_num == 'q') break;
        } else {
#endif
            char buff[256];
            sprintf(buff, "%s.png", "predictions");
            int success = stbi_write_bmp(buff, image_width, image_height, 4, ifm_buf);
            if(!success) 
                return -1;
            delete [] ifm_buf;
            fprintf(stdout, "Face-detector done. The result was saved in %s.png\n", "predictions");
            free(data);
#ifdef _OPENCV_
            break;
        }
    }
#endif
    return 0;
}
