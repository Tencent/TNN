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
        
        #ifdef _CUDA_
            detect_option->compute_units = TNN_NS::TNNComputeUnitsGPU;
        #elif _ARM_
            detect_option->compute_units = TNN_NS::TNNComputeUnitsCPU;
        #else
            detect_option->compute_units = TNN_NS::TNNComputeUnitsOpenvino;
        #endif
        
        detect_option->min_score_threshold = 0.75;
        detect_option->min_suppression_threshold = 0.3;
        detect_option->anchor_path = anchor_path;
    }

    predictor = std::make_shared<BlazeFaceDetector>();
    auto status = predictor->Init(detect_option);
    return status;
}

Status initFaceAlignPredictor(std::shared_ptr<Facemesh>& predictor, int argc, char **argv) {
    char align_path_buff[256];
    char *align_model_path = align_path_buff;
    if (argc < 2) {
        strncpy(align_model_path, "../../../../model/face_mesh/", 256);
    } else {
        strncpy(align_model_path, argv[1], 256);
    }

    std::string align_proto = std::string(align_model_path) + "face_mesh.tnnproto";
    std::string align_model = std::string(align_model_path) + "face_mesh.tnnmodel";
    auto align_proto_content = fdLoadFile(align_proto);
    auto align_model_content = fdLoadFile(align_model);

    auto align_option = std::make_shared<FacemeshOption>();
    {
        align_option->proto_content = align_proto_content;
        align_option->model_content = align_model_content;
        align_option->library_path = "";
        #ifdef _CUDA_
            align_option->compute_units = TNN_NS::TNNComputeUnitsGPU;
        #elif _ARM_
            align_option->compute_units = TNN_NS::TNNComputeUnitsCPU;
        #else
            align_option->compute_units = TNN_NS::TNNComputeUnitsOpenvino;
        #endif

        // set parameters
        align_option->face_presence_threshold = 0.1;
        align_option->flip_vertically = false;
        align_option->flip_horizontally = false;
        align_option->norm_z = 1.f;
        align_option->ignore_rotation = false;
    }

    predictor = std::make_shared<Facemesh>();
    auto status = predictor->Init(align_option);

    return status;
}

int main(int argc, char** argv) {
    std::shared_ptr<BlazeFaceDetector> detect_sdk;
    CHECK_TNN_STATUS(initDetectPredictor(detect_sdk, argc, argv));

    std::shared_ptr<Facemesh> align_sdk;
    CHECK_TNN_STATUS(initFaceAlignPredictor(align_sdk, argc, argv));

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
            cv::Mat img = frame.clone();
            data = img.ptr();
        }
#endif
        DimsVector nchw = {1, image_channel, image_height, image_width};
        auto image_mat = std::make_shared<Mat>(DEVICE_NAIVE, N8UC3, nchw, data);
        auto resize_mat = std::make_shared<Mat>(DEVICE_NAIVE, N8UC3, target_dims);
        CHECK_TNN_STATUS(detect_sdk->Resize(image_mat, resize_mat, TNNInterpLinear));

        CHECK_TNN_STATUS(detect_sdk->Predict(std::make_shared<BlazeFaceDetectorInput>(resize_mat), output));

        if (output && dynamic_cast<BlazeFaceDetectorOutput *>(output.get())) {
            auto face_output = dynamic_cast<BlazeFaceDetectorOutput *>(output.get());
            face_info = face_output->face_list;
        }

        uint8_t *ifm_buf = new uint8_t[image_width * image_height * 4];
        for (int i = 0; i < image_width * image_height; ++i) {
            ifm_buf[i*4]    = data[i*3];
            ifm_buf[i*4+1]  = data[i*3+1];
            ifm_buf[i*4+2]  = data[i*3+2];
            ifm_buf[i*4+3]  = 255;
        }

        for (auto face : face_info) {
            auto face_orig = face.AdjustToViewSize(image_height, image_width, 2);
            printf("%f %f %f %f\n", face_orig.x1, face_orig.x2, face_orig.y1, face_orig.y2);
            int crop_h = face_orig.y2 - face_orig.y1;
            int crop_w = face_orig.x2 - face_orig.x1;
            
            DimsVector crop_dims = {1, 3, (int)(1.5 * crop_h), (int)(1.5 * crop_w)};
            std::shared_ptr<Mat> croped_mat = std::make_shared<Mat>(DEVICE_NAIVE, N8UC3, crop_dims);
            CHECK_TNN_STATUS(detect_sdk->Crop(image_mat, croped_mat, (int)(face_orig.x1 - crop_w * 0.25), (int)(face_orig.y1 - crop_h * 0.25)));

            DimsVector align_dims = {1, 3, 192, 192};
            std::shared_ptr<Mat> input_mat = std::make_shared<Mat>(DEVICE_NAIVE, N8UC3, align_dims);
            CHECK_TNN_STATUS(detect_sdk->Resize(croped_mat, input_mat, TNNInterpLinear));

            std::shared_ptr<TNNSDKOutput> align_output = nullptr;
            CHECK_TNN_STATUS(align_sdk->Predict(std::make_shared<FacemeshInput>(input_mat), align_output));

            std::vector<FacemeshInfo> face_mesh_info;
            if (align_output && dynamic_cast<FacemeshOutput*>(align_output.get())) {
                auto face_output = dynamic_cast<FacemeshOutput*>(align_output.get());
                face_mesh_info = face_output->face_list;
            }

            for (auto face_mesh : face_mesh_info) {
                auto face_mesh_crop = face_mesh.AdjustToViewSize(1.5 * crop_h, 1.5 * crop_w, 2);
                face_mesh_crop = face_mesh_crop.AddOffset((int)(face_orig.x1 - crop_w * 0.25), (int)(face_orig.y1 - crop_h * 0.25));
                for (auto p : face_mesh_crop.key_points) {
                    TNN_NS::Point(ifm_buf, image_height, image_width, p.first, p.second, 0.f);
                }
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
            fprintf(stdout, "Face-detector done.\nNumber of faces: %d\n",int(face_info.size()));
            free(data);
#ifdef _OPENCV_
            break;
        }
    }
#endif
    return 0;
}
