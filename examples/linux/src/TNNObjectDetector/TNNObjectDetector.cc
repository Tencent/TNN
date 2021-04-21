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

#include "object_detector_ssd.h"
#include "macro.h"
#include "utils/utils.h"
#include "tnn_sdk_sample.h"
#include <iostream>

#define STB_IMAGE_IMPLEMENTATION
#include "../../../../third_party/stb/stb_image.h"
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "../../../../third_party/stb/stb_image_resize.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../../../../third_party/stb/stb_image_write.h"

int main(int argc, char **argv) {
    
    auto proto_path = "../../../model/mobilenet_v2-ssd/mobilenetv2_ssd.tnnproto";
    auto model_path = "../../../model/mobilenet_v2-ssd/mobilenetv2_ssd.tnnmodel";

    auto proto_content = fdLoadFile(proto_path);
    auto model_content = fdLoadFile(model_path);

    int target_height = 300;
    int target_width = 300;
    int target_channel = 3;

    auto option = std::make_shared<TNN_NS::TNNSDKOption>();
    {
        option->proto_content = proto_content;
        option->model_content = model_content;
        // if enable openvino/tensorrt, set option compute_units to openvino/tensorrt
        #ifdef _CUDA_
            option->compute_units = TNN_NS::TNNComputeUnitsGPU;
        #else
            option->compute_units = TNN_NS::TNNComputeUnitsCPU;
        #endif
        // option->input_shapes = nchw;
    }

    char img_buff[256];
    char* input_imgfn = img_buff;
    if (argc < 2) {
        strncpy(input_imgfn, "../../../assets/004545.jpg", 256);
    } else {
        strncpy(input_imgfn, argv[1], 256);
    }

    int image_width, image_height, image_channel;
    unsigned char *data = stbi_load(input_imgfn, &image_width, &image_height, &image_channel, 3);
    std::vector<int> nchw = {1, 3, image_height, image_width};

    if (!data) {
        fprintf(stderr, "ImageClassifier open file %s failed.\n", input_imgfn);
    }

    auto predictor = std::make_shared<TNN_NS::ObjectDetectorSSD>();
    auto status = predictor->Init(option);
    if (status != TNN_NS::TNN_OK) {
        std::cout << "Predictor Initing failed, please check the option parameters" << std::endl;
    }

    std::shared_ptr<TNN_NS::TNNSDKOutput> sdk_output = nullptr;

    auto image_mat = std::make_shared<TNN_NS::Mat>(TNN_NS::DEVICE_X86, TNN_NS::N8UC3, nchw, data);
    auto resize_mat = predictor->ProcessSDKInputMat(image_mat, "data_input");
    CHECK_TNN_STATUS(predictor->Predict(std::make_shared<TNN_NS::TNNSDKInput>(resize_mat), sdk_output));

    CHECK_TNN_STATUS(predictor->ProcessSDKOutput(sdk_output));
    std::vector<TNN_NS::ObjectInfo> object_list;
    if (sdk_output && dynamic_cast<TNN_NS::ObjectDetectorSSDOutput *>(sdk_output.get())) {
        auto obj_output = dynamic_cast<TNN_NS::ObjectDetectorSSDOutput *>(sdk_output.get());
        object_list = obj_output->object_list;
    }

    const int image_orig_height = int(image_height);
    const int image_orig_width  = int(image_width);
    float scale_x               = image_orig_width / (float)target_width;
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

    char buff[256];
    sprintf(buff, "%s.png", "predictions");
    int success = stbi_write_bmp(buff, image_orig_width, image_orig_height, 4, ifm_buf);
    if (!success) return -1;

    fprintf(stdout, "Object-Detector Done.\nNumber of objects: %d\n", int(object_list.size()));
    delete [] ifm_buf;
    free(data);

    if (argc < 2)
        printf("Or you can use \' %s image_path \' to run on different images.\n", argv[0]);
    return 0;
}