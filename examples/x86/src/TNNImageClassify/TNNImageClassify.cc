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

#include "image_classifier.h"
#include "macro.h"
#include "utils/utils.h"

#define STB_IMAGE_IMPLEMENTATION
#include "../../../../third_party/stb/stb_image.h"
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "../../../../third_party/stb/stb_image_resize.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../../../../third_party/stb/stb_image_write.h"

using namespace TNN_NS;

int main(int argc, char** argv) {
// int main(int argc, char** argv) {
    if (argc < 3) {
        printf("how to run:  %s proto model height width\n", argv[0]);
        return -1;
    }
    // 创建tnn实例
    auto proto_content = fdLoadFile(argv[1]);
    auto model_content = fdLoadFile(argv[2]);

    auto option = std::make_shared<TNNSDKOption>();
    {
        option->proto_content = proto_content;
        option->model_content = model_content;
        option->library_path = "";
        option->compute_units = TNN_NS::TNNComputeUnitsOpenvino;
    }

    auto predictor = std::make_shared<ImageClassifier>();

    char* temp_p;
    char line[256];
    FILE *fp_label;
#ifdef _WIN32
    if((fp_label = fopen("../../../assets/synset.txt", "r")) == NULL)
        return -1;
#else
    if((fp_label = fopen("../../assets/synset.txt", "r")) == NULL)
        return -1;
#endif
    static unsigned char labels[1000][256];
    for(int i = 0; i < 1000; i++){
        temp_p = fgets(line, 256 ,fp_label);
        memcpy(labels[i], line, 256);
    }
    fclose(fp_label);

    char img_buff[256];
    char *input_imgfn = img_buff;
    if(argc < 6)
#ifdef _WIN32
        strncpy(input_imgfn, "../../../assets/tiger_cat.jpg", 256);
#else
        strncpy(input_imgfn, "../../assets/tiger_cat.jpg", 256);
#endif
    else
        strncpy(input_imgfn, argv[5], 256);
    printf("Classify is about to start, and the picrture is %s\n",input_imgfn);

    int image_width, image_height, image_channel;
    unsigned char *data = stbi_load(input_imgfn, &image_width, &image_height, &image_channel, 3);
    std::vector<int> nchw = {1, image_channel, image_height, image_width};

    //Init
    std::shared_ptr<TNNSDKOutput> sdk_output = predictor->CreateSDKOutput();
    CHECK_TNN_STATUS(predictor->Init(option));
    //Predict
    auto image_mat = std::make_shared<TNN_NS::Mat>(TNN_NS::DEVICE_NAIVE, TNN_NS::N8UC3, nchw, data);
    CHECK_TNN_STATUS(predictor->Predict(std::make_shared<TNNSDKInput>(image_mat), sdk_output));

    int class_id = -1;
    if (sdk_output && dynamic_cast<ImageClassifierOutput *>(sdk_output.get())) {
        auto classfy_output = dynamic_cast<ImageClassifierOutput *>(sdk_output.get());
        class_id = classfy_output->class_id;
    }
    //完成计算，获取任意输出点
    fprintf(stdout, "Classify done. Result: %sOutput argmax %d\n",labels[class_id], class_id+1);
    fprintf(stdout, "%s\n", predictor->GetBenchResult().Description().c_str());
    free(data);
    return 0;
}
