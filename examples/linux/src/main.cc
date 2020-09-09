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
#include "utils.h"


// 随机初始化 0~255 BGR图像数据
static void InitRandom(uint8_t* ptr, size_t n) {
    for (unsigned long long i = 0; i < n; i++) {
        ptr[i] = static_cast<uint8_t>(rand() % 256);
    }
}
// Helper functions
std::string fdLoadFile(std::string path) {
    std::ifstream file(path, std::ios::in);
    if (file.is_open()) {
        file.seekg(0, file.end);
        int size      = file.tellg();
        char* content = new char[size];

        file.seekg(0, file.beg);
        file.read(content, size);
        std::string fileContent;
        fileContent.assign(content, size);
        delete[] content;
        file.close();
        return fileContent;
    } else {
        return "";
    }
}

int main(int argc, char** argv) {
    if (argc < 3) {
        printf("how to run:  %s proto model height width\n", argv[0]);
        return -1;
    }
    // 创建tnn实例
    int h = 224, w = 224;
    if(argc >= 5) {
        h = std::atoi(argv[3]);
        w = std::atoi(argv[4]);
    }

    std::shared_ptr<ImageClassifier>  classifier = std::make_shared<ImageClassifier>();
    std::vector<int> nchw = {1, 3, h, w};

    auto proto = fdLoadFile(argv[1]);
    auto model = fdLoadFile(argv[2]);

    CHECK_TNN_STATUS(classifier->Init(proto, model, "", TNN_NS::TNNComputeUnitsCPU));
    auto input_mat = std::make_shared<TNN_NS::Mat>(TNN_NS::DEVICE_ARM, TNN_NS::N8UC3, nchw);

    int result;
    CHECK_TNN_STATUS(classifier->Classify(input_mat, w, h, result));

    //完成计算，获取任意输出点
    fprintf(stdout, "Classify done, output argmax %d\n", result);
    return 0;
}
