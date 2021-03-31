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

#include "bert_tokenizer.h"
#include "tnn_sdk_sample.h"
#include "utils/utils.h"
#include "macro.h"

using namespace TNN_NS;

#define LETTER_MAX_COUNT 10000
#define MAX_SEQ_LENGTH 256
int main(int argc, char **argv) {
    auto tokenizer = std::make_shared<BertTokenizer>();
    std::cout << "Initializing..." << std::endl;
    tokenizer->Init("../vocab.txt");
    Status status;
    
    // char* paragraph = (char*)malloc(sizeof(char) * LETTER_MAX_COUNT);
    // char* question = (char*)malloc(sizeof(char) * LETTER_MAX_COUNT);

    // std::cout << "Please Enter the paragraph:(words limit count " << MAX_SEQ_LENGTH << ")" << std::endl;
    // std::cin.getline(paragraph, LETTER_MAX_COUNT);
    // std::cout << "Please Enter the question:" << std::endl;
    // std::cin.getline(question, LETTER_MAX_COUNT);

    std::string paragraph = "In its early years, the new convention center failed to meet attendance and revenue expectations.[12] By 2002, many Silicon Valley businesses were choosing the much larger Moscone Center in San Francisco over the San Jose Convention Center due to the latter's limited space. A ballot measure to finance an expansion via a hotel tax failed to reach the required two-thirds majority to pass. In June 2005, Team San Jose built the South Hall, a $6.77 million, blue and white tent, adding 80,000 square feet (7,400 m2) of exhibit space";

    std::string question = "how may votes did the ballot measure need?";

    auto bertInput = std::make_shared<BertTokenizerInput>();
    tokenizer->buildInput(paragraph, question, bertInput);

    // 创建tnn实例
    auto proto_content = fdLoadFile("bertsquad10_clean.tnnproto");
    auto model_content = fdLoadFile("bertsquad10_clean.tnnmodel");
    int h = 1, w = 256;
    std::vector<int> nchw = {1, 256};

    // auto inputId = std::make_shared<TNN_NS::Mat>(TNN_NS::DEVICE_X86, TNN_NS::MatType::NCHW_FLOAT, nchw, bertInput->inputIds);
    // auto inputMask = std::make_shared<TNN_NS::Mat>(TNN_NS::DEVICE_X86, TNN_NS::MatType::NCHW_FLOAT, nchw, bertInput->inputMasks);
    // auto segment = std::make_shared<TNN_NS::Mat>(TNN_NS::DEVICE_X86, TNN_NS::MatType::NCHW_FLOAT, nchw, bertInput->segments);

    std::cout << "Building Inputs..." << std::endl;

    auto option = std::make_shared<TNNSDKOption>();
    option->proto_content = proto_content;
    option->model_content = model_content;
    
    option->library_path = "";
    option->compute_units = TNN_NS::TNNComputeUnits::TNNComputeUnitsOpenvino;// TNNComputeUnitsOpenvino;
    
    option->input_shapes.insert(std::pair<std::string, DimsVector>("input_ids_0", nchw));
    option->input_shapes.insert(std::pair<std::string, DimsVector>("input_mask_0", nchw));
    option->input_shapes.insert(std::pair<std::string, DimsVector>("segment_ids_0", nchw));


    auto predictor = std::make_shared<TNNSDKSample>();
    std::cout << "initing predictor" << std::endl;

    auto bertOutput = predictor->CreateSDKOutput();
    
    CHECK_TNN_STATUS(predictor->Init(option));
    CHECK_TNN_STATUS(predictor->Predict(bertInput, bertOutput));

    float* data1 = reinterpret_cast<float*>(bertOutput->GetMat("unstack:0")->GetData());
    // for (int i = 0; i < 256; i++) {
    //     std::cout << data1[i] << ", ";
    // }
    // std::cout << std::endl;

    // auto index = tokenizer->_get_best_indexes(data1, 256, 20);
    // for (int i = 0; i < index.size(); i++) std::cout << index[i] << std::endl;
    std::string ans;
    tokenizer->ConvertResult(bertOutput, ans);


    // if (paragraph) free(paragraph);
}