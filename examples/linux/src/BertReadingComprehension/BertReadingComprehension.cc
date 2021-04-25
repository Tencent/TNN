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

#include "../flags.h"

using namespace TNN_NS;

static const char vocab_path_message[] = "(required) vocab file path";
DEFINE_string(v, "", vocab_path_message);

#define LETTER_MAX_COUNT 10000
#define MAX_SEQ_LENGTH 256
int main(int argc, char **argv) {
    if (!ParseAndCheckCommandLine(argc, argv, false)) {
        ShowUsage(argv[0], false);
        printf("\t-v, <vocab>    \t%s\n", vocab_path_message);
        return -1;
    }
    if (FLAGS_v.empty()) {
        printf("\t-v, <vocab>    \t%s\n", vocab_path_message);
        return -1;
    }

    auto tokenizer = std::make_shared<BertTokenizer>();

    std::cout << "Initializing Vocabularies..." << std::endl;
    tokenizer->Init(FLAGS_v.c_str());
    
    // 创建tnn实例
    std::cout << "Initializing TNN Instance..." << std::endl;
    auto proto_content = fdLoadFile(FLAGS_p.c_str());
    auto model_content = fdLoadFile(FLAGS_m.c_str());
    int h = 1, w = 256;
    std::vector<int> nchw = {1, 256};

    auto option = std::make_shared<TNNSDKOption>();
    {
        option->proto_content = proto_content;
        option->model_content = model_content;
        
        option->library_path = "";
        option->compute_units = TNN_NS::TNNComputeUnitsCPU;
        // if enable openvino/tensorrt, set option compute_units to openvino/tensorrt
        #ifdef _CUDA_
            option->compute_units = TNN_NS::TNNComputeUnitsTensorRT;
        #elif _OPENVINO_
            option->compute_units = TNN_NS::TNNComputeUnitsOpenvino;
        #endif
        
        option->input_shapes.insert(std::pair<std::string, DimsVector>("input_ids_0", nchw));
        option->input_shapes.insert(std::pair<std::string, DimsVector>("input_mask_0", nchw));
        option->input_shapes.insert(std::pair<std::string, DimsVector>("segment_ids_0", nchw));
    }

    
    auto bertInput = std::make_shared<BertTokenizerInput>(DEVICE_NAIVE);  
    auto predictor = std::make_shared<TNNSDKSample>();

    auto bertOutput = predictor->CreateSDKOutput();
    
    CHECK_TNN_STATUS(predictor->Init(option));

    char* paragraph = (char*)malloc(sizeof(char) * LETTER_MAX_COUNT);
    char* question = (char*)malloc(sizeof(char) * LETTER_MAX_COUNT);

    std::cout << "Please Enter the paragraph:(words limit count " << MAX_SEQ_LENGTH << ")" << std::endl;
    std::cin.getline(paragraph, LETTER_MAX_COUNT);
    std::cout << "Please Enter the question:" << std::endl;
    std::cin.getline(question, LETTER_MAX_COUNT);

    const std::string quit("exit");
    while (quit.compare(question) != 0) {
        // std::string paragraph = "In its early years, the new convention center failed to meet attendance and revenue expectations.[12] By 2002, many Silicon Valley businesses were choosing the much larger Moscone Center in San Francisco over the San Jose Convention Center due to the latter's limited space. A ballot measure to finance an expansion via a hotel tax failed to reach the required two-thirds majority to pass. In June 2005, Team San Jose built the South Hall, a $6.77 million, blue and white tent, adding 80,000 square feet (7,400 m2) of exhibit space";

        // std::string question = "where is the businesses choosing to go?";
        tokenizer->buildInput(paragraph, question, bertInput);
        CHECK_TNN_STATUS(predictor->Predict(bertInput, bertOutput));
        std::string ans;
        tokenizer->ConvertResult(bertOutput, ans);

        std::cout << "Please Enter the question: (Enter exit to quit)" << std::endl;
        std::cin.getline(question, LETTER_MAX_COUNT);
    }
}