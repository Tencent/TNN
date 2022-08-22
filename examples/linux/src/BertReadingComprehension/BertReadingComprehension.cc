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

static const char demo_guide[] = "If you don't know how to use this demo or got an execution error.\n"
    "step1. Check the model and vocab path (download from url: https://github.com/darrenyao87/tnn-models/tree/master/model/bertsquad10)\n"
    "step2. Enter a paragraph found in the wiki or elsewhere\n"
    "       eg. TNN README\n"
    "       TNN: A high-performance, lightweight neural network inference framework open sourced by Tencent Youtu Lab. It also has many outstanding advantages such as cross-platform, high performance, model compression, and code tailoring. The TNN framework further strengthens the support and performance optimization of mobile devices on the basis of the original Rapidnet and ncnn frameworks. At the same time, it refers to the high performance and good scalability characteristics of the industry's mainstream open source frameworks, and expands the support for X86 and NV GPUs. On the mobile phone, TNN has been used by many applications such as mobile QQ, weishi, and Pitu. As a basic acceleration framework for Tencent Cloud AI, TNN has provided acceleration support for the implementation of many businesses. Everyone is welcome to participate in the collaborative construction to promote the further improvement of the TNN inference framework.\n"
    "step3. Enter a question about the paragraph\n"
    "       what is TNN?\n"
    "       where TNN has been used?\n"
    "Quote: If you want to use tiny-bert model. Switch model in <path_to_TNN>/model/tiny-bert/, and change the input and output name\n"
    "       in <path_to_TNN>/examples/linux/src/BertRreadingComprehension/BertReadingComprehension.cc with line 86 and 110";

static const char vocab_path_message[] = "(required) vocab file path";
DEFINE_string(v, "", vocab_path_message);

#define LETTER_MAX_COUNT 10000
#define MAX_SEQ_LENGTH 256
int main(int argc, char **argv) {
    if (!ParseAndCheckCommandLine(argc, argv, false)) {
        printf("%s\n", demo_guide);

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
    
    printf("%s\n%s\n%s\n", FLAGS_v.c_str(), FLAGS_p.c_str(), FLAGS_m.c_str());
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
            option->compute_units = TNN_NS::TNNComputeUnitsGPU;
        #endif
        
    }

    // choose bertsquad or tiny-bert
    auto bertInput = std::make_shared<BertTokenizerInput>(DEVICE_NAIVE, "input_ids_0", "input_mask_0", "segment_ids_0");  
    // auto bertInput = std::make_shared<BertTokenizerInput>(DEVICE_NAIVE, "input_ids", "attention_mask", "token_type_ids");  
    
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
        tokenizer->buildInput(paragraph, question, bertInput);
        CHECK_TNN_STATUS(predictor->Predict(bertInput, bertOutput));
        std::string ans;

        //choose bertsquad or tiny-bert
        tokenizer->ConvertResult(bertOutput, "unstack:0", "unstack:1", ans);
        // tokenizer->ConvertResult(bertOutput, "output_0", "output_1", ans);

        std::cout << "Please Enter the question: (Enter exit to quit)" << std::endl;
        std::cin.getline(question, LETTER_MAX_COUNT);
    }
}