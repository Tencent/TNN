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

#ifndef TNN_EXAMPLES_BERT_TOKENIZER_H_
#define TNN_EXAMPLES_BERT_TOKENIZER_H_

#include "tnn_sdk_sample.h"
#include <algorithm>
#include <iostream>
#include <fstream>
#include <string>
#include <memory>
#include <vector>

namespace TNN_NS {

typedef struct prelim_prediction {
public:
    prelim_prediction(size_t start, size_t end, float start_logit, float end_logit) :
        start(start), end(end), start_logit(start_logit), end_logit(end_logit) {};
    size_t start, end;
    float start_logit, end_logit, prob = 1.0;
} prelim_prediction;

class BertTokenizerInput : public TNNSDKInput {
public:
    BertTokenizerInput(DeviceType device_type, const std::string& input_id_name,
            const std::string& mask_name, const std::string& segment_name);
    virtual ~BertTokenizerInput();
    void* inputIds;
    void* inputMasks;
    void* segments;
};

class BertTokenizer {
public:
    // @brief Init vocabulary with vocab file
    Status Init(std::string vocab);

    // @brief Init vocabulary with vocab content
    Status InitByFileContent(std::string content);

    // @brief Encode the text to tokens and features
    std::vector<size_t> Encode(std::string text, Status &status);

    // @brief find vocabulary by id
    size_t Word2Id(std::string word);

    // @brief find id by vocabulary
    std::string Id2Word(size_t id);
    
    size_t PadId();
    size_t MaskId();
    size_t SepId();
    size_t ClsId();
    size_t UnkId();

    // build inputBert with paragraph and question
    Status buildInput(std::string paragraph, std::string question, std::shared_ptr<BertTokenizerInput> input);

    // @brief get indexes from result
    std::vector<size_t> _get_best_indexes(float* logits, size_t size, size_t n_best_size);

    // @brief judge if a charactor is punctuate chracter
    bool is_punct_char(char cp);

    // @brief seperate text with whitespace and punctuate character
    std::string basic_separate(std::string text);

    // @brief set a string to lower case
    std::string toLower(std::string s);

    // @brief tokenize BertOutput to text result and probalities
    Status ConvertResult(std::shared_ptr<TNNSDKOutput> output, const std::string& start_logits_name, 
            const std::string& end_logits_name, std::string& ans);

    // @brief calculate probabilities for result
    Status CalProbs(std::vector<std::shared_ptr<prelim_prediction>> scores);
private:
    // @brief seperate token to indivisible one
    void max_seg_(std::string s, std::vector<size_t>& results);

    // @brief get vocabulary by lines of char
    Status InitFromLines(const std::vector<std::string>& lines);

    // @brief split strings by sepChar (usually '\\n')
    Status SplitString(const char *str, size_t len, char sepChar, std::vector<std::string> &pOut);

    // @brief split string ASCII
    std::string StripStringASCIIWhole(const std::string str);

    // @param map between token and id in vocabulary
    std::map<std::string, int> token_2_id_map_;
    std::vector<std::string> tokens_;

    // signs needed by vocabulary
    // @param [Unk] for Unknown
    static std::string kUnkToken;
    // @param [Mask] for Mask
    static std::string kMaskToken;
    // @param [Sep] for Seperate
    static std::string kSepToken;
    // @param [Pad] for Pad
    static std::string kPadToken;
    // @param [Cls] for Cls
    static std::string kClsToken;

    // @param index to origin word 
    std::vector<std::string> features_;
};
} // namespace TNN_NS

#endif
