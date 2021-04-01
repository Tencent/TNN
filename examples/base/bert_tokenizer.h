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

class BertTokenizerInput : public TNNSDKInput {
public:
    BertTokenizerInput();
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
    size_t TotalSize();

    Status buildInput(std::string paragraph, std::string question, std::shared_ptr<BertTokenizerInput> input);

    // @brief get indexed
    std::vector<size_t> _get_best_indexes(float* logits, size_t size, size_t n_best_size);

    bool is_punct_char(char cp);

    std::string basic_separate(std::string text);

    std::string toLower(std::string s);

    Status ConvertResult(std::shared_ptr<TNNSDKOutput> output, std::string& ans);
private:
    void max_seg_(std::string s, std::vector<size_t>& results);
    void load_vocab_(std::string path, std::vector<std::string>& lines);

    // @brief get vocabulary by lines of char
    Status InitFromLines(const std::vector<std::string>& lines);

    // @brief split strings by sepChar (usually '\\n')
    Status SplitString(const char *str, size_t len, char sepChar, std::vector<std::string> &pOut);

    std::string StripStringASCIIWhole(const std::string str);

    // UString _basic_tokenize(UString text);
    // UString _clean(UString text);
    std::map<std::string, int> token_2_id_map_;
    std::vector<std::string> tokens_;

    // signs needed by vocabulary
    static std::string kUnkToken;
    static std::string kMaskToken;
    static std::string kSepToken;
    static std::string kPadToken;
    static std::string kClsToken;

    std::vector<std::string> features_, features_low_;
    std::vector<size_t> token_map_;

};
} // namespace TNN_NS

#endif