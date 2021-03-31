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

#include "bert_tokenizer.h"
#include <cmath>
#include <fstream>
#include <cstring>
#include <time.h>
#include <set>
// #include "utf8.h"
#ifdef _WIN32
    #include <windows.h>
#else
    #include <sys/time.h>
#endif

namespace TNN_NS {

std::string BertTokenizer::kUnkToken = "[UNK]";
std::string BertTokenizer::kMaskToken = "[MASK]";
std::string BertTokenizer::kSepToken = "[SEP]";
std::string BertTokenizer::kPadToken = "[PAD]";
std::string BertTokenizer::kClsToken = "[CLS]";

const std::set<uint16_t> kChinesePunts = {
    12290, 65306, 65311, 8212, 8216, 12304, 12305, 12298, 12299, 65307};

const int kMaxCharsPerWords = 100;
const int MaxSeqCount = 256;

Status BertTokenizer::Init(std::string vocab_file) {
    std::ifstream ifs(vocab_file);
    if (!ifs) {
        return Status(TNNERR_INVALID_INPUT, "Vocab file not found!");
    }
    std::string content((std::istreambuf_iterator<char>(ifs)), (std::istreambuf_iterator<char>()));
    return InitByFileContent(content);
}

size_t BertTokenizer::PadId() { return token_2_id_map_.at(kPadToken); }
size_t BertTokenizer::SepId() { return token_2_id_map_.at(kSepToken); }
size_t BertTokenizer::MaskId() { return token_2_id_map_.at(kMaskToken); }
size_t BertTokenizer::UnkId() { return token_2_id_map_.at(kUnkToken); }
size_t BertTokenizer::ClsId() { return token_2_id_map_.at(kClsToken); }

Status BertTokenizer::SplitString(const char *str, size_t len, char sepChar, std::vector<std::string> &pOut) {
    const char *ptr = str;
    if (ptr == NULL || len == 0) {
        return Status(TNNERR_INVALID_INPUT, "The vocab file is empty!");
    }
    size_t start = 0;
    while (start < len && (str[start] == sepChar)) {
        start ++;
    }
    ptr = str + start;
    len = len - start;
    while (len > 0 && ptr[len - 1] == sepChar) {
        len --;
    }
    if (len <= 0) {
        return Status(TNNERR_INVALID_INPUT, "The vocab file is invalid, the seperate char should be \\n");
    }

    size_t ps = 0;
    for (size_t i = 0; i < len; i++) {
        if (ptr[i] == sepChar) {
            if (ptr[i - 1] != sepChar) {
                std::string ts(ptr, ps, i - ps);
                pOut.push_back(ts);
            }
            ps = i + 1;
        }
    }

    if (ps < len) {
        pOut.push_back(std::string(ptr, ps, len - ps));
    }
    return TNN_OK;
}

Status BertTokenizer::InitByFileContent(std::string content) {
    std::vector<std::string> lines;
    SplitString(content.c_str(), content.size(), '\n', lines);

    InitFromLines(lines);
    if (token_2_id_map_.find(kPadToken) == token_2_id_map_.end()) {
        return Status(TNNERR_INVALID_INPUT, "The vocab file is invalid, [PAD] needed.");
    }
    if (token_2_id_map_.find(kUnkToken) == token_2_id_map_.end()) {
        return Status(TNNERR_INVALID_INPUT, "The vocab file is invliad, [UNK] needed.");
    }
    if (token_2_id_map_.find(kClsToken) == token_2_id_map_.end()) {
        return Status(TNNERR_INVALID_INPUT, "The vocab file is invliad, [CLS] needed.");
    }
    if (token_2_id_map_.find(kSepToken) == token_2_id_map_.end()) {
        return Status(TNNERR_INVALID_INPUT, "The vocab file is invliad, [SEP] needed.");
    }
    if (token_2_id_map_.find(kMaskToken) == token_2_id_map_.end()) {
        return Status(TNNERR_INVALID_INPUT, "The vocab file is invliad, [MASK] needed.");
    }
    int v = token_2_id_map_.at(kPadToken);
    if (v != 0) {
        return Status(TNNERR_INVALID_INPUT, "The vocab file is invliad, [PAD] shoulde be at the head of file.");
    }
    return TNN_OK;
}

Status BertTokenizer::InitFromLines(const std::vector<std::string>& lines) {
    int idx = 0;

    for (size_t i = 0; i < lines.size(); i++) {
        
        std::string line(lines[i]);
        
        size_t nn = line.size();
        while (nn > 0 && ((line[nn - 1] == '\n') || (line[nn - 1] == '\r'))) {
            nn --;
        }
        if (nn == 0) {
            continue;
        }
        std::string token = line.substr(0, nn);
        tokens_.push_back(token);
        token_2_id_map_[token] = idx;
        idx ++;
    }
    return TNN_OK;
}

size_t BertTokenizer::Word2Id(std::string word) {
    if (word.size() > kMaxCharsPerWords) {
        return token_2_id_map_.at(kUnkToken);
    }
    auto it = token_2_id_map_.find(word);
    if (it == token_2_id_map_.end()) {
        return token_2_id_map_.at(kUnkToken);
    } else {
        return it->second;
    }
}

std::string BertTokenizer::Id2Word(size_t id) {
    if (id >= 0 && id < static_cast<int>(tokens_.size())) {
        return tokens_[id];
    }
    return kUnkToken;
}

void BertTokenizer::max_seg_(std::string s, std::vector<size_t>& results) {
  int end = s.size();
  int start = 0;
  // spdlog::info("now s:[{}]", s);
  bool firstOne = true;
  while (start < end) {
    std::string test(s.c_str() + start, end - start);
    if (!firstOne) {
      test = std::string("##") + test;
    }
    auto it = token_2_id_map_.find(test);
    if (it == token_2_id_map_.end()) {
      end -= 1;
    } else {
      // spdlog::info("now got :{}", test);
      results.push_back(it->second);
      start = end;
      end = s.size();
      firstOne = false;
    }
  }
  if (firstOne) {
    // not any one matched
    results.push_back(token_2_id_map_.at(kUnkToken));
  }
}

std::vector<size_t> BertTokenizer::Encode(std::string text, Status &status) {
    std::vector<size_t> results;
    text = StripStringASCIIWhole(text);
    for(size_t i = 0; i < text.length(); i++) {
        if (text[i] <= 'Z' && text[i] >= 'A') text[i] += 32;
    }
    std::vector<std::string> tokens;
    SplitString(text.c_str(), text.size(), ' ', tokens);
    for (auto s : tokens) {
        if (s.size() > kMaxCharsPerWords) {
            results.push_back(token_2_id_map_.at(kUnkToken));
        } else {
            max_seg_(s, results);
        }
    }

    status = TNN_OK;
    return results;

}

std::string BertTokenizer::StripStringASCIIWhole(const std::string str) {
    size_t nn = str.size();
    while (nn > 0 && (str[nn - 1] == ' ' || str[nn - 1] == '\t' ||
                      str[nn - 1] == '\r' || str[nn - 1] == '\n')) {
      nn -= 1;
    }
    size_t off = 0;
    while (off < nn && (str[off] == ' ' || str[off] == '\t' ||
                        str[off] == '\r' || str[off] == '\n')) {
      off += 1;
    }
    bool seeWhitespace = false;
    std::string ret;
    for (size_t k = off; k < nn; k++) {
      if (str[k] == ' ' || str[k] == '\t' || str[k] == '\r' || str[k] == '\n') {
        if (!seeWhitespace) {
          seeWhitespace = true;
          ret.append(1, ' ');
        }
      } else {
        seeWhitespace = false;
        ret.append(1, str[k]);
      }
    }
    return ret;
}

Status BertTokenizer::buildInput(std::string paragraph, std::string question, std::shared_ptr<BertTokenizerInput> input) {
    std::vector<size_t> code1, code2;
    
    Status status;

    code1 = Encode(question, status);
    code2 = Encode(paragraph, status);

    code1.insert(code1.begin(), ClsId());
    code1.insert(code1.end(), SepId());
    for (size_t i = 0; i < MaxSeqCount; i++) {
        if (i >= code1.size() && i < code1.size() + code2.size() + 1) reinterpret_cast<int*>(input->segments)[i] = 1;
        else reinterpret_cast<int*>(input->segments)[i] = 0;
    }
    code1.insert(code1.end(), code2.begin(), code2.end());
    code1.insert(code1.end(), SepId());

    for (int i = 0; i < code1.size(); i++) {
        features.push_back(Id2Word(code1[i]));
    }

    if (code1.size() < MaxSeqCount) {
        code1.insert(code1.end(), (MaxSeqCount - code1.size()), 0);
    }

    // code1 = { 101,  2073,  2003,  1996,  5661, 10549,  2000,  2175,  1029,
    //       102,  1999,  2049,  2220,  2086,  1010,  1996,  2047,  4680,
    //      2415,  3478,  2000,  3113,  5270,  1998,  6599, 10908,  1012,
    //      1031,  2260,  1033,  2011,  2526,  1010,  2116, 13773,  3028,
    //      5661,  2020, 10549,  1996,  2172,  3469,  9587,  9363,  2638,
    //      2415,  1999,  2624,  3799,  2058,  1996,  2624,  4560,  4680,
    //      2415,  2349,  2000,  1996,  3732,  1005,  1055,  3132,  2686,
    //      1012,  1037, 10428,  5468,  2000,  5446,  2019,  4935,  3081,
    //      1037,  3309,  4171,  3478,  2000,  3362,  1996,  3223,  2048,
    //      1011, 12263,  3484,  2000,  3413,  1012,  1999,  2238,  2384,
    //      1010,  2136,  2624,  4560,  2328,  1996,  2148,  2534,  1010,
    //      1037,  1002,  1020,  1012,  6255,  2454,  1010,  2630,  1998,
    //      2317,  9311,  1010,  5815,  3770,  1010,  2199,  2675,  2519,
    //      1006,  1021,  1010,  4278, 25525,  1007,  1997,  8327,  2686,
    //       102,     0,     0,     0,     0,     0,     0,     0,     0,
    //         0,     0,     0,     0,     0,     0,     0,     0,     0,
    //         0,     0,     0,     0,     0,     0,     0,     0,     0,
    //         0,     0,     0,     0,     0,     0,     0,     0,     0,
    //         0,     0,     0,     0,     0,     0,     0,     0,     0,
    //         0,     0,     0,     0,     0,     0,     0,     0,     0,
    //         0,     0,     0,     0,     0,     0,     0,     0,     0,
    //         0,     0,     0,     0,     0,     0,     0,     0,     0,
    //         0,     0,     0,     0,     0,     0,     0,     0,     0,
    //         0,     0,     0,     0,     0,     0,     0,     0,     0,
    //         0,     0,     0,     0,     0,     0,     0,     0,     0,
    //         0,     0,     0,     0,     0,     0,     0,     0,     0,
    //         0,     0,     0,     0,     0,     0,     0,     0,     0,
    //         0,     0,     0,     0,     0,     0,     0,     0,     0,
    //         0,     0,     0,     0};
    
    for (size_t i = 0; i < MaxSeqCount; i++) {
        if (code1[i]) {
            reinterpret_cast<int*>(input->inputIds)[i] = code1[i];
            reinterpret_cast<int*>(input->inputMasks)[i] = 1;
        } else {
            reinterpret_cast<int*>(input->inputMasks)[i] = 0;
            reinterpret_cast<int*>(input->segments)[i] = 0;
        }
    }

    return TNN_OK;
}

BertTokenizerInput::BertTokenizerInput() {
    inputIds = (void*)malloc(sizeof(float) * MaxSeqCount);
    inputMasks = (void*)malloc(sizeof(float) * MaxSeqCount);
    segments = (void*)malloc(sizeof(float) * MaxSeqCount);
    DimsVector nchw = {1, MaxSeqCount};
    mat_map_.insert(std::pair<std::string, std::shared_ptr<Mat>>("input_ids_0", 
        std::make_shared<TNN_NS::Mat>(DEVICE_X86, NC_INT32, nchw, inputIds)));
    mat_map_.insert(std::pair<std::string, std::shared_ptr<Mat>>("input_mask_0", 
        std::make_shared<TNN_NS::Mat>(DEVICE_X86, NC_INT32, nchw, inputMasks)));
    mat_map_.insert(std::pair<std::string, std::shared_ptr<Mat>>("segment_ids_0", 
        std::make_shared<TNN_NS::Mat>(DEVICE_X86, NC_INT32, nchw, segments)));
}

BertTokenizerInput::~BertTokenizerInput() {
    mat_map_.clear();
    if (inputIds) free(inputIds);
    if (inputMasks) free(inputMasks);
    if (segments) free(segments);
}
 
std::vector<size_t> BertTokenizer::_get_best_indexes(float* logits, size_t size, size_t n_best_size) {
    std::map<float, size_t, std::greater<float>> logits_index;
    for (int i = 0; i < size; i++) {
        logits_index.insert(std::pair<float, size_t>(logits[i], i));
    }
    std::vector<size_t> results;

    size_t index = 0;
    for (auto item : logits_index) {
        if (index >= n_best_size) break;
        results.push_back(item.second);
        index++;
    }
    return results;
}

struct prelim_prediction {
public:
    prelim_prediction(size_t start, size_t end, float start_logit, float end_logit) :
        start(start), end(end), start_logit(start_logit), end_logit(end_logit) {};
    size_t start, end;
    float start_logit, end_logit;
};

bool cmp(const std::shared_ptr<struct prelim_prediction> &a, const std::shared_ptr<struct prelim_prediction> &b) {
    return (a->start_logit + a->end_logit) > (b->start_logit + b->end_logit);
}

Status BertTokenizer::ConvertResult(std::shared_ptr<TNNSDKOutput> output, std::string& ans) {
    std::vector<size_t> start_index, end_index;
    float *start_logits, *end_logits;
    start_logits = reinterpret_cast<float*>(output->GetMat("unstack:0")->GetData());
    end_logits   = reinterpret_cast<float*>(output->GetMat("unstack:1")->GetData());
    start_index = _get_best_indexes(start_logits, MaxSeqCount, 20);
    end_index   = _get_best_indexes(end_logits, MaxSeqCount, 20);

    std::vector<std::shared_ptr<struct prelim_prediction>> prelim_predictions;

    for (auto start : start_index) {
        for (auto end : end_index) {
            if (start >= features.size()) continue;
            if (end >= features.size()) continue;
            if (end < start) continue;
            int length = end - start + 1;
            if (length > 20) continue;
            prelim_predictions.push_back(std::make_shared<struct prelim_prediction>(start, end, start_logits[start], end_logits[end]));
        }
    }

    std::sort(prelim_predictions.begin(), prelim_predictions.end(), cmp);
    for (auto item : prelim_predictions) {
        std::string tok;
        for (size_t i = item->start; i <= item->end; i++) {
            if (features[i].find("##") != std::string::npos) {
                auto s = features[i].replace(features[i].find("##"), 2, "");
                tok += s + " ";
            } else {
                tok += features[i] + " ";
            }
        }
        std::cout << tok << std::endl;
        // std::cout << item->start << " " << item->end << " " << item->start_logit << " " << item->end_logit << std::endl;
    }

    return TNN_OK;
}

} // namespace TNN_NS
