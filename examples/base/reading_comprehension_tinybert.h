//
// Created by rgb000000 on 2021/8/6.
//

#ifndef TNN_EXAMPLES_BASE_READING_COMPREHENSION_TINYBERT_H
#define TNN_EXAMPLES_BASE_READING_COMPREHENSION_TINYBERT_H

#include <algorithm>
#include <iostream>
#include <string>
#include <vector>

#include "tnn_sdk_sample.h"

namespace TNN_NS {

class ReadingComprehensionTinyBertOutput : public TNNSDKOutput {
public:
    ReadingComprehensionTinyBertOutput(std::shared_ptr<Mat> mat = nullptr) : TNNSDKOutput(mat) {};
    virtual ~ReadingComprehensionTinyBertOutput();

};

class ReadingComprehensionTinyBert : public TNN_NS::TNNSDKSample {
public:
    virtual ~ReadingComprehensionTinyBert();
    virtual std::shared_ptr<TNNSDKOutput> CreateSDKOutput();
    virtual Status ProcessSDKOutput(std::shared_ptr<TNNSDKOutput> output_);
    virtual std::shared_ptr<TNN_NS::Mat> ProcessSDKInputMat(std::shared_ptr<TNN_NS::Mat> input_mat,
                                                            std::string name);

};

}

#endif //TNN_EXAMPLES_BASE_READING_COMPREHENSION_TINYBERT_H
