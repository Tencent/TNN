#include <algorithm>
#include <iostream>
#include <string>
#include <vector>

#include "tnn_sdk_sample.h"

namespace TNN_NS {

class ImageClassifierEvalOutput : public TNNSDKOutput {
public:
    ImageClassifierEvalOutput(std::shared_ptr<Mat> mat = nullptr) : TNNSDKOutput(mat) {};
    virtual ~ImageClassifierEvalOutput();
    
    int *class_id;
};

class ImageClassifierEval : public TNN_NS::TNNSDKSample {
public:
    virtual ~ImageClassifierEval();
    virtual MatConvertParam GetConvertParamForInput(std::string tag = "");
    virtual std::shared_ptr<TNNSDKOutput> CreateSDKOutput();
    virtual Status ProcessSDKOutput(std::shared_ptr<TNNSDKOutput> output);
    virtual std::shared_ptr<TNN_NS::Mat> ProcessSDKInputMat(std::shared_ptr<TNN_NS::Mat> mat,
                                                              std::string name);
    Status SetOMPThreads(int num_threads);
};

}