#pragma once
#include "tnn/core/tnn.h"
#include "tnn/core/instance.h"

#define CL_TARGET_OPENCL_VERSION 200
#define CL_HPP_TARGET_OPENCL_VERSION 110
#define CL_HPP_MINIMUM_OPENCL_VERSION 110

#include "CL/cl2.hpp"

class TNNLib {
public:
    TNNLib();

    int Init(const std::string& proto_file, const std::string& model_file, const std::string& device);

    std::vector<float> Forward(void* sourcePixelscolor);

    ~TNNLib();

private:

    TNN_NS::TNN tnn_;
    std::shared_ptr<TNN_NS::Instance> instance_;

};
