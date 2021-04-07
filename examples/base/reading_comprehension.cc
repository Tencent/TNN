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

#include "reading_comprehension.h"
#include "sample_timer.h"
#include <cmath>
#include <cstring>

const size_t MaxSeqCount = 256;
namespace TNN_NS {
ReadingComprehensionInput::ReadingComprehensionInput() {
    inputIds = (void*)malloc(sizeof(float) * MaxSeqCount);
    inputMask = (void*)malloc(sizeof(float) * MaxSeqCount);
    segmentIds = (void*)malloc(sizeof(float) * MaxSeqCount);
    DimsVector nchw = {1, 1, 1, MaxSeqCount};
    mat_map_.insert(std::pair<std::string, std::shared_ptr<Mat>>("input_ids_0", 
        std::make_shared<TNN_NS::Mat>(DEVICE_X86, NCHW_FLOAT, nchw, inputIds)));
    mat_map_.insert(std::pair<std::string, std::shared_ptr<Mat>>("input_mask_0", 
        std::make_shared<TNN_NS::Mat>(DEVICE_X86, NCHW_FLOAT, nchw, inputMask)));
    mat_map_.insert(std::pair<std::string, std::shared_ptr<Mat>>("segment_ids_0", 
        std::make_shared<TNN_NS::Mat>(DEVICE_X86, NCHW_FLOAT, nchw, segmentIds)));
}

ReadingComprehensionInput::~ReadingComprehensionInput() {
    mat_map_.clear();
    if (inputIds) free(inputIds);
    if (inputMask) free(inputMask);
    if (segmentIds) free(segmentIds);
}

ReadingComprehensionOutput::ReadingComprehensionOutput() {
    unstack0 = (void*)malloc(sizeof(float) * MaxSeqCount);
    unstack1 = (void*)malloc(sizeof(float) * MaxSeqCount);
    DimsVector nchw = {1, 1, 1, MaxSeqCount};
    AddMat(std::make_shared<Mat>(DEVICE_X86, NCHW_FLOAT, nchw, unstack0), "unstack:0");
    AddMat(std::make_shared<Mat>(DEVICE_X86, NCHW_FLOAT, nchw, unstack0), "unstack:1");
}

ReadingComprehensionOutput::~ReadingComprehensionOutput() {
    mat_map_.clear();
    if (unstack0) free(unstack0);
    if (unstack1) free(unstack1);
}

ReadingComprehension::ReadingComprehension() {};
ReadingComprehension::~ReadingComprehension() {};

std::shared_ptr<TNNSDKOutput> ReadingComprehension::CreateSDKOutput() {
    return std::make_shared<ReadingComprehensionOutput>();
}

Status ReadingComprehension::Predict(std::shared_ptr<TNNSDKInput> input, std::shared_ptr<TNNSDKOutput> &output) {
    Status status = TNN_OK;
    if (!input || input->IsEmpty()) {
        status = Status(TNNERR_PARAM_ERR, "input image is empty ,please check!");
        LOGE("input image is empty ,please check!\n");
        return status;
    }

    // step 1. set input mat
    auto input_names = GetInputNames();
}

}