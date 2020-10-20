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

#include "image_classifier.h"
#include "sample_timer.h"
#include <cmath>

namespace TNN_NS {

ImageClassifierOutput::~ImageClassifierOutput() {}

ImageClassifier::~ImageClassifier() {}

MatConvertParam ImageClassifier::GetConvertParamForInput(std::string tag) {
    MatConvertParam input_cvt_param;
    input_cvt_param.scale = {1.0 / (255 * 0.229), 1.0 / (255 * 0.224), 1.0 / (255 * 0.225), 0.0};
    input_cvt_param.bias  = {-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225, 0.0};
    return input_cvt_param;
}

std::shared_ptr<TNNSDKOutput> ImageClassifier::CreateSDKOutput() {
    return std::make_shared<ImageClassifierOutput>();
}

Status ImageClassifier::ProcessSDKOutput(std::shared_ptr<TNNSDKOutput> output_) {
    Status status = TNN_OK;
    auto output = dynamic_cast<ImageClassifierOutput *>(output_.get());
    RETURN_VALUE_ON_NEQ(!output, false,
                        Status(TNNERR_PARAM_ERR, "TNNSDKOutput is invalid"));
    
    auto output_mat_scores = output->GetMat();
    RETURN_VALUE_ON_NEQ(!output_mat_scores, false,
                        Status(TNNERR_PARAM_ERR, "output_mat_scores is invalid"));
    
    int class_id           = 0;
    float *scores_data = (float *)output_mat_scores.get()->GetData();
    float max_v        = scores_data[0];
    for (int i = 1; i < output_mat_scores->GetChannel(); ++i) {
        if (max_v < scores_data[i]) {
            max_v    = scores_data[i];
            class_id = i;
        }
    }
    output->class_id = class_id;
    return status;
}

}
