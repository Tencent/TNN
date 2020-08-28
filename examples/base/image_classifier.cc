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

ImageClassifier::ImageClassifier() {}

ImageClassifier::~ImageClassifier() {}

int ImageClassifier::Classify(std::shared_ptr<TNN_NS::Mat> image_mat, int image_height, int image_width, int &class_id) {
    if (!image_mat || !image_mat->GetData()) {
        std::cout << "image is empty ,please check!" << std::endl;
        return -1;
    }

#if TNN_SDK_ENABLE_BENCHMARK
    bench_result_.Reset();
    tnn::SampleTimer timer;

    for (int fcount = 0; fcount < bench_option_.forward_count; fcount++) {
        timer.Start();
#endif
        
        // step 1. set input mat
        TNN_NS::MatConvertParam input_cvt_param;
        input_cvt_param.scale = {1.0 / (255 * 0.229), 1.0 / (255 * 0.224), 1.0 / (255 * 0.225), 0.0};
        input_cvt_param.bias  = {-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225, 0.0};
        auto status = instance_->SetInputMat(image_mat, input_cvt_param);
        RETURN_ON_NEQ(status, TNN_NS::TNN_OK);
        
        // step 2. Forward
        status = instance_->ForwardAsync(nullptr);
        RETURN_ON_NEQ(status, TNN_NS::TNN_OK);
        
        // step 3. get output mat
        std::shared_ptr<TNN_NS::Mat> output_mat_scores = nullptr;
        status = instance_->GetOutputMat(output_mat_scores);
        RETURN_ON_NEQ(status, TNN_NS::TNN_OK);

#if TNN_SDK_ENABLE_BENCHMARK
        timer.Stop();
        bench_result_.AddTime(timer.GetTime());
        timer.Reset();
#endif

        class_id           = -1;
        float *scores_data = (float *)output_mat_scores.get()->GetData();
        float max_v        = scores_data[0];
        for (int i = 1; i < output_mat_scores->GetChannel(); ++i) {
            if (max_v < scores_data[i]) {
                max_v    = scores_data[i];
                class_id = i;
            }
        }

#if TNN_SDK_ENABLE_BENCHMARK
    }
#endif
    return 0;
}
