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

#include "FaceGrayTransfer.h"
#include <sys/time.h>
#include <cmath>

FaceGrayTransfer::FaceGrayTransfer() {}

FaceGrayTransfer::~FaceGrayTransfer() {}

TNN_NS::Status FaceGrayTransfer::Trasfer(std::shared_ptr<TNN_NS::Mat> input_mat,
               std::shared_ptr<TNN_NS::Mat>&output_mat,
               int input_width, int input_length) {
    if (!input_mat || !input_mat->GetData()) {
        std::cout << "input mat is empty ,please check!" << std::endl;
        return TNN_NS::Status(-1, "input mat is empty ,please check!");
    }
    
    output_mat = nullptr;

#if TNN_SDK_ENABLE_BENCHMARK
    bench_result_.Reset();
    for (int fcount = 0; fcount < bench_option_.forward_count; fcount++) {
        timeval tv_begin, tv_end;
        gettimeofday(&tv_begin, NULL);
#endif
        
        // step 1. set input mat
        TNN_NS::MatConvertParam input_cvt_param;
        input_cvt_param.scale = {2.0 / 255, 2.0 / 255, 2.0 / 255, 0.0};
        input_cvt_param.bias  = {-1.0, -1.0, -1.0, 0.0};
        auto status = instance_->SetInputMat(input_mat, input_cvt_param);
        RETURN_ON_NEQ(status, TNN_NS::TNN_OK);
        
        // step 2. Forward
        status = instance_->ForwardAsync(nullptr);
        RETURN_ON_NEQ(status, TNN_NS::TNN_OK);
        
        // step 3. get output mat
        TNN_NS::MatConvertParam output_cvt_param;
//        output_cvt_param.scale = {255 / 2.0, 255 / 2.0, 255 / 2.0, 0.0};
//        output_cvt_param.bias  = {255 / 2.0, 255 / 2.0, 255 / 2.0, 0.0};
        status = instance_->GetOutputMat(output_mat, output_cvt_param);
        RETURN_ON_NEQ(status, TNN_NS::TNN_OK);

#if TNN_SDK_ENABLE_BENCHMARK
        gettimeofday(&tv_end, NULL);
        double elapsed = (tv_end.tv_sec - tv_begin.tv_sec) * 1000.0 + (tv_end.tv_usec - tv_begin.tv_usec) / 1000.0;
        bench_result_.AddTime(elapsed);
#endif
//
//        class_id           = -1;
//        float *scores_data = (float *)output_mat_scores.get()->GetData();
//        float max_v        = scores_data[0];
//        for (int i = 1; i < output_mat_scores->GetChannel(); ++i) {
//            if (max_v < scores_data[i]) {
//                max_v    = scores_data[i];
//                class_id = i;
//            }
//        }

#if TNN_SDK_ENABLE_BENCHMARK
    }
#endif
    return TNN_NS::TNN_OK;
}
