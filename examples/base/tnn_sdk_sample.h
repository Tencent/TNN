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

#ifndef TNN_EXAMPLES_BASE_TNN_SDK_SAMPLE_H_
#define TNN_EXAMPLES_BASE_TNN_SDK_SAMPLE_H_

#include <cmath>
#include <fstream>
#include <sstream>
#include "tnn/core/macro.h"
#include "tnn/core/tnn.h"
#include "tnn/utils/blob_converter.h"

#define TNN_SDK_ENABLE_BENCHMARK 1

#define TNN_SDK_USE_NCNN_MODEL 0

namespace TNN_NS {
struct BenchOption {
    int warm_count    = 0;
    int forward_count = 1;
    int create_count  = 1;

    std::string Description();
};

struct BenchResult {
    TNN_NS::Status status;

    // time
    float min   = FLT_MAX;
    float max   = FLT_MIN;
    float avg   = 0;
    float total = 0;
    int count   = 0;

    float diff = 0;

    void Reset();
    int AddTime(float time);
    std::string Description();
};

typedef enum {
    // run on cpu
    TNNComputeUnitsCPU = 0,
    // run on gpu, if failed run on cpu
    TNNComputeUnitsGPU = 1,
    // run on npu, if failed run on cpu
    TNNComputeUnitsNPU = 2,
} TNNComputeUnits;

class TNNSDKSample {
public:
    TNNSDKSample();
    virtual ~TNNSDKSample();
    virtual TNN_NS::Status Init(const std::string &proto_content, const std::string &model_path,
                                const std::string &library_path, TNNComputeUnits units, std::vector<int> nchw = {});
    TNNComputeUnits GetComputeUnits();
    void SetBenchOption(BenchOption option);
    BenchResult GetBenchResult();

protected:
    BenchOption bench_option_;
    BenchResult bench_result_;

protected:
    std::shared_ptr<TNN_NS::TNN> net_           = nullptr;
    std::shared_ptr<TNN_NS::Instance> instance_ = nullptr;
    TNN_NS::DeviceType device_type_             = DEVICE_ARM;
};

void Rectangle(void *data_rgba, int image_height, int image_width,
               int x0, int y0, int x1, int y1, float scale_x, float scale_y);
}  // namespace TNN_NS

#endif // TNN_EXAMPLES_BASE_TNN_SDK_SAMPLE_H_
