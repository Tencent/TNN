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

#ifndef TNNSDKSample_hpp
#define TNNSDKSample_hpp

#include <cmath>
#include <fstream>
#include <sstream>
#include "tnn/core/macro.h"
#include "tnn/core/tnn.h"
#include "tnn/utils/blob_converter.h"

#define TNN_SDK_ENABLE_BENCHMARK 1

#define TNN_SDK_USE_NCNN_MODEL 0

namespace TNN_NS {
struct ObjectInfo {
    int image_width = 0;
    int image_height = 0;
    
    float x1 = 0;
    float y1 = 0;
    float x2 = 0;
    float y2 = 0;
    
    float score = 0;
    int class_id = -1;
    
    ObjectInfo AdjustToImageSize(int image_height, int image_width);
    /**gravity 0:resize 1:resize and keep aspect 2:resize to fill the view and keep aspect*/
    ObjectInfo AdjustToViewSize(int view_height, int view_width, int gravity = 2);
    ObjectInfo FlipX();
    float IntersectionRatio(ObjectInfo *obj);
};

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

typedef  struct{
    unsigned char r,g,b,a;
}RGBA;


extern const std::string kTNNSDKDefaultName;
class TNNSDKInput {
public:
    TNNSDKInput(std::shared_ptr<TNN_NS::Mat> mat = nullptr);
    virtual ~TNNSDKInput();
    
    bool IsEmpty();
    std::shared_ptr<TNN_NS::Mat> GetMat(std::string name = kTNNSDKDefaultName);
    bool AddMat(std::shared_ptr<TNN_NS::Mat> mat, std::string name);
    
protected:
    std::map<std::string, std::shared_ptr<TNN_NS::Mat> > mat_map_ = {};
};

class TNNSDKOutput : public TNNSDKInput {
public:
    TNNSDKOutput(std::shared_ptr<Mat> mat = nullptr) : TNNSDKInput(mat) {};
    virtual ~TNNSDKOutput();
};

class TNNSDKOption {
public:
    TNNSDKOption();
    virtual ~TNNSDKOption();
    
    std::string proto_content = "";
    std::string model_content = "";
    std::string library_path = "";
    TNNComputeUnits compute_units = TNNComputeUnitsCPU;
    InputShapesMap input_shapes = {};
};

class TNNSDKSample {
public:
    TNNSDKSample();
    virtual ~TNNSDKSample();
    TNNComputeUnits GetComputeUnits();
    void SetBenchOption(BenchOption option);
    BenchResult GetBenchResult();
    DimsVector GetInputShape(std::string name = kTNNSDKDefaultName);
    
    
    virtual Status Predict(std::shared_ptr<TNNSDKInput> input, std::shared_ptr<TNNSDKOutput> &output);
    
    virtual Status Init(std::shared_ptr<TNNSDKOption> option);
    virtual MatConvertParam GetConvertParamForInput(std::string name = "");
    virtual MatConvertParam GetConvertParamForOutput(std::string name = "");
    virtual std::shared_ptr<TNNSDKOutput> CreateSDKOutput();
    virtual Status ProcessSDKOutput(std::shared_ptr<TNNSDKOutput> output);
    
    
    
protected:
    BenchOption bench_option_;
    BenchResult bench_result_;

    std::vector<std::string> GetInputNames();
    std::vector<std::string> GetOutputNames();
    
protected:
    std::shared_ptr<TNN> net_           = nullptr;
    std::shared_ptr<Instance> instance_ = nullptr;
    std::shared_ptr<TNNSDKOption> option_ = nullptr;
    DeviceType device_type_             = DEVICE_ARM;
};

void Rectangle(void *data_rgba, int image_height, int image_width,
               int x0, int y0, int x1, int y1, float scale_x, float scale_y);
}  // namespace TNN_NS

#endif /* TNNSDKSample_hpp */
