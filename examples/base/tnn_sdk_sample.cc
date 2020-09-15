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

#include "tnn_sdk_sample.h"

#if defined(__APPLE__)
#include "TargetConditionals.h"
#endif

namespace TNN_NS {
std::string BenchOption::Description() {
    std::ostringstream ostr;
    ostr << "create_count = " << create_count << "  warm_count = " << warm_count
         << "  forward_count = " << forward_count;

    ostr << std::endl;
    return ostr.str();
}

void BenchResult::Reset() {
    min   = FLT_MAX;
    max   = FLT_MIN;
    avg   = 0;
    total = 0;
    count = 0;

    diff = 0;
}

int BenchResult::AddTime(float time) {
    count++;
    total += time;
    min = std::min(min, time);
    max = std::max(max, time);
    avg = total / count;
    return 0;
}

std::string BenchResult::Description() {
    std::ostringstream ostr;
    ostr << "min = " << min << "  max = " << max << "  avg = " << avg;

    if (status != TNN_NS::TNN_OK) {
        ostr << "\nerror = " << status.description();
    }
    ostr << std::endl;

    return ostr.str();
}

TNNSDKSample::TNNSDKSample() {}

TNNSDKSample::~TNNSDKSample() {}

TNN_NS::Status TNNSDKSample::Init(const std::string &proto_content, const std::string &model_path,
                                const std::string &library_path, TNNComputeUnits units, std::vector<int> nchw) {
    //网络初始化
    TNN_NS::Status status;
    if (!net_) {
        TNN_NS::ModelConfig config;
#if TNN_SDK_USE_NCNN_MODEL
        config.model_type = TNN_NS::MODEL_TYPE_NCNN;
#else
        config.model_type = TNN_NS::MODEL_TYPE_TNN;
#endif
        config.params = {proto_content, model_path};

        auto net = std::make_shared<TNN_NS::TNN>();
        status   = net->Init(config);
        if (status != TNN_NS::TNN_OK) {
            LOGE("instance.net init failed %d", (int)status);
            return status;
        }
        net_ = net;
    }

    // network init
    device_type_ = TNN_NS::DEVICE_ARM;
    if (units >= TNNComputeUnitsGPU) {
#if defined(__APPLE__) && TARGET_OS_IPHONE
        device_type_ = TNN_NS::DEVICE_METAL;
#else
        device_type_      = TNN_NS::DEVICE_OPENCL;
#endif
    }
    InputShapesMap shapeMap;
    if (nchw.size() == 4) {
        shapeMap.insert(std::pair<std::string, DimsVector>("input", nchw));
    }
    //创建实例instance
    {
        TNN_NS::NetworkConfig network_config;
        network_config.library_path = {library_path};
        network_config.device_type  = device_type_;
        auto instance               = net_->CreateInst(network_config, status, shapeMap);
        if (status != TNN_NS::TNN_OK || !instance) {
            // try device_arm
            if (units >= TNNComputeUnitsGPU) {
                device_type_               = TNN_NS::DEVICE_ARM;
                network_config.device_type = TNN_NS::DEVICE_ARM;
                instance                   = net_->CreateInst(network_config, status, shapeMap);
            }
        }
        instance_ = instance;
    }

    return status;
}

TNNComputeUnits TNNSDKSample::GetComputeUnits() {
    switch (device_type_) {
        case DEVICE_METAL:
        case DEVICE_OPENCL:
            return TNNComputeUnitsGPU;
        default:
            return TNNComputeUnitsCPU;
    }
}

void TNNSDKSample::SetBenchOption(BenchOption option) {
    bench_option_ = option;
}

BenchResult TNNSDKSample::GetBenchResult() {
    return bench_result_;
}

/*
 * Rectangle
 */
void Rectangle(void *data_rgba, int image_height, int image_width,
               int x0, int y0, int x1, int y1, float scale_x, float scale_y)
{
    typedef  struct{
        unsigned char r,g,b,a;
    }RGBA;
    
    RGBA *image_rgba = (RGBA *)data_rgba;

    int x_min = std::min(x0, x1) * scale_x;
    int x_max = std::max(x0, x1) * scale_x;
    int y_min = std::min(y0, y1) * scale_y;
    int y_max = std::max(y0, y1) * scale_y;

    x_min = std::min(std::max(x_min, 0), image_width - 1);
    x_max = std::min(std::max(x_max, 0), image_width - 1);
    y_min = std::min(std::max(y_min, 0), image_height - 1);
    y_max = std::min(std::max(y_max, 0), image_height - 1);

    // top bottom
    for (int x = x_min; x <= x_max; x++) {
        int offset                       = y_min * image_width + x;
        image_rgba[offset]               = {0, 255, 0, 0};
        image_rgba[offset + image_width] = {0, 255, 0, 0};

        offset                           = y_max * image_width + x;
        image_rgba[offset]               = {0, 255, 0, 0};
        image_rgba[offset - image_width] = {0, 255, 0, 0};
    }

    // left right
    for (int y = y_min; y <= y_max; y++) {
        int offset             = y * image_width + x_min;
        image_rgba[offset]     = {0, 255, 0, 0};
        image_rgba[offset + 1] = {0, 255, 0, 0};

        offset                 = y * image_width + x_max;
        image_rgba[offset]     = {0, 255, 0, 0};
        image_rgba[offset - 1] = {0, 255, 0, 0};
    }
}

}  // namespace TNN_NS
