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

#include "monodepth.h"

namespace TNN_NS {

Status MonoDepth::Init(std::shared_ptr<TNNSDKOption> option_i) {
    Status status = TNN_OK;
    auto option = dynamic_cast<MonoDepthOption *>(option_i.get());
    RETURN_VALUE_ON_NEQ(!option, false, Status(TNNERR_PARAM_ERR, "TNNSDKOption is invalid"));
    
    status = TNNSDKSample::Init(option_i);
    RETURN_ON_NEQ(status, TNN_OK);
    
    return status;
}

MatConvertParam MonoDepth::GetConvertParamForInput(std::string tag) {
    MatConvertParam input_convert_param;
    input_convert_param.scale = {0.00392f, 0.00392f, 0.00392f, 0.0};
    input_convert_param.bias  = {0.0,    0.0,      0.0,      0.0};
    return input_convert_param;
}

std::shared_ptr<TNNSDKOutput> MonoDepth::CreateSDKOutput() {
    return std::make_shared<MonoDepthOutput>();
}

std::shared_ptr<Mat> MonoDepth::ProcessSDKInputMat(std::shared_ptr<Mat> input_image, std::string name) {
    RETURN_VALUE_ON_NEQ(input_image->GetMatType(), N8UC4, nullptr);
    this->orig_dims = input_image->GetDims();
    // save input image mat for merging
    auto dims = input_image->GetDims();
    //dims[1] = 4;
    this->input_image = std::make_shared<Mat>(DEVICE_NAIVE, N8UC4, dims);
    auto status = Copy(input_image, this->input_image);
    RETURN_VALUE_ON_NEQ(status, TNN_OK, nullptr);

    auto target_dims = GetInputShape(name);
    auto input_height = input_image->GetHeight();
    auto input_width = input_image->GetWidth();
    if (target_dims.size() >= 4 &&
        (input_height != target_dims[2] || input_width != target_dims[3])) {
        auto target_mat = std::make_shared<TNN_NS::Mat>(input_image->GetDeviceType(),
                                                        input_image->GetMatType(), target_dims);
        auto status = Resize(input_image, target_mat, TNNInterpLinear);
        if (status == TNN_OK) {
            return target_mat;
        } else {
            LOGE("%s\n", status.description().c_str());
            return nullptr;
        }
    }
    return input_image;
}

Status MonoDepth::ProcessSDKOutput(std::shared_ptr<TNNSDKOutput> output_) {
    Status status = TNN_OK;
    auto option = dynamic_cast<MonoDepthOption *>(option_.get());
    RETURN_VALUE_ON_NEQ(!option, false, Status(TNNERR_PARAM_ERR, "TNNOption is invalid"));
    
    auto output = dynamic_cast<MonoDepthOutput *>(output_.get());
    RETURN_VALUE_ON_NEQ(!output, false, Status(TNNERR_PARAM_ERR, "TNNSDKOutput is invalid"));

    auto depth = output->GetMat();
    depth = GenerateDepthImage(depth);
    output->depth = ImageInfo(depth);

    return status;
}

std::shared_ptr<Mat> MonoDepth::GenerateDepthImage(std::shared_ptr<Mat> alpha) {
    RETURN_VALUE_ON_NEQ(alpha->GetChannel(), 1, nullptr);
    auto alpha_image_dims = alpha->GetDims();
    alpha_image_dims[1]   = 4;
    auto alpha_image = std::make_shared<Mat>(alpha->GetDeviceType(), N8UC4, alpha_image_dims);
    auto alpha_data = static_cast<float *>(alpha->GetData());
    auto alpha_image_data = static_cast<uint8_t *>(alpha_image->GetData());
    
    auto clip = [](float v){
        return (std::min)(v>0.0?v:0.0, 1.0);
    };
    
    auto alpha_dims = alpha->GetDims();
    auto hw = alpha_dims[2] * alpha_dims[3];
    for(int i=0; i<hw; ++i) {
        float val = clip(alpha_data[i]) * 255;
        alpha_image_data[i*4 + 0] = val;
        alpha_image_data[i*4 + 1] = val;
        alpha_image_data[i*4 + 2] = val;
        alpha_image_data[i*4 + 3] = 0;
    }
    return alpha_image;
}

}
