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

#include "hair_segmentation.h"

namespace TNN_NS {

Status HairSegmentation::ConvertMat(std::shared_ptr<Mat> src, std::shared_ptr<Mat> dst) {
    if (DimsVectorUtils::Count(src->GetDims()) != DimsVectorUtils::Count(dst->GetDims()))
        return Status(TNNERR_PARAM_ERR, "src and dst mat have different dims!");
    if (src->GetMatType() == dst->GetMatType()) {
        return TNN_OK;
    } else {
        auto src_mat_type = src->GetMatType();
        auto dst_mat_type = dst->GetMatType();
        auto count = DimsVectorUtils::Count(src->GetDims());
        if (src_mat_type == NCHW_FLOAT && dst_mat_type == NGRAY) {
            auto src_ptr = static_cast<float *>(src->GetData());
            auto dst_ptr = static_cast<uint8_t *>(dst->GetData());
            for(int i=0; i<count; ++i) {
                dst_ptr[i] = static_cast<uint8_t>(src_ptr[i] * 255.0f);
            }
        } else if (src_mat_type == NGRAY && dst_mat_type == NCHW_FLOAT) {
            auto src_ptr = static_cast<uint8_t *>(src->GetData());
            auto dst_ptr = static_cast<float *>(dst->GetData());
            for(int i=0; i<count; ++i) {
                dst_ptr[i] = static_cast<float>(src_ptr[i] / 255.0f);
            }
        } else {
            return Status(TNNERR_INST_ERR, "unsupported mat type pair!");
        }
    }
    return TNN_OK;
}

Status HairSegmentation::Init(std::shared_ptr<TNNSDKOption> option_i) {
    Status status = TNN_OK;
    auto option = dynamic_cast<HairSegmentationOption *>(option_i.get());
    RETURN_VALUE_ON_NEQ(!option, false, Status(TNNERR_PARAM_ERR, "TNNSDKOption is invalid"));
    
    status = TNNSDKSample::Init(option_i);
    RETURN_ON_NEQ(status, TNN_OK);
    
    auto input_dims = GetInputShape();
    option->input_height = input_dims[2];
    option->input_width  = input_dims[3];
    
    return status;
}

MatConvertParam HairSegmentation::GetConvertParamForInput(std::string tag) {
    MatConvertParam input_convert_param;
    input_convert_param.scale = {1.0 / (255 * 0.229), 1.0 / (255 * 0.224), 1.0 / (255 * 0.225), 0.0};
    input_convert_param.bias  = {-0.485 / 0.229,     -0.456 / 0.224,       -0.406 / 0.225,      0.0};
    return input_convert_param;
}

std::shared_ptr<TNNSDKOutput> HairSegmentation::CreateSDKOutput() {
    return std::make_shared<HairSegmentationOutput>();
}

std::shared_ptr<Mat> HairSegmentation::ProcessSDKInputMat(std::shared_ptr<Mat> input_image, std::string name) {
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

std::shared_ptr<Mat> HairSegmentation::MergeImage(std::shared_ptr<Mat> alpha, RGBA color) {
    auto merged_image = std::make_shared<Mat>(alpha->GetDeviceType(), N8UC4, orig_dims);
    auto alpha_data = static_cast<float *>(alpha->GetData());
    auto image_data = static_cast<uint8_t *>(input_image->GetData());
    auto merged_image_data = static_cast<uint8_t *>(merged_image->GetData());

    auto hw = orig_dims[2] * orig_dims[3];
    auto channel = orig_dims[1];
    for(int s=0; s<hw; ++s) {
        float hair_conf = alpha_data[s];
        //float bg_conf   = 1 - hair_conf;
        const float bg_conf = 1.0f;
        const float merge_weight = color.a/255.0f;
        float c0 = bg_conf * image_data[s*channel + 0] + merge_weight * hair_conf * color.r;
        float c1 = bg_conf * image_data[s*channel + 1] + merge_weight * hair_conf * color.g;
        float c2 = bg_conf * image_data[s*channel + 2] + merge_weight * hair_conf * color.b;
        float c3 = 255;

        merged_image_data[s*4 + 0] = static_cast<uint8_t>(std::min(255.0f, std::max(0.0f, c0)));
        merged_image_data[s*4 + 1] = static_cast<uint8_t>(std::min(255.0f, std::max(0.0f, c1)));
        merged_image_data[s*4 + 2] = static_cast<uint8_t>(std::min(255.0f, std::max(0.0f, c2)));
        merged_image_data[s*4 + 3] = static_cast<uint8_t>(std::min(255.0f, std::max(0.0f, c3)));
    }
    return merged_image;
}

Status HairSegmentation::ProcessSDKOutput(std::shared_ptr<TNNSDKOutput> output_) {
    Status status = TNN_OK;
    auto option = dynamic_cast<HairSegmentationOption *>(option_.get());
    RETURN_VALUE_ON_NEQ(!option, false, Status(TNNERR_PARAM_ERR, "TNNOption is invalid"));
    
    auto output = dynamic_cast<HairSegmentationOutput *>(output_.get());
    RETURN_VALUE_ON_NEQ(!output, false, Status(TNNERR_PARAM_ERR, "TNNSDKOutput is invalid"));

    auto bg = output->GetMat("background");
    auto fg = output->GetMat("foreground");
    auto alpha = ProcessAlpha(fg, option->mode);
    auto merged_image = MergeImage(alpha, this->hair_color_);
    alpha = GenerateAlphaImage(alpha);

    output->hair_mask = ImageInfo(alpha);
    output->merged_image = ImageInfo(merged_image);

    return status;
}

std::shared_ptr<Mat> HairSegmentation::ProcessAlpha(std::shared_ptr<Mat> alpha, int mode) {
    std::shared_ptr<Mat> rtn = nullptr;
    auto alpha_dims = alpha->GetDims();
    if (mode == 0 || mode == 1) {
        auto resized_dims = orig_dims;
        resized_dims[0] = alpha_dims[0];
        resized_dims[1] = alpha_dims[1];
        // resize
        rtn = std::make_shared<Mat>(alpha->GetDeviceType(), alpha->GetMatType(), resized_dims);
        auto status = ResizeFloatMat(alpha, rtn, TNNInterpLinear);
        RETURN_VALUE_ON_NEQ(status, TNN_OK, nullptr);
        if (mode == 0) {
            auto data  = static_cast<float *>(rtn->GetData());
            auto count = DimsVectorUtils::Count(rtn->GetDims());
            // clip
            auto clip = [](float& val) {
                val = val > 0.5? 1 :(val < 0.5? 0:val);
            };
            std::for_each(data, data+count, clip);
        }
    } else if (mode == 2) {
        //downsample to 64*64
        auto resized_dims = alpha->GetDims();
        resized_dims[2] = 64;
        resized_dims[3] = 64;
        auto alpha_small = std::make_shared<Mat>(alpha->GetDeviceType(), alpha->GetMatType(), resized_dims);
        auto status = ResizeFloatMat(alpha, alpha_small, TNNInterpLinear);
        RETURN_VALUE_ON_NEQ(status, TNN_OK, nullptr);
        // step.1: Gaussian Blur
        auto alpha_data = static_cast<float *>(alpha_small->GetData());
        auto expander = [](float& val){
            val = val * 255.0;
        };
        std::for_each(alpha_data, alpha_data+DimsVectorUtils::Count(alpha_small->GetDims()), expander);
        //TODO:: gaussian blur on alpha_data
        
        // step.2: resize & clip
        rtn = std::make_shared<Mat>(alpha->GetDeviceType(), alpha->GetMatType(), orig_dims);
        status = ResizeFloatMat(alpha_small, rtn, TNNInterpLinear);
        RETURN_VALUE_ON_NEQ(status, TNN_OK, nullptr);
        alpha_data = static_cast<float *>(rtn->GetData());
        auto step2_processor = [](float& val) {
            double x = std::exp(2*val - 1);
            x = x / (x + 1.0);
            x = std::min(1.0, std::max(0.0, (x - 0.5) * 1.5 + 0.4));
            x = x < 0.45? 0 : (x>0.9 ? 1.0 : x);
            x = std::min(1.0, std::max(0.0, (x - 0.5) * 1.5 + 0.5));
            x = std::min(1.0, std::max(0.0, (x - 0.4) / 0.4));
            x = 3*x*x - 2*x*x*x;
            val = x;
        };
        std::for_each(alpha_data, alpha_data+DimsVectorUtils::Count(rtn->GetDims()), step2_processor);
    } else{
        LOGE("invalid alpha process mode!\n");
    }
    return rtn;
}

std::shared_ptr<Mat> HairSegmentation::GenerateAlphaImage(std::shared_ptr<Mat> alpha) {
    RETURN_VALUE_ON_NEQ(alpha->GetChannel(), 1, nullptr);
    auto alpha_image_dims = alpha->GetDims();
    alpha_image_dims[1]   = 4;
    auto alpha_image = std::make_shared<Mat>(alpha->GetDeviceType(), N8UC4, alpha_image_dims);
    auto alpha_data = static_cast<float *>(alpha->GetData());
    auto alpha_image_data = static_cast<uint8_t *>(alpha_image->GetData());
    
    auto alpha_dims = alpha->GetDims();
    auto hw = alpha_dims[2] * alpha_dims[3];
    for(int i=0; i<hw; ++i) {
        float val = static_cast<uint8_t>(std::min(255.0, std::max(0.0, alpha_data[i]*255.0)));
        alpha_image_data[i*4 + 0] = val;
        alpha_image_data[i*4 + 1] = val;
        alpha_image_data[i*4 + 2] = val;
        alpha_image_data[i*4 + 3] = 0;
    }
    return alpha_image;
}

/*
 Resize a NCHW_FLOAT mat
 allocate buffer N8UC4 mat to perform resize
 */
Status HairSegmentation::ResizeFloatMat(std::shared_ptr<Mat> input_mat, std::shared_ptr<Mat> output_mat, TNNInterpType type) {
    Status status = TNN_OK;
    RETURN_VALUE_ON_NEQ(input_mat->GetMatType(), NCHW_FLOAT, Status(TNNERR_PARAM_ERR, "invalid input mat, only NCHW_FLAOT supported!"));

    auto input_dims = input_mat->GetDims();
    auto buffer_mat_type = INVALID;
    if (input_dims[1] == 4)
        buffer_mat_type = N8UC4;
    else if (input_dims[1] == 3)
        buffer_mat_type = N8UC3;
    else if (input_dims[1] == 1)
        buffer_mat_type = NGRAY;

    // allocate temp buffer mat
    auto input_image_mat = std::make_shared<Mat>(input_mat->GetDeviceType(), buffer_mat_type, input_mat->GetDims());
    auto output_image_mat = std::make_shared<Mat>(output_mat->GetDeviceType(), buffer_mat_type, output_mat->GetDims());

    // copy input mat
    status = ConvertMat(input_mat, input_image_mat);
    RETURN_ON_NEQ(status, TNN_OK);

    // resize
    status = Resize(input_image_mat, output_image_mat, type);
    RETURN_ON_NEQ(status, TNN_OK);

    // copy back
    status = ConvertMat(output_image_mat, output_mat);
    RETURN_ON_NEQ(status, TNN_OK);

    return status;
}

}
