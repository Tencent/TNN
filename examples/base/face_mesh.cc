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

#include "face_mesh.h"
#include <cmath>
#include <fstream>
#include <cstring>
#include <tuple>

namespace TNN_NS {

Status Facemesh::Init(std::shared_ptr<TNNSDKOption> option_i) {
    Status status = TNN_OK;
    auto option = dynamic_cast<FacemeshOption *>(option_i.get());
    RETURN_VALUE_ON_NEQ(!option, false,
                        Status(TNNERR_PARAM_ERR, "TNNSDKOption is invalid"));
    
    status = TNNSDKSample::Init(option_i);
    RETURN_ON_NEQ(status, TNN_OK);
    
    auto input_dims = GetInputShape();
    option->input_height = input_dims[2];
    option->input_width  = input_dims[3];

    return status;
}

MatConvertParam Facemesh::GetConvertParamForInput(std::string tag) {
    MatConvertParam input_convert_param;
    //TODO: find proper preprocess parameters: do we need channel_swapping?
    //devandong: from face_landmark_cpu.pbtxt:TfLiteConverterCalculator
    input_convert_param.scale = {2.0 / 255.0, 2.0 / 255.0, 2.0 / 255.0, 0.0};
    input_convert_param.bias  = {-1.0, -1.0, -1.0, 0.0};
    return input_convert_param;
}

std::shared_ptr<TNNSDKOutput> Facemesh::CreateSDKOutput() {
    return std::make_shared<FacemeshOutput>();
}

std::shared_ptr<Mat> Facemesh::ProcessSDKInputMat(std::shared_ptr<Mat> input_mat,
                                                                   std::string name) {
    auto target_dims = GetInputShape(name);
    auto input_height = input_mat->GetHeight();
    auto input_width = input_mat->GetWidth();
    if (target_dims.size() >= 4 &&
        (input_height != target_dims[2] || input_width != target_dims[3])) {
        auto target_mat = std::make_shared<TNN_NS::Mat>(input_mat->GetDeviceType(),
                                                        input_mat->GetMatType(), target_dims);
        auto status = Resize(input_mat, target_mat, TNNInterpLinear);
        if (status == TNN_OK) {
            return target_mat;
        } else {
            LOGI("%s\n", status.description().c_str());
            return nullptr;
        }
    }
    return input_mat;
}

Status Facemesh::ProcessSDKOutput(std::shared_ptr<TNNSDKOutput> output_) {
    Status status = TNN_OK;
    auto option = dynamic_cast<FacemeshOption *>(option_.get());
    RETURN_VALUE_ON_NEQ(!option, false,
                           Status(TNNERR_PARAM_ERR, "TNNSDKOption is invalid"));
    auto output = dynamic_cast<FacemeshOutput *>(output_.get());
    RETURN_VALUE_ON_NEQ(!output, false,
    Status(TNNERR_PARAM_ERR, "TNNSDKOutput is invalid"));
    
    auto landmarkMat  = output->GetMat("conv2d_20");
    auto scoreMat     = output->GetMat("conv2d_30");
    
    std::vector<FacemeshInfo> landmarks;
    // dummy rect， indicating the crop is the same as the input
    //TODO: generate the rect from: 1) blazeface detection; 2) generated landmarks
    FaceRect rect{
        0,   //rotation
        1,   //width
        1,   //height,
        0.5, //x_center,
        0.5  //y_center
    };
    //decode landmarks
    GenerateLandmarks(landmarks, *(scoreMat.get()), *(landmarkMat.get()), option, rect);
    
    output->face_list = landmarks;
    return status;
}

/*
 devandong: implement post-processing according to mediapipe(face_landmark_cpu.pbtxt):
 1) get face_presence_score and compare it with threshold;
 2) normalize the landmarks to the input crop;
 3) normalize the landmarks to the original input image;
 */
void Facemesh::GenerateLandmarks(std::vector<FacemeshInfo> &detects, TNN_NS::Mat &score_mat, TNN_NS::Mat &landmark_mat, FacemeshOption* option, FaceRect& rect) {
    const int image_w = option->input_width;
    const int image_h = option->input_height;
    const float norm_z = option->norm_z;
    const float face_presence_threshold = option->face_presence_threshold;
    bool flip_horizontal = option->flip_horizontally;
    bool flip_vertical = option->flip_vertically;
    // check face presence score
    float face_presence_score = static_cast<float*>(score_mat.GetData())[0];
    if(face_presence_score < face_presence_threshold) {
        //No faces found
        LOGI("No faces found!\n");
        return;
    }
    // landmark decoding & normalization
    float* rawLandmarkData = static_cast<float*>(landmark_mat.GetData());
    
    FacemeshInfo info;
    info.image_width = image_w;
    info.image_height = image_h;
    
    for(int i=0; i < num_landmarks; ++i) {
        int offset = i * landmark_dimensions;
        
        //decode landmark
        float x = flip_horizontal ? image_w - rawLandmarkData[offset + 0] : rawLandmarkData[offset + 0];
        float y = flip_vertical ? image_h - rawLandmarkData[offset + 1] : rawLandmarkData[offset + 1];
        float z = rawLandmarkData[offset + 2];
        
        //normalization with the crop
        x = x / input_crop_width;
        y = y / input_crop_height;
        z = z / input_crop_width / norm_z;
        
        //transform the coords of the crop to coords of the image
        x = x - 0.5f;
        y = y - 0.5f;
        float angle = option->ignore_rotation ? 0 : rect.rotation;
        x = std::cos(angle) * x - std::sin(angle) * y;
        y = std::sin(angle) * x + std::cos(angle) * y;
        x = x * rect.width  + rect.x_center;
        y = y * rect.height + rect.y_center;
        z = z * rect.width;
        /*
         transform to the coords of the original input iamge, references:
         1) mediapipe/util/annnotation_renderer.cc
         2) mediapipe/calculators/util/annotation_overlay_calculator.cc
         3) mediapipe/util/render_data.ptoto
         4) mediapipe/graphs/face_mesh/subgraphs/face_renderer_cpu.pbtxt
         */
        x = round(x * image_w);
        y = round(y * image_h);
        
        info.key_points_3d.push_back(std::make_tuple(x, y, z));
        info.key_points.push_back(std::make_pair(x, y));
    }
    detects.push_back(std::move(info));
}

}
