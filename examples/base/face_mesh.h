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

#ifndef TNN_EXAMPLES_BASE_FACE_MESH_H_
#define TNN_EXAMPLES_BASE_FACE_MESH_H_

#include "tnn_sdk_sample.h"
#include <algorithm>
#include <iostream>
#include <string>
#include <memory>
#include <vector>
#include <tuple>

namespace TNN_NS {

struct FacemeshInfo : TNN_NS::ObjectInfo {
};

struct FaceRect {
    // rotate angle in radian
    float rotation;
    //TODO: what's the size?
    float width;
    float height;
    float x_center;
    float y_center;
};

class FacemeshInput : public TNNSDKInput {
public:
    FacemeshInput(std::shared_ptr<Mat> mat = nullptr) : TNNSDKInput(mat) {};
    virtual ~FacemeshInput(){}
};

class FacemeshOutput : public TNNSDKOutput {
public:
    FacemeshOutput(std::shared_ptr<Mat> mat = nullptr) : TNNSDKOutput(mat) {};
    virtual ~FacemeshOutput() {};
    std::vector<FacemeshInfo> face_list;
};

class FacemeshOption : public TNNSDKOption {
public:
    FacemeshOption() {}
    virtual ~FacemeshOption() {}
    int input_width;
    int input_height;
    int num_thread = 1;
    //TODO: add specific arguments, e.g., thresholds
    float face_presence_threshold = 0.1;
    bool flip_vertically   = false;
    bool flip_horizontally = false;
    float norm_z = 1.0;
    bool ignore_rotation = false;
};

class Facemesh : public TNNSDKSample {
public:
    virtual ~Facemesh() {};
    
    virtual Status Init(std::shared_ptr<TNNSDKOption> option);
    virtual MatConvertParam GetConvertParamForInput(std::string name = "");
    virtual std::shared_ptr<TNNSDKOutput> CreateSDKOutput();
    virtual Status ProcessSDKOutput(std::shared_ptr<TNNSDKOutput> output);
    virtual std::shared_ptr<Mat> ProcessSDKInputMat(std::shared_ptr<Mat> mat, std::string name = kTNNSDKDefaultName);
    
private:
    void GenerateLandmarks(std::vector<FacemeshInfo> &landmarks, TNN_NS::Mat &score_mat, TNN_NS::Mat &landmark_mat, FacemeshOption* option, FaceRect& rect);
    // generate face rect accorfing to landmarks, used for the next frame
    void GenerateRectFromLandmarks(FaceRect& rect) {}
    // use blazeface to generate facerect
    void DetectFace(FaceRect& rect) {}
    
    int num_landmarks = 468;
    int landmark_dimensions = 3;
    int input_crop_height = 192;
    int input_crop_width  = 192;
};

}


#endif // TNN_EXAMPLES_BASE_FACE_MESH_H_
