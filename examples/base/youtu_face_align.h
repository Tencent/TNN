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

#ifndef TNN_EXAMPLES_BASE_YOUTU_FACE_ALIGN_H_
#define TNN_EXAMPLES_BASE_YOUTU_FACE_ALIGN_H_

#include "tnn_sdk_sample.h"

#include "stdlib.h"
#include <algorithm>
#include <cstring>
#include <memory>
#include <iostream>
#include <string>
#include <vector>
#include <array>
#include "tnn/utils/mat_utils.h"
#include "tnn/utils/dims_vector_utils.h"
#include "sample_timer.h"

namespace TNN_NS{

struct YoutuFaceAlignInfo : TNN_NS::ObjectInfo {
};

class YoutuFaceAlignInput : public TNNSDKInput {
public:
    YoutuFaceAlignInput(std::shared_ptr<Mat> mat = nullptr) : TNNSDKInput(mat) {};
    virtual ~YoutuFaceAlignInput(){}
};

class YoutuFaceAlignOutput : public TNNSDKOutput {
public:
    YoutuFaceAlignOutput(std::shared_ptr<Mat> mat = nullptr) : TNNSDKOutput(mat) {};
    virtual ~YoutuFaceAlignOutput() {};
    
    YoutuFaceAlignInfo face;
};

class YoutuFaceAlignOption : public TNNSDKOption {
public:
    YoutuFaceAlignOption() {}
    virtual ~YoutuFaceAlignOption() {}
    int input_width;
    int input_height;
    int num_thread = 1;
    float net_scale;
    float face_threshold = 0.75;
    int min_face_size = 20;
    //phase
    int phase = -1;
    // mean_pts file path
    std::string mean_pts_path;
};

class YoutuFaceAlign : public TNN_NS::TNNSDKSample {
public:
    virtual ~YoutuFaceAlign() {}
    
    virtual Status Init(std::shared_ptr<TNNSDKOption> option);
    virtual MatConvertParam GetConvertParamForInput(std::string name = "");
    virtual std::shared_ptr<TNNSDKOutput> CreateSDKOutput();
    virtual Status ProcessSDKOutput(std::shared_ptr<TNNSDKOutput> output);
    virtual Status Predict(std::shared_ptr<TNNSDKInput> input, std::shared_ptr<TNNSDKOutput> &output);

    bool SetFaceRegion(float x1, float y1, float x2, float y2) {
        bool isValidFace = IsValidFace(x1, y1, x2, y2);
        if(!isValidFace)
            return false;
        
        this->x1 = x1;
        this->y1 = y1;
        this->x2 = x2;
        this->y2 = y2;

        return true;
    }

    std::shared_ptr<TNN_NS::Mat> GetPrePts() {
        return this->pre_pts;
    }

    bool GetPrevFace() {
        return this->prev_face;
    }
    void SetPrevFace(bool b) {
        this->prev_face = b;
    }

    void SetPrePts(std::shared_ptr<Mat> p, bool deep_copy = false) {
        if(deep_copy) {
            this->pre_pts = std::make_shared<TNN_NS::Mat>(p->GetDeviceType(), p->GetMatType(), p->GetDims());
            auto count = TNN_NS::DimsVectorUtils::Count(p->GetDims());
            memcpy(this->pre_pts->GetData(), p->GetData(), sizeof(float)*count);
        } else {
            this->pre_pts = p;
        }
    }
    
private:
    //prep-rocessing methods
    std::shared_ptr<TNN_NS::Mat> WarpByRect(std::shared_ptr<TNN_NS::Mat> image, float x1, float y1, float x2, float y2, int net_width, float enlarge, std::vector<float>&M);
    
    std::shared_ptr<TNN_NS::Mat> AlignN(std::shared_ptr<TNN_NS::Mat> image, std::shared_ptr<TNN_NS::Mat> pre_pts, std::vector<float>mean, int net_h, int net_w, float net_scale, std::vector<float>&M);
    
    // methods used in pre-processing and post-processing
    std::shared_ptr<TNN_NS::Mat> BGRToGray(std::shared_ptr<TNN_NS::Mat> bgr_mat);
    
    std::vector<float> MatrixInverse2x3(std::vector<float>& mat, int rows, int cols, bool transMat=true);
    
    void LandMarkWarpAffine(std::shared_ptr<TNN_NS::Mat>pts, std::vector<float>& M);
    
    void MatrixMean(const float *ptr, unsigned int rows, unsigned int cols, int axis, std::vector<float>& means);
    
    void MatrixStd(const float *ptr, unsigned int rows, unsigned int cols,int axis, std::vector<float>& stds);
    
    void MatrixSVD2x2(const std::vector<float>a, int rows, int cols, std::vector<float>&u, std::vector<float>&vt);

    bool IsValidFace(float x1, float y1, float x2, float y2) {
        return (x2 - x1 >= min_face_size) && (y2-y1 >= min_face_size);
    }

private:
    // model phase
    int phase;
    // input shape
    int image_w;
    int image_h;
    // whether faces in the previous frame
    bool prev_face = false;
    // face region
    float x1, y1, x2, y2;
    // the minimum face size
    float min_face_size = 20;
    // the confident threshold
    float face_threshold = 0.5;
    // model configs
    float net_scale;
    std::vector<float> mean;
    // current pts data
    std::shared_ptr<TNN_NS::Mat> pre_pts;
    // warpAffine trans matrix
    std::vector<float> M;
};

}

#endif // TNN_EXAMPLES_BASE_YOUTU_FACE_ALIGN_H_
