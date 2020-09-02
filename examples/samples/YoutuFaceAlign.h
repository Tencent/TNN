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

#ifndef YoutuFaceAlign_hpp
#define YoutuFaceAlign_hpp

// using Eigen to do SVD
#define USE_EIGEN 1

#include "TNNSDKSample.h"
#include <algorithm>
#include <memory>
#include <iostream>
#include <string>
#include <vector>
#include <array>
#include "tnn/utils/mat_utils.h"
#include "tnn/utils/dims_vector_utils.h"

#if USE_EIGEN
#include <Eigen/Dense>
#endif

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
    
    std::shared_ptr<TNN_NS::Mat> BGRToGray(std::shared_ptr<TNN_NS::Mat>bgr_image);
    
    // methods used in pre-processing and post-processing
    std::shared_ptr<TNN_NS::Mat> BGR2Gray(std::shared_ptr<TNN_NS::Mat> bgr_mat);

    void Sigmoid(std::shared_ptr<TNN_NS::Mat>mat);
    
    std::vector<float> MatrixInverse(std::vector<float>& mat, int rows, int cols, bool transMat=true);
    
    void LandMarkWarpAffine(std::shared_ptr<TNN_NS::Mat>pts, std::vector<float>& M);
    
    void MatrixMean(const float *ptr, unsigned int rows, unsigned int cols, int axis, std::vector<float>& means);
    
    void MatrixStd(const float *ptr, unsigned int rows, unsigned int cols,int axis, std::vector<float>& stds);
    
    void DrawPointOnImage(int x, int y,TNN_NS::Mat* image, int radius, int shift, std::array<int, 3> color);
    
#if USE_EIGEN
    void MatrixSVD(const std::vector<float>m, int rows, int cols, std::vector<float>&u, std::vector<float>&vt);
#endif

    bool IsValidFace(float x1, float y1, float x2, float y2) {
        return (x2 - x1 >= min_face_size) && (y2-y1 >= min_face_size);
    }
    
    void SaveMat(std::shared_ptr<TNN_NS::Mat> mat) {
        void* data_ = mat->GetData();
        auto dims = mat->GetDims();
        int offset = 0;
        const std::string save_path("/Users/devandong/Desktop/0_phase1.txt");
        std::ofstream out(save_path);
        if(mat->GetMatType() == TNN_NS::N8UC4){
            unsigned char* data = static_cast<unsigned char*>(data_);
            for(int h=0; h<dims[2]; ++h){
                for(int w=0; w<dims[3]; ++w){
                    unsigned r, g, b, a;
                    r = static_cast<unsigned>(data[offset + 0]);
                    g = static_cast<unsigned>(data[offset + 1]);
                    b = static_cast<unsigned>(data[offset + 2]);
                    a = static_cast<unsigned>(data[offset + 3]);
                    out<<r<<","<<g<<","<<b<<","<<a<<std::endl;
                    offset += 4;
                }
                out<<std::endl;
            }
        } else if(mat->GetMatType() == TNN_NS::NCHW_FLOAT){
            float* data = static_cast<float*>(data_);
            for(int c=0; c<dims[1]; ++c){
                for(int h=0; h<dims[2]; ++h){
                    for(int w=0; w<dims[3]; ++w){
                        out<<data[offset]<<std::endl;
                        offset += 1;
                    }
                    out<<std::endl;
                }
            }
        }
        out.close();
    }
    void PrintMat(std::shared_ptr<TNN_NS::Mat> mat) {
        void* data_ = mat->GetData();
        auto dims = mat->GetDims();
        int offset = 0;
        if(mat->GetMatType() == TNN_NS::N8UC4){
            unsigned char* data = static_cast<unsigned char*>(data_);
            for(int h=0; h<dims[2]; ++h){
                for(int w=0; w<dims[3]; ++w){
                    unsigned r, g, b, a;
                    r = static_cast<unsigned>(data[offset + 0]);
                    g = static_cast<unsigned>(data[offset + 1]);
                    b = static_cast<unsigned>(data[offset + 2]);
                    a = static_cast<unsigned>(data[offset + 3]);
                    printf("%u,%u,%u,%u\n", r, g, b, a);
                    offset += 4;
                }
                printf("\n");
            }
        } else if(mat->GetMatType() == TNN_NS::NCHW_FLOAT){
            float* data = static_cast<float*>(data_);
            for(int c=0; c<dims[1]; ++c){
                for(int h=0; h<dims[2]; ++h){
                    for(int w=0; w<dims[3]; ++w){
                        printf("%f,\n", data[offset]);
                        offset += 1;
                    }
                }
            }
        }
    }

    void Normalize(std::shared_ptr<TNN_NS::Mat> mat, float* scale, float* bias){
        auto dims = mat->GetDims();
        void* data_ = mat->GetData();
        auto offset = 0;
        if(mat->GetMatType() == TNN_NS::NCHW_FLOAT && mat->GetDeviceType() == TNN_NS::DEVICE_ARM){
            float* data = static_cast<float*>(data_);
            for(int c=0; c<dims[1]; ++c){
                for(int h=0; h<dims[2]; ++h){
                    for(int w=0; w<dims[3]; ++w){
                        auto val = data[offset] * scale[c] + bias[c];
                        data[offset] = val;
                        offset += 1;
                    }
                }
            }
        }
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
    std::shared_ptr<tnn::Mat> pre_pts;
    // warpAffine trans matrix
    std::vector<float> M;
};

}

#endif /* YoutuFaceAlign_hpp */
