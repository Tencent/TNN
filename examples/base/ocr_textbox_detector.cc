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

#include "ocr_textbox_detector.h"

#if HAS_OPENCV

#include "clipper.h"
#include "opencv2/core/mat.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc.hpp"

#include <iostream>
#include <fstream>
#include <algorithm>
#include <string>
#include <unordered_set>
#include <cmath>


namespace TNN_NS {

static ScaleParam getScaleParam(cv::Mat &src, const int targetSize) {
    int srcWidth, srcHeight, dstWidth, dstHeight;
    srcWidth = dstWidth = src.cols;
    srcHeight = dstHeight = src.rows;

    float ratio = 1.f;
    if (srcWidth > srcHeight) {
        ratio = float(targetSize) / float(srcWidth);
    } else {
        ratio = float(targetSize) / float(srcHeight);
    }
    dstWidth = int(float(srcWidth) * ratio);
    dstHeight = int(float(srcHeight) * ratio);
    if (dstWidth % 32 != 0) {
        dstWidth = (dstWidth / 32) * 32;
        dstWidth = (std::max)(dstWidth, 32);
    }
    if (dstHeight % 32 != 0) {
        dstHeight = (dstHeight / 32) * 32;
        dstHeight = (std::max)(dstHeight, 32);
    }
    float ratioWidth = (float) dstWidth / (float) srcWidth;
    float ratioHeight = (float) dstHeight / (float) srcHeight;
    return {srcWidth, srcHeight, dstWidth, dstHeight, ratioWidth, ratioHeight};
}

static std::vector<cv::Point> getMinBoxes(const std::vector<cv::Point> &inVec, float &minSideLen, float &allEdgeSize) {
    std::vector<cv::Point> minBoxVec;
    cv::RotatedRect textRect = cv::minAreaRect(inVec);
    cv::Mat boxPoints2f;
    cv::boxPoints(textRect, boxPoints2f);

    float *p1 = reinterpret_cast<float *>(boxPoints2f.data);
    std::vector<cv::Point> tmpVec;
    for (int i = 0; i < 4; ++i, p1 += 2) {
        tmpVec.emplace_back(int(p1[0]), int(p1[1]));
    }

    const auto& cvPointCompare= [](const cv::Point &a, const cv::Point &b) {
        return a.x < b.x;
    };
    std::sort(tmpVec.begin(), tmpVec.end(), cvPointCompare);

    int index1, index2, index3, index4;
    if (tmpVec[1].y > tmpVec[0].y) {
        index1 = 0;
        index4 = 1;
    } else {
        index1 = 1;
        index4 = 0;
    }

    if (tmpVec[3].y > tmpVec[2].y) {
        index2 = 2;
        index3 = 3;
    } else {
        index2 = 3;
        index3 = 2;
    }

    minBoxVec.push_back(tmpVec[index1]);
    minBoxVec.push_back(tmpVec[index2]);
    minBoxVec.push_back(tmpVec[index3]);
    minBoxVec.push_back(tmpVec[index4]);

    minSideLen = (std::min)(textRect.size.width, textRect.size.height);
    allEdgeSize = 2.f * (textRect.size.width + textRect.size.height);

    return minBoxVec;
}

static float boxScoreFast(const cv::Mat &inMat, const std::vector<cv::Point> &inBox) {
    std::vector<cv::Point> box = inBox;
    int width = inMat.cols;
    int height = inMat.rows;
    int maxX = -INFINITY, minX = INFINITY, maxY = -INFINITY, minY = INFINITY;
    for (int i = 0; i < box.size(); ++i) {
        if (maxX < box[i].x)
            maxX = box[i].x;
        if (minX > box[i].x)
            minX = box[i].x;
        if (maxY < box[i].y)
            maxY = box[i].y;
        if (minY > box[i].y)
            minY = box[i].y;
    }
    maxX = std::min(std::max(maxX, 0), width - 1);
    minX = std::max(std::min(minX, width - 1), 0);
    maxY = std::min(std::max(maxY, 0), height - 1);
    minY = std::max(std::min(minY, height - 1), 0);

    for (int i = 0; i < box.size(); ++i) {
        box[i].x = box[i].x - minX;
        box[i].y = box[i].y - minY;
    }

    std::vector<std::vector<cv::Point>> maskBox;
    maskBox.push_back(box);
    cv::Mat maskMat(maxY - minY + 1, maxX - minX + 1, CV_8UC1, cv::Scalar(0, 0, 0));
    cv::fillPoly(maskMat, maskBox, cv::Scalar(1, 1, 1), 1);

    return cv::mean(inMat(cv::Rect(cv::Point(minX, minY), cv::Point(maxX + 1, maxY + 1))).clone(),
                    maskMat).val[0];
}

static std::vector<cv::Point> unClip(const std::vector<cv::Point> &inBox, float perimeter, float unClipRatio) {
    std::vector<cv::Point> outBox;

    ClipperLib::Path poly;

    for (int i = 0; i < inBox.size(); ++i) {
        poly.push_back(ClipperLib::IntPoint(inBox[i].x, inBox[i].y));
    }

    double distance = unClipRatio * ClipperLib::Area(poly) / (double) perimeter;

    ClipperLib::ClipperOffset clipperOffset;
    clipperOffset.AddPath(poly, ClipperLib::JoinType::jtRound, ClipperLib::EndType::etClosedPolygon);
    ClipperLib::Paths polys;
    polys.push_back(poly);
    clipperOffset.Execute(polys, distance);
    
    outBox.clear();
    std::vector<cv::Point> rsVec;
    for (int i = 0; i < polys.size(); ++i) {
        ClipperLib::Path tmpPoly = polys[i];
        for (int j = 0; j < tmpPoly.size(); ++j) {
            outBox.emplace_back(tmpPoly[j].X, tmpPoly[j].Y);
        }
    }

    return outBox;
}

OCRTextboxDetectorOutput::~OCRTextboxDetectorOutput() {}

Status OCRTextboxDetector::Init(std::shared_ptr<TNNSDKOption> option_i) {
    Status status = TNN_OK;

    auto option = dynamic_cast<OCRTextboxDetectorOption *>(option_i.get());
    RETURN_VALUE_ON_NEQ(!option, false, Status(TNNERR_PARAM_ERR, "TNNSDKOption is invalid"));

    int max_size = max_size_ + 2*padding_;
    // input size must be multiple of 32
    if (max_size % 32 != 0) {
        max_size = (max_size + 31 ) / 32 * 32;
    }
    option->input_shapes.insert( {"input0", DimsVector({1, 3, max_size, max_size})} );
    status = TNNSDKSample::Init(option_i);
    RETURN_ON_NEQ(status, TNN_OK);

    padding_ = option->padding;
    box_score_thresh_ = option->box_score_threshold;
    scale_down_ratio_ = option->scale_down_ratio;
    return status;
}

MatConvertParam OCRTextboxDetector::GetConvertParamForInput(std::string name) {
    MatConvertParam input_convert_param;
    input_convert_param.scale = {1.0 / (0.229 * 255), 1.0 / (0.224 * 255), 1.0 / (0.225 * 255), 0.0};
    input_convert_param.bias  = {-0.485 / 0.229,      -0.456 / 0.224,      -0.406 / 0.225,      0.0};
    // model requires RGB input
    input_convert_param.reverse_channel = false;

    return input_convert_param;
}

std::shared_ptr<Mat> OCRTextboxDetector::ProcessSDKInputMat(std::shared_ptr<Mat> input_mat,
                                                                   std::string name) {
    Status status = TNN_OK;

    auto scale_down_dims = input_mat->GetDims();
    scale_down_dims[2] = static_cast<int>(scale_down_dims[2] * scale_down_ratio_);
    scale_down_dims[3] = static_cast<int>(scale_down_dims[3] * scale_down_ratio_);

    // 0) copy if necessary
    bool need_copy = false;
    DeviceType origin_dev = input_mat->GetDeviceType();
    if (input_mat->GetDeviceType() != DEVICE_ARM && device_type_ == DEVICE_ARM) {
        need_copy = true;
        auto input_arm_mat = std::make_shared<Mat>(DEVICE_ARM, input_mat->GetMatType(),
                                                   input_mat->GetDims());
        status = Copy(input_mat, input_arm_mat);
        RETURN_VALUE_ON_NEQ(status, TNN_OK, nullptr);
        // sacle down
        auto scale_down_mat = std::make_shared<Mat>(DEVICE_ARM, input_arm_mat->GetMatType(),
                                                    scale_down_dims);
        status = Resize(input_arm_mat, scale_down_mat, TNNInterpLinear);
        RETURN_VALUE_ON_NEQ(status, TNN_OK, nullptr);
        //input_mat = input_arm_mat;
        input_mat = scale_down_mat;
    } else if (device_type_ == DEVICE_ARM) {
        // sacle down
        auto scale_down_mat = std::make_shared<Mat>(DEVICE_ARM, input_mat->GetMatType(),
                                                    scale_down_dims);
        status = Resize(input_mat, scale_down_mat, TNNInterpLinear);
        RETURN_VALUE_ON_NEQ(status, TNN_OK, nullptr);
        input_mat = scale_down_mat;
    } else {
        auto scale_down_mat = std::make_shared<Mat>(DEVICE_NAIVE, input_mat->GetMatType(),
                                                    scale_down_dims);
        status = Resize(input_mat, scale_down_mat, TNNInterpLinear);
        input_mat = scale_down_mat;
    }
    
    // 1) TNN::Mat to opencv Mat
    int img_height = input_mat->GetHeight();
    int img_width  = input_mat->GetWidth();
    void *pixel = input_mat->GetData();
    cv::Mat cv_src(img_height, img_width, CV_8UC4, pixel);
    // 2) padding + resize
    int max_side_size = std::min(max_size_,
                    std::max(img_width, img_height));
    int resize_size = max_side_size + 2 * padding_;
    if (padding_ > 0) {
        cv::Scalar scalar = {255, 255, 255};
        cv::copyMakeBorder(cv_src, padded_input_, padding_, padding_, padding_, padding_,
                           cv::BORDER_ISOLATED, scalar);
        cv_src = padded_input_;
    } else {
        // TODO: hold data 'pixel' to avoid copy
        padded_input_ = cv_src.clone();
    }
    
    scale_ = getScaleParam(cv_src, resize_size);
    cv::Mat resized_src;
    cv::resize(cv_src, resized_src, cv::Size(scale_.dstWidth, scale_.dstHeight));
    // 3) cv::Mat to TNN::Mat
    int input_height = resized_src.rows;
    int input_width  = resized_src.cols;
    auto input_shape = input_mat->GetDims();
    input_shape[1] = 4;
    input_shape[2] = input_height;
    input_shape[3] = input_width;
    input_height_ = input_height;
    input_width_  = input_width;
    
    std::shared_ptr<Mat> result_mat = nullptr;
    if (need_copy) {
        auto input_arm_mat = std::make_shared<Mat>(DEVICE_ARM, input_mat->GetMatType(),
                                                   input_shape, resized_src.data);
        result_mat = std::make_shared<Mat>(origin_dev, input_mat->GetMatType(), input_shape);
        status = Copy(input_arm_mat, result_mat);
        RETURN_VALUE_ON_NEQ(status, TNN_OK, nullptr);
    } else {
        result_mat = std::make_shared<Mat>(input_mat->GetDeviceType(), N8UC4, input_shape);
        memcpy(result_mat->GetData(), resized_src.data, sizeof(uint8_t)*DimsVectorUtils::Count(input_shape));
    }
    // reshape
    InputShapesMap input_shape_map;
    const auto input_name = GetInputNames()[0];
    input_shape[1] = 3;
    input_shape_map.insert({input_name, input_shape});
    status = instance_->Reshape(input_shape_map);
    if (status != TNN_OK) {
        LOGE("instance Reshape failed in ocr textbox detector\n");
        return nullptr;
    }
    
    return result_mat;
}

std::shared_ptr<TNNSDKOutput> OCRTextboxDetector::CreateSDKOutput() {
    return std::make_shared<OCRTextboxDetectorOutput>();
}

Status OCRTextboxDetector::ProcessSDKOutput(std::shared_ptr<TNNSDKOutput> output_) {
    Status status = TNN_OK;
    auto output_mat = output_->GetMat("out1");
    const auto output_count = DimsVectorUtils::Count(output_mat->GetDims());
    auto output = dynamic_cast<OCRTextboxDetectorOutput *>(output_.get());
    // mask for boxThresh
    cv::Mat fMapMat(output_mat->GetHeight(), output_mat->GetWidth(), CV_32FC1);
    memcpy(fMapMat.data, output_mat->GetData(), output_count * sizeof(float));
    
    std::vector<TextBox> text_boxes;

    cv::Mat norfMapMat = fMapMat > box_thresh_;
    // find rs boxes
    const float minArea = 3;

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(norfMapMat, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);
    for (int i = 0; i < contours.size(); ++i) {
        float minSideLen, perimeter;
        std::vector<cv::Point> minBox = getMinBoxes(contours[i], minSideLen, perimeter);
        if (minSideLen < minArea)
            continue;
        float score = boxScoreFast(fMapMat, contours[i]);
        if (score < box_score_thresh_)
            continue;

        std::vector<cv::Point> clipBox = unClip(minBox, perimeter, un_clip_ratio_);
        std::vector<cv::Point> clipMinBox = getMinBoxes(clipBox, minSideLen, perimeter);

        if (minSideLen < minArea + 2)
            continue;

        for (int j = 0; j < clipMinBox.size(); ++j) {
            clipMinBox[j].x = (clipMinBox[j].x / scale_.ratioWidth);
            clipMinBox[j].x = (std::min)((std::max)(clipMinBox[j].x, 0), scale_.srcWidth);

            clipMinBox[j].y = (clipMinBox[j].y / scale_.ratioHeight);
            clipMinBox[j].y = (std::min)((std::max)(clipMinBox[j].y, 0), scale_.srcHeight);
        }
        std::vector<cv::Point> box_to_input(clipMinBox.size());
        for (int j = 0; j < clipMinBox.size(); ++j) {
            box_to_input[j].x = static_cast<int>((clipMinBox[j].x - padding_) / scale_down_ratio_);
            box_to_input[j].y = static_cast<int>((clipMinBox[j].y - padding_) / scale_down_ratio_);
        }
        text_boxes.emplace_back(TextBox{clipMinBox, box_to_input, score,
                                static_cast<int>((scale_.srcWidth - 2*padding_) / scale_down_ratio_),
                                static_cast<int>((scale_.srcHeight - 2*padding_) / scale_down_ratio_)});
    }
    reverse(text_boxes.begin(), text_boxes.end());
    output->text_boxes = text_boxes;
    
    return status;
}

OCRTextboxDetector::~OCRTextboxDetector() {}

}

#endif // HAS_OPENCV
