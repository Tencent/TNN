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

#if HAS_OPENCV

#include "ocr_textbox_detector.h"
#include "ocr_angle_predictor.h"
#include "ocr_text_recognizer.h"
#include "ocr_driver.h"
#include "tnn/utils/dims_vector_utils.h"

#include "opencv2/core/mat.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc.hpp"

#include <vector>

namespace TNN_NS {
static cv::Mat getRotateCropImage(const cv::Mat &src, std::vector<cv::Point>& box) {
    cv::Mat image;
    src.copyTo(image);
    std::vector<cv::Point>& points = box;

    int collectX[4] = {box[0].x, box[1].x, box[2].x, box[3].x};
    int collectY[4] = {box[0].y, box[1].y, box[2].y, box[3].y};
    int left = int(*std::min_element(collectX, collectX + 4));
    int right = int(*std::max_element(collectX, collectX + 4));
    int top = int(*std::min_element(collectY, collectY + 4));
    int bottom = int(*std::max_element(collectY, collectY + 4));

    cv::Mat imgCrop;
    image(cv::Rect(left, top, right - left, bottom - top)).copyTo(imgCrop);

    for (int i = 0; i < points.size(); i++) {
        points[i].x -= left;
        points[i].y -= top;
    }

    int imgCropWidth = int(sqrt(pow(points[0].x - points[1].x, 2) +
                                pow(points[0].y - points[1].y, 2)));
    int imgCropHeight = int(sqrt(pow(points[0].x - points[3].x, 2) +
                                 pow(points[0].y - points[3].y, 2)));

    cv::Point2f ptsDst[4];
    ptsDst[0] = cv::Point2f(0., 0.);
    ptsDst[1] = cv::Point2f(imgCropWidth, 0.);
    ptsDst[2] = cv::Point2f(imgCropWidth, imgCropHeight);
    ptsDst[3] = cv::Point2f(0.f, imgCropHeight);

    cv::Point2f ptsSrc[4];
    ptsSrc[0] = cv::Point2f(points[0].x, points[0].y);
    ptsSrc[1] = cv::Point2f(points[1].x, points[1].y);
    ptsSrc[2] = cv::Point2f(points[2].x, points[2].y);
    ptsSrc[3] = cv::Point2f(points[3].x, points[3].y);

    cv::Mat M = cv::getPerspectiveTransform(ptsSrc, ptsDst);

    cv::Mat partImg;
    cv::warpPerspective(imgCrop, partImg, M,
                        cv::Size(imgCropWidth, imgCropHeight),
                        cv::BORDER_REPLICATE);

    if (float(partImg.rows) >= float(partImg.cols) * 1.5) {
        cv::Mat srcCopy = cv::Mat(partImg.rows, partImg.cols, partImg.depth());
        cv::transpose(partImg, srcCopy);
        cv::flip(srcCopy, srcCopy, 0);
        return srcCopy;
    } else {
        return partImg;
    }
}

static std::vector<cv::Mat> getPartImages(cv::Mat &src, std::vector<TextBox> &textBoxes) {
    std::vector<cv::Mat> partImages;
    for (int i = 0; i < textBoxes.size(); ++i) {
        cv::Mat partImg = getRotateCropImage(src, textBoxes[i].box_points);
        partImages.emplace_back(partImg);
    }
    return partImages;
}

// TODO:  a more general rorate method
void matRotateClockwise180(cv::Mat& src) {
    cv::flip(src, src, 0);
    cv::flip(src, src, 1);
}

Status OCRDriver::Init(std::vector<std::shared_ptr<TNNSDKSample>> sdks) {
    if (sdks.size() < 3) {
        return Status(TNNERR_INST_ERR, "OCRDriver::Init has invalid sdks, its size < 3");
    }
    
    textbox_detector_ = sdks[0];
    angle_predictor_  = sdks[1];
    text_recognizer_  = sdks[2];
    return TNNSDKComposeSample::Init(sdks);
}

Status OCRDriver::MatToTNNMat(const cv::Mat& mat, std::shared_ptr<Mat>& tnn_mat, bool try_share_data) {
    const auto device = tnn_mat->GetDeviceType();
    Status status = TNN_OK;
    bool is_cpu_mat = device == DEVICE_ARM || device == DEVICE_NAIVE;
    bool can_share_data = is_cpu_mat && (tnn_mat->GetData());
    if (can_share_data && try_share_data) {
        tnn_mat = std::make_shared<Mat>(device, tnn_mat->GetMatType(),
                                        tnn_mat->GetDims(), mat.data);
    } else {
        // new memory
        tnn_mat = std::make_shared<Mat>(device, tnn_mat->GetMatType(),
                                        tnn_mat->GetDims());
        if (is_cpu_mat) {
            memcpy(tnn_mat->GetData(), mat.data, sizeof(uint8_t)*DimsVectorUtils::Count(tnn_mat->GetDims()));
        } else {
            auto tmp_arm_mat = std::make_shared<Mat>(DEVICE_ARM, tnn_mat->GetMatType(),
                                            tnn_mat->GetDims());
            memcpy(tmp_arm_mat->GetData(), mat.data, sizeof(uint8_t)*DimsVectorUtils::Count(tnn_mat->GetDims()));
            auto status = Copy(tmp_arm_mat, tnn_mat);
        }
    }
    return status;
}

bool OCRDriver::hideTextBox() {
    return true;
}

Status OCRDriver::Predict(std::shared_ptr<TNNSDKInput> sdk_input,
                                  std::shared_ptr<TNNSDKOutput> &sdk_output) {
    Status status = TNN_OK;
    
    if (!sdk_input || sdk_input->IsEmpty()) {
        status = Status(TNNERR_PARAM_ERR, "input image is empty ,please check!");
        LOGE("input image is empty ,please check!\n");
        return status;
    }
    auto predictor_textbox_detector_async = textbox_detector_;
    auto predictor_angle_predictor_async  = angle_predictor_;
    auto predictor_text_recognizer_async  = text_recognizer_;
    
    auto predictor_textbox_detector_cast = dynamic_cast<OCRTextboxDetector *>(predictor_textbox_detector_async.get());
    auto predictor_angle_predictor_cast = dynamic_cast<OCRAnglePredictor *>(predictor_angle_predictor_async.get());
    auto predictor_text_recognizer_cast = dynamic_cast<OCRTextRecognizer *>(predictor_text_recognizer_async.get());
    
    const auto input_mat = sdk_input->GetMat();
    cv::Mat origin_input(input_mat->GetHeight(), input_mat->GetWidth(), CV_8UC4, input_mat->GetData());

    std::vector<TextBox> text_boxes;
    std::shared_ptr<TNNSDKOutput> textbox_det;
    {
        // phase1: textbox detection
        status = predictor_textbox_detector_cast->Predict(sdk_input, textbox_det);
        if (textbox_det && dynamic_cast<OCRTextboxDetectorOutput *>(textbox_det.get())) {
            auto output = dynamic_cast<OCRTextboxDetectorOutput *>(textbox_det.get());
            text_boxes = output->text_boxes;
        }
        if(text_boxes.size() <= 0) {
            printf("Error no text boxes found!\n");
            return -1;
        }
    }
    std::vector<cv::Mat> part_images = getPartImages(predictor_textbox_detector_cast->GetPaddedInput(), text_boxes);
    auto dims = input_mat->GetDims();
    
    if (predictor_angle_predictor_cast->DoAngle()) {
        // phase2: angle prediction
        std::vector<std::shared_ptr<TNNSDKOutput>>angles;
        for(int i=0; i<part_images.size(); ++i) {
            // cv::Mat to TNN::Mat
            auto& cv_mat   = part_images[i];
            dims[2] = cv_mat.rows;
            dims[3] = cv_mat.cols;
            auto tnn_mat = std::make_shared<Mat>(input_mat->GetDeviceType(), input_mat->GetMatType(), dims, nullptr);
            status = MatToTNNMat(cv_mat, tnn_mat, true);
            RETURN_ON_NEQ(status, TNN_OK);
            auto input   = std::make_shared<TNNSDKInput>(tnn_mat);
            
            std::shared_ptr<TNNSDKOutput> angle;
            predictor_angle_predictor_cast->Predict(input, angle);
            angles.push_back(angle);
        }
        predictor_angle_predictor_cast->ProcessAngles(angles);
        for(int i=0; i<part_images.size(); ++i) {
            auto angle = dynamic_cast<OCRAnglePredictorOutput *>(angles[i].get());
            if(angle->index == 0) {
                matRotateClockwise180(part_images[i]);
            }
        }
    }
    std::vector<std::shared_ptr<TNNSDKOutput>> texts;
    {
        // phase3: text recognize
        for(int i=0; i<part_images.size(); ++i) {
            // cv::Mat to TNN::Mat
            auto& cv_mat   = part_images[i];
            dims[2] = cv_mat.rows;
            dims[3] = cv_mat.cols;
            auto tnn_mat = std::make_shared<Mat>(input_mat->GetDeviceType(), input_mat->GetMatType(), dims, nullptr);
            status = MatToTNNMat(cv_mat, tnn_mat, true);
            RETURN_ON_NEQ(status, TNN_OK);
            auto input   = std::make_shared<TNNSDKInput>(tnn_mat);
            
            std::shared_ptr<TNNSDKOutput> text;
            predictor_text_recognizer_cast->Predict(input, text);
            if (text && dynamic_cast<OCRTextRecognizerOutput *>(text.get())) {
                texts.push_back(text);
            }
        }
    }

    {
        auto ocr_output = std::make_shared<OCROutput>();
        for(int i=0; i<texts.size(); ++i) {
            const auto& o = texts[i];
            const auto& box = text_boxes[i];
            auto text_output = dynamic_cast<OCRTextRecognizerOutput *>(o.get());
            const auto& text = text_output->text;
            ocr_output->texts.push_back(text);

            for(const auto&p : box.box_points_input) {
                ocr_output->box.push_back({p.x, p.y});
            }

            ocr_output->image_height = sdk_input->GetMat()->GetHeight();
            ocr_output->image_width  = sdk_input->GetMat()->GetWidth();
        }
        // fill output
        sdk_output = ocr_output;
    }

    return TNN_OK;
}

}

#endif // HAS_OPENCV
