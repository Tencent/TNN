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
// #include "ocr_angle_predictor.h"
// #include "ocr_text_recognizer.h"
#include "broadcast_text_matcher.h"
#include "ultra_face_detector.h"
#include "broadcast_driver.h"
#include "tnn/utils/dims_vector_utils.h"

#include "opencv2/core/mat.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc.hpp"

#include <vector>
//#include "math.h"

namespace TNN_NS {
// 从原图中获取旋转裁剪图像
static cv::Mat getRotateTextCropImage(const cv::Mat &src, std::vector<cv::Point> box) {
    cv::Mat image;
    src.copyTo(image);
    std::vector<cv::Point> points(box.size());

    int collectX[4] = {box[0].x, box[1].x, box[2].x, box[3].x};
    int collectY[4] = {box[0].y, box[1].y, box[2].y, box[3].y};
    // 外接矩形框
    int left = int(*std::min_element(collectX, collectX + 4));
    int right = int(*std::max_element(collectX, collectX + 4));
    int top = int(*std::min_element(collectY, collectY + 4));
    int bottom = int(*std::max_element(collectY, collectY + 4));

    cv::Mat imgCrop;
    image(cv::Rect(left, top, right - left, bottom - top)).copyTo(imgCrop);

    for (int i = 0; i < points.size(); i++) {
        points[i].x = box[i].x - left;
        points[i].y = box[i].y - top;
    }

    // 旋转框宽度
    int imgCropWidth = int(sqrt(pow(points[0].x - points[1].x, 2) +
                                pow(points[0].y - points[1].y, 2)));
    // 旋转框高度
    int imgCropHeight = int(sqrt(pow(points[0].x - points[3].x, 2) +
                                 pow(points[0].y - points[3].y, 2)));
    // （变换后）目标点坐标
    cv::Point2f ptsDst[4];
    ptsDst[0] = cv::Point2f(0., 0.);
    ptsDst[1] = cv::Point2f(imgCropWidth, 0.);
    ptsDst[2] = cv::Point2f(imgCropWidth, imgCropHeight);
    ptsDst[3] = cv::Point2f(0.f, imgCropHeight);
    // （变换前）点坐标
    cv::Point2f ptsSrc[4];
    ptsSrc[0] = cv::Point2f(points[0].x, points[0].y);
    ptsSrc[1] = cv::Point2f(points[1].x, points[1].y);
    ptsSrc[2] = cv::Point2f(points[2].x, points[2].y);
    ptsSrc[3] = cv::Point2f(points[3].x, points[3].y);
    // 计算透视变换矩阵
    cv::Mat M = cv::getPerspectiveTransform(ptsSrc, ptsDst);
    // 变换后图像
    cv::Mat partImg;
    cv::warpPerspective(imgCrop, partImg, M,
                        cv::Size(imgCropWidth, imgCropHeight),
                        cv::BORDER_REPLICATE);
    // 竖排文字转换
    if (float(partImg.rows) >= float(partImg.cols) * 1.5) {
        cv::Mat srcCopy = cv::Mat(partImg.rows, partImg.cols, partImg.depth());
        cv::transpose(partImg, srcCopy);
        cv::flip(srcCopy, srcCopy, 0);
        return srcCopy;
    } else {
        return partImg;
    }
}

// 从原图中根据文字检测框获得头像图
static void getHeroCropImage(const cv::Mat &src, std::vector<cv::Point> box, HeroSide& winnerCrop, HeroSide& loserCrop) {
    cv::Mat image;
    src.copyTo(image);
    int img_width = image.cols;
    int img_height = image.rows;
    
    int collectX[4] = {box[0].x, box[1].x, box[2].x, box[3].x};
    int collectY[4] = {box[0].y, box[1].y, box[2].y, box[3].y};
    
//    LOGI("text x: %i, %i, %i, %i\n", collectX[0], collectX[1], collectX[2], collectX[3]);
//    LOGI("text y: %i, %i, %i, %i\n", collectY[0], collectY[1], collectY[2], collectY[3]);
    // 外接矩形框
    int left = int(*std::min_element(collectX, collectX + 4));
    int right = int(*std::max_element(collectX, collectX + 4));
    int top = int(*std::min_element(collectY, collectY + 4));
    int bottom = int(*std::max_element(collectY, collectY + 4));
//    LOGI("text box: %i, %i, %i, %i\n", left, right, top, bottom);
    
    // winner / loser box
    int text_h = bottom - top;
    float ratio = 0.9f;
    float shift_ratio = 0.9f;
    int hero_h = int(text_h * (1 + ratio * 2));
    int hero_w = int(hero_h * 2.667);
    int shift_x = int(text_h * shift_ratio);
    int mid_x = (left + right) / 2;
    int hero_top = std::max(int(top - text_h * ratio), 0);
    int hero_bottom = std::min(int(bottom + text_h * ratio), img_height);
    int winner_left = std::max(mid_x - shift_x - hero_w, 0);
    int winner_right = std::max(mid_x - shift_x, 0);
    int loser_left = std::min(mid_x + shift_x, img_width);
    int loser_right = std::min(mid_x + shift_x + hero_w, img_width);
//    LOGI("top: %i, bottom: %i\n", hero_top, hero_bottom);
//    LOGI("winner: %i, %i | loser: %i, %i\n", winner_left, winner_right, loser_left, loser_right);
    
    cv::Mat winnerImage, loserImage;
    image(cv::Rect(winner_left, hero_top, winner_right - winner_left, hero_bottom - hero_top)).copyTo(winnerImage);
    image(cv::Rect(loser_left, hero_top, loser_right - loser_left, hero_bottom - hero_top)).copyTo(loserImage);
    winnerCrop = HeroSide{winner_left, hero_top, winner_right, hero_bottom, winnerImage};
    loserCrop = HeroSide{loser_left, hero_top, loser_right, hero_bottom, loserImage};

    return;
}


static std::vector<cv::Mat> getPartTextImages(cv::Mat &src, std::vector<TextBox> &textBoxes) {
    std::vector<cv::Mat> partImages;
    for (int i = 0; i < textBoxes.size(); ++i) {
        cv::Mat partImg = getRotateTextCropImage(src, textBoxes[i].box_points);
        partImages.emplace_back(partImg);
    }
    return partImages;
}

static std::vector<HeroSide> getPartHeroImages(cv::Mat &src, TextBox &textBox) {
    std::vector<HeroSide> partImages;
    HeroSide winnerCrop, loserCrop;
    getHeroCropImage(src, textBox.box_points, winnerCrop, loserCrop);
    partImages.emplace_back(winnerCrop);
    partImages.emplace_back(loserCrop);
    return partImages;
}


Status BroadcastDriver::Init(std::vector<std::shared_ptr<TNNSDKSample>> sdks) {
    if (sdks.size() < 3) {
        return Status(TNNERR_INST_ERR, "BroadcastDriver::Init has invalid sdks, its size < 3");
    }
    
    textbox_detector_ = sdks[0];
    text_matcher_ = sdks[1];
    hero_detector_ = sdks[2];
    return TNNSDKComposeSample::Init(sdks);
}

Status BroadcastDriver::MatToTNNMat(const cv::Mat& mat, std::shared_ptr<Mat>& tnn_mat, bool try_share_data) {
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

bool BroadcastDriver::hideTextBox() {
    return true;
}

Status BroadcastDriver::Predict(std::shared_ptr<TNNSDKInput> sdk_input,
                                std::shared_ptr<TNNSDKOutput> &sdk_output) {
    Status status = TNN_OK;
    if (!sdk_input || sdk_input->IsEmpty()) {
        status = Status(TNNERR_PARAM_ERR, "input image is empty ,please check!");
        LOGE("input image is empty, please check!\n");
        return status;
    }
    auto predictor_textbox_detector_async = textbox_detector_;
    auto predictor_text_matcher_async  = text_matcher_;
    auto predictor_hero_detector_async  = hero_detector_;
    
    auto predictor_textbox_detector_cast = dynamic_cast<OCRTextboxDetector *>(predictor_textbox_detector_async.get());
    auto predictor_text_matcher_cast = dynamic_cast<BroadcastTextMatcher *>(predictor_text_matcher_async.get());
    auto predictor_hero_detector_cast = dynamic_cast<UltraFaceDetector *>(predictor_hero_detector_async.get());
    
    const auto input_mat = sdk_input->GetMat();

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
            // LOGE("Error no text boxes found!");
            return status;
        }
    }

    cv::Mat pad_input = predictor_textbox_detector_cast->GetPaddedInput();
    std::vector<cv::Mat> part_images = getPartTextImages(pad_input, text_boxes);
    int padding = predictor_textbox_detector_cast->GetPadding();
    float scale_up_ratio = 1.0f / predictor_textbox_detector_cast->GetScaleDownRatio();

    auto dims = input_mat->GetDims();
    
    std::vector<TextBox> keep_text_boxes;
    std::vector<std::shared_ptr<TNNSDKOutput>> texts;
    std::vector<std::shared_ptr<TNNSDKOutput>> hero_boxes;
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
            predictor_text_matcher_cast->Predict(input, text);
            if (!text)
                continue;
            auto text_output = dynamic_cast<BroadcastTextMatcherOutput *>(text.get());
            if (!text_output || text_output->text_id == 0)
                continue;

            keep_text_boxes.push_back(text_boxes[i]);
            texts.push_back(text);

            std::vector<HeroSide> hero_sides = getPartHeroImages(pad_input, text_boxes[i]);
            
            // phase4: hero detect
            for (int j=0; j<2; ++j) {
                auto hero_side = hero_sides[j];
                // LOGI("%d offset: %d, %d\n", j, hero_side.x1, hero_side.y1);
                auto& cv_mat   = hero_side.image;
                // for debug
                // cv::cvtColor(cv_mat, cv_mat, cv::COLOR_RGB2BGR);
                // if (j == 0)
                //     cv::imwrite("left.png", cv_mat);
                // else
                //     cv::imwrite("right.png", cv_mat);
                dims[2] = cv_mat.rows;
                dims[3] = cv_mat.cols;
                // LOGI("%d hero input size: %d, %d, %d, %d\n", j,  dims[0], dims[1], dims[2], dims[3]);
                auto tnn_mat = std::make_shared<Mat>(input_mat->GetDeviceType(), input_mat->GetMatType(), dims, nullptr);
                status = MatToTNNMat(cv_mat, tnn_mat, true);
                RETURN_ON_NEQ(status, TNN_OK);
                auto input   = std::make_shared<TNNSDKInput>(tnn_mat);
                std::shared_ptr<TNNSDKOutput> hero_box;
                predictor_hero_detector_cast->Predict(input, hero_box);
                if (!hero_box)
                    continue;
                auto hero_output = dynamic_cast<UltraFaceDetectorOutput *>(hero_box.get());
                if (hero_output) {
                    for (auto& object: hero_output->object_list) {
                        // printf("%d class id %d\n", j, object.class_id);
                        // printf("object position before: %.2f, %.2f, %.2f, %.2f\n", object.x1, object.y1, object.x2, object.y2);
                        object = object.AddOffset((float)(hero_side.x1-padding), (float)(hero_side.y1-padding), scale_up_ratio, scale_up_ratio);
                        LOGE("---julis object position after: %.2f, %.2f, %.2f, %.2f\n", object.x1, object.y1, object.x2, object.y2);
                    }
                    hero_boxes.push_back(hero_box);
                }
            }
        }
    }


    {
        auto ocr_output = std::make_shared<BroadcastOutput>();
        for(int i=0; i<texts.size(); ++i) {
            const auto& text_out = texts[i];
            const auto& box = keep_text_boxes[i];
            auto text_output = dynamic_cast<BroadcastTextMatcherOutput *>(text_out.get());
            const auto& text = text_output->text;
            ocr_output->texts.push_back(text);

            for(const auto&p : box.box_points_input) {
                ocr_output->box.push_back({p.x, p.y});
            }
            // hero boxes
//            for (int j=0; j<2; ++j) {
//                const auto& hero = hero_boxes[2*i+j];
//                auto hero_output = dynamic_cast<UltraFaceDetectorOutput *>(hero.get());
//                const auto& hero_box = hero_output->object_list;
//                ocr_output->heros.insert(ocr_output->heros.end(), hero_box.begin(), hero_box.end());
//            }

            ocr_output->image_height = sdk_input->GetMat()->GetHeight();
            ocr_output->image_width  = sdk_input->GetMat()->GetWidth();
        }
        
        // hero boxes
        for (int j=0; j<hero_boxes.size(); ++j) {
            const auto& hero = hero_boxes[j];
            auto hero_output = dynamic_cast<UltraFaceDetectorOutput *>(hero.get());
            const auto& hero_box = hero_output->object_list;
            ocr_output->heros.insert(ocr_output->heros.end(), hero_box.begin(), hero_box.end());
        }
        // fill output
        sdk_output = ocr_output;
    }

    return TNN_OK;
}

}

#endif // HAS_OPENCV
