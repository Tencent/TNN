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

#include "broadcast_text_matcher.h"
#include <iostream>
#include <fstream>
#include <algorithm>
#include "math.h"
#include <string>
#include <unordered_set>
#include <opencv2/core/mat.hpp>
#include <charconv>


#if HAS_OPENCV
// #include <opencv2/highgui.hpp>


namespace TNN_NS {

    template<class ForwardIterator>
    inline size_t argmax(ForwardIterator first, ForwardIterator last) {
        return std::distance(first, std::max_element(first, last));
    }

    BroadcastTextMatcherOutput::~BroadcastTextMatcherOutput() {}


    Status BroadcastTextMatcher::Init(std::shared_ptr<TNNSDKOption> option_i) {
        Status status = TNN_OK;
        auto option = dynamic_cast<BroadcastTextMatcherOption *>(option_i.get());
        RETURN_VALUE_ON_NEQ(!option, false,
                            Status(TNNERR_PARAM_ERR, "TNNSDKOption is invalid"));

        // 非模型，不需要调用模型初始化
        // status = TNNSDKSample::Init(option_i);
        // RETURN_ON_NEQ(status, TNN_OK);

        // 1. 读取特征文件
        cv::FileStorage store(option->feature_path, cv::FileStorage::READ);
        // 2. 加载特征
        cv::FileNode node;
        node = store["id"];
        cv::read(node, ids_);
        node = store["text"];

//        cv::read(node, texts_);
        LOGE("---julis %s", typeid(node).name());
        node = store["word"];
        cv::read(node, words_);
        for (int i = 0; i < ids_.size(); i++) {
            char key[20];
            // std::vector<cv::KeyPoint> keypoint;
            cv::Mat descriptor;
            // keypoints
            // sprintf(key, "keypoints_%d", i);
            // node = store[key];
            // cv::read(node, keypoint);
            // keypoints_.push_back(keypoint);
            // descriptors
            sprintf(key, "descriptors_%d", i);
            node = store[key];
            cv::read(node, descriptor);
            descriptors_.push_back(descriptor);
        }
        store.release();
        // printf("[Init] load features\n");

        // 3. 特征检测&匹配器
        detector_ = cv::SIFT::create();
        matcher_ = cv::BFMatcher::create();
        // printf("[Init] create detector and matcher\n");

//     3. 参数设置
//     height_norm_ = option->height_norm;
        printf("[Init] set parameters\n");

        return status;
    }

    std::shared_ptr<Mat> BroadcastTextMatcher::ProcessSDKInputMat(std::shared_ptr<Mat> input_mat,
                                                                  std::string name) {
        return input_mat;
    }


    Status BroadcastTextMatcher::Predict(std::shared_ptr<TNNSDKInput> input, std::shared_ptr<TNNSDKOutput> &output) {
        Status status = TNN_OK;
        if (!input || input->IsEmpty()) {
            status = Status(TNNERR_PARAM_ERR, "input image is empty, please check!");
            LOGE("input image is empty, please check!\n");
            return status;
        }

        // step 1. set input mat
        auto input_mat = input->GetMat();
        // input_mat = ProcessSDKInputMat(input_mat);

        // step 2. process
        // 2.1 TNN::Mat to opencv Mat
        int img_height = input_mat->GetHeight();
        int img_width = input_mat->GetWidth();
        int img_channel = input_mat->GetChannel();
        // SetResolution(img_width, img_height);
        // printf("set resolution done. roi: %d, %d, %d, %d\n", roi_[0], roi_[1], roi_[2], roi_[3]);
        void *pixel = input_mat->GetData();
        cv::Mat img_src(img_height, img_width, CV_MAKETYPE(CV_8U, img_channel), pixel);
        // 2.2 get roi
        // cv::Mat img_roi = img_src(cv::Rect(roi_[0], roi_[1], roi_[2], roi_[3]));
        // cv::imshow("roi", img_roi);
        // cv::waitKey(0);
        // printf("get roi done.\n");
        // 2.3 resize roi
        // if (text_ratio_ != 1.0f) {
        //     cv::resize(img_roi, img_roi, cv::Size(), text_ratio_, text_ratio_, cv::INTER_CUBIC);
        // }
        // printf("resize done. ratio: %.3f\n", text_ratio_);
        // 2.4 convert color to gray
        int color_code;
        if (img_channel == 4)
            color_code = cv::COLOR_RGBA2GRAY;
        else
            color_code = cv::COLOR_RGB2GRAY;
        cv::cvtColor(img_src, img_src, color_code);
        // printf("convert to gray done.\n");
        // 2.5 feature match
//        int text_idx = TextMatch(img_src, text_thresh_, text_dist_ratio_);
        int text_idx = 1;
        // printf("text match done. text idx: %d, bbox: %d, %d, %d, %d\n", text_idx, bbox[0], bbox[1], bbox[2], bbox[3]);
        // step 3. get output
        output = CreateSDKOutput();
        auto output_ = dynamic_cast<BroadcastTextMatcherOutput *>(output.get());
        if (text_idx < 0) {
            output_->text_id = 0;
            return status;
        }
        output_->text_id = ids_[text_idx];
        output_->text = "aaaa";
        // printf("[TextMatch] text: %s\n", output_->text.c_str());
        // printf("get output done\n");
        return status;
    }

    std::shared_ptr<TNNSDKOutput> BroadcastTextMatcher::CreateSDKOutput() {
        return std::make_shared<BroadcastTextMatcherOutput>();
    }

    Status BroadcastTextMatcher::ProcessSDKOutput(std::shared_ptr<TNNSDKOutput> output_) {
        Status status = TNN_OK;
        return status;
    }


    BroadcastTextMatcher::~BroadcastTextMatcher() {}


    int BroadcastTextMatcher::TextMatch(cv::Mat image, float threshold, float dist_ratio) {
        // (1) detect feature
        std::vector<cv::KeyPoint> kpt;
        cv::Mat des;
        detector_->detectAndCompute(image, cv::Mat(), kpt, des);
        int min_keypoints = (int) ceil(threshold * 1.62f);   // pow(2, 0.7)
        if (kpt.size() < min_keypoints) {
            // printf("[TextMatch] no keypoints in roi\n");
            return -1;
        }
        // (2) match
        std::vector<std::vector<cv::DMatch>> matches_list;
        std::vector<int> num_matches;
        std::vector<int> norm_matches;
        for (int i = 0; i < descriptors_.size(); i++) {
            // printf("[TextMatch] id: %d, text: %s\n", i, texts_[i].c_str());
            std::vector<cv::DMatch> matches;
            int num;
            float norm;
            Match(matches, descriptors_[i], des, dist_ratio);
            matches_list.push_back(matches);
            num = matches.size();
            // printf("[TextMatch] num_match: %d\n", num);
            num_matches.push_back(num);
            norm = num * 1.0f / pow(words_[i], 0.7);
            // printf("[TextMatch] norm_match: %.3f\n", norm);
            norm_matches.push_back(norm);
        }
        // (3) optimal match
        size_t max_idx = argmax(norm_matches.begin(), norm_matches.end());
        // printf("[TextMatch] max idx: %d\n", max_idx);
        if (norm_matches[max_idx] <= threshold)
            return -1;
        if (max_idx == 13 && (num_matches[14] - num_matches[13] > 10))
            max_idx += 1;
        if (max_idx == 23 && (num_matches[23] - num_matches[0]) <= ceil(num_matches[0] / 3.0))
            max_idx = 0;
        return max_idx;
    }

    int BroadcastTextMatcher::Match(std::vector<cv::DMatch> &matches, cv::Mat des1, cv::Mat des2, float dist_ratio) {
//        std::vector<std::vector<cv::DMatch>> knn_matches;
//        matcher_->knnMatch(des1, des2, knn_matches, 2);
//        for (auto pair : knn_matches) {
//            if (pair[0].distance < dist_ratio * pair[1].distance)
//                matches.push_back(pair[0]);
//        }
        return 0;
    }

}

#endif // HAS_OPENCV
