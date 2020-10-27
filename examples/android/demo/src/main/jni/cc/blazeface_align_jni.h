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

#ifndef ANDROID_BLAZEFACE_ALIGN_JNI_H_
#define ANDROID_BLAZEFACE_ALIGN_JNI_H_

#include "blazeface_detector.h"
#include "youtu_face_align.h"
#include "face_detect_aligner.h"
#include <jni.h>
#include "helper_jni.h"
#include <android/bitmap.h>

#define TNN_BLAZEFACE_ALIGN(sig) Java_com_tencent_tnn_demo_FaceAlign_##sig
#ifdef __cplusplus
extern "C"{
#endif

std::shared_ptr<TNN_NS::BlazeFaceDetector> CreateBlazeFaceDetector(JNIEnv *env, jobject thiz, jstring modelPath,
                           jint width, jint height, jint topk,
                           jint computUnitType);

std::shared_ptr<TNN_NS::YoutuFaceAlign> CreateBlazeFaceAlign(JNIEnv *env, jobject thiz, jstring modelPath,
                           jint width, jint height, jint topk,
                           jint computUnitType, jint phase);

void makeBlazefaceAlignDetectOption(std::shared_ptr<TNN_NS::BlazeFaceDetectorOption>& option, std::string& lib_path, std::string& proto_content, std:: string& model_content);

JNIEXPORT jint JNICALL TNN_BLAZEFACE_ALIGN(init)(JNIEnv *env, jobject thiz, jstring modelPath,
                                                    jint width, jint height, jfloat scoreThreshold,
                                                    jfloat iouThreshold, jint topk,
                                                    jint computUnitType);
JNIEXPORT JNICALL jboolean TNN_BLAZEFACE_ALIGN(checkNpu)(JNIEnv *env, jobject thiz, jstring modelPath);
JNIEXPORT JNICALL jint TNN_BLAZEFACE_ALIGN(deinit)(JNIEnv *env, jobject thiz);
JNIEXPORT JNICALL jobjectArray TNN_BLAZEFACE_ALIGN(detectFromStream)(JNIEnv *env, jobject thiz, jbyteArray yuv420sp, jint width, jint height, jint view_width, jint view_height, jint rotate);

#ifdef __cplusplus
}
#endif
#endif //ANDROID_BLAZEFACE_ALIGN_JNI_H_
