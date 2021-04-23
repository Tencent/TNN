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

#ifndef ANDROID_OCR_DETECTOR_JNI_H_
#define ANDROID_OCR_DETECTOR_JNI_H_

#include <jni.h>
#define TNN_OCR_DETECTOR(sig) Java_com_tencent_tnn_demo_OCRDetector_##sig
#ifdef __cplusplus
extern "C" {
#endif
JNIEXPORT JNICALL jint TNN_OCR_DETECTOR(init)(JNIEnv *env, jobject thiz, jstring modelPath, jint width, jint height, jint computUnitType);
JNIEXPORT JNICALL jint TNN_OCR_DETECTOR(deinit)(JNIEnv *env, jobject thiz);
JNIEXPORT JNICALL jboolean TNN_OCR_DETECTOR(checkNpu)(JNIEnv *env, jobject thiz, jstring modelPath);
JNIEXPORT JNICALL jobjectArray TNN_OCR_DETECTOR(detectFromStream)(JNIEnv *env, jobject thiz, jbyteArray yuv420sp, jint width, jint height, jint view_width, jint view_height, jint rotate);
JNIEXPORT JNICALL jobjectArray TNN_OCR_DETECTOR(detectFromImage)(JNIEnv *env, jobject thiz, jobject imageSource, jint width, jint height);

#ifdef __cplusplus
}
#endif

#endif // ANDROID_OCR_DETECTOR_JNI_H_
