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

#ifndef ANDROID_SKELETON_DETECTOR_JNI_H
#define ANDROID_SKELETON_DETECTOR_JNI_H
#include <jni.h>
#define TNN_SKELETON_DETECTOR(sig) Java_com_tencent_tnn_demo_SkeletonDetector_##sig
#ifdef __cplusplus
extern "C" {
#endif

JNIEXPORT JNICALL jint TNN_SKELETON_DETECTOR(init)(JNIEnv *env, jobject thiz, jstring modelPath, jint width, jint height, jint computUnitType);
JNIEXPORT JNICALL jboolean TNN_SKELETON_DETECTOR(checkNpu)(JNIEnv *env, jobject thiz, jstring modelPath);
JNIEXPORT JNICALL jint TNN_SKELETON_DETECTOR(deinit)(JNIEnv *env, jobject thiz);
JNIEXPORT JNICALL jobjectArray TNN_SKELETON_DETECTOR(detectFromStream)(JNIEnv *env, jobject thiz, jbyteArray yuv420sp, jint width, jint height, jint view_width, jint view_height, jint rotate, jint detector_type);

#ifdef __cplusplus
}
#endif
#endif //ANDROID_SKELETON_DETECTOR_JNI_H
