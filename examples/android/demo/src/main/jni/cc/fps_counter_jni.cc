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

#include "fps_counter_jni.h"
#include "tnn_fps_counter.h"
#include <jni.h>
#include "helper_jni.h"

static std::shared_ptr<TNNFPSCounter> gFpsCounter;
// Jni functions

JNIEXPORT JNICALL jint TNN_FPS_COUNTER(init)(JNIEnv *env, jobject thiz) {
    gFpsCounter = std::make_shared<TNNFPSCounter>();
    return 0;
}

JNIEXPORT JNICALL jint TNN_FPS_COUNTER(deinit)(JNIEnv *env, jobject thiz) {
    gFpsCounter = nullptr;
    return 0;
}

JNIEXPORT JNICALL jint TNN_FPS_COUNTER(begin)(JNIEnv *env, jobject thiz, jstring tag) {
    std::string tagStr(jstring2string(env, tag));
    gFpsCounter->Begin(tagStr);

    return 0;
}

JNIEXPORT JNICALL jint TNN_FPS_COUNTER(end)(JNIEnv *env, jobject thiz, jstring tag) {
    std::string tagStr(jstring2string(env, tag));
    gFpsCounter->End(tagStr);

    return 0;
}

JNIEXPORT JNICALL jdouble TNN_FPS_COUNTER(getFps)(JNIEnv *env, jobject thiz, jstring tag) {
    std::string tagStr(jstring2string(env, tag));
    double fps = gFpsCounter->GetFPS(tagStr);

    return (jdouble)fps;
}

JNIEXPORT JNICALL jdouble TNN_FPS_COUNTER(getTime)(JNIEnv *env, jobject thiz, jstring tag) {
    std::string tagStr(jstring2string(env, tag));
    double time = gFpsCounter->GetTime(tagStr);

    return (jdouble)time;
}
