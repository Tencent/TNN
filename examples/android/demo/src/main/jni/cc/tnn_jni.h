//
// Created by neiltian on 17-3-2.
//

#ifndef YT_ANDROID_TNN_TNN_JNI_H_H
#define YT_ANDROID_TNN_TNN_JNI_H_H

#include <jni.h>
#include <string>
#include <android/log.h>

#ifdef __cplusplus
extern "C" {
#endif

JNIEXPORT JNICALL jint Java_com_tencent_tnn_demo_TNNLib_init(
        JNIEnv *env, jobject thiz, jstring protoFilePath, jstring modelFilePath, jstring device);

JNIEXPORT JNICALL jfloatArray Java_com_tencent_tnn_demo_TNNLib_forward(
        JNIEnv *env, jobject thiz, jobject imageSource);

JNIEXPORT JNICALL jint Java_com_tencent_tnn_demo_TNNLib_deinit(
        JNIEnv *env, jobject thiz);


#ifdef __cplusplus
}
#endif

#endif //YT_ANDROID_TNN_TNN_JNI_H_H
