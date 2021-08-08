//
// Created by rgb000000 on 2021/8/6.
//

#ifndef ANDROID_READING_COMPREHENSION_JNI_H
#define ANDROID_READING_COMPREHENSION_JNI_H

#include <jni.h>
#define TNN_READING_COMPREHENSION(sig) Java_com_tencent_tnn_demo_ReadingComprehensionTinyBert_##sig
#ifdef __cplusplus
extern "C" {
#endif
JNIEXPORT JNICALL jint TNN_READING_COMPREHENSION(init)(JNIEnv *env, jobject thiz, jstring modelPath, jint computeUnitType);
JNIEXPORT JNICALL jint TNN_READING_COMPREHENSION(deinit)(JNIEnv *env, jobject thiz);
JNIEXPORT JNICALL jstring TNN_READING_COMPREHENSION(ask)(JNIEnv *env, jobject thiz, jstring modelPath, jstring material, jstring question);

#ifdef __cplusplus
}
#endif

#endif //ANDROID_READING_COMPREHENSION_JNI_H
