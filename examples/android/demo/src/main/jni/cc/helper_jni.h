//
// Created by tencent on 2020-04-30.
//

#ifndef ANDROID_HELPER_JNI_H
#define ANDROID_HELPER_JNI_H
#include <string>
#include <jni.h>
#define TNN_HELPER(sig) Java_com_tencent_tnn_demo_Helper_##sig
#ifdef __cplusplus
extern "C" {
#endif
std::string fdLoadFile(std::string path);
char* jstring2string(JNIEnv* env, jstring jstr);
jstring string2jstring(JNIEnv* env, const char* pat);
void setBenchResult(std::string result);
JNIEXPORT JNICALL jstring TNN_HELPER(getBenchResult)(JNIEnv *env, jobject thiz);
#ifdef __cplusplus
}
#endif
#endif //ANDROID_HELPER_JNI_H
