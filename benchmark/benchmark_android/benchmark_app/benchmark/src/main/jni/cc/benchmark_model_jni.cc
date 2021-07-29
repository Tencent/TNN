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

#include <jni.h>

#include <sstream>
#include <string>
#include <fstream>

#include "benchmark_model_jni.h"
#include "test.h"

#ifdef __ANDROID__
#include <android/log.h>
#endif

JNIEXPORT jint JNICALL TNN_BENCHMARK_MODEL(nativeRun)(JNIEnv* env, jobject thiz, jstring args_obj, jstring file_dir) {
    const char* args_chars = env->GetStringUTFChars(args_obj, nullptr);
    const char* file_chars = env->GetStringUTFChars(file_dir, nullptr);

    // Split the args string into individual arg tokens.
    std::istringstream iss(args_chars);
    std::vector<std::string> args_split{std::istream_iterator<std::string>(iss),
                                        {}};

    // Construct a fake argv command-line object for the benchmark.
    std::vector<char*> argv;
    std::string arg0 = "(BenchmarkModelAndroid)";
    std::string model_file;
    bool model_path_option = false;
    argv.push_back(const_cast<char*>(arg0.data()));
    for (auto& arg : args_split) {
        // Deal with the model path
        if (!model_path_option) {
            argv.push_back(const_cast<char*>(arg.data()));
        } else {
            model_file = arg;
            std::ifstream fin(arg);
            if (!fin) {
                model_file = std::string(file_chars) + "/" + arg;
            }
            argv.push_back(const_cast<char*>(model_file.data()));
        }
        model_path_option = (arg.find("-mp") != std::string::npos);
    }

    int result = TNN_NS::test::Run(static_cast<int>(argv.size()), argv.data());
    env->ReleaseStringUTFChars(args_obj, args_chars);
    return result;
}
