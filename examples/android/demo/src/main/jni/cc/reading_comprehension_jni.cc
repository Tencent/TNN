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

#include "reading_comprehension_jni.h"
#include <jni.h>
#include "helper_jni.h"
#import "bert_tokenizer.h"

static std::shared_ptr<TNN_NS::TNNSDKSample> gResponder;
static int gComputeUnitType = 0; // 0 is cpu, 1 is gpu, 2 is huawei_npu

JNIEXPORT JNICALL jint TNN_READING_COMPREHENSION(init)(JNIEnv *env, jobject thiz, jstring modelPath, jint computeUnitType){

    setBenchResult("");
    gResponder = std::make_shared<TNN_NS::TNNSDKSample>();
    std::string protoContent, modelContent, vocabContent;
    std::string modelPathStr(jstring2string(env, modelPath));
    protoContent = fdLoadFile(modelPathStr + "/tiny-bert-squad.tnnproto");
    modelContent = fdLoadFile(modelPathStr + "/tiny-bert-squad.tnnmodel");
    LOGI("proto content size %d model content size %d", protoContent.length(), modelContent.length());
    TNN_NS::Status status = TNN_NS::TNN_OK;
    gComputeUnitType = computeUnitType;

    auto option = std::make_shared<TNN_NS::TNNSDKOption>();
    option->compute_units = TNN_NS::TNNComputeUnitsCPU;
    option->library_path="";
    option->proto_content = protoContent;
    option->model_content = modelContent;
//    option->precision = TNN_NS::PRECISION_HIGH;
    option->precision = TNN_NS::PRECISION_NORMAL;

    if (gComputeUnitType == 1) {
        option->compute_units = TNN_NS::TNNComputeUnitsGPU;
    } else if (gComputeUnitType == 2) {
        LOGI("the device type  %d device huawei_npu" ,gComputeUnitType);
        gResponder->setNpuModelPath(modelPathStr + "/");
        gResponder->setCheckNpuSwitch(false);
        option->compute_units = TNN_NS::TNNComputeUnitsHuaweiNPU;

        LOGE("tiny bert does not support NPU");
        return -1;
    } else {
        option->compute_units = TNN_NS::TNNComputeUnitsCPU;
    }

    status = gResponder->Init(option);

    if (status != TNN_NS::TNN_OK) {
        LOGE("detector init failed %d", (int)status);
        return -1;
    }

    return 0;
}

JNIEXPORT JNICALL jint TNN_READING_COMPREHENSION(deinit)(JNIEnv *env, jobject thiz){
    gResponder = nullptr;
    return 0;
}

JNIEXPORT JNICALL jstring TNN_READING_COMPREHENSION(ask)(JNIEnv *env, jobject thiz, jstring modelPath, jstring material, jstring question){

    TNN_NS::BenchOption bench_option;
    bench_option.forward_count = 10;
    gResponder->SetBenchOption(bench_option);

    std::string vocabPath;
    std::string modelPathStr(jstring2string(env, modelPath));
    vocabPath = modelPathStr + "/vocab.txt";

    auto tokenizer = std::make_shared<TNN_NS::BertTokenizer>();
    auto status = tokenizer->Init(vocabPath);
    if (status != TNN_NS::TNN_OK) {
        LOGE("tokenizer init failed %d", (int)status);
        return nullptr;
    }

    LOGE("%s", jstring2string(env, material));
    LOGE("%s", jstring2string(env, question));

    TNN_NS::DeviceType dt = TNN_NS::DEVICE_ARM;

    auto bertInput = std::make_shared<TNN_NS::BertTokenizerInput>(dt, "input_ids", "attention_mask", "token_type_ids");
    auto bertOutput = gResponder->CreateSDKOutput();

    std::string material_utf8 = std::string(jstring2string(env, material));
    std::string question_utf8 = std::string(jstring2string(env, question));

    tokenizer->buildInput(material_utf8, question_utf8, bertInput);

    status = gResponder->Predict(bertInput, bertOutput);

    if (status != TNN_NS::TNN_OK) {
        LOGE("gResponder Pedict failed %d", (int)status);
        return string2jstring(env, "Predict Failed");
    }

    std::string ans;
    tokenizer->ConvertResult(bertOutput, "output_0", "output_1", ans);

    auto bench_result = gResponder->GetBenchResult();
    LOGE("avg time is %f", bench_result.avg);

    return string2jstring(env, ans.c_str());
}
