//
// Created by rgb000000 on 2021/8/6.
//

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

        LOGE("tiny bert does not support GPU");
        return -1;
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

    jintArray resultArray;
    int ret = -1;

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
//    TNN_NS::DeviceType dt = TNN_NS::DEVICE_NAIVE;
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
