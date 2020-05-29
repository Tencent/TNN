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

#include <fstream>
#include <string>
#include <vector>

#include "tnn_inst.h"
#include "utils.h"

using arm_linux_demo::TNNInst;

// 随机初始化 0~255 BGR图像数据
static void InitRandom(uint8_t* ptr, size_t n) {
    for (unsigned long long i = 0; i < n; i++) {
        ptr[i] = static_cast<uint8_t>(rand() % 256);
    }
}

int main(int argc, char** argv) {
    if (argc < 3) {
        printf("go with:%s proto model input\n", argv[0]);
        return -1;
    }
    // 创建tnn实例
    TNNInst inst;
    //初始化实例，输入proto & model位置
    CHECK_API(inst.Init(argv[1], argv[2]));

    //获取输入、输出尺寸
    auto input_dims  = inst.GetInputSize();
    auto output_dims = inst.GetOutputSize();
    //构造输入mat, 设置格式为uint8 BGR图片数据
    TNN_NS::Mat input_mat(TNN_NS::DEVICE_ARM, TNN_NS::N8UC3, input_dims);
    auto input_count   = input_dims[1] * input_dims[2] * input_dims[3];
    InitRandom((uint8_t *)input_mat.GetData(), input_count);

    //构造输出mat, 设置格式为NCHW 浮点数据
    TNN_NS::Mat output_mat(TNN_NS::DEVICE_ARM, TNN_NS::NCHW_FLOAT, output_dims);
    
    //执行网络计算
    CHECK_API(inst.Forward(input_mat, output_mat));

    //完成计算，获取任意输出点
    fprintf(stdout, "Inst forward done, first output data %.3f\n",
            reinterpret_cast<float*>(output_mat.GetData())[0]);
    return 0;
}
