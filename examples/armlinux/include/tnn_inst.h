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

#pragma once
#include <fstream>
#include <vector>
#include "tnn/core/instance.h"
#include "tnn/core/macro.h"
#include "tnn/core/status.h"
#include "tnn/core/tnn.h"
#include "tnn/utils/blob_converter.h"

namespace arm_linux_demo {

class TNNInst {
public:
    /**
     * @brief 初始化函数
     * 输入为mat格式，输入类型支持NBGRA, NBGR, NGRAY, NCHW_FLOAT
     * 输出为mat格式，输出支持NCHW_FLOAT
     * @return 返回0，表示成功
     */
    int Init(const std::string& proto_file, const std::string& model_file);

    /**
     * @brief 前向计算函数
     * 输入为mat格式，输入类型支持NBGRA, NBGR, NGRAY, NCHW_FLOAT
     * 输出为mat格式，输出支持NCHW_FLOAT
     * @return 返回0，表示成功
     */
    int Forward(TNN_NS::Mat& input_mat, TNN_NS::Mat& output_mat);

    /**
     * @brief 获取输入输出尺寸
     * 默认仅有一个输入，一个输出，多输入输出需要对该部分进行修改
     * @return 返回{N, C, H, W}
     */
    std::vector<int> GetInputSize() const;
    std::vector<int> GetOutputSize() const;

    ~TNNInst();

private:
    TNN_NS::TNN tnn_;
    std::shared_ptr<TNN_NS::Instance> net_instance_;
    TNN_NS::MatConvertParam input_convert_param_;
    TNN_NS::MatConvertParam output_convert_param_;
    TNN_NS::BlobMap input_blobs_;
    TNN_NS::BlobMap output_blobs_;
};

}  // namespace arm_linux_demo
