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

#include "face_gray_transfer.h"
#include <sys/time.h>
#include <cmath>

namespace TNN_NS {

FaceGrayTransfer::~FaceGrayTransfer() {}

MatConvertParam FaceGrayTransfer::GetConvertParamForInput(std::string tag) {
    MatConvertParam input_cvt_param;
    input_cvt_param.scale = {2.0 / 255, 2.0 / 255, 2.0 / 255, 0.0};
    input_cvt_param.bias  = {-1.0, -1.0, -1.0, 0.0};
    return input_cvt_param;
}

}
