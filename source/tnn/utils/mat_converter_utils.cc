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

#include "tnn/utils/mat_converter_utils.h"

namespace TNN_NS {

Status CheckMatConverterParams(Mat& src, Mat& dst, bool check_same_device) {
    if (src.GetData() == nullptr) {
        return Status(TNNERR_NULL_PARAM, "input mat is null");
    }

    if (check_same_device && (src.GetDeviceType() != dst.GetDeviceType())) {
        return Status(TNNERR_PARAM_ERR, "src and dst mat type must be same");
    }

    if (dst.GetData() == nullptr) {
        dst = Mat(dst.GetDeviceType(), dst.GetMatType(), dst.GetDims());
    }

    return TNN_OK;
}

}  // namespace TNN_NS