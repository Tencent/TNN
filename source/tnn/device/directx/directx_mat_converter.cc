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

#include "tnn/device/directx/directx_mat_converter.h"


#include "tnn/utils/dims_utils.h"
#include "tnn/utils/mat_converter_utils.h"

namespace TNN_NS {

Status DirectXMatConverterAcc::Copy(Mat& src, Mat& dst, void* command_queue) {
    Status ret = TNN_OK;

    ret = CheckMatConverterParams(src, dst, false);
    if (ret != TNN_OK)
        return ret;

    auto elem_num = DimsVectorUtils::Count(src.GetDims());

    if (src.GetMatType() == NGRAY || src.GetMatType() == NNV21 || src.GetMatType() == NNV12 || 
        src.GetMatType() == N8UC3 || src.GetMatType() == N8UC4) {
        memcpy(dst.GetData(), src.GetData(), elem_num * sizeof(uint8_t));
    } else if(src.GetMatType() == NCHW_FLOAT) {
        memcpy(dst.GetData(), src.GetData(), elem_num * sizeof(float));
    } else {
        return Status(TNNERR_PARAM_ERR, "DirectXMatConverterAcc::Copy, convert type not support yet");
    }
    return ret;
}

}
