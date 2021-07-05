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

#include "tnn/utils/data_flag_utils.h"

#include "tnn/core/macro.h"

namespace TNN_NS {

bool DataFlagUtils::AllocateInForward(int flag) {
    return flag & DATA_FLAG_ALLOCATE_IN_FORWARD;
}

int DataFlagUtils::ChangeStatus(int flag) {
    return flag & 0x0000FFFF;
}

int DataFlagUtils::MinChangeStatus(int flag0, int flag1) {
    auto allocate_in_forward = DataFlagUtils::AllocateInForward(flag0) || DataFlagUtils::AllocateInForward(flag1);
    flag0 = ChangeStatus(flag0);
    flag1 = ChangeStatus(flag1);
    flag0 = flag0 < flag1 ? flag0 : flag1;
    return allocate_in_forward ? (flag0 | DATA_FLAG_ALLOCATE_IN_FORWARD): flag0;
}

}  // namespace TNN_NS
