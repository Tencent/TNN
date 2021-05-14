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

#ifndef TNN_SOURCE_TNN_UTILS_DATA_FLAG_UTILS_H_
#define TNN_SOURCE_TNN_UTILS_DATA_FLAG_UTILS_H_

#include <string>

#include "tnn/core/common.h"
#include "tnn/core/macro.h"

namespace TNN_NS {

class DataFlagUtils {
public:
    // @brief to check wether the data is allocated in forward
    // @param flag data flag
    static bool AllocateInForward(int flag);

    // @brief to check the data change flag
    // @param flag data flag
    static int ChangeStatus(int flag);
    
    // @brief get the minimal change flag, ignore allocate flag
    // @param flag data flag
    static int MinChangeStatus(int flag0, int  flag1);
};

}  // namespace TNN_NS

#endif  // TNN_SOURCE_TNN_UTILS_DATA_TYPE_UTILS_H_
