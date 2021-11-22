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

#ifndef TNN_ARM_BLOB_COMPUTE_H_
#define TNN_ARM_BLOB_COMPUTE_H_

#include "tnn/core/blob.h"
#include "tnn/core/common.h"

namespace TNN_NS {

DimsVector GetNCXHWXRoundDims(const DimsVector &dims, const int round);

void SplitvCommon(Blob *input, const std::vector<Blob *> &outputs, const int axis);
void SplitvChannel(Blob *input, const std::vector<Blob *> &outputs, const int axis);
void SplitvChannelC4(Blob *input, const std::vector<Blob *> &outputs, const int axis);

}  // namespace TNN_NS

#endif  // TNN_ARM_BLOB_COMPUTE_H_
