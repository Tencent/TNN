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

#ifndef TNN_SOURCE_TNN_UTILS_STRING_UTILS_TEA_H_
#define TNN_SOURCE_TNN_UTILS_STRING_UTILS_TEA_H_

#include <stdint.h>
#include "tnn/core/common.h"
#include "tnn/core/macro.h"

namespace TNN_NS {
#define PROTO_ENCRYPTION_ENABLED "proto_encryption_enabled"
#define PROTO_ENCRYPTION_UNKNOWN "proto_encryption_unknown"
#define MODEL_ENCRYPTION_ENABLED "model_encryption_enabled"
#define MODEL_ENCRYPTION_UNKNOWN "model_encryption_unknown"

int TeaEncrypt(const uint8_t *plain, int plain_len, const uint8_t *key, uint8_t *buf, int buf_len);

int TeaDecrypt(const uint8_t *cryptograph, int cryptograph_len, const uint8_t *key, uint8_t *buf, int buf_len);

}  // namespace TNN_NS
#endif
