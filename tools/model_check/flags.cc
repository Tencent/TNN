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

#include "flags.h"

namespace TNN_NS {

DEFINE_bool(h, false, help_message);

DEFINE_string(p, "", proto_path_message);

DEFINE_string(m, "", model_path_message);

DEFINE_string(d, "", device_type_message);

DEFINE_string(i, "", input_path_message);

DEFINE_string(f, "", output_ref_path_message);

DEFINE_bool(e, false, cmp_end_message);

DEFINE_string(n, "", bias_message);

DEFINE_string(s, "", scale_message);

DEFINE_string(do, "", dump_output_path_message);

DEFINE_bool(b, false, check_batch_message);

DEFINE_string(a, "", align_all_message);

DEFINE_string(sp, "", set_precision_message);

DEFINE_string(du, "", dump_unaligned_layer_path_message);

}  // namespace TNN_NS
