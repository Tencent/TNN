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

#include "test/flags.h"

namespace TNN_NS {

DEFINE_bool(h, false, help_message);

DEFINE_string(mt, "", model_type_message);

DEFINE_string(nt, "", network_type_message);

DEFINE_string(mp, "", model_path_message);

DEFINE_string(dt, "ARM", device_type_message);

DEFINE_string(lp, "", library_path_message);

DEFINE_int32(di, 0, device_id_message);

DEFINE_int32(ic, 1, iterations_count_message);

DEFINE_int32(wc, 0, warm_up_count_message);

DEFINE_string(ip, "", input_path_message);

DEFINE_string(op, "", output_path_message);

DEFINE_bool(fc, false, output_format_cmp_message);

DEFINE_string(dl, "", device_list_message);

DEFINE_bool(ub, false, unit_test_benchmark_message);

DEFINE_int32(th, 1, cpu_thread_num_message);

DEFINE_int32(it, 0, input_format_message);

DEFINE_string(pr, "HIGH", precision_message);

DEFINE_string(is, "", input_shape_message);

}  // namespace TNN_NS
