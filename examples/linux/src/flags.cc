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

DEFINE_bool(h, false, help_message);
DEFINE_string(p, "", proto_path_message);
DEFINE_string(m, "", model_path_message);
DEFINE_string(i, "", input_path_message);

void ShowUsage(const char *exe, bool input_required) {
    printf("usage:\n%s [-h] [-p] tnnproto [-m] tnnmodel [-i] <input>\n", exe);
    printf("\t-h, <help>     \t%s\n", help_message);
    printf("\t-p, <proto>    \t%s\n", proto_path_message);
    printf("\t-m, <model>    \t%s\n", model_path_message);
    if (input_required) {
        printf("\t-i, <input>    \t%s\n", input_path_message);
    }
}

bool ParseAndCheckCommandLine(int argc, char* argv[], bool input_required) {
    gflags::ParseCommandLineNonHelpFlags(&argc, &argv, true);
    if (FLAGS_h) {
        return false;
    }

    if (FLAGS_m.empty() || FLAGS_p.empty()) {
        printf("Parameter -m and -p should be set \n");
        return false;
    }

    if (FLAGS_i.empty() && input_required) {
        printf("Parameter -i should be set \n");

        return false;
    }

    return true;
}