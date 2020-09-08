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

#include "command.h"

#include <iostream>

#include "gflags/gflags.h"
#include "tools/converter/source/utils/flags.h"

namespace TNN_CONVERTER {

void ShowHelpMessage() {
    // TODO
    std::cout << "show help message!" << std::endl;
}

void ShowModelPathMessage() {
    // TODO
    std::cout << "please special the tensorflow lite path!" << std::endl;
}

void ShowOnnxPathMessage() {
    // TODO
}

bool ParseCommandLine(int argc, char* argv[]) {
    gflags::ParseCommandLineNonHelpFlags(&argc, &argv, true);
    if (FLAGS_h) {
        ShowHelpMessage();
        return false;
    }
    if (FLAGS_mp.empty()) {
        ShowModelPathMessage();
        ShowHelpMessage();
        return false;
    }
    if (FLAGS_od.empty()) {
        ShowOnnxPathMessage();
        ShowHelpMessage();
        return false;
    }
    // TODO
    return true;
}
}  // namespace TNN_CONVERTER
