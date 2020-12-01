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

#include "utils.h"

#include <string>
#include <fstream>

// Helper functions
std::string fdLoadFile(std::string path) {
    std::ifstream file(path, std::ios::binary);
    if (file.is_open()) {
        file.seekg(0, file.end);
        int size      = file.tellg();
        char* content = new char[size];
        file.seekg(0, file.beg);
        file.read(content, size);
        std::string fileContent;
        fileContent.assign(content, size);
        delete[] content;
        file.close();
        return fileContent;
    } else {
        return "";
    }
}