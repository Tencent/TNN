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

#ifndef TNN_EXAMPLES_BASE_TNN_FPS_COUNTER_H_
#define TNN_EXAMPLES_BASE_TNN_FPS_COUNTER_H_
#include <algorithm>
#include <map>
#include <memory>
#include <string>
#include <vector>

class TNNFPSCounter {
public:
    TNNFPSCounter();
    void Begin(std::string tag);
    void End(std::string tag);
    double GetFPS(std::string tag);
    double GetTime(std::string tag);
    std::map<std::string, double> GetAllFPS();
    std::map<std::string, double> GetAllTime();
    
protected:
    std::map<std::string, double> map_fps_ = {};
    std::map<std::string, double> map_start_time_ = {};
    std::map<std::string, double> map_time_ = {};
    
private:
    std::string RetifiedTag(std::string tag);
    double GetStartTime(std::string tag);
};

#endif //TNN_EXAMPLES_BASE_TNN_FPS_COUNTER_H_
