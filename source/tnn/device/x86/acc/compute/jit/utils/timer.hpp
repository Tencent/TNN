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

#ifndef TNN_DEVICE_X86_ACC_COMPUTE_JIT_UTILS_TIMER_H_
#define TNN_DEVICE_X86_ACC_COMPUTE_JIT_UTILS_TIMER_H_

#include <map>
#include <chrono>

namespace TNN_NS {

using std::chrono::time_point;
using std::chrono::system_clock;
using std::chrono::duration_cast;
using std::chrono::microseconds;

class NiceTimer {

public:

    float tick(int cur_id, int cmp_id = 0) {

        time_point<system_clock> cur = system_clock::now();
        m_footprint_list[cur_id] = cur;

        time_point<system_clock> last = cur;
        if (m_footprint_list.find(cmp_id) != m_footprint_list.end()) {
            last = m_footprint_list[cmp_id];
        }

        return cmp(last, cur);
    }


private:

    float cmp(time_point<system_clock> t1, time_point<system_clock> t2) {
        return duration_cast<microseconds>(t2 - t1).count() / 1000.0f;
    }


std::map<int, time_point<system_clock> > m_footprint_list;

};


};

#endif // TNN_DEVICE_X86_ACC_COMPUTE_JIT_UTILS_TIMER_H_