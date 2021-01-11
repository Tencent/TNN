/*
* @Author: Dandi Ding
* @Date:   2019-09-21 16:59:43
* @Last Modified by:   Dandiding
* @Last Modified time: 2019-09-21 17:29:27
*/

#pragma once

#include <map>
#include <chrono>

namespace rpngpu
{

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