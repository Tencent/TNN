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

#ifndef TNN_SOURCE_TNN_UTILS_SPLIT_UTILS_H_
#define TNN_SOURCE_TNN_UTILS_SPLIT_UTILS_H_

#include <map>
#include <string>
#include <vector>

#include "tnn/core/macro.h"
#include "tnn/core/status.h"

namespace TNN_NS {

typedef std::vector<std::string> str_arr;
typedef std::map<int, std::string> str_dict;

class SplitUtils {
public:
    static Status SplitStr(const char *str, str_arr &subs_array, const char spliter[] = ",;:", bool trim = true,
                           bool ignore_blank = false, bool supp_quote = false, bool trim_quote = true,
                           bool supp_quanjiao = false);

    static Status SplitParamList(const str_arr input_arr, str_dict &subs_array, const char spliter[] = "=");

private:
    static bool IsFullWidth(const char *pstr);
    static bool IsQuote(char c);
    static char *StrNCpy(char *dst, const char *src, int maxcnt);
    static int TrimStr(char *pstr, const char trim_char = ' ', bool trim_gb = false);
    static void ParseStr(const char *str, char *subs, const int len,
                         const bool supp_quote, const bool trim, const bool ignore_blank,
                         const bool trim_quote, const bool supp_quanjiao, const int i,
                         int& cursor, bool &left_quote, bool &right_quote);
};

}  // namespace TNN_NS

#endif  // TNN_SOURCE_TNN_UTILS_SPLIT_UTILS_H_
