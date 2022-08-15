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

#include "tnn/utils/split_utils.h"

#include <stdlib.h>
#include <algorithm>
#include <cstring>

namespace TNN_NS {

bool SplitUtils::IsQuote(char c) {
    return c == '\'' || c == '\"';
}

bool SplitUtils::IsFullWidth(const char *pstr) {
    if (pstr == 0)
        return false;
    if (*pstr == 0 || *(pstr + 1) == 0)
        return false;
    return ((*pstr) & 0x80) && ((unsigned char)(*pstr) != 0xFF);
}

int SplitUtils::TrimStr(char *pstr, const char trim_char /* = ' ' */, bool trim_gb /* = false */) {
    const char *p = pstr;
    if (0 == p)
        return 0;

    // Get start and end position
    int start = 0, end = 0;
    while (*p) {
        // 全角
        if (trim_gb && IsFullWidth(p)) {
            if (*((unsigned short *)p) == 0xa1a1) {
                if (end == 0)
                    start += 2;
            } else {
                end = (int)(p - pstr + 2);
            }
            p += 2;
            continue;
        }
        // 半角
        if (((unsigned char)*p < 0x20 || trim_char == *p)) {
            if (end == 0)
                start++;
        } else
            end = (int)(p - pstr + 1);
        p++;
    }

    // trim it
    end > 0 ? pstr[end] = 0 : end = (int)(p - pstr);
    if (end == start)
        pstr[0] = 0;
    else if (start > 0)
        memmove(pstr, pstr + start, end - start + sizeof(char));

    return end - start;
}

char *SplitUtils::StrNCpy(char *dst, const char *src, int maxcnt) {
    char *rdst       = dst;
    const char *rsrc = src;
    int rmaxlen      = maxcnt;

    if (rmaxlen > 0) {
        if (rdst != rsrc) {
            *rdst = '\0';
            if (rsrc != 0)
                strncat(rdst, rsrc, --rmaxlen);
        } else {
            rdst += (rmaxlen - 1);
            *rdst = '\0';
        }
    }
    return dst;
}

void SplitUtils::ParseStr(const char *str, char *subs, const int len,
                          const bool supp_quote, const bool trim, const bool ignore_blank,
                          const bool trim_quote, const bool supp_quanjiao, const int i,
                          int& cursor, bool &left_quote, bool &right_quote) {
    if (len > 0) {
        if (!supp_quote)
            StrNCpy(subs, str + cursor, len + 1);
        else {
            if (trim_quote && IsQuote(str[cursor])) {
                left_quote = true;
                if (str[i - 1] == str[cursor])
                    right_quote = true;

                StrNCpy(subs, str + cursor + left_quote, len + 1 - left_quote - right_quote);
            } else
                StrNCpy(subs, str + cursor, len + 1);

            right_quote = false;
        }
    }
    cursor = i + 1;

    // trim it
    if (trim || ignore_blank)
        TrimStr(subs, ' ', supp_quanjiao);
}

Status SplitUtils::SplitStr(const char *str, str_arr &subs_array, const char spliter[] /* = ",;" */,
                            bool trim /* = true */, bool ignore_blank /* = false */, bool supp_quote /* = false */,
                            bool trim_quote /* = true */, bool supp_quanjiao /* = false */) {
    bool quote_start = false;
    char last_quote  = 0;
    bool left_quote = false, right_quote = false;
    int step = 1;

    if (str[0] == 0) {
        return TNN_OK;
    }
    
    const int subs_length = 4096;
    char *subs = (char *)calloc(subs_length, sizeof(char));

    for (int i = 0, cursor = 0;; i += step) {
        const char &c = str[i];

        // check full width
        step = 1;
        if (supp_quanjiao && IsFullWidth(str + i)) {
            step = 2;
            continue;
        }

        // check quote
        if (supp_quote) {
            if (IsQuote(c)) {
                if (quote_start == false) {
                    quote_start = true;
                    left_quote  = true;
                    last_quote  = c;
                    continue;
                } else if (c == last_quote) {
                    quote_start = false;
                    right_quote = true;
                    last_quote  = 0;
                }
            }
        }

        // segment check
        if (c == 0 || (!quote_start && strchr(spliter, c))) {
            subs[0] = 0;
#if defined(WIN32) && _MSC_VER < 1300  // VC++ 6.0
            int len =  (i - cursor)>=(subs_length - 1)?(subs_length - 1):(i - cursor);
#else
            int len = std::min<int>(i - cursor, subs_length - 1);
#endif

            ParseStr(str, subs, len, supp_quote, trim, ignore_blank,
                     trim_quote, supp_quanjiao, i, cursor, left_quote, right_quote);

            std::string subs_str(subs);  // for gnu_stl
            if (!ignore_blank || subs[0] != 0)
                subs_array.push_back(subs_str);
        }

        // end loop
        if (str[i] == 0)
            break;
    }
    
    free(subs);
    return TNN_OK;
}

Status SplitUtils::SplitParamList(const str_arr kv_pairs, str_dict &subs_dict, const char spliter[]) {
    for (int i = 0; i < kv_pairs.size(); i++) {
        str_arr kv;
        auto ret = SplitStr(kv_pairs[i].c_str(), kv, spliter, true, false);
        if (ret != TNN_OK || kv.size() != 2) {
            return Status(TNNERR_PARAM_ERR, "split param list failed");
        }
        int key = atoi(kv[0].c_str());

        bool is_array = key <= -23300;
        if (is_array) {
            key = -key - 23300;
        }

        std::string value(kv[1]);
        subs_dict[key] = value;
    }

    return TNN_OK;
}

}  // namespace TNN_NS
