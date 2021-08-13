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

#include "tnn/utils/string_format.h"

#include <algorithm>
#include <iomanip>
#include <numeric>

namespace TNN_NS {

std::string DoubleToString(double val) {
    std::stringstream stream;
    stream << std::setprecision(3) << std::setiosflags(std::ios::fixed) << val;
    return stream.str();
}

std::string DoubleToStringFilter(double val) {
    if (0 == val) {
        return "";
    } else {
        return DoubleToString(val);
    }
}

std::string MatTypeToString(MatType mat_type) {
    if (N8UC3 == mat_type) {
        return "N8UC3";
    } else if (N8UC4 == mat_type) {
        return "N8UC4";
    } else if (NGRAY == mat_type) {
        return "NGRAY";
    } else if (NNV21 == mat_type) {
        return "NNV21";
    } else if (NNV12 == mat_type) {
        return "NNV12";
    } else if (NCHW_FLOAT == mat_type) {
        return "NCHW_FLOAT";
    } else if (NC_INT32 == mat_type) {
        return "NC_INT32";
    } else if (RESERVED_BFP16_TEST == mat_type) {
        return "RESERVED_BFP16_TEST";
    } else if (RESERVED_FP16_TEST == mat_type) {
        return "RESERVED_FP16_TEST";
    } else if (RESERVED_INT8_TEST == mat_type) {
        return "RESERVED_INT8_TEST";
    } else {
        return "INVALID Mat Type";
    }
}

std::string DimsToString(std::vector<int> dims) {
    std::stringstream stream;
    stream << "[";
    for (int i = 0; i < dims.size() - 1; ++i) {
        stream << dims[i] << ", ";
    }
    stream << dims[dims.size() - 1] << "]";

    return stream.str();
}

struct CmpByValue {
    bool operator()(const std::pair<std::string, std::vector<float>> &lhs,
                    const std::pair<std::string, std::vector<float>> &rhs) {
        return lhs.second[0] > rhs.second[0];
    }
};

std::vector<std::pair<std::string, std::vector<float>>> SortMapByValue(std::map<std::string, std::vector<float>> &map) {
    std::vector<std::pair<std::string, std::vector<float>>> pair_vec(map.begin(), map.end());
    sort(pair_vec.begin(), pair_vec.end(), CmpByValue());
    return pair_vec;
}

std::ostream &FormatRow(std::ostream &stream, int width) {
    stream << std::right << std::setw(width);
    return stream;
}

std::string StringFormatter::Table(const std::string &title, const std::vector<std::string> &header,
                                   const std::vector<std::vector<std::string>> &data) {
    if (header.empty())
        return "";
    const size_t column_size = header.size();
    const size_t data_size   = data.size();
    std::vector<int> max_column_len(header.size(), 0);
    for (size_t col_idx = 0; col_idx < column_size; ++col_idx) {
        max_column_len[col_idx] = std::max<int>(max_column_len[col_idx], static_cast<int>(header[col_idx].size()));
        for (size_t data_idx = 0; data_idx < data_size; ++data_idx) {
            if (col_idx < data[data_idx].size()) {
                max_column_len[col_idx] =
                    std::max<int>(max_column_len[col_idx], static_cast<int>(data[data_idx][col_idx].size()));
            }
        }
    }
    const size_t row_length = std::accumulate(max_column_len.begin(), max_column_len.end(), 0, std::plus<size_t>()) +
                              2 * column_size + column_size + 1;
    const std::string dash_line(row_length, '-');
    std::stringstream stream;
    stream << dash_line << std::endl;
    FormatRow(stream, static_cast<int>(row_length / 2 + title.size() / 2)) << title << std::endl;
    stream << dash_line << std::endl;
    // format header
    stream << "|";
    for (size_t h_idx = 0; h_idx < column_size; ++h_idx) {
        stream << " ";
        FormatRow(stream, max_column_len[h_idx]) << header[h_idx];
        stream << " |";
    }
    stream << std::endl << dash_line << std::endl;
    // format data
    for (size_t data_idx = 0; data_idx < data_size; ++data_idx) {
        stream << "|";
        for (size_t h_idx = 0; h_idx < column_size; ++h_idx) {
            stream << " ";
            FormatRow(stream, max_column_len[h_idx]) << data[data_idx][h_idx];
            stream << " |";
        }
        stream << std::endl;
    }
    stream << dash_line << std::endl;
    return stream.str();
}

}  // namespace TNN_NS
