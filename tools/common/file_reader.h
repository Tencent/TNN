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

#ifndef TNN_TOOLS_COMMON_FILE_READER_H_
#define TNN_TOOLS_COMMON_FILE_READER_H_

#include <map>
#include <string>
#include <vector>

#include "tnn/core/blob.h"
#include "tnn/core/mat.h"
#include "tnn/core/status.h"

namespace TNN_NS {

typedef enum {
    /* Not Support */
    NOTSUPPORT = 0,
    /* text file */
    TEXT = 1,
    /* numpy file */
    NPY = 2,
    /* image file */
    IMAGE = 3,
} FileFormat;

class FileReader {
public:
    // @brief FileReader constructor
    FileReader();

    // @brief FileReader virtual Destructor
    ~FileReader();

public:
    // @brief Read the file into Blob
    // param 0 : output_blob
    // param 1 : file_path, the file_path of the input
    // param 2 : format, the format of the input file. txt or npy
    Status Read(Blob* output_blob, const std::string file_path, const FileFormat format);

    // @brief Read the file into Mat Map
    // param 0 : mat_map
    // param 1 : file_path, the file_path of the input
    // param 2 : format, the format of the input file. txt or npy
    Status Read(std::map<std::string, std::shared_ptr<Mat>>& mat_map, const std::string file_path,
                const FileFormat format);

    // @brief set bias_ value
    // param 0 : bias val
    void SetBiasValue(std::vector<float> bias);

    // @brief set scale_ value
    // param 0 : scale val
    void SetScaleValue(std::vector<float> scale);

    // @brief set reverse_channel_ value
    // param 0 : reverse_channel val
    void SetReverseChannel(bool reverse_channel);

private:
    Status PreProcessImage(unsigned char* img_data, Blob* blob, int width, int height, int channel);

    std::vector<float> bias_;
    std::vector<float> scale_;
    bool reverse_channel_;

    size_t length_in_elements_;
};

}  // namespace TNN_NS

#endif  // TNN_TOOLS_COMMON_FILE_READER_H_
