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

#include "file_reader.h"
#include <fstream>
#include "tnn/utils/dims_vector_utils.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb_image_resize.h"

namespace TNN_NS {

static void ProcessNHWC2NCHW(unsigned char* img_data, float* blob_data, int channel, int height, int width,
                             std::vector<float> bias, std::vector<float> scale) {
    ASSERT(bias.size() >= channel)
    ASSERT(scale.size() >= channel)
    for (int h = 0; h < height; ++h) {
        for (int w = 0; w < width; ++w) {
            for (int c = 0; c < channel; ++c) {
                int idx_src        = h * width * channel + w * channel + c;
                int idx_dst        = c * height * width + h * width + w;
                blob_data[idx_dst] = ((float)img_data[idx_src] - bias[c]) * scale[c];
            }
        }
    }
}

FileReader::FileReader() {
    bias_  = {0.0f, 0.0f, 0.0f, 0.0f};
    scale_ = {1.0f, 1.0f, 1.0f, 1.0f};
}

FileReader::~FileReader() {}

Status FileReader::Read(Blob* output_blob, const std::string file_path, const FileFormat format) {
    if (output_blob->GetBlobDesc().data_type != DATA_TYPE_FLOAT) {
        LOGE("The blob data type is not support yet!\n");
        return TNNERR_INVALID_INPUT;
    }

    Status ret = TNN_OK;
    if (format == TEXT) {
        std::ifstream f_stream(file_path);
        int count       = DimsVectorUtils::Count(output_blob->GetBlobDesc().dims);
        float* data_ptr = static_cast<float*>(output_blob->GetHandle().base);
        for (int i = 0; i < count; ++i) {
            f_stream >> data_ptr[i];
        }
        f_stream.close();

    } else if (format == IMAGE) {
        int blob_c = 0;
        if (output_blob->GetBlobDesc().data_format == DATA_FORMAT_NCHW) {
            blob_c = output_blob->GetBlobDesc().dims[1];
        } else {
            LOGE("The blob data format is not support yet!\n");
            return TNNERR_INVALID_INPUT;
        }

        int w                   = 0;
        int h                   = 0;
        int c                   = 0;
        unsigned char* img_data = stbi_load(file_path.c_str(), &w, &h, &c, blob_c);
        if (img_data == nullptr) {
            LOGE("load image data falied!\n");
            return TNNERR_INVALID_INPUT;
        }
        ret = PreProcessImage(img_data, output_blob, w, h, blob_c);
        stbi_image_free(img_data);

    } else {
        LOGE("The input format is not support yet!\n");
        return TNNERR_INVALID_INPUT;
    }

    return ret;
}

Status FileReader::Read(std::map<std::string, std::shared_ptr<Mat>>& mat_map, const std::string file_path,
                        const FileFormat format) {
    Status ret = TNN_OK;

    if (format == TEXT) {
        std::ifstream f_stream(file_path);

        int blob_count = 1;
        f_stream >> blob_count;

        for (int i = 0; i < blob_count; ++i) {
            std::string blob_name;
            uint32_t dims_size;
            DimsVector dims;

            f_stream >> blob_name;
            f_stream >> dims_size;
            for (int j = 0; j < dims_size; ++j) {
                uint32_t dim_value;
                f_stream >> dim_value;
                dims.push_back(dim_value);
            }

            int data_type;
            f_stream >> data_type;

            MatType mat_type = INVALID;
            if (DATA_TYPE_FLOAT == data_type) {
                mat_type = NCHW_FLOAT;
            } else if (DATA_TYPE_INT32 == data_type) {
                mat_type = NC_INT32;
            } else {
                f_stream.close();
                return Status(TNNERR_INVALID_INPUT, "the data type is not support in txt in file reader");
            }

            std::shared_ptr<Mat> mat(new Mat(DEVICE_NAIVE, mat_type, dims));

            int count = DimsVectorUtils::Count(dims);
            if (DATA_TYPE_FLOAT == data_type) {
                float* data_ptr = static_cast<float*>(mat->GetData());
                for (int i = 0; i < count; ++i) {
                    f_stream >> data_ptr[i];
                }
            } else if (DATA_TYPE_INT32 == data_type) {
                int* data_ptr = static_cast<int*>(mat->GetData());
                for (int i = 0; i < count; ++i) {
                    f_stream >> data_ptr[i];
                }
            }

            mat_map[blob_name] = mat;
        }
        f_stream.close();

    } else if (format == IMAGE) {
        // TO-DO: support image input
        LOGE("The input format is not support yet!\n");
        return TNNERR_INVALID_INPUT;
    } else {
        LOGE("The input format is not support yet!\n");
        return TNNERR_INVALID_INPUT;
    }

    return TNN_OK;
}

void FileReader::SetBiasValue(std::vector<float> bias) {
    bias_ = bias;
}

void FileReader::SetScaleValue(std::vector<float> scale) {
    scale_ = scale;
}

Status FileReader::PreProcessImage(unsigned char* img_data, Blob* blob, int width, int height, int channel) {
    float* data_ptr = static_cast<float*>(blob->GetHandle().base);
    if (blob->GetBlobDesc().data_format == DATA_FORMAT_NCHW) {
        int blob_c = blob->GetBlobDesc().dims[1];
        int blob_h = blob->GetBlobDesc().dims[2];
        int blob_w = blob->GetBlobDesc().dims[3];

        if (blob_c != channel) {
            LOGE("input channel not match!\n");
            return TNNERR_INVALID_INPUT;
        }

        if (blob_h != height || blob_w != width) {
            // resize img_data
            printf("\t\tresize from %dx%dx%d to %dx%dx%d\n", height, width, channel, blob_h, blob_w, blob_c);
            unsigned char* img_resized = (unsigned char*)malloc(blob_w * blob_h * blob_c);
            int ret = stbir_resize_uint8(img_data, width, height, 0, img_resized, blob_w, blob_h, 0, channel);
            if (ret == 0) {
                free(img_resized);
                LOGE("resize image falied!\n");
                return TNNERR_INVALID_INPUT;
            }
            ProcessNHWC2NCHW(img_resized, data_ptr, blob_c, blob_h, blob_w, bias_, scale_);
            free(img_resized);
        } else {
            ProcessNHWC2NCHW(img_data, data_ptr, blob_c, blob_h, blob_w, bias_, scale_);
        }

    } else {
        LOGE("The blob data format is not support yet!\n");
        return TNNERR_INVALID_INPUT;
    }

    return TNN_OK;
}

}  // namespace TNN_NS
