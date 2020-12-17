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

#ifndef TNN_SOURCE_TNN_INTERPRETER_NCNN_SERIALIZER_H_
#define TNN_SOURCE_TNN_INTERPRETER_NCNN_SERIALIZER_H_

#include <string.h>
#include <fstream>
#include <string>
#include <typeinfo>
#include "tnn/interpreter/raw_buffer.h"
#include "tnn/core/common.h"
#include "tnn/interpreter/tnn/objseri.h"

namespace TNN_NS {

namespace ncnn {

    typedef union
    {
        struct
        {
            unsigned char f0;
            unsigned char f1;
            unsigned char f2;
            unsigned char f3;
        };
        unsigned int tag;
    } ncnn_header_t;

    static inline size_t AlignSize(size_t sz, int n)
    {
        return (sz + n-1) & -n;
    }

    class Deserializer : TNN_NS::Deserializer {
    public:
        explicit Deserializer(std::istream &is) : TNN_NS::Deserializer(is) {}

        int GetInt() {
            return get_basic_t<int>();
        }

        void GetRaw(RawBuffer &value, size_t w) {

            ncnn_header_t flag_struct;

            _istream.read((char*)&flag_struct,
                static_cast<std::streamsize>(sizeof(ncnn_header_t)));
            if (_istream.eof()) {
                return;
            }
            unsigned int flag = flag_struct.f0 +
                                flag_struct.f1 +
                                flag_struct.f2 +
                                flag_struct.f3;

            size_t read_size;
            DataType data_type;

            if (flag_struct.tag == 0x01306B47)
            {
                // fp16
                read_size = AlignSize(w * sizeof(unsigned short), 4);
                data_type = DATA_TYPE_HALF;
            }
            else if (flag_struct.tag == 0x000D4B38)
            {
                // int8 data
                read_size = AlignSize(w, 4);
                data_type = DATA_TYPE_INT8;
            }
            else if (flag_struct.tag == 0x0002C056)
            {
                // float
                read_size = w * sizeof(float);
                data_type = DATA_TYPE_FLOAT;
            }
            else if (flag !=0)
            {
                // quantized data
                float quantization_value[256];
                _istream.read(reinterpret_cast<char *>(quantization_value),
                    static_cast<std::streamsize>(256 * sizeof(float)));

                size_t index_size = AlignSize(w, 4);
                std::vector<unsigned char> index_array;
                index_array.resize(index_size);

                _istream.read(reinterpret_cast<char *>(index_array.data()),
                    static_cast<std::streamsize>(256 * sizeof(uint8_t)));

                value = RawBuffer(256 * sizeof(float));
                value.SetDataType(DATA_TYPE_FLOAT);

                float* ptr = value.force_to<float *>();
                for (size_t i = 0; i < w; i++)
                {
                    ptr[i] = quantization_value[ index_array[i] ];
                }

                return;
            }
            else if (flag_struct.f0 == 0)
            {
                // float
                read_size = w * sizeof(float);
                data_type = DATA_TYPE_FLOAT;

            }

            value = RawBuffer(static_cast<int>(read_size));
            value.SetDataType(data_type);

            char *buffer = value.force_to<char *>();
            if (_istream.eof())
                return;
            _istream.read(buffer, static_cast<std::streamsize>(read_size));

            return;

        }

        void GetRawSimple(RawBuffer &value, size_t w) {

            size_t read_size = w * sizeof(float);
            value = RawBuffer(static_cast<int>(read_size));
            value.SetDataType(DATA_TYPE_FLOAT);

            char *buffer = value.force_to<char *>();
            if (_istream.eof())
                return;
            _istream.read(buffer, static_cast<std::streamsize>(read_size));

            return;
        }

    private:
        Deserializer &operator=(const Deserializer &);
    };

}  // namespace ncnn

}  // namespace TNN_NS

#endif  // TNN_SOURCE_TNN_INTERPRETER_NCNN_SERIALIZER_H_
