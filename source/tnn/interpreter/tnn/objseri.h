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

#ifndef TNN_SOURCE_TNN_INTERPRETER_TNN_OBJSERI_H_
#define TNN_SOURCE_TNN_INTERPRETER_TNN_OBJSERI_H_

#include <string>
#include <fstream>
#include <string>
#include <typeinfo>
#include "tnn/core/common.h"
#include "tnn/interpreter/raw_buffer.h"

#define BLOB_SCALE_SUFFIX "_scale_data_"


namespace TNN_NS {
    static const uint32_t g_version_magic_number = 0x0FABC0002;
    static const uint32_t g_version_magic_number_v2 = 0x0FABC0004;

    class Serializer {
    public:
        explicit Serializer(std::ostream &os) : _ostream(os) {}

        void PutBool(bool value) {
            return put_basic_t<bool>(value);
        }
        void PutShort(short value) {
            return put_basic_t<short>(value);
        }
        void PutInt(int value) {
            return put_basic_t<int>(value);
        }
        void PutString(const std::string &value) {
            return PutString_t<std::string>(value);
        }

        virtual void PutRaw(TNN_NS::RawBuffer &value) {
            int length = value.GetBytesSize();
            auto data_type = (TNN_NS::DataType)value.GetDataType();
            DimsVector  dims  = value.GetBufferDims();
            char *buffer = value.force_to<char *>();
            PutRaw(length, buffer, dims, data_type);
        }
        
        void PutRaw(int length, char* buffer, std::vector<int> dims, DataType data_type = DATA_TYPE_FLOAT)
        {
#ifdef TNN_V2
            PutInt(g_version_magic_number_v2);
            PutInt(data_type);
            PutInt(static_cast<int>(length));
            if (length <= 0) {
                return;
            }
           
            PutInt((int)(dims.size()));
            if (dims.size() > 0) {
                _ostream.write(reinterpret_cast<char *>(dims.data()),
                               static_cast<std::streamsize>(dims.size() * sizeof(int32_t)));
            }
#else
            PutInt(g_version_magic_number);
            PutInt(data_type);
            PutInt(static_cast<int>(length));
#endif
            if (_ostream.bad())
                return;
 
            _ostream.write(reinterpret_cast<char *>(buffer),
                           static_cast<std::streamsize>(length));
            return;
        }


    protected:
        std::ostream &_ostream;
        
        template <typename T>
        void put_basic_t(T value);
        template <typename T>
        void PutString_t(const T &value);

    private:
        Serializer &operator=(const Serializer &);
    };

    template <typename T>
    void Serializer::put_basic_t(T value) {
        _ostream.write(reinterpret_cast<char *>(&value), sizeof(T));
        if (_ostream.bad())
            return;
    }

    template <typename T>
    void Serializer::PutString_t(const T &value) {
        if (typeid(T) == typeid(std::string)) {
            int len = static_cast<int>(value.length() *
                                       sizeof(std::string::value_type));
            PutInt(len);
            _ostream.write(reinterpret_cast<const char *>(value.data()), len);
            if (_ostream.bad())
                return;
        } else
            return;
    }

    class Deserializer {
    public:
        explicit Deserializer(std::istream &is) : _istream(is) {}

        bool GetBool() {
            return get_basic_t<bool>();
        }
        short GetShort() {
            return get_basic_t<short>();
        }
        int GetInt() {
            return get_basic_t<int>();
        }
        std::string GetString() {
            return get_string_t<std::string>();
        }

        virtual void GetDims(std::vector<int>& dims) {
            auto magic_number = GetInt();
            auto data_type = (TNN_NS::DataType)GetInt();
            int size = GetInt();
            if (size <= 0) {
                return;
            }
            for (int i = 0; i < size; ++i) {
                dims.push_back(GetInt());
            }
        }

        virtual void GetRaw(TNN_NS::RawBuffer &value) {
            auto magic_number  = static_cast<uint32_t>(GetInt());
            auto data_type = (TNN_NS::DataType)GetInt();
            int length = GetInt();
            if (length <= 0) {
                return;
            }

            DimsVector dims;
            if(magic_number == g_version_magic_number_v2) {
                int size = GetInt();
                for (int i = 0; i < size; ++i) {
                    dims.push_back(GetInt());
                }
            }
 
            value = TNN_NS::RawBuffer(length);
            value.SetDataType(data_type);
            value.SetBufferDims(dims);

            char *buffer = value.force_to<char *>();
            if (_istream.eof())
                return;
            
            _istream.read(buffer, static_cast<std::streamsize>(length));
            return;
        }

    protected:
        std::istream &_istream;
        
        template <typename T>
        T get_basic_t();
        template <typename T>
        T get_string_t();

    private:
        Deserializer &operator=(const Deserializer &);
    };

    template <typename T>
    T Deserializer::get_basic_t() {
        T value = 0;
        if (_istream.eof())
            // throw std::exception("unexpected_eof");
            return value;
        // T value;
        _istream.read(reinterpret_cast<char *>(&value), sizeof(T));
        return value;
    }

    template <typename T>
    T Deserializer::get_string_t() {
        int len = GetInt();
        T value;
        if (typeid(T) == typeid(std::string)) {
            value.resize(len / sizeof(std::string::value_type));
            if (_istream.eof())
                return value;
            _istream.read(reinterpret_cast<char *>(&value[0]), len);
            // if (_istream.bad())
            // return value;
        }
        return value;
    }

    class Serializable {
    public:
        Serializable() {}
        virtual ~Serializable() {}

    public:
        virtual void serialize(Serializer &out)    = 0;
        virtual void deserialize(Deserializer &in) = 0;
    };

}  // namespace TNN_NS



#endif  // TNN_SOURCE_TNN_INTERPRETER_TNN_OBJSERI_H_
