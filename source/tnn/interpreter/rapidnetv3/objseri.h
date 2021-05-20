// Copyright 2019 Tencent. All Rights Reserved

#ifndef TNN_SOURCE_TNN_INTERPRETER_RAPIDNETV3_OBJSERI_H_
#define TNN_SOURCE_TNN_INTERPRETER_RAPIDNETV3_OBJSERI_H_

#include <string>
#include <fstream>
#include <string>
#include <typeinfo>
#include "tnn/core/common.h"
#include "tnn/interpreter/raw_buffer.h"
#include "tnn/interpreter/rapidnetv3/encryption/encryption.h"
#include "tnn/interpreter/tnn/objseri.h"

#define FLOAT_16_BIT_MASK 0x80000000
#define INT_8_BIT_MASK 0x40000000
#define INT_32_BIT_MASK 0x20000000
#define INT_64_BIT_MASK 0x10000000


namespace rapidnetv3 {

    enum ModelVersion {
        MV_RPNV1 = 0,
        MV_TNN = 1,
        MV_RPNV3 = 2,
        MV_TNN_V2 = 3
    };

    static const uint32_t g_version_magic_number_tnn = 0x0FABC0002;
    static const uint32_t g_version_magic_number_rapidnet_v3 = 0x0FABC0003;
    static const uint32_t g_version_magic_number_tnn_v2 = 0x0FABC0004;

class Serializer : public TNN_NS::Serializer {
    public:
        Serializer(std::ostream &os) : TNN_NS::Serializer(os), model_version_(MV_RPNV3){}
        Serializer(std::ostream &os, int ver) : TNN_NS::Serializer(os), model_version_(ver){}

        virtual void PutRaw(TNN_NS::RawBuffer &value) {
            int length = value.GetBytesSize();
            auto data_type = (TNN_NS::DataType)value.GetDataType();
            char *buffer = value.force_to<char *>();

            if (MV_RPNV1 == model_version_) {
                int len_with_flag  = length;
                if (TNN_NS::DATA_TYPE_HALF == data_type) {
                    len_with_flag |= FLOAT_16_BIT_MASK;
                } else if (TNN_NS::DATA_TYPE_INT8 == data_type) {
                    len_with_flag |= INT_8_BIT_MASK;
                } else if (TNN_NS::DATA_TYPE_INT32 == data_type) {
                    len_with_flag |= INT_32_BIT_MASK;
                } else if (TNN_NS::DATA_TYPE_INT64 == data_type) {
                    len_with_flag |= INT_64_BIT_MASK;
                }

                PutInt(len_with_flag);
            } else {
                if (MV_TNN == model_version_) {
                    PutInt(g_version_magic_number_tnn);
                } else if (MV_RPNV3 == model_version_) {
                    PutInt(g_version_magic_number_rapidnet_v3);
                } else if (MV_TNN_V2 == model_version_) {
                    PutInt(g_version_magic_number_tnn_v2);
                } else {
                    return;
                }
                PutInt(data_type);
                PutInt(static_cast<int>(length));
            }
            if (length <= 0) {
                return;
            }

            if (MV_RPNV3 == model_version_) {
                TNN_NS::RawBuffer mixed(length);
                char *mixed_buffer = value.force_to<char *>();
                BlurMix(buffer, mixed_buffer, length);
                _ostream.write(reinterpret_cast<char *>(mixed_buffer),
                        static_cast<std::streamsize>(length));
            } else if (MV_TNN_V2 == model_version_) {
                auto dims = value.GetBufferDims();
                PutInt((int)(dims.size()));
                if (dims.size() > 0) {
                    _ostream.write(reinterpret_cast<char *>(dims.data()),
                                   static_cast<std::streamsize>(dims.size() * sizeof(int32_t)));
                }
                if (_ostream.bad()) {
                    return;
                }
                _ostream.write(reinterpret_cast<char *>(buffer), static_cast<std::streamsize>(length));
            } else {
                _ostream.write(reinterpret_cast<char *>(buffer),
                        static_cast<std::streamsize>(length));
            }

            if (_ostream.bad())
                return;
            return;
        }

    private:
        int model_version_;
    };

    class Deserializer : public TNN_NS::Deserializer{
    public:
        Deserializer(std::istream &is) : TNN_NS::Deserializer(is) {}
        
        virtual void GetRaw(TNN_NS::RawBuffer &value) {
            auto pos = _istream.tellg();
            uint32_t magic_number = 0;
            _istream.read((char *)&magic_number, sizeof(uint32_t));
            
            if (magic_number == g_version_magic_number_tnn ||
                magic_number == g_version_magic_number_tnn_v2 ||
                magic_number == g_version_magic_number_rapidnet_v3) {
                auto data_type = (TNN_NS::DataType)GetInt();
                 int length = GetInt();
                 if (length <= 0) {
                     return;
                 }

                 value = TNN_NS::RawBuffer(length);
                 value.SetDataType(data_type);
                 if (magic_number == g_version_magic_number_tnn_v2) {
                     DimsVector dims;
                     int size = GetInt();
                     for (int i = 0; i < size; ++i) {
                         dims.push_back(GetInt());
                     }
                     value.SetBufferDims(dims);
                 }
                 char *buffer = value.force_to<char *>();
                 if (_istream.eof())
                     return;
                 
                if (magic_number == g_version_magic_number_rapidnet_v3) {
                    TNN_NS::RawBuffer mixed(length);
                    char *mixed_buffer = value.force_to<char *>();
                    _istream.read(mixed_buffer, static_cast<std::streamsize>(length));
                    BlurMix(mixed_buffer, buffer, length);
                } else {
                    _istream.read(buffer, static_cast<std::streamsize>(length));
                }
            } else {
                _istream.seekg(pos, std::ios::beg);
                
                int length = static_cast<int>(GetInt());
                TNN_NS::DataType data_type = TNN_NS::DATA_TYPE_FLOAT;
                if ((length & FLOAT_16_BIT_MASK) != 0) {
                    length    = length & (~FLOAT_16_BIT_MASK);
                    data_type = TNN_NS::DATA_TYPE_HALF;
                } else if ((length & INT_8_BIT_MASK) != 0) {
                    length    = length & (~INT_8_BIT_MASK);
                    data_type = TNN_NS::DATA_TYPE_INT8;
                } else if ((length & INT_32_BIT_MASK) != 0) {
                    length    = length & (~INT_32_BIT_MASK);
                    data_type = TNN_NS::DATA_TYPE_INT32;
                } else if ((length & INT_64_BIT_MASK) != 0) {
                    length    = length & (~INT_64_BIT_MASK);
                    data_type = TNN_NS::DATA_TYPE_INT64;
                }
                value = TNN_NS::RawBuffer(length);
                value.SetDataType(data_type);

                char *buffer = value.force_to<char *>();
                if (_istream.eof())
                    return;
                // RawBuffer value(length);
                _istream.read(buffer, static_cast<std::streamsize>(length));
            }
            
            return;
        }

    private:
        Deserializer &operator=(const Deserializer &);
    };

    class Serializable : public TNN_NS::Serializable {
    public:
        Serializable() {}
        virtual ~Serializable() {}

    public:
        virtual void serialize(Serializer &out)    = 0;
        virtual void deserialize(Deserializer &in) = 0;
    };

}  // namespace rapidnetv3



#endif  // TNN_SOURCE_TNN_INTERPRETER_RAPIDNETV3_OBJSERI_H_
