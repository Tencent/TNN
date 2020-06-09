#ifndef _TNN_SERILIZE_20160513_H_
#define _TNN_SERILIZE_20160513_H_
#include "onnx2tnn_prefix.h"

#include <fstream>
#include <string>
#include <string.h>
#include <typeinfo>

namespace parser{
    static const uint32_t g_version_magic_number_tnn = 0x0FABC0002;

    typedef enum {
        // float
        DATA_TYPE_FLOAT = 0,
        // half float
        DATA_TYPE_HALF = 1,
        // int8
        DATA_TYPE_INT8 = 2,
        // int32
        DATA_TYPE_INT32 = 3
    } DataType;
    
    static const int FLOAT_32_BIT_MASK = 0x00000000;
    static const int FLOAT_16_BIT_MASK = 0x80000000;
    static const int BIT_MASK_PLACEHOLDER = 0xf8000000; // front 5 bits
    enum MODEL_DATA_BITS{MODEL_DATA_FLOAT_32, MODEL_DATA_FLOAT_16, MODEL_DATA_INT_8};
    
    class raw_buffer
    {
    public:
        raw_buffer(){
            buff_ = NULL;
            len_ = 0;
            data_bits_ = MODEL_DATA_FLOAT_32;
            _effective_data_bits = 0;
        }
        raw_buffer(long len)
        {
            buff_ = NULL;
            buff_ = alloc_memory(len);
            memset (buff_, 0, len);
            len_ = len;
        }
        raw_buffer(long len, char* buffer)
        {
            buff_ = alloc_memory(len);
            memcpy(buff_, buffer, len);
            len_ = len;
        }
        raw_buffer(const raw_buffer& buf)
        {
            this->len_ = buf.len_;
            this->buff_ = alloc_memory(len_) ;
            if (this->buff_ != NULL){
                memcpy(this->buff_, buf.buff_, len_);
            }
        }
        raw_buffer& operator = (raw_buffer buf)
        {
            this->len_ = buf.len_;
            this->buff_ = alloc_memory(len_);
            if (this->buff_ != NULL){
                memcpy(this->buff_, buf.buff_, len_);
            }
            return *this;
        }
        ~raw_buffer()
        {
            if (buff_ != NULL)
                delete[] buff_;
            buff_ = NULL;
        }
        char* buffer() {
            return buff_;
        }
        void buffer(char* buf, int len){
            if (len > len_){
                return;
            }
            if (buff_ == NULL){
                buff_ = alloc_memory(len_);
            }
            memcpy(buff_, buf, len_);
        }
        void len(int len){
            len_ = len;
        }
        
        void set_effective_data_bits(int effective_data_bits){
            // 0 : fp32 or int8(legacy model)
            // 16: fp16
            // 9: int32
            // 8:int8, 7:int7, 6:int6, 5:int5, 4:int4, 3:int3, 2:int2, 1:int1
            _effective_data_bits = effective_data_bits;
            
            if(_effective_data_bits == 16){
                set_float16();
            }
            else if(_effective_data_bits <= 8){
                set_int8();
            }
        }
        
        int get_effective_data_bits() const{
            // 0 : fp32 or int8(legacy model)
            // 16: fp16
            // 8:int8, 7:int7, 6:int6, 5:int5, 4:int4, 3:int3, 2:int2, 1:int1
            return _effective_data_bits;
        }
        
        void set_float16(){
            data_bits_ = MODEL_DATA_FLOAT_16;
        }
        
        bool is_float16() const{
            return data_bits_ == MODEL_DATA_FLOAT_16;
        }
        
        void set_int8(){
            data_bits_ = MODEL_DATA_INT_8;
        }
        
        bool is_int8(){
            return data_bits_ == MODEL_DATA_INT_8;
        }
        
        MODEL_DATA_BITS get_data_bits() const{
            return data_bits_;
        }
        inline char* alloc_memory(int len) {
            // in convolution_3x3 or other neon code, vld1q_f32 may read extra bytes out of bound, so + 3 * sizeof(float)
            return new char[len + 3 * sizeof(float)];
        }
        
    public:
        template <typename T>T force_to(){
            return reinterpret_cast<T>(buff_);
        }
        long len(){
            return len_;
        }
    private:
        char* buff_;
        int len_;
        MODEL_DATA_BITS data_bits_;
        int _effective_data_bits;
    };
    
    
    class serializer
    {
    public:
        serializer(std::ostream& os) : _ostream(os) {}
        
        void put_bool(bool value) { return put_basic_t<bool>(value); }
        void put_short(short value) { return put_basic_t<short>(value); }
        void put_int(int value) { return put_basic_t<int>(value); }
        void put_string(const std::string& value) { return put_string_t<std::string>(value); }
        
        void put_raw(int length, char* buff, DataType data_type = DATA_TYPE_FLOAT)
        {
            put_int(g_version_magic_number_tnn);
            put_int(static_cast<int>(data_type));
            put_int(static_cast<int>(length));
            if (length <= 0) {
                return;
            }
            _ostream.write(reinterpret_cast<char*>(buff),
                           static_cast<std::streamsize>(length));
            if (_ostream.bad())
                return;
        }
        
        
    protected:
        template<typename T> void put_basic_t(T value);
        template<typename T> void put_string_t(const T& value);
        
    private:
        std::ostream& _ostream;
        serializer& operator =(const serializer&);
    };
    
    template<typename T>
    void serializer::put_basic_t(T value)
    {
        _ostream.write(reinterpret_cast<char*>(&value), sizeof(T));
        if (_ostream.bad())
            return;
    }
    
    template<typename T>
    void serializer::put_string_t(const T& value)
    {
        if (typeid(T) == typeid(std::string)){
            int len = static_cast<int>(value.length() * sizeof(std::string::value_type));
            put_int(len);
            _ostream.write(reinterpret_cast<const char*>(value.data()), len);
            if (_ostream.bad())
                return;
        }
        else
            return;
    }
    
    class deserializer
    {
    public:
        deserializer(std::istream& is) : _istream(is) {}
        
        bool get_bool() { return get_basic_t<bool>(); }
        short get_short() { return get_basic_t<short>(); }
        int get_int() { return get_basic_t<int>(); }
        std::string get_string() { return get_string_t<std::string>(); }
        
        void get_raw(raw_buffer& value)
        {
            auto ignore = get_int();
            
            DataType data_type = (DataType)get_int();
            int length = get_int();
            if (length <= 0) {
                return;
            }
            
            value.len(length);
            
            char* buffer = new char[length];
            if (_istream.eof())
                return;
            //raw_buffer value(length);
            _istream.read(buffer, static_cast<std::streamsize>(length));
            value.buffer(buffer, length);
            delete[] buffer;
            return;
        }
    protected:
        template <typename T> T get_basic_t();
        template<typename T> T get_string_t();
    private:
        std::istream& _istream;
        deserializer& operator =(const deserializer&);
    };
    
    template <typename T> T deserializer::get_basic_t()
    {
        T value = 0;
        if (_istream.eof())
            //throw std::exception("unexpected_eof");
            return value;
        //T value;
        _istream.read(reinterpret_cast<char *>(&value), sizeof(T));
        return value;
    }
    
    template<typename T> T deserializer::get_string_t()
    {
        int len = get_int();
        T value;
        if (typeid(T) == typeid(std::string)){
            
            value.resize(len / sizeof(std::string::value_type));
            if (_istream.eof())
                return value;
            _istream.read(reinterpret_cast<char*>(&value[0]), len);
            //if (_istream.bad())
            //return value;
        }
        return value;
    }
    
    class serializable
    {
    public:
        serializable(){}
        virtual ~serializable(){}
    public:
        virtual void serialize(serializer& out) = 0;
        virtual void deserialize(deserializer& in) = 0;
    };
    
}

#endif /* _TNN_SERILIZE_20160513_H_ */

