#ifndef GE_BUFFER_H
#define GE_BUFFER_H

#include <memory>
#include <vector>
#include <string>
#include "detail/attributes_holder.h"

namespace ge {
#ifdef HOST_VISIBILITY
    #define GE_FUNC_HOST_VISIBILITY __attribute__((visibility("default")))
#else
    #define GE_FUNC_HOST_VISIBILITY
#endif
#ifdef DEV_VISIBILITY
    #define GE_FUNC_DEV_VISIBILITY __attribute__((visibility("default")))
#else
    #define GE_FUNC_DEV_VISIBILITY
#endif

using std::shared_ptr;

class GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY Buffer {
public:
    Buffer();
    Buffer(const Buffer& other);

    explicit Buffer(std::size_t bufferSize, std::uint8_t defualtVal = 0);

    ~Buffer() = default;

    Buffer& operator=(const Buffer& other);

    static Buffer CopyFrom(std::uint8_t* data, std::size_t bufferSize);

    const std::uint8_t* GetData() const;
    std::uint8_t* GetData();
    std::size_t GetSize() const;
    void ClearBuffer();

    // for compatibility
    inline const std::uint8_t* data() const{
        return GetData();
    }
    inline std::uint8_t* data(){
        return GetData();
    } //lint !e659
    inline std::size_t size() const{
        return GetSize();
    }
    inline void clear(){
        return ClearBuffer();
    }
    uint8_t operator[](size_t index) const{ //lint !e1022 !e1042
        if(buffer_ != nullptr && index < buffer_->size()){  //lint !e574
            return (uint8_t)(*buffer_)[index];
        }
        return 0xff;
    }
private:
    GeIrProtoHelper<proto::AttrDef> data_;
    std::string* buffer_ = nullptr;

    // create from protobuf obj
    Buffer(const ProtoMsgOwner& protoOnwer, proto::AttrDef* buffer);
    Buffer(const ProtoMsgOwner& protoOnwer, std::string* buffer);

    friend class AttrValueImp;
    friend class Tensor;

};

} // namespace ge

#endif //GE_BUFFER_H
