// Copyright 2019 Tencent. All Rights Reserved

#include "atlas_common_types.h"

namespace TNN_NS {

template <class Archive>
void serialize(Archive& ar, DimInfo& data) {
    ar(data.batch, data.channel, data.height, data.width);
}

template <class Archive>
void serialize(Archive& ar, TransferDataInfo& data) {
    ar(data.cmd_type, data.query_type, data.dim_info, data.size_in_bytes,
       data.name);
}

template <class Archive>
void serialize(Archive& ar, OutputDataInfo& data) {
    ar(data.output_map, data.output_cv_addr, data.time_s, data.time_ns,
       data.process_duration_ms);
}

template <class Archive>
void serialize(Archive& ar, TransferDataType& data) {
    ar(data.info, data.output_info, data.data_len, data.data);
}

template <class Archive>
void serialize(Archive& ar, DvppTransDataType& data) {
    ar(data.output_info, data.b_info, data.img_data);
}

void GetTransferDataTypeSearPtr(void* input_ptr, std::string& ctrl_str,
                                uint8_t*& data_ptr, uint32_t& data_len) {
    if (input_ptr == nullptr) {
        return;
    }
    TransferDataType* trans_data = static_cast<TransferDataType*>(input_ptr);

    std::ostringstream output_stream;
    cereal::PortableBinaryOutputArchive archive(output_stream);
    archive(*trans_data);
    ctrl_str = output_stream.str();

    data_ptr = trans_data->data.get();
    data_len = trans_data->data_len;
}

std::shared_ptr<void> GetTransferDataTypeDearPtr(const char* ctrl_ptr,
                                                 const uint32_t& ctr_len,
                                                 const uint8_t* data_ptr,
                                                 const uint32_t& data_len) {
    if (ctrl_ptr == nullptr) {
        return nullptr;
    }
    std::shared_ptr<TransferDataType> data_handle =
        std::make_shared<TransferDataType>();

    std::istringstream input_stream;
    input_stream.str(std::string(ctrl_ptr, ctr_len));
    cereal::PortableBinaryInputArchive archive(input_stream);
    archive(*data_handle);

    data_handle->data.reset((uint8_t*)data_ptr, hiai::Graph::ReleaseDataBuffer);

    return std::static_pointer_cast<void>(data_handle);
}

void GetDvppTransDataTypeSearPtr(void* input_ptr, std::string& ctrl_str,
                                 uint8_t*& data_ptr, uint32_t& data_len) {
    if (input_ptr == nullptr) {
        return;
    }
    DvppTransDataType* trans_data = static_cast<DvppTransDataType*>(input_ptr);

    std::ostringstream output_stream;
    cereal::PortableBinaryOutputArchive archive(output_stream);
    archive(*trans_data);
    ctrl_str = output_stream.str();

    data_ptr = trans_data->img_data.data.get();
    data_len = trans_data->img_data.size;
}

std::shared_ptr<void> GetDvppTransDataTypeDearPtr(const char* ctrl_ptr,
                                                  const uint32_t& ctr_len,
                                                  const uint8_t* data_ptr,
                                                  const uint32_t& data_len) {
    if (ctrl_ptr == nullptr) {
        return nullptr;
    }
    std::shared_ptr<DvppTransDataType> data_handle =
        std::make_shared<DvppTransDataType>();

    std::istringstream input_stream;
    input_stream.str(std::string(ctrl_ptr, ctr_len));
    cereal::PortableBinaryInputArchive archive(input_stream);
    archive(*data_handle);

    data_handle->img_data.data.reset((uint8_t*)data_ptr,
                                     hiai::Graph::ReleaseDataBuffer);

    return std::static_pointer_cast<void>(data_handle);
}

HIAI_REGISTER_SERIALIZE_FUNC("TransferDataType", TransferDataType,
                             GetTransferDataTypeSearPtr,
                             GetTransferDataTypeDearPtr);
HIAI_REGISTER_SERIALIZE_FUNC("DvppTransDataType", DvppTransDataType,
                             GetDvppTransDataTypeSearPtr,
                             GetDvppTransDataTypeDearPtr);

}  // namespace TNN_NS
