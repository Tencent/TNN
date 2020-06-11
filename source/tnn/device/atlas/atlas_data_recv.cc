// Copyright 2019 Tencent. All Rights Reserved

#include "atlas_data_recv.h"
#include "atlas_common_types.h"
#include "tnn/utils/dims_vector_utils.h"
#include "unistd.h"

namespace TNN_NS {

AtlasDataRecv::AtlasDataRecv() {
    input_blob_status_  = TS_NotBegin;
    output_blob_status_ = TS_NotBegin;

    input_blob_map_.clear();
    output_blob_map_.clear();
}

HIAI_StatusT AtlasDataRecv::RecvData(const std::shared_ptr<void>& message) {
    std::shared_ptr<TransferDataType> data =
        std::static_pointer_cast<TransferDataType>(message);

    LOGD("RecvData: cmd_type: %d\n", data->info.cmd_type);
    if (data->info.cmd_type == CT_InfoQuery) {
        if (data->info.query_type == QT_InputDims &&
            input_blob_status_ != TS_Complete) {
            input_blob_status_ = TS_InProgress;

            // add input blob
            BlobDesc desc;
            desc.data_format = DATA_FORMAT_NCHW;
            desc.name        = data->info.name;
            desc.dims.push_back(data->info.dim_info.batch);
            desc.dims.push_back(data->info.dim_info.channel);
            desc.dims.push_back(data->info.dim_info.height);
            desc.dims.push_back(data->info.dim_info.width);
            int bytes_per_elem =
                data->info.size_in_bytes / DimsVectorUtils::Count(desc.dims);
            if (bytes_per_elem == 1) {
                desc.data_type = DATA_TYPE_INT8;
            } else if (bytes_per_elem == 2) {
                desc.data_type = DATA_TYPE_HALF;
            } else if (bytes_per_elem == 4) {
                desc.data_type = DATA_TYPE_FLOAT;
            }
            BlobHandle handle;
            input_blob_map_[data->info.name] = new Blob(desc, handle);
        } else if (data->info.query_type == QT_OutputDims &&
                   output_blob_status_ != TS_Complete) {
            output_blob_status_ = TS_InProgress;

            // add output blob
            BlobDesc desc;
            desc.data_format = DATA_FORMAT_NCHW;
            desc.name        = data->info.name;
            desc.dims.push_back(data->info.dim_info.batch);
            desc.dims.push_back(data->info.dim_info.channel);
            desc.dims.push_back(data->info.dim_info.height);
            desc.dims.push_back(data->info.dim_info.width);
            int bytes_per_elem =
                data->info.size_in_bytes / DimsVectorUtils::Count(desc.dims);
            if (bytes_per_elem == 1) {
                desc.data_type = DATA_TYPE_INT8;
            } else if (bytes_per_elem == 2) {
                desc.data_type = DATA_TYPE_HALF;
            } else if (bytes_per_elem == 4) {
                desc.data_type = DATA_TYPE_FLOAT;
            }
            BlobHandle handle;
            output_blob_map_[data->info.name] = new Blob(desc, handle);
        }
    } else if (data->info.cmd_type == CT_InfoQuery_End) {
        if (data->info.query_type == QT_InputDims) {
            input_blob_status_ = TS_Complete;
        } else if (data->info.query_type == QT_OutputDims) {
            output_blob_status_ = TS_Complete;
        }
    }

    return HIAI_OK;
}

int AtlasDataRecv::WaitInputBlobMap(int timeout_ms) {
    int count = timeout_ms * 1000 / 100;

    while (count > 0) {
        if (input_blob_status_ == TS_Complete) {
            input_blob_status_ = TS_NotBegin;
            return 0;
        }
        usleep(100);
        count--;
    }

    return -1;
}

int AtlasDataRecv::WaitOutputBlobMap(int timeout_ms) {
    int count = timeout_ms * 1000 / 100;

    while (count > 0) {
        if (output_blob_status_ == TS_Complete) {
            return 0;
        }
        usleep(100);
        count--;
    }

    return -1;
}

BlobMap& AtlasDataRecv::GetInputBlobMap() {
    return input_blob_map_;
}

BlobMap& AtlasDataRecv::GetOutputBlobMap() {
    return output_blob_map_;
}

}  // namespace TNN_NS
