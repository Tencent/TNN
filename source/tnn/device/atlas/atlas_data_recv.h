// Copyright 2019 Tencent. All Rights Reserved

#ifndef TNN_SOURCE_DEVICE_ATLAS_ATLAS_DATA_RECV_H_
#define TNN_SOURCE_DEVICE_ATLAS_ATLAS_DATA_RECV_H_

#include <memory>
#include <string>
#include <vector>
#include "hiaiengine/api.h"
#include "tnn/core/macro.h"
#include "tnn/core/blob.h"

namespace TNN_NS {

class AtlasDataRecv : public hiai::DataRecvInterface {
public:
    AtlasDataRecv();

    enum TransferStatus { TS_NotBegin, TS_InProgress, TS_Complete };

    HIAI_StatusT RecvData(const std::shared_ptr<void>& message);

    int WaitInputBlobMap(int timeout_ms);
    int WaitOutputBlobMap(int timeout_ms);

    BlobMap& GetInputBlobMap();
    BlobMap& GetOutputBlobMap();

private:
    TransferStatus input_blob_status_;
    TransferStatus output_blob_status_;

    BlobMap input_blob_map_;
    BlobMap output_blob_map_;
};

}  // namespace TNN_NS

#endif  // TNN_SOURCE_DEVICE_ATLAS_ATLAS_DATA_RECV_H_
