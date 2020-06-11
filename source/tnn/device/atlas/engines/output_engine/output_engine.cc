// Copyright 2019 Tencent. All Rights Reserved

#include "output_engine.h"
#include <time.h>
#include <condition_variable>
#include <memory>
#include "atlas_utils.h"
#include "hiaiengine/ai_memory.h"
#include "hiaiengine/log.h"

namespace TNN_NS {

HIAI_StatusT OutputEngine::Init(
    const hiai::AIConfig& config,
    const std::vector<hiai::AIModelDescription>& model_desc) {
    HIAI_ENGINE_LOG(HIAI_IDE_INFO, "[OutputEngine] start init!");

    HIAI_ENGINE_LOG(HIAI_IDE_INFO, "[OutputEngine] end init!");
    return HIAI_OK;
}

HIAI_IMPL_ENGINE_PROCESS("OutputEngine", OutputEngine, DT_INPUT_SIZE) {
    HIAI_StatusT ret = HIAI_OK;
    std::shared_ptr<TransferDataType> input_arg =
        std::static_pointer_cast<TransferDataType>(arg0);
    if (nullptr == input_arg) {
        HIAI_ENGINE_LOG(HIAI_IDE_ERROR, "Fail to process invalid message\n");
        return HIAI_ERROR;
    }

    if (input_arg->info.cmd_type == CT_DataTransfer) {
        if (0 == output_count_) {
            output_count_ = input_arg->output_info.output_map.size();
        }
        // Save output data
        if (input_arg->output_info.output_map.find(input_arg->info.name) ==
            input_arg->output_info.output_map.end()) {
            HIAI_ENGINE_LOG(HIAI_IDE_ERROR,
                            "Fail to find output buffer (name: %s)\n",
                            input_arg->info.name);
            return HIAI_ERROR;
        }
        memcpy(reinterpret_cast<void*>(
                   input_arg->output_info.output_map[input_arg->info.name]),
               input_arg->data.get(), input_arg->info.size_in_bytes);

        struct timespec time_stamp;
        clock_gettime(CLOCK_MONOTONIC, &time_stamp);
        double duration_ms =
            (time_stamp.tv_sec - input_arg->output_info.time_s) * 1e3 +
            (time_stamp.tv_nsec - input_arg->output_info.time_ns) * 1e-6;

        if (duration_ms < FORWARD_TIMEOUT) {
            output_count_--;
            if (0 == output_count_) {
                std::condition_variable* cv_ptr =
                    reinterpret_cast<std::condition_variable*>(
                        input_arg->output_info.output_cv_addr);
                cv_ptr->notify_one();
            }
        } else {
            output_count_ = 0;
            HIAI_ENGINE_LOG(HIAI_IDE_ERROR, "Process time out!\n");
        }
        HIAI_ENGINE_LOG(
            HIAI_IDE_ERROR, "Process time: %lf ms   Forward time: %lf ms\n",
            input_arg->output_info.process_duration_ms, duration_ms);

        return HIAI_OK;
    }

    std::shared_ptr<TransferDataType> trans_data =
        std::make_shared<TransferDataType>();
    trans_data->info = input_arg->info;

    if (trans_data->info.cmd_type == CT_InfoQuery) {
        HIAI_ENGINE_LOG(
            HIAI_IDE_ERROR,
            "[OutputEngine] trans_data: dims:[%d %d %d %d] size: %d!\n",
            trans_data->info.dim_info.batch, trans_data->info.dim_info.channel,
            trans_data->info.dim_info.height, trans_data->info.dim_info.width,
            trans_data->info.size_in_bytes);
    }

    ret = SendData(0, "TransferDataType",
                   std::static_pointer_cast<void>(trans_data));
    if (HIAI_OK != ret) {
        HIAI_ENGINE_LOG(HIAI_IDE_ERROR, "OutputEngine send data failed\n");
        return HIAI_ERROR;
    }

    return HIAI_OK;
}

}  // namespace TNN_NS
