// Copyright 2019 Tencent. All Rights Reserved

#include "inference_engine.h"
#include <time.h>
#include <memory>
#include "atlas_device_utils.h"
#include "atlas_utils.h"
#include "hiaiengine/ai_memory.h"
#include "hiaiengine/c_graph.h"
#include "hiaiengine/data_type.h"
#include "hiaiengine/log.h"

namespace TNN_NS {

int InferenceEngine::ParseConfig(const hiai::AIConfig& config,
                                 hiai::AIModelDescription& model_desc) {
    auto kvcfg = Kvmap(config);
    if (kvcfg.count("model_path") <= 0) {
        HIAI_ENGINE_LOG(HIAI_IDE_ERROR,
                        "[PareseConfig] can't find model in config.");
        return -1;
    }

    if (kvcfg.count("dynamic_aipp") > 0) {
        if (kvcfg["dynamic_aipp"] == "1")
            use_dynamic_aipp_ = true;
    }

    if (kvcfg.count("daipp_swap_rb") > 0) {
        if (kvcfg["daipp_swap_rb"] == "1")
            aipp_swap_rb_ = true;
    }

    if (kvcfg.count("daipp_norm") > 0) {
        if (kvcfg["daipp_norm"] == "1")
            aipp_normalize_ = true;
    }

    std::string model_path = kvcfg["model_path"];
    std::set<char> delims{'\\', '/'};
    std::vector<std::string> path = SplitPath(model_path, delims);
    std::string model_name        = path.back();
    HIAI_ENGINE_LOG(HIAI_IDE_INFO, "model path %s", model_path.c_str());
    HIAI_ENGINE_LOG(HIAI_IDE_INFO, "model name %s", model_name.c_str());
    model_desc.set_path(model_path);
    model_desc.set_name(model_name);
    model_desc.set_key(kvcfg["passcode"]);
    return 0;
}

HIAI_StatusT InferenceEngine::SetDynamicAipp() {
    HIAI_ENGINE_LOG(HIAI_IDE_INFO,
                    "[SetDynamicAipp] start to set dynamic aipp ...\n");
    hiai::AITensorDescription desc =
        hiai::AippDynamicParaTensor::GetDescription(
            std::to_string(input_tensor_dims_[0].n));
    shared_ptr<hiai::IAITensor> tensor =
        hiai::AITensorFactory::GetInstance()->CreateTensor(desc);
    aipp_params_ =
        std::static_pointer_cast<hiai::AippDynamicParaTensor>(tensor);

    hiai::AippInputFormat aipp_input_format = hiai::RGB888_U8;
    aipp_params_->SetInputFormat(aipp_input_format);

    hiai::AippModelFormat aipp_output_format = hiai::MODEL_RGB888_U8;
    if (aipp_swap_rb_) {
        aipp_output_format = hiai::MODEL_BGR888_U8;
        aipp_params_->SetCscParams(aipp_input_format, aipp_output_format,
                                   hiai::JPEG);
    }

    aipp_params_->SetSrcImageSize(input_tensor_dims_[0].w,
                                  input_tensor_dims_[0].h);

    aipp_params_->SetCropParams(true, 0, 0, input_tensor_dims_[0].w,
                                input_tensor_dims_[0].h);
    // set scale
    for (int i = 0; i < input_tensor_dims_[0].n; ++i) {
        aipp_params_->SetPixelVarReci(0.0039216, 0.0039216, 0.0039216, 0, i);
    }

    return HIAI_OK;
}

HIAI_StatusT InferenceEngine::Init(
    const hiai::AIConfig& config,
    const std::vector<hiai::AIModelDescription>& model_desc) {
    HIAI_ENGINE_LOG(HIAI_IDE_INFO, "[InferenceEngine] start init!\n");
    HIAI_StatusT ret = HIAI_OK;
    if (nullptr == model_manager_) {
        model_manager_ = std::make_shared<hiai::AIModelManager>();
    }
    hiai::AIModelDescription model_desc_t;
    if (ParseConfig(config, model_desc_t) != 0) {
        HIAI_ENGINE_LOG(HIAI_IDE_ERROR,
                        "[InferenceEngine] load model desc failed!\n");
        return HIAI_ERROR;
    }

    // init ai model manager
    ret = model_manager_->Init(config, {model_desc_t});
    if (hiai::SUCCESS != ret) {
        HIAI_ENGINE_LOG(HIAI_IDE_ERROR,
                        "[InferenceEngine] ai model manager init failed!\n");
        return HIAI_ERROR;
    }

    // input/output buffer allocation
    ret = model_manager_->GetModelIOTensorDim(
        model_desc_t.name(), input_tensor_dims_, output_tensor_dims_);
    if (ret != hiai::SUCCESS) {
        HIAI_ENGINE_LOG(
            HIAI_IDE_ERROR,
            "[InferenceEngine] hiai ai model manager init failed.\n");
        return HIAI_ERROR;
    }

    // input dims
    if ((!use_dynamic_aipp_ && 1 != input_tensor_dims_.size()) ||
        (use_dynamic_aipp_ && 2 != input_tensor_dims_.size())) {
        HIAI_ENGINE_LOG(
            HIAI_IDE_ERROR,
            "[InferenceEngine] input_tensor_dims_.size() invalid\n");
        for (auto input_dim : input_tensor_dims_) {
            HIAI_ENGINE_LOG(HIAI_IDE_ERROR,
                            "[InferenceEngine] input[] n:%d c:%d h:%d w:%d "
                            "data_type:%d size:%d name:%s\n",
                            input_dim.n, input_dim.c, input_dim.h, input_dim.w,
                            input_dim.data_type, input_dim.size,
                            input_dim.name.c_str());
        }
        return HIAI_ERROR;
    }

    ret = CreatIOTensors(input_tensor_dims_, input_tensor_vec_,
                         input_data_buffer_);
    if (HIAI_OK != ret) {
        HIAI_ENGINE_LOG(HIAI_IDE_ERROR,
                        "[InferenceEngine] creat input tensors failed!\n");
        return HIAI_ERROR;
    }

    ret = CreatIOTensors(output_tensor_dims_, output_tensor_vec_,
                         output_data_buffer_);
    if (HIAI_OK != ret) {
        HIAI_ENGINE_LOG(HIAI_IDE_ERROR,
                        "[InferenceEngine] creat output tensors failed!\n");
        return HIAI_ERROR;
    }

    if (use_dynamic_aipp_) {
        SetDynamicAipp();
    }

    HIAI_ENGINE_LOG(HIAI_IDE_INFO, "[InferenceEngine] end init!\n");
    return HIAI_OK;
}

HIAI_IMPL_ENGINE_PROCESS("InferenceEngine", InferenceEngine, DT_INPUT_SIZE) {
    HIAI_ENGINE_LOG(HIAI_IDE_INFO, "[InferenceEngine] process start\n");
    HIAI_StatusT ret = HIAI_OK;
    std::shared_ptr<TransferDataType> input_arg =
        std::static_pointer_cast<TransferDataType>(arg0);
    if (nullptr == input_arg) {
        HIAI_ENGINE_LOG(HIAI_IDE_ERROR, "Fail to process invalid message\n");
        return HIAI_ERROR;
    }

    if (input_arg->info.cmd_type == CT_DataTransfer) {
        // Process Data Transfer packet
        HIAI_ENGINE_LOG(HIAI_IDE_INFO,
                        "[InferenceEngine] cmd type: CT_DataTransfer\n");

        memcpy(input_data_buffer_[0].get(), input_arg->data.get(),
               input_arg->info.size_in_bytes);

        // set aipp
        if (use_dynamic_aipp_) {
            int ret = model_manager_->SetInputDynamicAIPP(input_tensor_vec_,
                                                          aipp_params_);
            if (ret != 0) {
                HIAI_ENGINE_LOG(
                    HIAI_IDE_ERROR,
                    "[InferenceEngine] set input dynamic aipp failed!\n");
                return HIAI_ERROR;
            }
        }

        struct timespec start;
        struct timespec stop;

        // inference
        hiai::AIContext ai_context;
        clock_gettime(CLOCK_MONOTONIC, &start);
        ret = model_manager_->Process(ai_context, input_tensor_vec_,
                                      output_tensor_vec_, 0);
        clock_gettime(CLOCK_MONOTONIC, &stop);
        if (hiai::SUCCESS != ret) {
            HIAI_ENGINE_LOG(HIAI_IDE_ERROR,
                            "AI Model Manager Process failed\n");
            return HIAI_ERROR;
        }

        double time_taken = (stop.tv_sec - start.tv_sec) * 1e3 +
                            (stop.tv_nsec - start.tv_nsec) * 1e-6;
        input_arg->output_info.process_duration_ms = time_taken;

        // send output data
        for (auto item : output_tensor_vec_) {
            SendOutputData(item, input_arg);
        }

    } else if (input_arg->info.cmd_type == CT_InfoQuery) {
        // Process Info Query packet
        HIAI_ENGINE_LOG(HIAI_IDE_INFO,
                        "[InferenceEngine] cmd type: CT_InfoQuery\n");
        if (input_arg->info.query_type == QT_InputDims) {
            HIAI_ENGINE_LOG(HIAI_IDE_INFO,
                            "[InferenceEngine] query type: QT_InputDims\n");
            for (auto tensor_dim : input_tensor_dims_) {
                SendQueryDimInfo(tensor_dim, QT_InputDims);
            }
            SendTransDataEnd(CT_InfoQuery_End, QT_InputDims);
        } else if (input_arg->info.query_type == QT_OutputDims) {
            HIAI_ENGINE_LOG(HIAI_IDE_INFO,
                            "[InferenceEngine] query type: QT_OutputDims\n");
            for (auto tensor_dim : output_tensor_dims_) {
                SendQueryDimInfo(tensor_dim, QT_OutputDims);
            }
            SendTransDataEnd(CT_InfoQuery_End, QT_OutputDims);
        } else {
            HIAI_ENGINE_LOG(HIAI_IDE_ERROR,
                            "Invalid query parameters failed\n");
            return HIAI_ERROR;
        }
    }

    return HIAI_OK;
}

int InferenceEngine::SendOutputData(
    std::shared_ptr<hiai::IAITensor>& output_tensor,
    std::shared_ptr<TransferDataType>& input_trans_data) {
    HIAI_ENGINE_LOG(HIAI_IDE_INFO, "[InferenceEngine] SendOutputData\n");
    shared_ptr<hiai::AINeuralNetworkBuffer> output =
        std::static_pointer_cast<hiai::AINeuralNetworkBuffer>(output_tensor);
    uint32_t size_in_bytes = output->ByteSizeLong();

    uint8_t* buffer = nullptr;
    HIAI_StatusT get_ret =
        hiai::HIAIMemory::HIAI_DMalloc(size_in_bytes, (void*&)buffer);
    if (HIAI_OK != get_ret || nullptr == buffer) {
        HIAI_ENGINE_LOG(HIAI_IDE_ERROR,
                        "[InferenceEngine] DMalloc buffer error!\n");
        return -1;
    }
    memcpy(buffer, output->GetBuffer(), size_in_bytes);

    std::shared_ptr<TransferDataType> trans_data =
        std::make_shared<TransferDataType>();
    trans_data->info.cmd_type         = CT_DataTransfer;
    trans_data->info.query_type       = QT_None;
    trans_data->info.dim_info.batch   = output->GetNumber();
    trans_data->info.dim_info.channel = output->GetChannel();
    trans_data->info.dim_info.height  = output->GetHeight();
    trans_data->info.dim_info.width   = output->GetWidth();
    trans_data->info.size_in_bytes    = size_in_bytes;
    trans_data->output_info           = input_trans_data->output_info;
    trans_data->data_len              = size_in_bytes;
    trans_data->data.reset(buffer, [](uint8_t*) {});
    strncpy(trans_data->info.name, output->GetName().c_str(), 31);
    trans_data->info.name[31] = '\0';

    HIAI_StatusT ret = SendData(0, "TransferDataType",
                                std::static_pointer_cast<void>(trans_data));
    if (HIAI_OK != ret) {
        HIAI_ENGINE_LOG(HIAI_IDE_ERROR, "InferenceEngine send data failed\n");
        return -1;
    }

    return 0;
}

int InferenceEngine::SendQueryDimInfo(hiai::TensorDimension& dim_info,
                                      QueryType qt) {
    HIAI_ENGINE_LOG(HIAI_IDE_INFO, "[InferenceEngine] SendQueryDimInfo\n");
    uint8_t* buffer_tmp = nullptr;
    int buffer_size     = 16;
    HIAI_StatusT get_ret =
        hiai::HIAIMemory::HIAI_DMalloc(buffer_size, (void*&)buffer_tmp);
    if (HIAI_OK != get_ret || nullptr == buffer_tmp) {
        HIAI_ENGINE_LOG(HIAI_IDE_INFO,
                        "DMalloc buffer error (size=%d) (ret=0x%x)!\n",
                        buffer_size, get_ret);
        return -1;
    }

    uint32_t dim_size      = dim_info.n * dim_info.c * dim_info.h * dim_info.w;
    uint32_t size_in_bytes = dim_size;
    if (dim_info.data_type == hiai::HIAI_DATA_FLOAT) {
        size_in_bytes *= 4;
    } else if (dim_info.data_type == hiai::HIAI_DATA_INT32) {
        size_in_bytes *= 4;
    } else if (dim_info.data_type == hiai::HIAI_DATA_HALF) {
        size_in_bytes *= 2;
    }

    std::shared_ptr<TransferDataType> trans_data =
        std::make_shared<TransferDataType>();
    trans_data->info.cmd_type         = CT_InfoQuery;
    trans_data->info.query_type       = qt;
    trans_data->info.dim_info.batch   = dim_info.n;
    trans_data->info.dim_info.channel = dim_info.c;
    trans_data->info.dim_info.height  = dim_info.h;
    trans_data->info.dim_info.width   = dim_info.w;
    trans_data->info.size_in_bytes    = size_in_bytes;
    trans_data->data_len              = buffer_size;
    trans_data->data.reset(buffer_tmp, [](uint8_t*) {});
    strncpy(trans_data->info.name, dim_info.name.c_str(), 31);
    trans_data->info.name[31] = '\0';

    HIAI_StatusT ret = SendData(0, "TransferDataType",
                                std::static_pointer_cast<void>(trans_data));
    if (HIAI_OK != ret) {
        HIAI_ENGINE_LOG(HIAI_IDE_ERROR, "InferenceEngine send data failed\n");
        return -1;
    }

    return 0;
}

int InferenceEngine::SendTransDataEnd(CommandType ct, QueryType qt) {
    HIAI_ENGINE_LOG(HIAI_IDE_INFO,
                    "[InferenceEngine] SendTransDataEnd (ct %d, qt %d)\n", ct,
                    qt);
    uint8_t* buffer_tmp = nullptr;
    int buffer_size     = 16;
    HIAI_StatusT get_ret =
        hiai::HIAIMemory::HIAI_DMalloc(buffer_size, (void*&)buffer_tmp);
    if (HIAI_OK != get_ret || nullptr == buffer_tmp) {
        HIAI_ENGINE_LOG(HIAI_IDE_INFO,
                        "DMalloc buffer error (size=%d) (ret=0x%x)!\n",
                        buffer_size, get_ret);
        return -1;
    }

    std::shared_ptr<TransferDataType> trans_data =
        std::make_shared<TransferDataType>();
    trans_data->info.cmd_type   = ct;
    trans_data->info.query_type = qt;
    trans_data->data_len        = buffer_size;
    trans_data->data.reset(buffer_tmp, [](uint8_t*) {});

    HIAI_StatusT ret = SendData(0, "TransferDataType",
                                std::static_pointer_cast<void>(trans_data));
    if (HIAI_OK != ret) {
        HIAI_ENGINE_LOG(HIAI_IDE_ERROR, "InferenceEngine send data failed\n");
        return -1;
    }

    return 0;
}

}  // namespace TNN_NS
