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

#include "tnn/device/arm/acc/arm_reshape_layer_acc.h"

#include "tnn/device/arm/arm_common.h"
#include "tnn/utils/bfp16.h"
#include "tnn/utils/data_format_converter.h"
#include "tnn/utils/data_type_utils.h"
#include "tnn/utils/dims_utils.h"

namespace TNN_NS {

bool ArmReshapeLayerAcc::UseNaiveConstantBlobs() {
    return true;
}

Status ArmReshapeLayerAcc::Init(Context *context, LayerParam *param, LayerResource *resource,
                                const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    LOGD("Init Reshape Acc\n");
    Status ret = ArmLayerAcc::Init(context, param, resource, inputs, outputs);
    RETURN_ON_NEQ(ret, TNN_OK);

    ReshapeLayerParam *reshape_param = dynamic_cast<ReshapeLayerParam *>(param_);
    if (!reshape_param) {
        FlattenLayerParam *flatten_param = dynamic_cast<FlattenLayerParam *>(param_);
        if (!flatten_param) {
            LOGE("Error: layer param is null\n");
            return Status(TNNERR_MODEL_ERR, "Error: layer param is null");
        } else {
            reshape_type_ = 0;
        }
    } else {
        reshape_type_ = reshape_param->reshape_type;
    }

    return TNN_OK;
}

template <typename T>
Status ArmReshapeLayerAcc::Exec(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto dims_input  = inputs[0]->GetBlobDesc().dims;
    auto dims_output = outputs[0]->GetBlobDesc().dims;

    char *input_origin  = GetBlobHandlePtr(inputs[0]->GetHandle());
    char *output_origin = GetBlobHandlePtr(outputs[0]->GetHandle());

    auto ic    = DimsFunctionUtils::GetDim(dims_input, 1);;
    auto ic_r4 = ROUND_UP(ic, 4);
    auto ihw   = DimsVectorUtils::Count(dims_input, 2);
    auto oc    = DimsFunctionUtils::GetDim(dims_output, 1);
    auto oc_r4 = ROUND_UP(oc, 4);
    auto ohw   = DimsVectorUtils::Count(dims_output, 2);

    auto input_plane     = ic * ihw;
    auto input_plane_r4  = ic_r4 * ihw;
    auto output_plane    = oc * ohw;
    auto output_plane_r4 = oc_r4 * ohw;

    for (int b = 0; b < dims_input[0]; b++) {
        auto input_data     = reinterpret_cast<T *>(input_origin) + b * input_plane_r4;
        auto workspace_data = reinterpret_cast<T *>(workspace_) + b * input_plane;
        if (reshape_type_ == 0)
            UnpackC4(workspace_data, input_data, ihw, ic);  // NC4HWC4 -> NCHW
        else if (reshape_type_ == 1)
            UnpackC4ToNHWC(workspace_data, input_data, ihw, ic);  // NC4HW4 -> NHWC
        else
            return Status(TNNERR_LAYER_ERR, "Unsupport reshape type");
    }
    for (int b = 0; b < dims_output[0]; b++) {
        auto workspace_data = reinterpret_cast<T *>(workspace_) + b * output_plane;
        auto output_data    = reinterpret_cast<T *>(output_origin) + b * output_plane_r4;
        if (reshape_type_ == 0)
            PackC4(output_data, workspace_data, ohw, oc);  // NCHW -> NC4HW4
        else if (reshape_type_ == 1)
            PackC4FromNHWC(output_data, workspace_data, ohw, oc);  // NHWC -> NC4HW4
        else
            return Status(TNNERR_LAYER_ERR, "Unsupport reshape type");
    }

    return TNN_OK;
}

#if TNN_ARM82
template <>
Status ArmReshapeLayerAcc::Exec<fp16_t>(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto dims_input  = inputs[0]->GetBlobDesc().dims;
    auto dims_output = outputs[0]->GetBlobDesc().dims;

    char *input_origin  = GetBlobHandlePtr(inputs[0]->GetHandle());
    char *output_origin = GetBlobHandlePtr(outputs[0]->GetHandle());
    
    const int ib = dims_input.size() > 0 ? dims_input[0] : 1;
    auto ic    = DimsFunctionUtils::GetDim(dims_input, 1);
    auto ic_r8 = ROUND_UP(ic, 8);
    auto ihw   = DimsVectorUtils::Count(dims_input, 2);
    
    const int ob = dims_output.size() > 0 ? dims_output[0] : 1;
    auto oc    = DimsFunctionUtils::GetDim(dims_output, 1);
    auto oc_r8 = ROUND_UP(oc, 8);
    auto ohw   = DimsVectorUtils::Count(dims_output, 2);

    auto input_plane     = ic * ihw;
    auto input_plane_r8  = ic_r8 * ihw;
    auto output_plane    = oc * ohw;
    auto output_plane_r8 = oc_r8 * ohw;

    for (int b = 0; b < ib; b++) {
        auto input_data     = reinterpret_cast<fp16_t *>(input_origin) + b * input_plane_r8;
        auto workspace_data = reinterpret_cast<fp16_t *>(workspace_) + b * input_plane;
        if (reshape_type_ == 0)
            UnpackC8(workspace_data, input_data, ihw, ic);
        else if (reshape_type_ == 1)
            UnpackC8ToNHWC(workspace_data, input_data, ihw, ic);
        else
            return Status(TNNERR_LAYER_ERR, "Unsupport reshape type");
    }
    for (int b = 0; b < ob; b++) {
        auto workspace_data = reinterpret_cast<fp16_t *>(workspace_) + b * output_plane;
        auto output_data    = reinterpret_cast<fp16_t *>(output_origin) + b * output_plane_r8;
        if (reshape_type_ == 0)
            PackC8(output_data, workspace_data, ohw, oc);
        else if (reshape_type_ == 1)
            PackC8FromNHWC(output_data, workspace_data, ohw, oc);
        else
            return Status(TNNERR_LAYER_ERR, "Unsupport reshape type");
    }

    return TNN_OK;
}
#endif

template <typename T>
Status ArmReshapeLayerAcc::ExecNchw(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto dims_input  = inputs[0]->GetBlobDesc().dims;
    auto dims_output = outputs[0]->GetBlobDesc().dims;

    char *input_origin  = GetBlobHandlePtr(inputs[0]->GetHandle());
    char *output_origin = GetBlobHandlePtr(outputs[0]->GetHandle());

    auto ele_size = DataTypeUtils::GetBytesSize(inputs[0]->GetBlobDesc().data_type);

    if (reshape_type_ == 0) {
        if (input_origin != output_origin) {
            memcpy(output_origin, input_origin, DimsVectorUtils::Count(dims_input) * ele_size);
        }
    } else if (reshape_type_ == 1) {
        if (inputs[0]->GetBlobDesc().data_type == DATA_TYPE_FLOAT) {
            DataFormatConverter::ConvertFromNCHWToNHWC<float>(inputs[0], outputs[0]);
            DataFormatConverter::ConvertFromNHWCToNCHW<float>(outputs[0], nullptr);
        } else if (inputs[0]->GetBlobDesc().data_type == DATA_TYPE_BFP16) {
            DataFormatConverter::ConvertFromNCHWToNHWC<bfp16_t>(inputs[0], outputs[0]);
            DataFormatConverter::ConvertFromNHWCToNCHW<bfp16_t>(outputs[0], nullptr);
        }
#if TNN_ARM82
        else if (inputs[0]->GetBlobDesc().data_type == DATA_TYPE_HALF) {
            DataFormatConverter::ConvertFromNCHWToNHWC<fp16_t>(inputs[0], outputs[0]);
            DataFormatConverter::ConvertFromNHWCToNCHW<fp16_t>(outputs[0], nullptr);
        }
#endif
        else if (inputs[0]->GetBlobDesc().data_type == DATA_TYPE_INT8) {
            DataFormatConverter::ConvertFromNCHWToNHWC<int8_t>(inputs[0], outputs[0]);
            DataFormatConverter::ConvertFromNHWCToNCHW<int8_t>(outputs[0], nullptr);
        } else if (inputs[0]->GetBlobDesc().data_type == DATA_TYPE_INT32) {
            DataFormatConverter::ConvertFromNCHWToNHWC<int32_t>(inputs[0], outputs[0]);
            DataFormatConverter::ConvertFromNHWCToNCHW<int32_t>(outputs[0], nullptr);
        } else {
            return Status(TNNERR_LAYER_ERR, "NO IMPLEMENT FOR int8 reshape, in todo list");
        }
    } else {
        return Status(TNNERR_LAYER_ERR, "Unsupport reshape type");
    }
    return TNN_OK;
}

template <typename T>
Status ArmReshapeLayerAcc::ExecNHWC4(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto dims_input  = inputs[0]->GetBlobDesc().dims;
    auto dims_output = outputs[0]->GetBlobDesc().dims;

    char *input_origin  = GetBlobHandlePtr(inputs[0]->GetHandle());
    char *output_origin = GetBlobHandlePtr(outputs[0]->GetHandle());

    auto ic    = DimsFunctionUtils::GetDim(dims_input, 1);
    auto ic_r4 = ROUND_UP(ic, 4);
    auto ihw   = DimsVectorUtils::Count(dims_input, 2);
    auto oc    = DimsFunctionUtils::GetDim(dims_output, 1);
    auto oc_r4 = ROUND_UP(oc, 4);
    auto ohw   = DimsVectorUtils::Count(dims_output, 2);

    auto input_plane     = ic * ihw;
    auto input_plane_r4  = ihw * ic_r4;
    auto output_plane    = oc * ohw;
    auto output_plane_r4 = ohw * oc_r4;

    for (int b = 0; b < dims_input[0]; b++) {
        auto input_data     = reinterpret_cast<T *>(input_origin) + b * input_plane_r4;
        auto workspace_data = reinterpret_cast<T *>(workspace_) + b * input_plane;
        if (reshape_type_ == 0) {
            UnpackNHWC4(workspace_data, input_data, ihw, ic);  // NHWC4 -> NCHW
        } else if (reshape_type_ == 1) {
            UnpackNHWC4ToNHWC(workspace_data, input_data, ihw, ic);  // NHWC4 -> NHWC
        } else {
            return Status(TNNERR_LAYER_ERR, "Unsupport reshape type");
        }
    }
    for (int b = 0; b < dims_output[0]; b++) {
        auto workspace_data = reinterpret_cast<T *>(workspace_) + b * output_plane;
        auto output_data    = reinterpret_cast<T *>(output_origin) + b * output_plane_r4;
        if (reshape_type_ == 0) {
            PackNHWC4(output_data, workspace_data, ohw, oc);  // NCHW -> NHWC4
        } else if (reshape_type_ == 1) {
            PackNHWC4FromNHWC(output_data, workspace_data, ohw, oc);  // NHWC -> NHWC4
        } else {
            return Status(TNNERR_LAYER_ERR, "Unsupport reshape type");
        }
    }

    return TNN_OK;
}

Status ArmReshapeLayerAcc::DoForward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    if (inputs.size() < 1) {
        LOGE("Error: invalid inputs count\n");
        return Status(TNNERR_LAYER_ERR, "layer's inputs size must >= 2");
    }

    auto in_data_type   = inputs[0]->GetBlobDesc().data_type;
    auto in_data_format = inputs[0]->GetBlobDesc().data_format;
    auto input          = inputs[0];
    auto output         = outputs[0];
    int data_byte_size  = DataTypeUtils::GetBytesSize(output->GetBlobDesc().data_type);
    auto size_in_bytes  = DimsVectorUtils::Count(input->GetBlobDesc().dims) * data_byte_size;
    workspace_          = context_->GetSharedWorkSpace(size_in_bytes);

    if (DATA_FORMAT_NC4HW4 == in_data_format || DATA_FORMAT_NC8HW8 == in_data_format) {
        if (DATA_TYPE_FLOAT == in_data_type) {
            return Exec<float>(inputs, outputs);
        } else if (DATA_TYPE_BFP16 == in_data_type) {
            return Exec<bfp16_t>(inputs, outputs);
        }
#if TNN_ARM82
        else if (DATA_TYPE_HALF == in_data_type) {
            return Exec<fp16_t>(inputs, outputs);
        }
#endif
        else {
            return Status(TNNERR_LAYER_ERR, "NO IMPLEMENT FOR int8 reshape, in todo list");
        }
    } else if (DATA_FORMAT_NCHW == in_data_format) {
        if (DATA_TYPE_FLOAT == in_data_type) {
            return ExecNchw<float>(inputs, outputs);
        } else if (DATA_TYPE_BFP16 == in_data_type) {
            return ExecNchw<bfp16_t>(inputs, outputs);
        }
#if TNN_ARM82
        else if (DATA_TYPE_HALF == in_data_type) {
            return ExecNchw<fp16_t>(inputs, outputs);
        }
#endif
        else {
            return Status(TNNERR_LAYER_ERR, "NO IMPLEMENT FOR int8 reshape, in todo list");
        }
    } else if (DATA_FORMAT_NHWC4 == in_data_format) {
        return ExecNHWC4<int8_t>(inputs, outputs);
    } else {
        return Status(TNNERR_LAYER_ERR, "Unsupported data format in reshape");
    }
}

REGISTER_ARM_ACC(Reshape, LAYER_RESHAPE);
REGISTER_ARM_ACC(Reshape, LAYER_FLATTEN);
REGISTER_ARM_PRECISION_FP16(LAYER_RESHAPE)
REGISTER_ARM_PRECISION_FP16(LAYER_FLATTEN)
REGISTER_ARM_LAYOUT(LAYER_RESHAPE, DATA_FORMAT_NC4HW4)
REGISTER_ARM_LAYOUT(LAYER_FLATTEN, DATA_FORMAT_NC4HW4)
REGISTER_ARM_LAYOUT(LAYER_RESHAPE, DATA_FORMAT_NCHW)
REGISTER_ARM_LAYOUT(LAYER_FLATTEN, DATA_FORMAT_NCHW)

}  // namespace TNN_NS
