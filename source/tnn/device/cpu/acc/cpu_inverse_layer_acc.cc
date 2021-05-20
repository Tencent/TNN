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

#include <cmath>
#include "tnn/device/cpu/acc/cpu_layer_acc.h"
#include "tnn/utils/data_type_utils.h"
#include "tnn/utils/dims_vector_utils.h"

namespace TNN_NS {

DECLARE_CPU_ACC(Inverse, LAYER_INVERSE);

Status CpuInverseLayerAcc::Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    return TNN_OK;
}

Status CpuInverseLayerAcc::Forward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    //see https://en.wikipedia.org/wiki/Invertible_matrix#Inversion_of_2.C3.972_matrices
    //http://rodolphe-vaillant.fr/?e=7

    auto input_dims = inputs[0]->GetBlobDesc().dims;
    if (input_dims.size()<2) {
        return Status(TNNERR_PARAM_ERR, "CpuInverseLayerAcc has invalid input dims");
    }
    
    float *input_data  = (float *)((char *)inputs[0]->GetHandle().base+ inputs[0]->GetHandle().bytes_offset);
    float *output_data = (float *)((char *)outputs[0]->GetHandle().base + outputs[0]->GetHandle().bytes_offset);
    
    const int batch = DimsVectorUtils::Count(input_dims, 0, (int)input_dims.size()-2);
    if (input_dims[input_dims.size()-1] ==2 && input_dims[input_dims.size()-2] ==2) {
        for (int b=0; b<batch; b++) {
            float det = input_data[0]*input_data[3] - input_data[1]*input_data[2];
            float det_inverse = 1.0f / det;
            
            output_data[0] = input_data[3]*det_inverse;
            output_data[1] = -input_data[1]*det_inverse;
            output_data[2] = -input_data[2]*det_inverse;
            output_data[3] = input_data[0]*det_inverse;
            
            input_data += 4;
            output_data += 4;
        }
    } else {
        return Status(TNNERR_PARAM_ERR, "CpuInverseLayerAcc now only support inverse of matrix batchx2x2");
    }
    return TNN_OK;
}

REGISTER_CPU_ACC(Inverse, LAYER_INVERSE);

}  // namespace TNN_NS
