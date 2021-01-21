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

#include "tnn/layer/base_layer.h"
#include "tnn/utils/dims_vector_utils.h"

namespace TNN_NS {
DECLARE_LAYER(MatMul, LAYER_MATMUL);

Status MatMulLayer::InferOutputDataType() {
    return BaseLayer::InferOutputDataType();
}

// @brief matmul op to MatMul matrix_a_dims and matrix_b_dims
//    see https://pytorch.org/docs/stable/generated/torch.matmul.html?highlight=matmul#torch.matmul
//    Matrix product of two tensors.
//
//    The behavior depends on the dimensionality of the tensors as follows:
//
//    If both tensors are 1-dimensional, the dot product (scalar) is returned.
//
//    If both arguments are 2-dimensional, the matrix-matrix product is returned.
//
//    If the first argument is 1-dimensional and the second argument is 2-dimensional, a 1 is prepended to its dimension for the purpose of the matrix multiply. After the matrix multiply, the prepended dimension is removed.
//
//    If the first argument is 2-dimensional and the second argument is 1-dimensional, the matrix-vector product is returned.
//
//    If both arguments are at least 1-dimensional and at least one argument is N-dimensional (where N > 2), then a batched matrix multiply is returned. If the first argument is 1-dimensional, a 1 is prepended to its dimension for the purpose of the batched matrix multiply and removed after. If the second argument is 1-dimensional, a 1 is appended to its dimension for the purpose of the batched matrix multiple and removed after. The non-matrix (i.e. batch) dimensions are broadcasted (and thus must be broadcastable). For example, if input is a (j \times 1 \times n \times m)(j×1×n×m) tensor and other is a (k \times m \times p)(k×m×p) tensor, out will be an (j \times k \times n \times p)(j×k×n×p) tensor.
DimsVector CalculateOutputDim(DimsVector matrix_a_dims, DimsVector matrix_b_dims) {
    DimsVector output_dims;
    bool squeeze_matrix_a = false;
    bool squeeze_matrix_b = false;
    if (matrix_a_dims.size() == 1) {
        matrix_a_dims.insert(matrix_a_dims.begin(), 1);
        squeeze_matrix_a = true;
    }
    if (matrix_b_dims.size() == 1) {
        matrix_b_dims.push_back(1);
        squeeze_matrix_b = true;
    }
    if (matrix_a_dims.size() == 2 && matrix_b_dims.size() == 2) {
        output_dims = {matrix_a_dims[0], matrix_b_dims[1]};
    } else if (matrix_a_dims.size() == 2 && matrix_b_dims.size() > 2) {
        if (matrix_a_dims[1] == matrix_b_dims[matrix_b_dims.size() - 2]) {
            output_dims                           = matrix_b_dims;
            output_dims[matrix_b_dims.size() - 2] = matrix_a_dims[matrix_a_dims.size() - 2];
        } else {
            LOGE("MatMul get wrong matrix_a or matrix_b\n");
            assert(-1);
        }
    } else if (matrix_a_dims.size() > 2 && matrix_b_dims.size() == 2) {
        if (matrix_a_dims.back() == matrix_b_dims[matrix_b_dims.size() - 2]) {
            output_dims                           = matrix_a_dims;
            output_dims[matrix_a_dims.size() - 1] = matrix_b_dims.back();
        } else {
            LOGE("MatMul get wrong matrix_a or matrix_b\n");
            assert(-1);
        }
    } else if (matrix_a_dims.size() > 2 && matrix_b_dims.size() > 2) {
        // check matrix_a and matrix_b
        if (matrix_a_dims.back() != matrix_b_dims[matrix_b_dims.size() - 2]) {
            LOGE("MatMul get wrong matrix_a or matrix_b\n");
            assert(-1);
        }
        output_dims = matrix_a_dims.size() >= matrix_b_dims.size() ? matrix_a_dims : matrix_b_dims;
        output_dims[output_dims.size() - 2] = matrix_a_dims[matrix_a_dims.size() - 2];
        output_dims[output_dims.size() - 1] = matrix_b_dims[matrix_b_dims.size() - 1];

        int count = matrix_a_dims.size() <= matrix_b_dims.size() ? matrix_a_dims.size() : matrix_b_dims.size();
        for (int i = count - 1 - 2; i >= 0; --i) {
            if (matrix_a_dims[i] != 1 && matrix_b_dims[i] != 1 && matrix_a_dims[i] != matrix_b_dims[i]) {
                LOGE("MatMul get wrong matrix_a or matrix_b\n");
                assert(-1);
            } else {
                output_dims[i] = matrix_a_dims[i] >= matrix_b_dims[i] ? matrix_a_dims[i] : matrix_b_dims[i];
            }
        }
    }
    if (squeeze_matrix_a && output_dims[output_dims.size() - matrix_a_dims.size()] == 1) {
        output_dims.erase(output_dims.end() - matrix_a_dims.size());
    }
    if (squeeze_matrix_b && output_dims[output_dims.size() - 1] == 1) {
        output_dims.erase(output_dims.end() - 1);
    }
    return output_dims;
}

Status MatMulLayer::InferOutputShape(bool ignore_error) {
    auto status = BaseLayer::InferOutputShape(ignore_error);
    RETURN_ON_NEQ(status, TNN_OK);
    
    auto param    = dynamic_cast<MatMulLayerParam*>(param_);
    auto resource = dynamic_cast<MatMulLayerResource*>(resource_);
    DimsVector matrix_a_dims;
    DimsVector matrix_b_dims;
    if (input_blobs_.size() == 1) {
        if (param->weight_position == 0) {
            matrix_a_dims = resource->weight.GetBufferDims();
            matrix_b_dims = input_blobs_[0]->GetBlobDesc().dims;
        } else if (param->weight_position == 1) {
            matrix_a_dims = input_blobs_[0]->GetBlobDesc().dims;
            matrix_b_dims = resource->weight.GetBufferDims();
        } else {
            return Status(TNNERR_INVALID_MODEL, "MatMul input size is error");
        }
    } else if (input_blobs_.size() == 2) {
        matrix_a_dims = input_blobs_[0]->GetBlobDesc().dims;
        matrix_b_dims = input_blobs_[1]->GetBlobDesc().dims;
    } else {
        return Status(TNNERR_INVALID_MODEL, "MatMul input size is error");
    }
    param->matrix_a_dims = matrix_a_dims;
    param->matrix_b_dims = matrix_b_dims;

    auto output_dims = CalculateOutputDim(matrix_a_dims, matrix_b_dims);
    output_blobs_[0]->GetBlobDesc().dims = output_dims;
    return TNN_OK;
}

REGISTER_LAYER(MatMul, LAYER_MATMUL);

}  // namespace TNN_NS
