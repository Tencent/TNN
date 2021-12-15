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

#include "tnn/device/cpu/acc/compute/compute_elewise.h"
#include "tnn/layer/base_layer.h"
#include "tnn/utils/dims_vector_utils.h"
#include "tnn/utils/naive_compute.h"
#include "tnn/utils/string_utils_inner.h"

namespace TNN_NS {

DECLARE_LAYER(Einsum, LAYER_EINSUM);

inline int count(std::vector<int> dimes, int start_axis) {
    const int end_axis = int(dimes.size());
    ASSERT(start_axis <= end_axis);
    ASSERT(start_axis >= 0);
    ASSERT(end_axis >= 0);
    int count = 1;
    for (int i = start_axis; i < end_axis; ++i) {
        count *= dimes[i];
    }
    return count;
};

DimsVector CalPermuteOutputShape(const DimsVector &input_dims, const std::vector<int> &orders) {
    auto output_dims    = input_dims;
    const int dims_size = input_dims.size();
    for (int i = 0; i < dims_size; i++) {
        output_dims[i] = input_dims[orders[i]];
    }

    return output_dims;
}

DimsVector CalSqueezeOutputShape(const DimsVector &input_dims, const int axis) {
    auto output_dims = input_dims;
    output_dims.erase(output_dims.begin() + axis);

    return output_dims;
}

DimsVector CalSumOutputShape(const DimsVector &input_dims, const int axis) {
    auto output_dims = input_dims;
    output_dims.erase(output_dims.begin() + axis);

    return output_dims;
}

DimsVector CalMulOutputShape(const DimsVector &input_dims_a, const DimsVector &input_dims_b) {
    return DimsVectorUtils::Max(input_dims_a, input_dims_b);
}

DimsVector CalFlattenOutputShape(const DimsVector &input_dims) {
    return {DimsVectorUtils::Count(input_dims)};
}

DimsVector CalDotOutputShape() {
    return {1};
}

Status EinsumLayer::InferOutputDataType() {
    return BaseLayer::InferOutputDataType();
}

Status EinsumLayer::InferOutputShape(bool ignore_error) {
    BaseLayer::InferOutputShape(ignore_error);

    auto inputs  = input_blobs_;
    auto outputs = output_blobs_;

    auto param = dynamic_cast<EinsumLayerParam *>(param_);
    if (!param) {
        return Status(TNNERR_MODEL_ERR, "Error: EinsumLayerParam is nil");
    }

    param->perm_shapes.clear();
    param->dim_last_op.clear();
    param->operand_dims.clear();
    param->has_zero_size_dim = false;
    const auto equation    = param->equation;
    constexpr int ELLIPSIS = '.';

    // Find arrow (->) to split equation into lhs and rhs
    const auto arrow_pos = equation.find("->");
    const auto lhs       = equation.substr(0, arrow_pos);

    const auto num_ops = inputs.size();

    // Convert labels for input operands into an index in [0, 25] and store
    // them in op_labels for each operand along with ELLIPSIS if present.
    std::vector<std::vector<int>> op_labels(num_ops);
    bool found_ell      = false;
    std::size_t curr_op = 0;
    for (auto i = decltype(lhs.length()){0}; i < lhs.length(); ++i) {
        switch (lhs[i]) {
            case ' ':
                // Ignore spaces
                break;

            case '.':
                if (found_ell) {
                    const std::string message = "Error: einsum() found \'.\' for operand " + ToString(curr_op) +
                                                " for which an ellipsis was already found";
                    return Status(TNNERR_MODEL_ERR, message);
                }
                if (!(i + 2 < lhs.length() && lhs[++i] == '.' && lhs[++i] == '.')) {
                    const std::string message = "einsum() found \'.\' for operand " + ToString(curr_op) +
                                                " that is not part of any ellipsis";
                    return Status(TNNERR_MODEL_ERR, message);
                }
                op_labels[curr_op].push_back(ELLIPSIS);
                found_ell = true;
                break;

            case ';':
                // Move onto next operand
                ++curr_op;
                if (curr_op >= num_ops) {
                    return Status(TNNERR_MODEL_ERR,
                                  "einsum() fewer operands were provided than specified in the equation");
                }
                found_ell = false;
                break;

            default:
                // Parse label
                if (lhs[i] < 'a' && lhs[i] > 'z') {
                    const std::string message = "einsum() operand subscript must be in range [a, z] but found " +
                                                ToString(lhs[i]) + " for operand " + ToString(curr_op);
                    return Status(TNNERR_MODEL_ERR, message);
                }
                // Convert label to index in [0, 25] and store
                op_labels[curr_op].push_back(lhs[i] - 'a');
        }
    }

    if (curr_op != num_ops - 1) {
        return Status(TNNERR_MODEL_ERR, "einsum() more operands were provided than specified in the equation");
    }

    // Labels must be within [a, z].
    constexpr int TOTAL_LABELS = 'z' - 'a' + 1;
    std::vector<int> label_count(TOTAL_LABELS, 0);

    // The maximum number of dimensions covered by any ellipsis, needed when
    // unsqueezing missing dimensions from operands to permute and broadcast
    int ell_num_dim = 0;

    // Compute label frequency and number of dimensions covered by ellipsis
    // We do this after parsing labels to make it more readable and simpler
    // to compute the number of dimensions covered by ellipsis.
    for (int i = 0; i < num_ops; i++) {
        const auto operand_dims = inputs[i]->GetBlobDesc().dims;
        const auto labels       = op_labels[i];
        const int ndims         = operand_dims.size();
        int nlabels             = labels.size();
        bool has_ellipsis       = false;

        for (const auto &label : labels) {
            if (label == ELLIPSIS) {
                --nlabels;
                has_ellipsis = true;
                ell_num_dim  = std::max(ell_num_dim, ndims - nlabels);
            } else {
                ++label_count[label];
            }
        }

        if (!(has_ellipsis ? nlabels <= ndims : nlabels == ndims)) {
            const std::string message = "einsum() the number of subscripts in the equation (" +
                                        ToString(nlabels) +
                                        (has_ellipsis ? ") is more than the number of dimensions ("
                                                      : ") does not match the number of dimensions (") +
                                        ToString(ndims) + ") for operand " + ToString(i) +
                                        (has_ellipsis ? "" : " and no ellipsis was given");

            return Status(TNNERR_MODEL_ERR, message);
        }
    }

    // We want to align the dimensions of every input tensor to have
    // shape out_dims + sum_dims. For this, we create a mapping of label
    // to index into the permuted shape.
    std::vector<int> label_perm_index(TOTAL_LABELS, -1);

    // Current index in the permuted shape
    int perm_index = 0;

    // Start index of ellipsis dimensions in the permuted shape
    int ell_index = 0;
    found_ell     = false;

    if (arrow_pos == std::string::npos) {
        // Implicit output is ellipsis (...) + labels seen only once
        perm_index = ell_num_dim;
        found_ell  = true;
        for (int label = 0; label < TOTAL_LABELS; label++) {
            if (label_count[label] == 1) {
                label_perm_index[label] = perm_index++;
            }
        }
    } else {
        // Parse explicit output
        const auto rhs = equation.substr(arrow_pos + 2);
        for (auto i = decltype(rhs.length()){0}; i < rhs.length(); ++i) {
            switch (rhs[i]) {
                case ' ':
                    // Ignore spaces
                    break;

                case '.':
                    if (found_ell) {
                        return Status(TNNERR_MODEL_ERR,
                                      "einsum() found \'.\' for output but an ellipsis (...) was already found");
                    }
                    if (!(i + 2 < rhs.length() && rhs[++i] == '.' && rhs[++i] == '.')) {
                        return Status(TNNERR_MODEL_ERR,
                                      "einsum() found \'.\' for output that is not part of any ellipsis (...)");
                    }
                    ell_index = perm_index;
                    perm_index += ell_num_dim;
                    found_ell = true;
                    break;

                default:
                    if (rhs[i] < 'a' && rhs[i] > 'z') {
                        const std::string message = "einsum() subscripts must be in range [a, z] but found " +
                                                    ToString(rhs[i]) + " for the output";
                        return Status(TNNERR_MODEL_ERR, message);
                    }
                    const auto label = rhs[i] - 'a';
                    if (!(label_count[label] > 0 && label_perm_index[label] == -1)) {
                        const std::string message =
                            "einsum() output subscript " + ToString(rhs[i]) +
                            (label_perm_index[label] > -1 ? " appears more than once in the output"
                                                          : " does not appear in the equation for any input operand");
                        return Status(TNNERR_MODEL_ERR, message);
                    }
                    label_perm_index[label] = perm_index++;
            }
        }
    }

    // Save output size before adding contraction dims (dims to sum out)
    const int out_size = perm_index;
    param->out_size = out_size;

    // If ellipsis is not part of the output, add to contraction dimensions
    if (!found_ell) {
        ell_index = perm_index;
        perm_index += ell_num_dim;
    }

    // Add contraction labels (labels not present in output)
    for (int label = 0; label < TOTAL_LABELS; label++) {
        if (label_count[label] > 0 && label_perm_index[label] == -1) {
            label_perm_index[label] = perm_index++;
        }
    }

    // Here we unsqueeze missing dimensions to make all operands have the same
    // number of dimensions. We take diagonals for repeated labels within the
    // same operand. Finally we permute the operands to align dimensions as
    // per the perm_out_index we computed above.
    std::vector<DimsVector> permuted_operands_dims;
    for (int i = 0; i < num_ops; i++) {
        std::vector<int> perm_shape(perm_index, -1);
        std::vector<int> label_dim(TOTAL_LABELS, -1);
        TNN_NS::DimsVector operand_dims(inputs[i]->GetBlobDesc().dims);
        const auto labels         = op_labels[i];
        const auto original_sizes = operand_dims;

        std::size_t j = 0;
        for (const auto &label : labels) {
            if (label == ELLIPSIS) {
                // Add missing dimensions covered by the ellipsis
                const int num_missing_dim = ell_num_dim - (original_sizes.size() - labels.size() + 1);
                for (int k = 0; k < num_missing_dim; k++) {
                    // unsqueeze
                    operand_dims.insert(operand_dims.begin() + j, 1);
                }
                for (int k = 0; k < ell_num_dim; k++) {
                    perm_shape[ell_index + k] = j++;
                }
            } else if (label_dim[label] != -1) {
                // Repeated label, take diagonal
                const auto dim = label_dim[label];
                if (operand_dims[j] != operand_dims[dim]) {
                    const std::string message = "einsum() subscript " + ToString(char(label + 'a')) +
                                                " is repeated for operand " + ToString(i) +
                                                " but the sizes don't match, " + ToString(operand_dims[j]) +
                                                " != " + ToString(operand_dims[dim]);
                    return Status(TNNERR_MODEL_ERR, message);
                }
                // diagonal is not supported
                // TODO
                // operand = operand.diagonal(0, dim, j).movedim(-1, dim);
                return Status(TNNERR_MODEL_ERR, "diagonal in einsum is not supported");
            } else {
                // Lookup output index for label
                label_dim[label]                    = j;
                perm_shape[label_perm_index[label]] = j++;
            }
        }

        // Add dimensions for missing labels
        for (int &index : perm_shape) {
            if (index == -1) {
                auto dims = operand_dims;
                dims.push_back(1);
                operand_dims = dims;
                index        = j++;
            }
        }
        param->operand_dims.push_back(operand_dims);
        param->perm_shapes.push_back(perm_shape);
        permuted_operands_dims.push_back(CalPermuteOutputShape(operand_dims, perm_shape));

    }

    // Check if operands broadcast and keep track of last operand with
    // dimension size != 1 for optimizing reductions
    std::vector<std::size_t> dim_last_op(perm_index, 0);
    bool has_zero_size_dim = false;
    for (int dim = 0; dim < perm_index; dim++) {
        auto broadcast_size = permuted_operands_dims[0][dim];
        for (int i = 1; i < num_ops; i++) {
            const auto dim_size = permuted_operands_dims[i][dim];
            if (broadcast_size != dim_size && broadcast_size != 1 && dim_size != 1) {
                const std::string message =
                    "einsum() operands do not broadcast with remapped shapes [original->remapped]";
                return Status(TNNERR_MODEL_ERR, message);
            }
            if (dim_size != 1) {
                broadcast_size   = dim_size;
                dim_last_op[dim] = i;
            }
        }
        has_zero_size_dim |= broadcast_size == 0;
    }
    param->has_zero_size_dim = has_zero_size_dim;
    param->dim_last_op = dim_last_op;

    // Compute result
    auto result = permuted_operands_dims[0];

    // Fast path for when an operand has zero sized dim
    if (has_zero_size_dim) {
        std::vector<int> out_shape(out_size);
        for (int i = 0; i < out_size; i++) {
            out_shape[i] = permuted_operands_dims[dim_last_op[i]][i];
        }
        output_blobs_[0]->GetBlobDesc().dims = out_shape;

        return TNN_OK;
    }

    // Sum out or squeeze dimensions that are size 1 for all later operands
    int dim = out_size;
    for (int i = dim; i < perm_index; ++i, ++dim) {
        if (dim_last_op[i] == 0) {
            if (result[dim] == 1) {
                result = CalSqueezeOutputShape(result, dim--);
            } else {
                result = CalSumOutputShape(result, dim--);
            }
        }
    }

    for (int i = 1; i < num_ops; i++) {
        auto operand_dims = permuted_operands_dims[i];
        std::vector<int> sum_dims;

        // Sum out or squeeze dimensions that are size 1 for all later operands
        dim = out_size;
        for (int j = dim; j < perm_index; ++j, ++dim) {
            if (dim_last_op[j] < i) {
                operand_dims = CalSqueezeOutputShape(operand_dims, dim);
                --dim;
            } else if (dim_last_op[j] == i) {
                if (result[dim] == 1) {
                    operand_dims = CalSumOutputShape(operand_dims, dim);
                    result       = CalSqueezeOutputShape(result, dim);
                    --dim;
                } else {
                    sum_dims.push_back(dim);
                }
            }
        }

        // Multiply tensors and sum out dimensions in sum_dims
        if (sum_dims.empty()) {
            result = CalMulOutputShape(result, operand_dims);
        } else if (sum_dims.size() == result.size()) {
            result = CalDotOutputShape();
        } else {
            result = CalMulOutputShape(result, operand_dims);
            for (const auto axis : sum_dims) {
                result = CalSumOutputShape(result, axis);
            }
        }
    }

    output_blobs_[0]->GetBlobDesc().dims = result;

    return TNN_OK;
}

REGISTER_LAYER(Einsum, LAYER_EINSUM);

}  // namespace TNN_NS

