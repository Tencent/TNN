/*
 * Copyright 1993-2019 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO LICENSEE:
 *
 * This source code and/or documentation ("Licensed Deliverables") are
 * subject to NVIDIA intellectual property rights under U.S. and
 * international Copyright laws.
 *
 * These Licensed Deliverables contained herein is PROPRIETARY and
 * CONFIDENTIAL to NVIDIA and is being provided under the terms and
 * conditions of a form of NVIDIA software license agreement by and
 * between NVIDIA and Licensee ("License Agreement") or electronically
 * accepted by Licensee.  Notwithstanding any terms or conditions to
 * the contrary in the License Agreement, reproduction or disclosure
 * of the Licensed Deliverables to any third party without the express
 * written consent of NVIDIA is prohibited.
 *
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
 * SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  IT IS
 * PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
 * NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
 * DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
 * NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY
 * SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
 * DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
 * WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
 * ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
 * OF THESE LICENSED DELIVERABLES.
 *
 * U.S. Government End Users.  These Licensed Deliverables are a
 * "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
 * 1995), consisting of "commercial computer software" and "commercial
 * computer software documentation" as such terms are used in 48
 * C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
 * only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
 * 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
 * U.S. Government End Users acquire the Licensed Deliverables with
 * only those rights set forth herein.
 *
 * Any use of the Licensed Deliverables in individual and commercial
 * software must include, in the user documentation and internal
 * comments to the code, the above Disclaimer and U.S. Government End
 * Users Notice.
 */

#ifndef NV_UTILS_H
#define NV_UTILS_H

#include "NvInfer.h"

namespace nvinfer1
{
namespace utils
{

    //!
    //! \param input The input weights to reshape.
    //! \param shape The shape of the weights.
    //! \param shapeOrder The order of the dimensions to process for the output.
    //! \param data The location where the output data is placed.
    //! \param nbDims The number of dimensions to process.
    //!
    //! \brief Reformat the input weights of the given shape based on the new
    //! order of dimensions.
    //!
    //! Take the weights specified by \p input with the dimensions specified by
    //! \p shape and re-order the weights based on the new dimensions specified
    //! by \p shapeOrder. The size of each dimension and the input data is not
    //! modified. The output volume pointed to by \p data must be the same as
    //! he \p input volume.
    //!
    //! Example usage:
    //! float *out = new float[N*C*H*W];
    //! Weights input{DataType::kFLOAT, {0 ... N*C*H*W-1}, N*C*H*W size};
    //! int order[4]{1, 0, 3, 2};
    //! int shape[4]{C, N, W, H};
    //! reshapeWeights(input, shape, order, out, 4);
    //! Weights reshaped{input.type, out, input.count};
    //!
    //! Input Matrix{3, 2, 3, 2}:
    //! { 0  1}, { 2  3}, { 4  5} <-- {0, 0, *, *}
    //! { 6  7}, { 8  9}, {10 11} <-- {0, 1, *, *}
    //! {12 13}, {14 15}, {16 17} <-- {1, 0, *, *}
    //! {18 19}, {20 21}, {22 23} <-- {1, 1, *, *}
    //! {24 25}, {26 27}, {28 29} <-- {2, 0, *, *}
    //! {30 31}, {32 33}, {34 35} <-- {2, 1, *, *}
    //!
    //! Output Matrix{2, 3, 2, 3}:
    //! { 0  2  4}, { 1  3  5} <-- {0, 0, *, *}
    //! {12 14 16}, {13 15 17} <-- {0, 1, *, *}
    //! {24 26 28}, {25 27 29} <-- {0, 2, *, *}
    //! { 6  8 10}, { 7  9 11} <-- {1, 0, *, *}
    //! {18 20 22}, {19 21 23} <-- {1, 1, *, *}
    //! {30 32 34}, {31 33 35} <-- {1, 2, *, *}
    //!
    //! \return True on success, false on failure.
    //!
    TENSORRTAPI bool reshapeWeights(const Weights &input, const int *shape, const int *shapeOrder, void *data, int nbDims);

    //!
    //! \param input The input data to re-order.
    //! \param order The new order of the data sub-buffers.
    //! \param num The number of data sub-buffers to re-order.
    //! \param size The size of each data sub-buffer in bytes.
    //!
    //! \brief Takes an input stream and re-orders \p num chunks of the data
    //! given the \p size and \p order.
    //!
    //! In some frameworks, the ordering of the sub-buffers within a dimension
    //! is different than the way that TensorRT expects them.
    //! TensorRT expects the gate/bias sub-buffers for LSTM's to be in fico order.
    //! TensorFlow however formats the sub-buffers in icfo order.
    //! This helper function solves this in a generic fashion.
    //!
    //! Example usage output of reshapeWeights above:
    //! int indir[1]{1, 0}
    //! int stride = W*H;
    //! for (int x = 0, y = N*C; x < y; ++x)
    //! reorderSubBuffers(out + x * stride, indir, H, W);
    //!
    //! Input Matrix{2, 3, 2, 3}:
    //! { 0  2  4}, { 1  3  5} <-- {0, 0, *, *}
    //! {12 14 16}, {13 15 17} <-- {0, 1, *, *}
    //! {24 26 28}, {25 27 29} <-- {0, 2, *, *}
    //! { 6  8 10}, { 7  9 11} <-- {1, 0, *, *}
    //! {18 20 22}, {19 21 23} <-- {1, 1, *, *}
    //! {30 32 34}, {31 33 35} <-- {1, 2, *, *}
    //!
    //! Output Matrix{2, 3, 2, 3}:
    //! { 1  3  5}, { 0  2  4} <-- {0, 0, *, *}
    //! {13 15 17}, {12 14 16} <-- {0, 1, *, *}
    //! {25 27 29}, {24 26 28} <-- {0, 2, *, *}
    //! { 7  9 11}, { 6  8 10} <-- {1, 0, *, *}
    //! {19 21 23}, {18 20 22} <-- {1, 1, *, *}
    //! {31 33 35}, {30 32 34} <-- {1, 2, *, *}
    //!
    //! \return True on success, false on failure.
    //!
    //! \see reshapeWeights()
    //!
    TENSORRTAPI bool reorderSubBuffers(void *input, const int *order, int num, int size);

    //!
    //! \param input The input data to transpose.
    //! \param type The type of the data to transpose.
    //! \param num The number of data sub-buffers to transpose.
    //! \param height The size of the height dimension to transpose.
    //! \param width The size of the width dimension to transpose.
    //!
    //! \brief Transpose \p num sub-buffers of \p height * \p width.
    //!
    //! \return True on success, false on failure.
    //!
    TENSORRTAPI bool transposeSubBuffers(void *input, DataType type, int num, int height, int width);

} // utils namespace
} // nvinfer1 namespace
#endif // NV_UTILS_H
