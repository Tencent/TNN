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

#include "tnn/utils/winograd_generator.h"

#include <math.h>
#include <memory.h>

#include "tnn/core/macro.h"
#include "tnn/utils/dims_vector_utils.h"

#ifdef TNN_USE_NEON
#include <arm_neon.h>
#endif

namespace TNN_NS {

/*
create matrix with w&h, alloc matrix buffer inner
*/
CMatrix CMatrixCreate(int w, int h) {
    std::shared_ptr<float> result = std::shared_ptr<float>(new float[w * h], [](float* p) { delete[] p; });
    DimsVector dims               = {w, h};
    return make_tuple(result, dims);
}

/*
create matrix with dims, alloc matrix buffer inner
*/
CMatrix CMatrixCreate(DimsVector dims) {
    int count = 1;
    for (auto iter : dims) {
        count *= iter;
    }
    std::shared_ptr<float> result = std::shared_ptr<float>(new float[count], [](float* p) { delete[] p; });
    return make_tuple(result, dims);
}

/*
get cmatrix stride
*/
DimsVector CMatrixGetStrides(CMatrix& matrix) {
    auto dims = std::get<1>(matrix);
    DimsVector strides;
    // strides.push_back(1);
    // int count = 1;
    // for (auto iter : dims) {
    //     count *= iter;
    //     strides.push_back(count);
    // }
    int count = 1;
    for (auto iter : dims) {
        count *= iter;
    }
    for (auto iter : dims) {
        count /= iter;
        strides.push_back(count);
    }
    return strides;
}

/*
matrix mul: M*K x K*N = M*N
*/
static void matmul(CMatrix& C, const CMatrix& A, const float* B, DimsVector B_dims) {
    auto C_dims = std::get<1>(C);
    auto A_dims = std::get<1>(A);

    ASSERT(2 == C_dims.size());
    ASSERT(2 == A_dims.size());
    ASSERT(2 == B_dims.size());

    const auto a = std::get<0>(A).get();
    const auto b = B;
    auto c       = std::get<0>(C).get();

    const int h = A_dims[1];
    const int k = A_dims[0];
    const int w = B_dims[0];

    const int aw = A_dims[0];
    const int bw = B_dims[0];
    const int cw = C_dims[0];

    ASSERT(k == B_dims[1]);

    int y = 0;
    for (; y < h; ++y) {
        int x            = 0;
        const auto aLine = a + y * aw;
        auto cLine       = c + y * cw;
#ifdef TNN_USE_NEON
        // firstly, compute 16 together
        for (; x <= w - 16; x += 16) {
            auto bColumn     = b + x;
            float32x4_t sum0 = vdupq_n_f32(0.0);
            float32x4_t sum1 = vdupq_n_f32(0.0);
            float32x4_t sum2 = vdupq_n_f32(0.0);
            float32x4_t sum3 = vdupq_n_f32(0.0);
            for (int i = 0; i < k; ++i) {
                const auto bLine = bColumn + i * bw;
                float32x4_t a0   = vdupq_n_f32(aLine[i]);
                float32x4_t b0   = vld1q_f32(bLine);
                float32x4_t b1   = vld1q_f32(bLine + 4);
                float32x4_t b2   = vld1q_f32(bLine + 8);
                float32x4_t b3   = vld1q_f32(bLine + 12);
                sum0             = vmlaq_f32(sum0, a0, b0);
                sum1             = vmlaq_f32(sum1, a0, b1);
                sum2             = vmlaq_f32(sum2, a0, b2);
                sum3             = vmlaq_f32(sum3, a0, b3);
            }
            vst1q_f32(cLine + x, sum0);
            vst1q_f32(cLine + x + 4, sum1);
            vst1q_f32(cLine + x + 8, sum2);
            vst1q_f32(cLine + x + 12, sum3);
        }
        // secondly, compute 4 together
        for (; x <= w - 4; x += 4) {
            auto bColumn    = b + x;
            float32x4_t sum = vdupq_n_f32(0.0);
            for (int i = 0; i < k; ++i) {
                const auto bLine = bColumn + i * bw;
                float32x4_t a4   = vdupq_n_f32(aLine[i]);
                float32x4_t b4   = vld1q_f32(bLine);
                sum              = vmlaq_f32(sum, a4, b4);
            }
            vst1q_f32(cLine + x, sum);
        }
#endif
        // naive matrix mul
        for (; x < w; ++x) {
            auto bColumn = b + x;
            float sum    = 0.0f;
            for (int i = 0; i < k; ++i) {
                sum += aLine[i] * bColumn[i * bw];
            }
            cLine[x] = sum;
        }
    }
}

static void matmul(CMatrix& C, const CMatrix& A, const CMatrix& B) {
    matmul(C, A, std::get<0>(B).get(), std::get<1>(B));
}

static void divPerLine(CMatrix& C, const CMatrix& A, const CMatrix& Line) {
    auto C_dims    = std::get<1>(C);
    auto A_dims    = std::get<1>(A);
    auto Line_dims = std::get<1>(Line);

    auto c = std::get<0>(C).get();
    auto a = std::get<0>(A).get();
    auto l = std::get<0>(Line).get();
    auto w = C_dims[0];
    auto h = C_dims[1];

    ASSERT(Line_dims[0] >= h);
    ASSERT(A_dims[1] == h);
    ASSERT(A_dims[0] == w);
    ASSERT(Line_dims[1] == 1);

    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            c[x + y * w] = a[x + y * w] / l[y];
        }
    }
}

static void transpose(CMatrix dst, CMatrix src) {
    auto src_data = std::get<0>(src).get();
    auto dst_data = std::get<0>(dst).get();

    auto src_dims = std::get<1>(src);
    auto dst_dims = std::get<1>(dst);

    int w = dst_dims[0];
    int h = dst_dims[1];

    ASSERT(w == src_dims[1] || h == src_dims[0]);

    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            dst_data[w * y + x] = src_data[h * x + y];
        }
    }
}

static CMatrix polyMulti(CMatrix A, CMatrix B) {
    auto a = std::get<0>(A).get();
    auto b = std::get<0>(B).get();

    auto a_dims = std::get<1>(A);
    auto b_dims = std::get<1>(B);

    ASSERT(a_dims[1] == 1);
    ASSERT(b_dims[1] == 1);

    auto aw = a_dims[0];
    auto bw = b_dims[0];

    std::shared_ptr<float> result = std::shared_ptr<float>(new float[aw + bw - 1], [](float* p) { delete[] p; });
    DimsVector result_dims        = {aw + bw - 1, 1};

    auto c = result.get();
    for (int i = 0; i < aw + bw - 1; ++i) {
        c[i] = 0.0f;
    }
    for (int y = 0; y < bw; ++y) {
        auto bValue = b[y];
        for (int x = 0; x < aw; ++x) {
            auto aValue = a[x];
            c[x + y] += bValue * aValue;
        }
    }
    return make_tuple(result, result_dims);
}

static CMatrix computeF(const float* a, int alpha) {
    auto res      = CMatrixCreate(alpha, 1);
    auto diagData = std::get<0>(res).get();
    for (int x = 0; x < alpha; ++x) {
        float product = 1.0f;
        for (int i = 0; i < alpha; ++i) {
            if (x == i) {
                continue;
            }
            product *= (a[x] - a[i]);
        }
        diagData[x] = product;
    }
    return res;
}

static CMatrix computeT(const float* a, int n) {
    auto result = CMatrixCreate(n + 1, n);
    for (int y = 0; y < n; ++y) {
        auto line = std::get<0>(result).get() + (n + 1) * y;
        ::memset(line, 0, (n + 1) * sizeof(float));
        line[y] = 1.0f;
        line[n] = -::powf(a[y], (float)n);
    }
    return result;
}

static CMatrix computeL(const float* a, int n) {
    ASSERT(n >= 1);
    auto result = CMatrixCreate(n, n);
    for (int k = 0; k < n; ++k) {
        auto poly = CMatrixCreate(1, 1);
        auto p    = std::get<0>(poly).get();
        p[0]      = 1.0f;

        auto poly2 = CMatrixCreate(2, 1);
        auto p2    = std::get<0>(poly2).get();
        for (int i = 0; i < n; ++i) {
            if (i == k) {
                continue;
            }
            p2[0] = -a[i];
            p2[1] = 1.0f;
            poly  = polyMulti(poly, poly2);
        }
        ::memcpy(std::get<0>(result).get() + n * k, std::get<0>(poly).get(), n * sizeof(float));
    }
    return result;
}

static CMatrix computeB(const float* a, int alpha) {
    std::shared_ptr<float> res;
    auto LT    = computeL(a, alpha - 1);
    auto fdiag = computeF(a, alpha - 1);
    divPerLine(LT, LT, fdiag);

    auto L = CMatrixCreate(alpha - 1, alpha - 1);
    transpose(L, LT);

    auto T  = computeT(a, alpha - 1);
    auto BT = CMatrixCreate(alpha, alpha - 1);
    matmul(BT, L, T);

    auto B = CMatrixCreate(alpha, alpha);
    for (int y = 0; y < alpha - 1; ++y) {
        ::memcpy(std::get<0>(B).get() + alpha * y, std::get<0>(BT).get() + alpha * y, alpha * sizeof(float));
    }
    auto BLast = std::get<0>(B).get() + alpha * (alpha - 1);
    for (int x = 0; x < alpha - 1; ++x) {
        BLast[x] = 0;
    }
    BLast[alpha - 1] = 1.0f;

    return B;
}

static CMatrix computeA(const float* a, int m, int n) {
    auto res = CMatrixCreate(m, n);
    for (int y = 0; y < n; ++y) {
        float* line = std::get<0>(res).get() + m * y;
        for (int x = 0; x < m - 1; ++x) {
            if (x == 0 && y == 0) {
                line[x] = 1.0f;
            } else {
                line[x] = ::powf(a[x], (float)y);
            }
        }
        if (y == n - 1) {
            line[m - 1] = 1.0f;
        } else {
            line[m - 1] = 0.0f;
        }
    }
    return res;
}

static CMatrix computeFDiag(const float* a, int alpha) {
    auto res      = CMatrixCreate(alpha, 1);
    auto diagData = std::get<0>(res).get();
    for (int x = 0; x < alpha - 1; ++x) {
        float product = 1.0f;
        for (int i = 0; i < alpha - 1; ++i) {
            if (x == i) {
                continue;
            }
            product *= (a[x] - a[i]);
        }
        diagData[x] = product;
    }
    diagData[alpha - 1] = 1.0f;
    if (diagData[0] < 0) {
        diagData[0] = -diagData[0];
    }
    return res;
}

/*
1D: AT*((G*g)(BT*d))
2D: AT*((G*g*GT)(BT*d*B))*A
https://github.com/andravin/wincnn
*/
WinogradGenerator::WinogradGenerator(int computeUnit, int kernelSize, float interp, bool transform_inner) {
    ASSERT(computeUnit > 0 && kernelSize > 0);
    unit_        = computeUnit;
    kernel_size_ = kernelSize;
    transform_inner_ = transform_inner;

    int n     = computeUnit;
    int r     = kernelSize;
    int alpha = n + r - 1;
    G_        = CMatrixCreate(r, alpha);
    B_        = CMatrixCreate(alpha, alpha);
    A_        = CMatrixCreate(n, alpha);

    auto polyBuffer = CMatrixCreate(alpha, 1);
    auto a          = std::get<0>(polyBuffer).get();
    a[0]            = 0.0f;
    int sign        = 1;
    for (int i = 0; i < alpha - 1; ++i) {
        int value = 1 + i / 2;
        a[i + 1]  = sign * value * interp;
        sign *= -1;
    }
    // Matrix::print(polyBuffer.get());
    {
        auto A = computeA(a, alpha, n);
        transpose(A_, A);
    }
    auto fdiag = computeFDiag(a, alpha);
    // Matrix::print(fdiag.get());
    {
        auto A = computeA(a, alpha, r);
        transpose(G_, A);
    }
    {
        auto B = computeB(a, alpha);
        transpose(B_, B);
        transpose(B, B_);
        B_ = B;
    }
}

/*
transform weight size: unit*unit*ROUND_UP(oc, 4)*ROUND_UP(ic, 4)
*/
CMatrix WinogradGenerator::allocTransformWeight(int batch, int channel, int height, int width, int unitCi, int unitCo) {
    int ci = channel;
    int co = batch;
    ASSERT(width == height && width == std::get<1>(G_)[0]);
    int ciC4 = UP_DIV(ci, unitCi);
    int coC4 = UP_DIV(co, unitCo);
    if(transform_inner_) {
        return CMatrixCreate({coC4, std::get<1>(B_)[0] * std::get<1>(B_)[1], ciC4, unitCi, unitCo});
    } else {
        return CMatrixCreate({std::get<1>(B_)[0] * std::get<1>(B_)[1], coC4, ciC4, unitCi, unitCo});
    }
}

/*
transform weight from [oc][ic][kh][kw] to [unit][unit][co4][ci4][16]
*/
void WinogradGenerator::transformWeight(CMatrix& weightDest, const float* source, int batch, int channel, int height,
                                        int width) {
    auto GT = CMatrixCreate(std::get<1>(G_)[1], std::get<1>(G_)[0]);
    transpose(GT, G_);

    auto weight_dest_data    = std::get<0>(weightDest).get();
    auto weight_dest_dims    = std::get<1>(weightDest);
    auto weight_dest_strides = CMatrixGetStrides(weightDest);
    auto B_dims              = std::get<1>(B_);

    int ci          = channel;
    int co          = batch;
    int kernelCount = height;
    int unitCi      = weight_dest_dims[3];
    int unitCo      = weight_dest_dims[4];
    auto alpha      = B_dims[0];

    if (ci % unitCi != 0 || co % unitCo != 0) {
        ::memset(weight_dest_data, 0, DimsVectorUtils::Count(weight_dest_dims) * sizeof(float));
    }

    auto M           = CMatrixCreate(kernelCount, alpha);
    auto K_Transform = CMatrixCreate(alpha, alpha);

    auto weightPtr      = source;
    auto KTransformData = std::get<0>(K_Transform).get();

    int oz_index, alpha_index;
    if(transform_inner_) {
        oz_index = 0;
        alpha_index = 1;
    } else {
        oz_index = 1;
        alpha_index = 0;
    }

    for (int oz = 0; oz < co; ++oz) {
        auto srcOz = weightPtr + oz * ci * kernelCount * kernelCount;

        int ozC4 = oz / unitCo;
        int mx   = oz % unitCo;

        auto dstOz = weight_dest_data + weight_dest_strides[oz_index] * ozC4 + mx;
        for (int sz = 0; sz < ci; ++sz) {
            int szC4   = sz / unitCi;
            int my     = sz % unitCi;
            auto srcSz = srcOz + kernelCount * kernelCount * sz;

            // M = G * K
            matmul(M, G_, srcSz, {kernelCount, kernelCount});
            // K_Transform = M*GT
            matmul(K_Transform, M, GT);

            auto dstSz = dstOz + szC4 * weight_dest_strides[2] + unitCo * my;
            // [alpha][alpha][oc4][ic4][16]
            for (int i = 0; i < alpha * alpha; ++i) {
                *(dstSz + i * weight_dest_strides[alpha_index]) = KTransformData[i];
            }
        }
    }
}

}  // namespace TNN_NS
