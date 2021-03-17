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

#include <algorithm>

#include <immintrin.h>
#include <xmmintrin.h>

#include "tnn/device/x86/acc/compute/jit/cblas.h"

extern "C" {

#define ABS_(a) ((a) > 0 ? (a) : (-(a)))

void cblas_sgemv(OPENBLAS_CONST enum CBLAS_ORDER order,  OPENBLAS_CONST enum CBLAS_TRANSPOSE trans,  OPENBLAS_CONST blasint m, OPENBLAS_CONST blasint n,
    OPENBLAS_CONST float alpha, OPENBLAS_CONST float  *a, OPENBLAS_CONST blasint lda,  OPENBLAS_CONST float  *x, OPENBLAS_CONST blasint incx,  OPENBLAS_CONST float beta,  float  *y, OPENBLAS_CONST blasint incy) 
{

    blasint n_stride  = (order == CblasRowMajor) ?  1 : lda;
    blasint m_stride  = (order == CblasRowMajor) ?  lda : 1;

    // blasint first_dim  = (trans == CblasNoTrans) ? m : n;
    // blasint second_dim = (trans == CblasNoTrans) ? n : m;
    blasint first_dim;
    blasint second_dim;
    blasint a_first_stride;
    blasint a_second_stride;


    if (trans == CblasNoTrans) {
        first_dim = m;
        second_dim = n;
        a_first_stride = m_stride;
        a_second_stride = n_stride;
    } else {
        first_dim = n;
        second_dim = m;
        a_first_stride = n_stride;
        a_second_stride = m_stride;
    }

    if (ABS_(beta) < 1e-6) {
        for(int i=0;i<first_dim;i++) {
            y[i * incy] = 0.f;
        }
    }

    for(int i=0;i<first_dim;i++) {
        float accu = 0.f;

        OPENBLAS_CONST float * x_ptr = x;
        OPENBLAS_CONST float * a_ptr = a + i * a_first_stride;

        for(int j=0;j<second_dim;j++) {
            accu += (*a_ptr) * (*x_ptr);

            x_ptr += incx;
            a_ptr += a_second_stride;
        }

        accu = accu * alpha + beta * y[i * incy];
        y[i * incy] = accu;

    }
}

void cblas_saxpy(OPENBLAS_CONST blasint n, OPENBLAS_CONST float alpha, OPENBLAS_CONST float *x, OPENBLAS_CONST blasint incx, float *y, OPENBLAS_CONST blasint incy) 
{
    for(int i=0;i<n;i++) {
        y[i * incy] += alpha * x[i * incx];
    }

}


float cblas_sasum(OPENBLAS_CONST blasint n, OPENBLAS_CONST float  *x, OPENBLAS_CONST blasint incx) 
{
    float res = 0;
    for(int i=0;i<n;i++) {
        res += ABS_(x[i * incx]);
    }
    return res;
}


void cblas_sscal(OPENBLAS_CONST blasint N, OPENBLAS_CONST float alpha, float *X, OPENBLAS_CONST blasint incX) 
{
    for(int i=0;i<N;i++) {
        X[i*incX] *= alpha;
    }
}


void cblas_scopy(OPENBLAS_CONST blasint n, OPENBLAS_CONST float *x, OPENBLAS_CONST blasint incx, float *y, OPENBLAS_CONST blasint incy) 
{
    for(int i=0;i<n;i++)  {
        y[i*incy] = x[i*incx];
    }
}

#undef ABS_

}