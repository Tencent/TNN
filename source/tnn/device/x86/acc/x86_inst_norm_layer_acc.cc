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

#include "tnn/device/x86/acc/x86_layer_acc.h"
#include "tnn/utils/data_type_utils.h"
#include "tnn/utils/dims_utils.h"
#include <math.h>
#include "tnn/device/x86/x86_common.h"

// #define AVX2 1
namespace TNN_NS {

DECLARE_X86_ACC(InstanceNorm, LAYER_INST_BATCH_NORM);

Status X86InstanceNormLayerAcc::DoForward(const std::vector<Blob*> &inputs, const std::vector<Blob*> &outputs) {
    auto resource = dynamic_cast<InstanceNormLayerResource*>(resource_);
    if (!resource) {
        return Status(TNNERR_MODEL_ERR, "Error, InstanceNormLayerResource is nil");
    }

    auto input_blob         = inputs[0];
    auto output_blob        = outputs[0];
    float *input_data       = handle_ptr<float*>(input_blob->GetHandle());
    float *output_data      = handle_ptr<float*>(output_blob->GetHandle());

    int batch    = output_blob->GetBlobDesc().dims[0];
    int channels = output_blob->GetBlobDesc().dims[1];
    int area     = DimsVectorUtils::Count(output_blob->GetBlobDesc().dims, 2);

    if (area == 0) {
        LOGE("Error: blob count is zero\n");
        return Status(TNNERR_COMMON_ERROR, "Error: blob count is zero");
    }

    float *k_data = resource->scale_handle.force_to<float*>();
    float *b_data = resource->bias_handle.force_to<float*>();

    float epsilon = 0.00001f;

    if (output_blob->GetBlobDesc().data_type == DATA_TYPE_FLOAT) {
        for (int b = 0; b < batch; b++) {
            if (1) {
                for (int c = 0; c < channels; c++) {
#ifdef __AVX__
                    __m256 _sum_x, _sum_x2;
                    float buffer[8];
                    _sum_x = _mm256_setzero_ps();
                    _sum_x2 = _mm256_setzero_ps();
                    int head = 0;
                    const int tail = area - area % 8;
                    double temp;
                    __m256 _temp;
                    for (size_t i = head; i < tail; i += 8) {
                        _temp = _mm256_loadu_ps(input_data + i);
                        _sum_x = _mm256_add_ps(_sum_x, _temp);
#ifdef __AVX2__
                        _sum_x2 = _mm256_fmadd_ps(_temp, _temp, _sum_x2);
#else
                        _sum_x2 = _mm256_add_ps(_mm256_mul_ps(_temp, _temp), _sum_x2);
#endif
                    }

                    float sum_x, sum_x2;
                    _mm256_storeu_ps(buffer, _sum_x);
                    sum_x = buffer[0] + buffer[1] + buffer[2] + buffer[3] + buffer[4] + buffer[5] + buffer[6] + buffer[7];
                    _mm256_storeu_ps(buffer, _sum_x2);
                    sum_x2 = buffer[0] + buffer[1] + buffer[2] + buffer[3] + buffer[4] + buffer[5] + buffer[6] + buffer[7];
                    for (size_t i = tail; i < area; i++) {
                        temp = input_data[i];
                        sum_x += temp;
                        sum_x2 += temp * temp;
                    }

                    auto mean_x = sum_x / area;
                    auto mean_x2 = sum_x2 / area;
                    float variance = mean_x2 - mean_x * mean_x;
                    variance = variance > 0 ? variance : 0;
                    variance = 1.0f / sqrt(variance + epsilon);
                    float k = k_data[c];
                    variance *= k;

                    float b = b_data == NULL ? 0.0f : b_data[c];
                    b -= mean_x * variance;

                    _sum_x = _mm256_broadcast_ss(&variance);
                    _sum_x2 = _mm256_broadcast_ss(&b);
                    const float *tail_p = output_data + tail;
                    for (; output_data < tail_p; output_data += 8, input_data += 8) {
                        // std::cout << i << std::endl;
                        _temp = _mm256_loadu_ps(input_data);
#ifdef __AVX2__
                        _temp = _mm256_fmadd_ps(_temp, _sum_x, _sum_x2);
#else
                        _temp = _mm256_add_ps(_mm256_mul_ps(_temp, _sum_x), _sum_x2);
#endif
                        _mm256_storeu_ps(output_data, _temp);
                    }
                    for (size_t i = tail; i < area; i++, output_data++, input_data++) {
                        *output_data = (*input_data) * variance + b;
                    }
#else
                    __m128 _sum_x, _sum_x2;
                    float buffer[4];
                    _sum_x = _mm_setzero_ps();
                    _sum_x2 = _mm_setzero_ps();

                    int head = 0;
                    const int tail = area - area % 4;
                    double temp;
                    __m128 _temp;
                    for (size_t i = head; i < tail; i += 4) {
                        _temp = _mm_loadu_ps(input_data + i);
                        _sum_x = _mm_add_ps(_sum_x, _temp);
                        _sum_x2 = _mm_add_ps(_mm_mul_ps(_temp, _temp), _sum_x2);
                    }

                    float sum_x, sum_x2;
                    _mm_storeu_ps(buffer, _sum_x);
                    sum_x = buffer[0] + buffer[1] + buffer[2] + buffer[3];
                    _mm_storeu_ps(buffer, _sum_x2);
                    sum_x2 = buffer[0] + buffer[1] + buffer[2] + buffer[3];
                    for (size_t i = tail; i < area; i++) {
                        temp = input_data[i];
                        sum_x += temp;
                        sum_x2 += temp * temp;
                    }

                    auto mean_x = sum_x / area;
                    auto mean_x2 = sum_x2 / area;
                    float variance = mean_x2 - mean_x * mean_x;
                    variance = variance > 0 ? variance : 0;
                    variance = 1.0f / sqrt(variance + epsilon);
                    float k = k_data[c];
                    variance *= k;

                    float b = b_data == NULL ? 0.0f : b_data[c];
                    b -= mean_x * variance;

                    _sum_x = _mm_load1_ps(&variance);
                    _sum_x2 = _mm_load1_ps(&b);
                    const float *tail_p = output_data + tail;
                    for (; output_data < tail_p; output_data += 4, input_data += 4) {
                        // std::cout << i << std::endl;
                        _temp = _mm_loadu_ps(input_data);
                        _temp = _mm_add_ps(_mm_mul_ps(_temp, _sum_x), _sum_x2);
                        _mm_storeu_ps(output_data, _temp);
                    }
                    for (size_t i = tail; i < area; i++, output_data++, input_data++) {
                        *output_data = (*input_data) * variance + b;
                    }
#endif
                }
            } else {
                for (int c = 0; c < channels; c += 4) {
#ifdef __AVX__
                    float buffer[8];
                    __m256 _sum1_x  = _mm256_setzero_ps();
                    __m256 _sum2_x  = _mm256_setzero_ps();
                    __m256 _sum3_x  = _mm256_setzero_ps();
                    __m256 _sum4_x  = _mm256_setzero_ps();
                    __m256 _sum1_x2 = _mm256_setzero_ps();
                    __m256 _sum2_x2 = _mm256_setzero_ps();
                    __m256 _sum3_x2 = _mm256_setzero_ps();
                    __m256 _sum4_x2 = _mm256_setzero_ps();

                    const int tail = area - area % 8;
                    double temp;
                    __m256 _temp1, _temp2, _temp3, _temp4;
                    float *input_data1 = input_data + c * area;
                    float *input_data2 = input_data + (c + 1) * area;
                    float *input_data3 = input_data + (c + 2) * area;
                    float *input_data4 = input_data + (c + 3) * area;
                    float *output_data1 = output_data + c * area;
                    float *output_data2 = output_data + (c + 1) * area;
                    float *output_data3 = output_data + (c + 2) * area;
                    float *output_data4 = output_data + (c + 3) * area;

                    for (size_t i = 0; i < tail; i += 8) {
                        _temp1 = _mm256_loadu_ps(input_data1 + i);
                        _temp2 = _mm256_loadu_ps(input_data2 + i);
                        _temp3 = _mm256_loadu_ps(input_data3 + i);
                        _temp4 = _mm256_loadu_ps(input_data4 + i);
                        _sum1_x = _mm256_add_ps(_sum1_x, _temp1);
                        _sum2_x = _mm256_add_ps(_sum2_x, _temp2);
                        _sum3_x = _mm256_add_ps(_sum3_x, _temp3);
                        _sum4_x = _mm256_add_ps(_sum4_x, _temp4);
#ifdef __AVX2__
                        _sum1_x2 = _mm256_fmadd_ps(_temp1, _temp1, _sum1_x2);
                        _sum2_x2 = _mm256_fmadd_ps(_temp2, _temp2, _sum2_x2);
                        _sum3_x2 = _mm256_fmadd_ps(_temp3, _temp3, _sum3_x2);
                        _sum4_x2 = _mm256_fmadd_ps(_temp4, _temp4, _sum4_x2);
#else
                        _sum1_x2 = _mm256_add_ps(_mm256_mul_ps(_temp1, _temp1), _sum1_x2);
                        _sum2_x2 = _mm256_add_ps(_mm256_mul_ps(_temp2, _temp2), _sum2_x2);
                        _sum3_x2 = _mm256_add_ps(_mm256_mul_ps(_temp3, _temp3), _sum3_x2);
                        _sum4_x2 = _mm256_add_ps(_mm256_mul_ps(_temp4, _temp4), _sum4_x2);
#endif
                    }

                    _mm256_storeu_ps(buffer, _sum1_x);
                    float sum1_x = buffer[0] + buffer[1] + buffer[2] + buffer[3] + buffer[4] + buffer[5] + buffer[6] + buffer[7];
                    _mm256_storeu_ps(buffer, _sum2_x);
                    float sum2_x = buffer[0] + buffer[1] + buffer[2] + buffer[3] + buffer[4] + buffer[5] + buffer[6] + buffer[7];
                    _mm256_storeu_ps(buffer, _sum3_x);
                    float sum3_x = buffer[0] + buffer[1] + buffer[2] + buffer[3] + buffer[4] + buffer[5] + buffer[6] + buffer[7];
                    _mm256_storeu_ps(buffer, _sum4_x);
                    float sum4_x = buffer[0] + buffer[1] + buffer[2] + buffer[3] + buffer[4] + buffer[5] + buffer[6] + buffer[7];
                    _mm256_storeu_ps(buffer, _sum1_x2);
                    float sum1_x2 = buffer[0] + buffer[1] + buffer[2] + buffer[3] + buffer[4] + buffer[5] + buffer[6] + buffer[7];
                    _mm256_storeu_ps(buffer, _sum2_x2);
                    float sum2_x2 = buffer[0] + buffer[1] + buffer[2] + buffer[3] + buffer[4] + buffer[5] + buffer[6] + buffer[7];
                    _mm256_storeu_ps(buffer, _sum3_x2);
                    float sum3_x2 = buffer[0] + buffer[1] + buffer[2] + buffer[3] + buffer[4] + buffer[5] + buffer[6] + buffer[7];
                    _mm256_storeu_ps(buffer, _sum4_x2);
                    float sum4_x2 = buffer[0] + buffer[1] + buffer[2] + buffer[3] + buffer[4] + buffer[5] + buffer[6] + buffer[7];

                    for (size_t i = tail; i < area; i++) {
                        sum1_x += input_data1[i];
                        sum1_x2 += input_data1[i] * input_data1[i];

                        sum2_x += input_data2[i];
                        sum2_x2 += input_data2[i] * input_data2[i];

                        sum3_x += input_data3[i];
                        sum3_x2 += input_data3[i] * input_data3[i];

                        sum4_x += input_data4[i];
                        sum4_x2 += input_data4[i] * input_data4[i];
                    }

                    float mean1_x = sum1_x / area;
                    float mean2_x = sum2_x / area;
                    float mean3_x = sum3_x / area;
                    float mean4_x = sum4_x / area;
                    float mean1_x2 = sum1_x2 / area;
                    float mean2_x2 = sum2_x2 / area;
                    float mean3_x2 = sum3_x2 / area;
                    float mean4_x2 = sum4_x2 / area;
                    float variance1 = mean1_x2 - mean1_x * mean1_x;
                    float variance2 = mean2_x2 - mean2_x * mean2_x;
                    float variance3 = mean3_x2 - mean3_x * mean3_x;
                    float variance4 = mean4_x2 - mean4_x * mean4_x;
                    variance1 = variance1 > 0 ? variance1 : 0;
                    variance2 = variance2 > 0 ? variance2 : 0;
                    variance3 = variance3 > 0 ? variance3 : 0;
                    variance4 = variance4 > 0 ? variance4 : 0;
                    variance1 = 1.0f / sqrt(variance1 + epsilon);
                    variance2 = 1.0f / sqrt(variance2 + epsilon);
                    variance3 = 1.0f / sqrt(variance3 + epsilon);
                    variance4 = 1.0f / sqrt(variance4 + epsilon);
                    variance1 *= k_data[c];
                    variance2 *= k_data[c + 1];
                    variance3 *= k_data[c + 2];
                    variance4 *= k_data[c + 3];
                    float b1 = b_data == NULL ? 0.0f : b_data[c];
                    float b2 = b_data == NULL ? 0.0f : b_data[c + 1];
                    float b3 = b_data == NULL ? 0.0f : b_data[c + 2];
                    float b4 = b_data == NULL ? 0.0f : b_data[c + 3];
                    b1 -= mean1_x * variance1;
                    b2 -= mean2_x * variance2;
                    b3 -= mean3_x * variance3;
                    b4 -= mean4_x * variance4;

                    _sum1_x = _mm256_broadcast_ss(&variance1);
                    _sum2_x = _mm256_broadcast_ss(&variance2);
                    _sum3_x = _mm256_broadcast_ss(&variance3);
                    _sum4_x = _mm256_broadcast_ss(&variance4);
                    _sum1_x2 = _mm256_broadcast_ss(&b1);
                    _sum2_x2 = _mm256_broadcast_ss(&b2);
                    _sum3_x2 = _mm256_broadcast_ss(&b3);
                    _sum4_x2 = _mm256_broadcast_ss(&b4);

                    for (size_t i = 0; i < tail; i += 8) {
                        _temp1 = _mm256_loadu_ps(input_data1 + i);
                        _temp2 = _mm256_loadu_ps(input_data2 + i);
                        _temp3 = _mm256_loadu_ps(input_data3 + i);
                        _temp4 = _mm256_loadu_ps(input_data4 + i);
#ifdef __AVX2__
                        _temp1 = _mm256_fmadd_ps(_temp1, _sum1_x, _sum1_x2);
                        _temp2 = _mm256_fmadd_ps(_temp2, _sum2_x, _sum2_x2);
                        _temp3 = _mm256_fmadd_ps(_temp3, _sum3_x, _sum3_x2);
                        _temp4 = _mm256_fmadd_ps(_temp4, _sum4_x, _sum4_x2);
#else
                        _temp1 = _mm256_add_ps(_mm256_mul_ps(_temp1, _sum1_x), _sum1_x2);
                        _temp2 = _mm256_add_ps(_mm256_mul_ps(_temp2, _sum2_x), _sum2_x2);
                        _temp3 = _mm256_add_ps(_mm256_mul_ps(_temp3, _sum3_x), _sum3_x2);
                        _temp4 = _mm256_add_ps(_mm256_mul_ps(_temp4, _sum4_x), _sum4_x2);
#endif
                        _mm256_storeu_ps(output_data1 + i, _temp1);
                        _mm256_storeu_ps(output_data2 + i, _temp2);
                        _mm256_storeu_ps(output_data3 + i, _temp3);
                        _mm256_storeu_ps(output_data4 + i, _temp4);
                    }

                    for (size_t i = tail; i < area; i++) {
                        *(output_data1 + i) = *(input_data1 + i) * variance1 + b1;
                        *(output_data2 + i) = *(input_data2 + i) * variance2 + b2;
                        *(output_data3 + i) = *(input_data3 + i) * variance3 + b3;
                        *(output_data4 + i) = *(input_data4 + i) * variance4 + b4;
                    }
#else
                    float buffer[4];
                    __m128 _sum1_x  = _mm_setzero_ps();
                    __m128 _sum2_x  = _mm_setzero_ps();
                    __m128 _sum3_x  = _mm_setzero_ps();
                    __m128 _sum4_x  = _mm_setzero_ps();
                    __m128 _sum1_x2 = _mm_setzero_ps();
                    __m128 _sum2_x2 = _mm_setzero_ps();
                    __m128 _sum3_x2 = _mm_setzero_ps();
                    __m128 _sum4_x2 = _mm_setzero_ps();

                    const int tail = area - area % 4;
                    double temp;
                    __m128 _temp1, _temp2, _temp3, _temp4;
                    float *input_data1 = input_data + c * area;
                    float *input_data2 = input_data + (c + 1) * area;
                    float *input_data3 = input_data + (c + 2) * area;
                    float *input_data4 = input_data + (c + 3) * area;
                    float *output_data1 = output_data + c * area;
                    float *output_data2 = output_data + (c + 1) * area;
                    float *output_data3 = output_data + (c + 2) * area;
                    float *output_data4 = output_data + (c + 3) * area;

                    for (size_t i = 0; i < tail; i += 4) {
                        _temp1 = _mm_loadu_ps(input_data1 + i);
                        _temp2 = _mm_loadu_ps(input_data2 + i);
                        _temp3 = _mm_loadu_ps(input_data3 + i);
                        _temp4 = _mm_loadu_ps(input_data4 + i);
                        _sum1_x = _mm_add_ps(_sum1_x, _temp1);
                        _sum2_x = _mm_add_ps(_sum2_x, _temp2);
                        _sum3_x = _mm_add_ps(_sum3_x, _temp3);
                        _sum4_x = _mm_add_ps(_sum4_x, _temp4);

                        _sum1_x2 = _mm_add_ps(_mm_mul_ps(_temp1, _temp1), _sum1_x2);
                        _sum2_x2 = _mm_add_ps(_mm_mul_ps(_temp2, _temp2), _sum2_x2);
                        _sum3_x2 = _mm_add_ps(_mm_mul_ps(_temp3, _temp3), _sum3_x2);
                        _sum4_x2 = _mm_add_ps(_mm_mul_ps(_temp4, _temp4), _sum4_x2);
                    }

                    float sum1_x, sum2_x, sum3_x, sum4_x, sum1_x2, sum2_x2, sum3_x2, sum4_x2;
                    _mm_storeu_ps(buffer, _sum1_x);
                    sum1_x = buffer[0] + buffer[1] + buffer[2] + buffer[3];
                    _mm_storeu_ps(buffer, _sum2_x);
                    sum2_x = buffer[0] + buffer[1] + buffer[2] + buffer[3];
                    _mm_storeu_ps(buffer, _sum3_x);
                    sum3_x = buffer[0] + buffer[1] + buffer[2] + buffer[3];
                    _mm_storeu_ps(buffer, _sum4_x);
                    sum4_x = buffer[0] + buffer[1] + buffer[2] + buffer[3];
                    _mm_storeu_ps(buffer, _sum1_x2);
                    sum1_x2 = buffer[0] + buffer[1] + buffer[2] + buffer[3];
                    _mm_storeu_ps(buffer, _sum2_x2);
                    sum2_x2 = buffer[0] + buffer[1] + buffer[2] + buffer[3];
                    _mm_storeu_ps(buffer, _sum3_x2);
                    sum3_x2 = buffer[0] + buffer[1] + buffer[2] + buffer[3];
                    _mm_storeu_ps(buffer, _sum4_x2);
                    sum4_x2 = buffer[0] + buffer[1] + buffer[2] + buffer[3];

                    for (size_t i = tail; i < area; i++) {
                        sum1_x += input_data1[i];
                        sum1_x2 += input_data1[i] * input_data1[i];

                        sum2_x += input_data2[i];
                        sum2_x2 += input_data2[i] * input_data2[i];

                        sum3_x += input_data3[i];
                        sum3_x2 += input_data3[i] * input_data3[i];

                        sum4_x += input_data4[i];
                        sum4_x2 += input_data4[i] * input_data4[i];
                    }

                    float mean1_x = sum1_x / area;
                    float mean2_x = sum2_x / area;
                    float mean3_x = sum3_x / area;
                    float mean4_x = sum4_x / area;
                    float mean1_x2 = sum1_x2 / area;
                    float mean2_x2 = sum2_x2 / area;
                    float mean3_x2 = sum3_x2 / area;
                    float mean4_x2 = sum4_x2 / area;
                    float variance1 = mean1_x2 - mean1_x * mean1_x;
                    float variance2 = mean2_x2 - mean2_x * mean2_x;
                    float variance3 = mean3_x2 - mean3_x * mean3_x;
                    float variance4 = mean4_x2 - mean4_x * mean4_x;
                    variance1 = variance1 > 0 ? variance1 : 0;
                    variance2 = variance2 > 0 ? variance2 : 0;
                    variance3 = variance3 > 0 ? variance3 : 0;
                    variance4 = variance4 > 0 ? variance4 : 0;
                    variance1 = 1.0f / sqrt(variance1 + epsilon);
                    variance2 = 1.0f / sqrt(variance2 + epsilon);
                    variance3 = 1.0f / sqrt(variance3 + epsilon);
                    variance4 = 1.0f / sqrt(variance4 + epsilon);
                    variance1 *= k_data[c];
                    variance2 *= k_data[c + 1];
                    variance3 *= k_data[c + 2];
                    variance4 *= k_data[c + 3];
                    float b1 = b_data == NULL ? 0.0f : b_data[c];
                    float b2 = b_data == NULL ? 0.0f : b_data[c + 1];
                    float b3 = b_data == NULL ? 0.0f : b_data[c + 2];
                    float b4 = b_data == NULL ? 0.0f : b_data[c + 3];
                    b1 -= mean1_x * variance1;
                    b2 -= mean2_x * variance2;
                    b3 -= mean3_x * variance3;
                    b4 -= mean4_x * variance4;

                    _sum1_x = _mm_load1_ps(&variance1);
                    _sum2_x = _mm_load1_ps(&variance2);
                    _sum3_x = _mm_load1_ps(&variance3);
                    _sum4_x = _mm_load1_ps(&variance4);
                    _sum1_x2 = _mm_load1_ps(&b1);
                    _sum2_x2 = _mm_load1_ps(&b2);
                    _sum3_x2 = _mm_load1_ps(&b3);
                    _sum4_x2 = _mm_load1_ps(&b4);

                    for (size_t i = 0; i < tail; i += 4) {
                        _temp1 = _mm_loadu_ps(input_data1 + i);
                        _temp2 = _mm_loadu_ps(input_data2 + i);
                        _temp3 = _mm_loadu_ps(input_data3 + i);
                        _temp4 = _mm_loadu_ps(input_data4 + i);
                        _temp1 = _mm_add_ps(_mm_mul_ps(_temp1, _sum1_x), _sum1_x2);
                        _temp2 = _mm_add_ps(_mm_mul_ps(_temp2, _sum2_x), _sum2_x2);
                        _temp3 = _mm_add_ps(_mm_mul_ps(_temp3, _sum3_x), _sum3_x2);
                        _temp4 = _mm_add_ps(_mm_mul_ps(_temp4, _sum4_x), _sum4_x2);
                        _mm_storeu_ps(output_data1 + i, _temp1);
                        _mm_storeu_ps(output_data2 + i, _temp2);
                        _mm_storeu_ps(output_data3 + i, _temp3);
                        _mm_storeu_ps(output_data4 + i, _temp4);
                    }

                    for (size_t i = tail; i < area; i++) {
                        *(output_data1 + i) = *(input_data1 + i) * variance1 + b1;
                        *(output_data2 + i) = *(input_data2 + i) * variance2 + b2;
                        *(output_data3 + i) = *(input_data3 + i) * variance3 + b3;
                        *(output_data4 + i) = *(input_data4 + i) * variance4 + b4;
                    }
#endif
                }
            }
        }
    } else {
        LOGE("Error: layer acc dont support datatype: %d\n", output_blob->GetBlobDesc().data_type);
        return Status(TNNERR_MODEL_ERR, "Error: layer acc dont support datatype");
    }

    return TNN_OK;
}

REGISTER_X86_ACC(InstanceNorm, LAYER_INST_BATCH_NORM);
}