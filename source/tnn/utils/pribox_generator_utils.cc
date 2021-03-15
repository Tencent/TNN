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

#include "tnn/utils/pribox_generator_utils.h"

#include <algorithm>
#include <cmath>

#include "tnn/utils/dims_utils.h"

namespace TNN_NS {

std::vector<float> GeneratePriorBox(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs,
                                    PriorBoxLayerParam *param) {
    // layer size
    const int layer_height = inputs[0]->GetBlobDesc().dims[2];
    const int layer_width  = inputs[0]->GetBlobDesc().dims[3];

    // image size
    int img_height = param->img_h;
    int img_width  = param->img_w;
    if (img_height == 0 || img_width == 0) {
        img_height = inputs[1]->GetBlobDesc().dims[2];
        img_width  = inputs[1]->GetBlobDesc().dims[3];
    }

    // step
    float step_h = param->step_h;
    float step_w = param->step_w;
    if (step_w == 0 || step_h == 0) {
        step_h = static_cast<float>(img_height) / layer_height;
        step_w = static_cast<float>(img_width) / layer_width;
    }

    int pribox_size = DimsVectorUtils::Count(outputs[0]->GetBlobDesc().dims, 1);
    std::vector<float> priorbox(pribox_size);

    int num_priors = outputs[0]->GetBlobDesc().dims[2] / (layer_height * layer_width * 4);

    float offset = param->offset;
    int dim      = outputs[0]->GetBlobDesc().dims[2];
    int idx      = 0;
    for (int h = 0; h < layer_height; ++h) {
        for (int w = 0; w < layer_width; ++w) {
            float center_x = (w + offset) * step_w;
            float center_y = (h + offset) * step_h;
            float box_width, box_height;

            for (int s = 0; s < param->min_sizes.size(); ++s) {
                int min_size = static_cast<int>(param->min_sizes[s]);
                box_width = box_height = static_cast<float>(min_size);
                // xmin
                priorbox[idx++] = (center_x - box_width / 2.f) / img_width;
                // ymin
                priorbox[idx++] = (center_y - box_height / 2.f) / img_height;
                // xmax
                priorbox[idx++] = (center_x + box_width / 2.f) / img_width;
                // ymax
                priorbox[idx++] = (center_y + box_height / 2.f) / img_height;
                // if we have max_size
                if (param->max_sizes.size() > 0) {
                    int max_size = static_cast<int>(param->max_sizes[s]);
                    // second prior: aspect_ratio = 1, size = sqrt(min_size *
                    // max_size)
                    box_width = box_height = static_cast<float>(sqrtf(float(min_size * max_size)));
                    // xmin
                    priorbox[idx++] = (center_x - box_width / 2.f) / img_width;
                    // ymin
                    priorbox[idx++] = (center_y - box_height / 2.f) / img_height;
                    // xmax
                    priorbox[idx++] = (center_x + box_width / 2.f) / img_width;
                    // ymax
                    priorbox[idx++] = (center_y + box_height / 2.f) / img_height;
                }
                for (int r = 0; r < param->aspect_ratios.size(); ++r) {
                    float ar = param->aspect_ratios[r];
                    if (fabs(ar - 1.) < 1e-6) {
                        continue;
                    }
                    box_width  = min_size * sqrt(ar);
                    box_height = min_size / sqrt(ar);
                    // xmin
                    priorbox[idx++] = (center_x - box_width / 2.f) / img_width;
                    // ymin
                    priorbox[idx++] = (center_y - box_height / 2.f) / img_height;
                    // xmax
                    priorbox[idx++] = (center_x + box_width / 2.f) / img_width;
                    // ymax
                    priorbox[idx++] = (center_y + box_height / 2.f) / img_height;
                }
            }
        }
    }

    // clip the prior's coordiate such that it is within [0, 1]
    if (param->clip) {
        for (int d = 0; d < dim; ++d) {
            priorbox[d] = std::min<float>(std::max<float>(priorbox[d], 0), 1);
        }
    }

    // set the variance.
    if (param->variances.size() == 1) {
        for (int vi = 0; vi < dim; vi++) {
            priorbox[dim + vi] = param->variances[0];
        }
    } else {
        int count = 0;
        for (int index = 0; index < layer_height * layer_width * num_priors; index++) {
            for (int j = 0; j < 4; ++j) {
                priorbox[dim + count] = param->variances[j];
                ++count;
            }
        }
    }
    return priorbox;
}

}  // namespace TNN_NS
