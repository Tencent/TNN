//  Copyright © 2019 tencent. All rights reserved.

#include <metal_math>
#include <metal_stdlib>
#include "tnn/device/metal/acc/metal_common.metal"

using namespace metal;

kernel void lrn_across_channel(const device ftype4 *src                [[buffer(0)]],
                               device ftype4 *dst                      [[buffer(1)]],
                               constant MetalLRNParams &params         [[buffer(2)]],
                               uint3 gid                               [[thread_position_in_grid]]) {
    if (any(gid >= uint3(params.output_width, params.output_height,
                         params.output_slice * params.batch))) {
        return;
    }

    int src_index = (int)gid.z * params.input_size +
                    (int)gid.y * params.input_width + (int)gid.x;
    int dst_index = (int)gid.z * params.output_size +
                    (int)gid.y * params.output_width + (int)gid.x;

    auto src_data_ptr = src + src_index;
    auto dst_data_ptr = dst + dst_index;

    int half_size = params.size / 2;
    float4 sum    = 0;
    for (int k = -half_size; k <= half_size; k++) {
        int4 j4  = int4(0, 1, 2, 3) + k;
        int4 z4  = int4(floor(float4(j4) / 4));
        int4 r4  = j4 - z4 * 4;
        int4 c4  = gid.z * 4 + j4;
        bool4 v4 = 0 <= c4 && c4 < params.input_channel;

        if (v4[0]) {
            float in4 = float(src_data_ptr[z4[0] * params.input_size][r4[0]]);
            sum[0] += in4 * in4;
        }
        if (v4[1]) {
            float in4 = float(src_data_ptr[z4[1] * params.input_size][r4[1]]);
            sum[1] += in4 * in4;
        }
        if (v4[2]) {
            float in4 = float(src_data_ptr[z4[2] * params.input_size][r4[2]]);
            sum[2] += in4 * in4;
        }
        if (v4[3]) {
            float in4 = float(src_data_ptr[z4[3] * params.input_size][r4[3]]);
            sum[3] += in4 * in4;
        }
    }

    *dst_data_ptr = *src_data_ptr *
                    ftype4(pow(params.bias + (params.alpha / params.size) * sum,
                               -params.beta));
}
