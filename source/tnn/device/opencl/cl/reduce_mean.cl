#include "base.inc"

__kernel void ReduceMeanC(GLOBAL_SIZE_2_DIMS __read_only image2d_t input,
                          __write_only image2d_t output,
                          __private const float channel_inv,
                          __private const int channels_block,
                          __private const int remain_channels) {
    const int width_idx        = get_global_id(0);
    const int batch_height_idx = get_global_id(1);

    DEAL_NON_UNIFORM_DIM2(width_idx, batch_height_idx);

    FLOAT4 temp_data;
    FLOAT4 accum_result = (FLOAT4)0.0f;
    int image_width_idx = width_idx;
    for (short i = 0; i < channels_block - 1; ++i) {
        temp_data =
            RI_F(input, SAMPLER, (int2)(image_width_idx, batch_height_idx));
        accum_result += temp_data;
        image_width_idx += global_size_dim0;
    }

    temp_data = RI_F(input, SAMPLER, (int2)(image_width_idx, batch_height_idx));
    if (remain_channels == 1) {
        temp_data.w = (FLOAT)0.0f;
    } else if (remain_channels == 2) {
        temp_data.z = (FLOAT)0.0f;
        temp_data.w = (FLOAT)0.0f;
    } else if (remain_channels == 3) {
        temp_data.y = (FLOAT)0.0f;
        temp_data.z = (FLOAT)0.0f;
        temp_data.w = (FLOAT)0.0f;
    }
    accum_result += temp_data;
    FLOAT result =
        accum_result.x + accum_result.y + accum_result.z + accum_result.w;
    result *= (FLOAT)channel_inv;

    WI_F(output, (int2)(width_idx, batch_height_idx),
         (FLOAT4)(result, (FLOAT)0.0f, (FLOAT)0.0f, (FLOAT)0.0f));
}
