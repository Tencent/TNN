#include "base.inc"

__kernel void ReduceMaxC(GLOBAL_SIZE_2_DIMS __read_only image2d_t input,
                         __write_only image2d_t output,
                         __private const int channels_block,
                         __private const int remain_channels) {
    const int width_idx        = get_global_id(0);
    const int batch_height_idx = get_global_id(1);

    DEAL_NON_UNIFORM_DIM2(width_idx, batch_height_idx);

    FLOAT4 temp_data;
    FLOAT4 max_result   = (FLOAT4)(-FLT_MAX);
    int image_width_idx = width_idx;
    for (short i = 0; i < channels_block - 1; ++i) {
        temp_data =
            RI_F(input, SAMPLER, (int2)(image_width_idx, batch_height_idx));
        max_result = max(temp_data, max_result);
        image_width_idx += global_size_dim0;
    }

    temp_data = RI_F(input, SAMPLER, (int2)(image_width_idx, batch_height_idx));
    if (remain_channels == 1) {
        temp_data.w = (FLOAT)(-FLT_MAX);
    } else if (remain_channels == 2) {
        temp_data.z = (FLOAT)(-FLT_MAX);
        temp_data.w = (FLOAT)(-FLT_MAX);
    } else if (remain_channels == 3) {
        temp_data.y = (FLOAT)(-FLT_MAX);
        temp_data.z = (FLOAT)(-FLT_MAX);
        temp_data.w = (FLOAT)(-FLT_MAX);
    }
    max_result = max(temp_data, max_result);
    FLOAT result =
        max(max(max_result.x, max_result.y), max(max_result.z, max_result.w));

    WI_F(output, (int2)(width_idx, batch_height_idx),
         (FLOAT4)(result, (FLOAT)0.0f, (FLOAT)0.0f, (FLOAT)0.0f));
}
