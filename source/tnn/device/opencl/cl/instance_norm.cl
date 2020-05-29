#include "base.inc"

__kernel void InstanceNormVarBias_LocalMem(
    GLOBAL_SIZE_2_DIMS __read_only image2d_t input, __read_only image2d_t scale,
    __read_only image2d_t bias, __private const int c_block, __private const int height,
    __private const int width, __private const int hxw,
    __write_only image2d_t var_out, __write_only image2d_t bias_out) {
    const int thread_idx = get_global_id(0);
    const int bc_idx     = get_global_id(1);

    DEAL_NON_UNIFORM_DIM2(thread_idx, bc_idx);

    __local float4 local_temp_x[THREAD_BLOCK_W * THREAD_BLOCK_W];
    __local float4 local_temp_x2[THREAD_BLOCK_W * THREAD_BLOCK_W];

    const int c_block_idx = bc_idx % c_block;
    const int b_idx       = bc_idx / c_block;
    const int w_offset = thread_idx % THREAD_BLOCK_W;
    const int h_offset = thread_idx / THREAD_BLOCK_W;

    float4 sum_x = (float4)0.0f;
    float4 sum_x2 = (float4)0.0f;
    float4 val;
    int h, w;
    int2 pos_base = (int2)(mul24(c_block_idx, width), mul24(b_idx, height));
    int2 pos;
    for (h = h_offset, pos.y = pos_base.y + h_offset; h < height; h += THREAD_BLOCK_W) {
        for (w = w_offset, pos.x = pos_base.x + w_offset; w < width; w += THREAD_BLOCK_W) {
            val = read_imagef(input, SAMPLER, pos);
            sum_x += val;
            sum_x2 = mad(val, val, sum_x2);
            pos.x += THREAD_BLOCK_W;
        }
        pos.y += THREAD_BLOCK_W;
    }
    local_temp_x[thread_idx] = sum_x;
    local_temp_x2[thread_idx] = sum_x2;

    barrier(CLK_LOCAL_MEM_FENCE);

    if (thread_idx == 0) {
        sum_x = (float4)0.0f;
        sum_x2 = (float4)0.0f;
        for (int i = 0; i < THREAD_BLOCK_W * THREAD_BLOCK_W; ++i) {
            sum_x += local_temp_x[i];
            sum_x2 += local_temp_x2[i];
        }
        float4 mean = sum_x / (float)(hxw);
        float4 varience = sum_x2 / (float)(hxw) - mean * mean;
        varience        = 1.0f / sqrt(varience + .00001f);

        float4 k_val = read_imagef(scale, SAMPLER, (int2)(c_block_idx, 0));
        float4 b_val = read_imagef(bias, SAMPLER, (int2)(c_block_idx, 0));

        varience *= k_val;
        b_val -= mean * varience;

        write_imagef(var_out, (int2)(c_block_idx, b_idx), varience);
        write_imagef(bias_out, (int2)(c_block_idx, b_idx), b_val);
    }
}
