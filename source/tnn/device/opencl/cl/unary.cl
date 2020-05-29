#include "base.inc"

__kernel void Unary(GLOBAL_SIZE_3_DIMS __read_only image2d_t input, __write_only image2d_t output) {
    const int w                 = get_global_id(0);
    const int channel_block_idx = get_global_id(1);
    const int hb                = get_global_id(2);

    DEAL_NON_UNIFORM_DIM3(w, channel_block_idx, hb);
    const int width = global_size_dim0;

    const int pos  = mad24(channel_block_idx, width, w);
    FLOAT4 in  = RI_F(input, SAMPLER, (int2)(pos, hb));
    FLOAT4 out = OPERATOR;
    WI_F(output, (int2)(pos, hb), out);
}
