#include "base.inc"

__kernel void PRelu(GLOBAL_SIZE_2_DIMS __read_only image2d_t input,
                    __private const int width, __read_only image2d_t scope,
                    __write_only image2d_t output) {
    const int cw_idx      = get_global_id(0);
    const int bh_idx      = get_global_id(1);
    const int c_block_idx = cw_idx / width;

    DEAL_NON_UNIFORM_DIM2(cw_idx, bh_idx);

    FLOAT4 in    = RI_F(input, SAMPLER, (int2)(cw_idx, bh_idx));
    FLOAT4 val_s = RI_F(scope, SAMPLER, (int2)(c_block_idx, 0));
    FLOAT4 out   = select(in, in * val_s, in < 0);
    WI_F(output, (int2)(cw_idx, bh_idx), out);
}

__kernel void PReluGS3D(GLOBAL_SIZE_3_DIMS __read_only image2d_t input,
                        __read_only image2d_t scope,
                        __write_only image2d_t output) {
    const int w_idx       = get_global_id(0);
    const int c_block_idx = get_global_id(1);
    const int hb_idx      = get_global_id(2);

    DEAL_NON_UNIFORM_DIM3(w_idx, c_block_idx, hb_idx);
    const int width = global_size_dim0;

    const int pos = mad24(c_block_idx, width, w_idx);

    FLOAT4 in    = RI_F(input, SAMPLER, (int2)(pos, hb_idx));
    FLOAT4 val_s = RI_F(scope, SAMPLER, (int2)(c_block_idx, 0));
    FLOAT4 out   = select(in, in * val_s, in < 0);
    WI_F(output, (int2)(pos, hb_idx), out);
}
