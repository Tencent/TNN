#include "base.inc"

__kernel void SignedMul(GLOBAL_SIZE_3_DIMS __read_only image2d_t input, __write_only image2d_t output, float alpha, float beta, float gamma_inv) {
    const int w_idx       = get_global_id(0);
    const int c_block_idx = get_global_id(1);
    const int hb_idx      = get_global_id(2);
    DEAL_NON_UNIFORM_DIM3(w_idx, c_block_idx, hb_idx);
    const int width = global_size_dim0;

    const int pos = mad24(c_block_idx, width, w_idx);

    FLOAT4 value    = RI_F(input, SAMPLER, (int2)(pos, hb_idx));    
    FLOAT4 mul   = RI_F(input, SAMPLER, (int2)(w_dix, hb_idx));
    value = value - (FLOAT4)(alpha);
    value = select(select(value,(FLOAT4)(-1),value<(FLOAT4)0),(FLOAT4)1,value>(FLOAT4)0);
    value = (value + (FLOAT4)(beta)) * (FLOAT4)(gamma_inv * mul.s0);
    WI_F(output, SAMPLER, (int2)(pos, hb_idx));    
}
