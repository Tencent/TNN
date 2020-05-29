#include "base.inc"

__kernel void NormalizeCommon0(GLOBAL_SIZE_2_DIMS __read_only image2d_t input,
                                 __private const int channel_blocks,
                                 __private const int width,
                                 __private const float eps,
                                 __write_only image2d_t output) {
    const int w_idx  = get_global_id(0);
    const int bh_idx = get_global_id(1);

    DEAL_NON_UNIFORM_DIM2(w_idx, bh_idx);

    FLOAT4 sum_xyzw = (FLOAT4)(0.0f, 0.0f, 0.0f, 0.0f);
    int cw_idx      = w_idx;
    for (int cb = 0; cb < channel_blocks; ++cb) {
        FLOAT4 val = RI_F(input, SAMPLER, (int2)(cw_idx, bh_idx));
#ifdef NORMALIZE_P2
        sum_xyzw += val * val;
#else
        sum_xyzw += fabs(val);
#endif
        cw_idx += width;
    }
    FLOAT sum = sum_xyzw.x + sum_xyzw.y + sum_xyzw.z + sum_xyzw.w;
#ifdef NORMALIZE_P2
    sum = max(sqrt(sum), (FLOAT)(eps));
#endif

    for (int cb = 0; cb < channel_blocks; ++cb) {
        int2 pos   = (int2)(mad24(cb, width, w_idx), bh_idx);
        FLOAT4 val = RI_F(input, SAMPLER, pos) / sum;
        WI_F(output, pos, val);
    }
}

__kernel void NormalizeCommon(GLOBAL_SIZE_2_DIMS __read_only image2d_t input,
                               __private const int channel_blocks,
                               __private const int channel_remain,
                               __private const int width,
                               __private const float eps,
                               __write_only image2d_t output) {
    const int w_idx  = get_global_id(0);
    const int bh_idx = get_global_id(1);

    DEAL_NON_UNIFORM_DIM2(w_idx, bh_idx);

    FLOAT4 sum_xyzw = (FLOAT4)(0.0f, 0.0f, 0.0f, 0.0f);
    int cw_idx      = w_idx;
    for (int cb = 0; cb < (channel_blocks - 1); ++cb) {
        FLOAT4 val = RI_F(input, SAMPLER, (int2)(cw_idx, bh_idx));
#ifdef NORMALIZE_P2
        sum_xyzw += val * val;
#else
        sum_xyzw += fabs(val);
#endif
        cw_idx += width;
    }
    FLOAT sum = sum_xyzw.x + sum_xyzw.y + sum_xyzw.z + sum_xyzw.w;

    FLOAT4 val_last = RI_F(input, SAMPLER, (int2)(cw_idx, bh_idx));
#ifdef NORMALIZE_P2
    val_last = val_last * val_last;
#else
    val_last = fabs(val_last);
#endif
    if (1 == channel_remain) {
        sum += val_last.x;
    } else if (2 == channel_remain) {
        sum += val_last.x;
        sum += val_last.y;
    } else if (3 == channel_remain) {
        sum += val_last.x;
        sum += val_last.y;
        sum += val_last.z;
    } else {
        sum += val_last.x;
        sum += val_last.y;
        sum += val_last.z;
        sum += val_last.w;
    }

#ifdef NORMALIZE_P2
    sum = max(sqrt(sum), (FLOAT)(eps));
#endif

    for (int cb = 0; cb < channel_blocks; ++cb) {
        int2 pos   = (int2)(mad24(cb, width, w_idx), bh_idx);
        FLOAT4 val = RI_F(input, SAMPLER, pos) / sum;
        WI_F(output, pos, val);
    }
}
