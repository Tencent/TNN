#include "base.inc"

__kernel void NHC4W4ImageToCNH4Image(GLOBAL_SIZE_2_DIMS __read_only image2d_t input, /* nhc4w4 */
                                     __private const int n,
                                     __private const int h,
                                     __write_only image2d_t output) {
    int h_updiv_4_idx  = get_global_id(0);
    int c_n_idx = get_global_id(1);

    DEAL_NON_UNIFORM_DIM2(h_updiv_4_idx, c_n_idx);

    const int c_idx = c_n_idx / n;
    const int n_idx = c_n_idx % n;

    const int h0 = h_updiv_4_idx << 2;
    const int h1 = h0 + 1;
    const int h2 = h1 + 1;
    const int h3 = h2 + 1;

    int c_4_idx         = c_idx / 4;
    int c_remain_4_idx  = c_idx % 4;

    int2 src_coord_h0 = (int2)(c_4_idx, n_idx * h + h0);
    int2 src_coord_h1 = (int2)(c_4_idx, n_idx * h + h1);
    int2 src_coord_h2 = (int2)(c_4_idx, n_idx * h + h2);
    int2 src_coord_h3 = (int2)(c_4_idx, n_idx * h + h3);
    int2 dst_coord = (int2)(h_updiv_4_idx, c_n_idx);

    FLOAT4 values_h0 = RI_F(input, SAMPLER, src_coord_h0);
    FLOAT4 values_h1 = RI_F(input, SAMPLER, src_coord_h1);
    FLOAT4 values_h2 = RI_F(input, SAMPLER, src_coord_h2);
    FLOAT4 values_h3 = RI_F(input, SAMPLER, src_coord_h3);

    FLOAT4 out;
    if (c_remain_4_idx == 0) {
        out.x = values_h0.x;
        out.y = values_h1.x;
        out.z = values_h2.x;
        out.w = values_h3.x;
    } else if (c_remain_4_idx == 1) {
        out.x = values_h0.y;
        out.y = values_h1.y;
        out.z = values_h2.y;
        out.w = values_h3.y;
    } else if (c_remain_4_idx == 2) {
        out.x = values_h0.z;
        out.y = values_h1.z;
        out.z = values_h2.z;
        out.w = values_h3.z;
    } else if (c_remain_4_idx == 3) {
        out.x = values_h0.w;
        out.y = values_h1.w;
        out.z = values_h2.w;
        out.w = values_h3.w;
    }

    WI_F(output, dst_coord, out);
}

__kernel void CNH4ImageToNHC4W4Image(GLOBAL_SIZE_2_DIMS __read_only image2d_t input, /* nhc4w4 */
                                     __private const int n,
                                     __private const int h,
                                     __write_only image2d_t output) {
    int c_updiv_4_idx  = get_global_id(0);
    int n_h_idx = get_global_id(1);

    DEAL_NON_UNIFORM_DIM2(c_updiv_4_idx, n_h_idx);

    const int h_idx = n_h_idx % h;
    const int n_idx = n_h_idx / h;

    const int c0 = c_updiv_4_idx << 2;
    const int c1 = c0 + 1;
    const int c2 = c1 + 1;
    const int c3 = c2 + 1;

    int h_4_idx        = h_idx / 4;
    int h_remain_4_idx = h_idx % 4;

    int2 src_coord_c0 = (int2)(h_4_idx, c0 * n + n_idx);
    int2 src_coord_c1 = (int2)(h_4_idx, c1 * n + n_idx);
    int2 src_coord_c2 = (int2)(h_4_idx, c2 * n + n_idx);
    int2 src_coord_c3 = (int2)(h_4_idx, c3 * n + n_idx);
    int2 dst_coord = (int2)(c_updiv_4_idx, n_h_idx);

    FLOAT4 values_c0 = RI_F(input, SAMPLER, src_coord_c0);
    FLOAT4 values_c1 = RI_F(input, SAMPLER, src_coord_c1);
    FLOAT4 values_c2 = RI_F(input, SAMPLER, src_coord_c2);
    FLOAT4 values_c3 = RI_F(input, SAMPLER, src_coord_c3);

    FLOAT4 out;
    if (h_remain_4_idx == 0) {
        out.x = values_c0.x;
        out.y = values_c1.x;
        out.z = values_c2.x;
        out.w = values_c3.x;
    } else if (h_remain_4_idx == 1) {
        out.x = values_c0.y;
        out.y = values_c1.y;
        out.z = values_c2.y;
        out.w = values_c3.y;
    } else if (h_remain_4_idx == 2) {
        out.x = values_c0.z;
        out.y = values_c1.z;
        out.z = values_c2.z;
        out.w = values_c3.z;
    } else if (h_remain_4_idx == 3) {
        out.x = values_c0.w;
        out.y = values_c1.w;
        out.z = values_c2.w;
        out.w = values_c3.w;
    }

    WI_F(output, dst_coord, out);
}