#include "base.inc"

__kernel void BilinearGridSample(GLOBAL_SIZE_2_DIMS __read_only image2d_t input, __read_only image2d_t grid,
                                 __write_only image2d_t output, __private const int input_height,
                                 __private const int input_width, __private const int out_height,
                                 __private const int out_width) {
    const int cw = get_global_id(0);
    const int hb = get_global_id(1);
    DEAL_NON_UNIFORM_DIM2(cw, hb);
    const int output_w_idx       = cw % out_width;
    const int output_c_block_idx = cw / out_width;
    const int output_b_idx       = hb / ((out_height + 3) / 4);
    const int output_h_idx       = hb % ((out_height + 3) / 4);
    FLOAT4 x = RI_F(grid, SAMPLER, (int2)(output_h_idx * 2, output_b_idx * out_width + output_w_idx));
    FLOAT4 y = RI_F(grid, SAMPLER, (int2)(output_h_idx * 2 + 1, output_b_idx * out_width + output_w_idx));

    FLOAT4 ix = (x + ((FLOAT)1.0f)) * input_width * ((FLOAT)0.5f) - ((FLOAT)0.5f);
    FLOAT4 iy = (y + ((FLOAT)1.0f)) * input_height * ((FLOAT)0.5f) - ((FLOAT)0.5f);

    FLOAT4 ix_nw = floor(ix);
    FLOAT4 iy_nw = floor(iy);

    FLOAT4 ix_ne = ix_nw + 1;
    FLOAT4 iy_ne = iy_nw;

    FLOAT4 ix_sw = ix_nw;
    FLOAT4 iy_sw = iy_nw + 1;

    FLOAT4 ix_se = ix_nw + 1;
    FLOAT4 iy_se = iy_nw + 1;

    // get nw_within_bound
    int4 low_ix_nw = (int4)(ix_nw.x >= 0 ? -1 : 0, ix_nw.y >= 0 ? -1 : 0, ix_nw.z >= 0 ? -1 : 0, ix_nw.w >= 0 ? -1 : 0);
    int4 low_iy_nw = (int4)(iy_nw.x >= 0 ? -1 : 0, iy_nw.y >= 0 ? -1 : 0, iy_nw.z >= 0 ? -1 : 0, iy_nw.w >= 0 ? -1 : 0);

    int4 up_ix_nw = (int4)(ix_nw.x < input_width ? -1 : 0, ix_nw.y < input_width ? -1 : 0,
                           ix_nw.z < input_width ? -1 : 0, ix_nw.w < input_width ? -1 : 0);
    int4 up_iy_nw = (int4)(iy_nw.x < input_height ? -1 : 0, iy_nw.y < input_height ? -1 : 0,
                           iy_nw.z < input_height ? -1 : 0, iy_nw.w < input_height ? -1 : 0);

    int4 nw_within_bound = select((int4)(0, 0, 0, 0), (int4)(-1, -1, -1, -1), low_ix_nw);
    nw_within_bound      = select((int4)(0, 0, 0, 0), nw_within_bound, low_iy_nw);
    nw_within_bound      = select((int4)(0, 0, 0, 0), nw_within_bound, up_ix_nw);
    nw_within_bound      = select((int4)(0, 0, 0, 0), nw_within_bound, up_iy_nw);

    // get ne_within_bound
    low_ix_nw = (int4)(ix_ne.x >= 0 ? -1 : 0, ix_ne.y >= 0 ? -1 : 0, ix_ne.z >= 0 ? -1 : 0, ix_ne.w >= 0 ? -1 : 0);
    low_iy_nw = (int4)(iy_ne.x >= 0 ? -1 : 0, iy_ne.y >= 0 ? -1 : 0, iy_ne.z >= 0 ? -1 : 0, iy_ne.w >= 0 ? -1 : 0);
    up_ix_nw  = (int4)(ix_ne.x < input_width ? -1 : 0, ix_ne.y < input_width ? -1 : 0, ix_ne.z < input_width ? -1 : 0,
                      ix_ne.w < input_width ? -1 : 0);
    up_iy_nw = (int4)(iy_ne.x < input_height ? -1 : 0, iy_ne.y < input_height ? -1 : 0, iy_ne.z < input_height ? -1 : 0,
                      iy_ne.w < input_height ? -1 : 0);
    int4 ne_within_bound = select((int4)(0, 0, 0, 0), (int4)(-1, -1, -1, -1), low_ix_nw);
    ne_within_bound      = select((int4)(0, 0, 0, 0), ne_within_bound, low_iy_nw);
    ne_within_bound      = select((int4)(0, 0, 0, 0), ne_within_bound, up_ix_nw);
    ne_within_bound      = select((int4)(0, 0, 0, 0), ne_within_bound, up_iy_nw);

    // get sw_within_bound
    low_ix_nw = (int4)(ix_sw.x >= 0 ? -1 : 0, ix_sw.y >= 0 ? -1 : 0, ix_sw.z >= 0 ? -1 : 0, ix_sw.w >= 0 ? -1 : 0);
    low_iy_nw = (int4)(iy_sw.x >= 0 ? -1 : 0, iy_sw.y >= 0 ? -1 : 0, iy_sw.z >= 0 ? -1 : 0, iy_sw.w >= 0 ? -1 : 0);
    up_ix_nw  = (int4)(ix_sw.x < input_width ? -1 : 0, ix_sw.y < input_width ? -1 : 0, ix_sw.z < input_width ? -1 : 0,
                      ix_sw.w < input_width ? -1 : 0);
    up_iy_nw = (int4)(iy_sw.x < input_height ? -1 : 0, iy_sw.y < input_height ? -1 : 0, iy_sw.z < input_height ? -1 : 0,
                      iy_sw.w < input_height ? -1 : 0);
    int4 sw_within_bound = select((int4)(0, 0, 0, 0), (int4)(-1, -1, -1, -1), low_ix_nw);
    sw_within_bound      = select((int4)(0, 0, 0, 0), sw_within_bound, low_iy_nw);
    sw_within_bound      = select((int4)(0, 0, 0, 0), sw_within_bound, up_ix_nw);
    sw_within_bound      = select((int4)(0, 0, 0, 0), sw_within_bound, up_iy_nw);

    // get se_within_bound
    low_ix_nw = (int4)(ix_se.x >= 0 ? -1 : 0, ix_se.y >= 0 ? -1 : 0, ix_se.z >= 0 ? -1 : 0, ix_se.w >= 0 ? -1 : 0);
    low_iy_nw = (int4)(iy_se.x >= 0 ? -1 : 0, iy_se.y >= 0 ? -1 : 0, iy_se.z >= 0 ? -1 : 0, iy_se.w >= 0 ? -1 : 0);
    up_ix_nw  = (int4)(ix_se.x < input_width ? -1 : 0, ix_se.y < input_width ? -1 : 0, ix_se.z < input_width ? -1 : 0,
                      ix_se.w < input_width ? -1 : 0);
    up_iy_nw = (int4)(iy_se.x < input_height ? -1 : 0, iy_se.y < input_height ? -1 : 0, iy_se.z < input_height ? -1 : 0,
                      iy_se.w < input_height ? -1 : 0);
    int4 se_within_bound = select((int4)(0, 0, 0, 0), (int4)(-1, -1, -1, -1), low_ix_nw);
    se_within_bound      = select((int4)(0, 0, 0, 0), se_within_bound, low_iy_nw);
    se_within_bound      = select((int4)(0, 0, 0, 0), se_within_bound, up_ix_nw);
    se_within_bound      = select((int4)(0, 0, 0, 0), se_within_bound, up_iy_nw);

    FLOAT4 nw = (ix_se - ix) * (iy_se - iy);
    FLOAT4 ne = (ix - ix_sw) * (iy_sw - iy);
    FLOAT4 sw = (ix_ne - ix) * (iy - iy_ne);
    FLOAT4 se = (ix - ix_nw) * (iy - iy_nw);

    nw = (FLOAT4)(nw_within_bound.x ? nw.x : 0, nw_within_bound.y ? nw.y : 0, nw_within_bound.z ? nw.z : 0,
                  nw_within_bound.w ? nw.w : 0);
    ne = (FLOAT4)(ne_within_bound.x ? ne.x : 0, ne_within_bound.y ? ne.y : 0, ne_within_bound.z ? ne.z : 0,
                  ne_within_bound.w ? ne.w : 0);
    sw = (FLOAT4)(sw_within_bound.x ? sw.x : 0, sw_within_bound.y ? sw.y : 0, sw_within_bound.z ? sw.z : 0,
                  sw_within_bound.w ? sw.w : 0);
    se = (FLOAT4)(se_within_bound.x ? se.x : 0, se_within_bound.y ? se.y : 0, se_within_bound.z ? se.z : 0,
                  se_within_bound.w ? se.w : 0);
    //    FLOAT4 ne = select((FLOAT)0, (ix - ix_sw) * (iy_sw - iy), ne_within_bound);
    //    FLOAT4 sw = select((FLOAT)0, (ix_ne - ix) * (iy - iy_ne), sw_within_bound);
    //    FLOAT4 se = select((FLOAT)0, (ix - ix_nw) * (iy - iy_nw), se_within_bound);

    int nw_index_x = select(0, (int)(ix_nw.x), nw_within_bound.x);
    int nw_index_y = select(0, (int)(iy_nw.x), nw_within_bound.x);
    FLOAT4 res     = (FLOAT4)(0, 0, 0, 0);
    FLOAT4 input_data_nw =
        RI_F(input, SAMPLER,
             (int2)(output_c_block_idx * input_width + nw_index_x, output_b_idx * input_height + nw_index_y));
    res            = res + input_data_nw * nw.x;
    int ne_index_x = select(0, (int)ix_ne.x, ne_within_bound.x);
    int ne_index_y = select(0, (int)iy_ne.x, ne_within_bound.x);
    FLOAT4 input_data_ne =
        RI_F(input, SAMPLER,
             (int2)(output_c_block_idx * input_width + ne_index_x, output_b_idx * input_height + ne_index_y));
    res            = res + input_data_ne * ne.x;
    int sw_index_x = select(0, (int)ix_sw.x, sw_within_bound.x);
    int sw_index_y = select(0, (int)iy_sw.x, sw_within_bound.x);
    FLOAT4 input_data_sw =
        RI_F(input, SAMPLER,
             (int2)(output_c_block_idx * input_width + sw_index_x, output_b_idx * input_height + sw_index_y));
    res            = res + input_data_sw * sw.x;
    int se_index_x = select(0, (int)ix_se.x, se_within_bound.x);
    int se_index_y = select(0, (int)iy_se.x, se_within_bound.x);
    FLOAT4 input_data_se =
        RI_F(input, SAMPLER,
             (int2)(output_c_block_idx * input_width + se_index_x, output_b_idx * input_height + se_index_y));
    res = res + input_data_se * se.x;
    WI_F(output, (int2)((output_c_block_idx * out_width + output_w_idx), output_b_idx * out_height + output_h_idx * 4),
         res);

    //
    if (output_h_idx * 4 + 1 >= out_height) {
        return;
    }

    nw_index_x = select(0, (int)(ix_nw.y), nw_within_bound.y);
    nw_index_y = select(0, (int)(iy_nw.y), nw_within_bound.y);
    res        = (FLOAT4)(0, 0, 0, 0);
    input_data_nw =
        RI_F(input, SAMPLER,
             (int2)(output_c_block_idx * input_width + nw_index_x, output_b_idx * input_height + nw_index_y));
    res        = res + input_data_nw * nw.y;
    ne_index_x = select(0, (int)ix_ne.y, ne_within_bound.y);
    ne_index_y = select(0, (int)iy_ne.y, ne_within_bound.y);
    input_data_ne =
        RI_F(input, SAMPLER,
             (int2)(output_c_block_idx * input_width + ne_index_x, output_b_idx * input_height + ne_index_y));
    res        = res + input_data_ne * ne.y;
    sw_index_x = select(0, (int)ix_sw.y, sw_within_bound.y);
    sw_index_y = select(0, (int)iy_sw.y, sw_within_bound.y);
    input_data_sw =
        RI_F(input, SAMPLER,
             (int2)(output_c_block_idx * input_width + sw_index_x, output_b_idx * input_height + sw_index_y));
    res        = res + input_data_sw * sw.y;
    se_index_x = select(0, (int)ix_se.y, se_within_bound.y);
    se_index_y = select(0, (int)iy_se.y, se_within_bound.y);
    input_data_se =
        RI_F(input, SAMPLER,
             (int2)(output_c_block_idx * input_width + se_index_x, output_b_idx * input_height + se_index_y));
    res = res + input_data_se * se.y;
    WI_F(output,
         (int2)((output_c_block_idx * out_width + output_w_idx), output_b_idx * out_height + output_h_idx * 4 + 1),
         res);

    //
    if (output_h_idx * 4 + 2 >= out_height) {
        return;
    }
    nw_index_x = select(0, (int)(ix_nw.z), nw_within_bound.z);
    nw_index_y = select(0, (int)(iy_nw.z), nw_within_bound.z);
    res        = (FLOAT4)(0, 0, 0, 0);
    input_data_nw =
        RI_F(input, SAMPLER,
             (int2)(output_c_block_idx * input_width + nw_index_x, output_b_idx * input_height + nw_index_y));
    res        = res + input_data_nw * nw.z;
    ne_index_x = select(0, (int)ix_ne.z, ne_within_bound.z);
    ne_index_y = select(0, (int)iy_ne.z, ne_within_bound.z);
    input_data_ne =
        RI_F(input, SAMPLER,
             (int2)(output_c_block_idx * input_width + ne_index_x, output_b_idx * input_height + ne_index_y));
    res        = res + input_data_ne * ne.z;
    sw_index_x = select(0, (int)ix_sw.z, sw_within_bound.z);
    sw_index_y = select(0, (int)iy_sw.z, sw_within_bound.z);
    input_data_sw =
        RI_F(input, SAMPLER,
             (int2)(output_c_block_idx * input_width + sw_index_x, output_b_idx * input_height + sw_index_y));
    res        = res + input_data_sw * sw.z;
    se_index_x = select(0, (int)ix_se.z, se_within_bound.z);
    se_index_y = select(0, (int)iy_se.z, se_within_bound.z);
    input_data_se =
        RI_F(input, SAMPLER,
             (int2)(output_c_block_idx * input_width + se_index_x, output_b_idx * input_height + se_index_y));
    res = res + input_data_se * se.z;
    WI_F(output,
         (int2)((output_c_block_idx * out_width + output_w_idx), output_b_idx * out_height + output_h_idx * 4 + 2),
         res);

    if (output_h_idx * 4 + 3 >= out_height) {
        return;
    }
    nw_index_x = select(0, (int)(ix_nw.w), nw_within_bound.w);
    nw_index_y = select(0, (int)(iy_nw.w), nw_within_bound.w);
    res        = (FLOAT4)(0, 0, 0, 0);
    input_data_nw =
        RI_F(input, SAMPLER,
             (int2)(output_c_block_idx * input_width + nw_index_x, output_b_idx * input_height + nw_index_y));
    res        = res + input_data_nw * nw.w;
    ne_index_x = select(0, (int)ix_ne.w, ne_within_bound.w);
    ne_index_y = select(0, (int)iy_ne.w, ne_within_bound.w);
    input_data_ne =
        RI_F(input, SAMPLER,
             (int2)(output_c_block_idx * input_width + ne_index_x, output_b_idx * input_height + ne_index_y));
    res        = res + input_data_ne * ne.w;
    sw_index_x = select(0, (int)ix_sw.w, sw_within_bound.w);
    sw_index_y = select(0, (int)iy_sw.w, sw_within_bound.w);
    input_data_sw =
        RI_F(input, SAMPLER,
             (int2)(output_c_block_idx * input_width + sw_index_x, output_b_idx * input_height + sw_index_y));
    res        = res + input_data_sw * sw.w;
    se_index_x = select(0, (int)ix_se.w, se_within_bound.w);
    se_index_y = select(0, (int)iy_se.w, se_within_bound.w);
    input_data_se =
        RI_F(input, SAMPLER,
             (int2)(output_c_block_idx * input_width + se_index_x, output_b_idx * input_height + se_index_y));
    res = res + input_data_se * se.w;
    WI_F(output,
         (int2)((output_c_block_idx * out_width + output_w_idx), output_b_idx * out_height + output_h_idx * 4 + 3),
         res);
}
