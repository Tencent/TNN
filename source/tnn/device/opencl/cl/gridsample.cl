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

    FLOAT4 ix = (x + ((FLOAT)1.0)) * input_width * ((FLOAT)0.5) - ((FLOAT)0.5);
    FLOAT4 iy = (y + ((FLOAT)1.0)) * input_height * ((FLOAT)0.5) - ((FLOAT)0.5);

    FLOAT4 ix_nw = floor(ix);
    FLOAT4 iy_nw = floor(iy);

    FLOAT4 ix_ne = ix_nw + 1;
    FLOAT4 iy_ne = iy_nw;

    FLOAT4 ix_sw = ix_nw;
    FLOAT4 iy_sw = iy_nw + 1;

    FLOAT4 ix_se = ix_nw + 1;
    FLOAT4 iy_se = iy_nw + 1;

    // get nw_within_bound
    int4 low_ix_nw = select((int4)(0, 0, 0, 0), (int4)(-1, -1, -1, -1), ix_nw >= (int)0);
    int4 low_iy_nw = select((int4)(0, 0, 0, 0), (int4)(-1, -1, -1, -1), iy_nw >= (int)0);

    int4 up_ix_nw = select((int4)(0, 0, 0, 0), (int4)(-1, -1, -1, -1), ix_nw < input_width);
    int4 up_iy_nw = select((int4)(0, 0, 0, 0), (int4)(-1, -1, -1, -1), iy_nw < input_height);

    int4 nw_within_bound = select((int4)(0, 0, 0, 0), (int4)(-1, -1, -1, -1), low_ix_nw);
    nw_within_bound      = select((int4)(0, 0, 0, 0), nw_within_bound, low_iy_nw);
    nw_within_bound      = select((int4)(0, 0, 0, 0), nw_within_bound, up_ix_nw);
    nw_within_bound      = select((int4)(0, 0, 0, 0), nw_within_bound, up_iy_nw);

    // get ne_within_bound
    low_ix_nw            = select((int4)(0, 0, 0, 0), (int4)(-1, -1, -1, -1), ix_ne >= 0);
    low_iy_nw            = select((int4)(0, 0, 0, 0), (int4)(-1, -1, -1, -1), iy_ne >= 0);
    up_ix_nw             = select((int4)(0, 0, 0, 0), (int4)(-1, -1, -1, -1), ix_ne < input_width);
    up_iy_nw             = select((int4)(0, 0, 0, 0), (int4)(-1, -1, -1, -1), iy_ne < input_height);
    int4 ne_within_bound = select((int4)(0, 0, 0, 0), (int4)(-1, -1, -1, -1), low_ix_nw);
    ne_within_bound      = select((int4)(0, 0, 0, 0), ne_within_bound, low_iy_nw);
    ne_within_bound      = select((int4)(0, 0, 0, 0), ne_within_bound, up_ix_nw);
    ne_within_bound      = select((int4)(0, 0, 0, 0), ne_within_bound, up_iy_nw);

    // get sw_within_bound
    low_ix_nw            = select((int4)(0, 0, 0, 0), (int4)(-1, -1, -1, -1), ix_sw >= 0);
    low_iy_nw            = select((int4)(0, 0, 0, 0), (int4)(-1, -1, -1, -1), iy_sw >= 0);
    up_ix_nw             = select((int4)(0, 0, 0, 0), (int4)(-1, -1, -1, -1), ix_sw < input_width);
    up_iy_nw             = select((int4)(0, 0, 0, 0), (int4)(-1, -1, -1, -1), iy_sw < input_height);
    int4 sw_within_bound = select((int4)(0, 0, 0, 0), (int4)(-1, -1, -1, -1), low_ix_nw);
    sw_within_bound      = select((int4)(0, 0, 0, 0), sw_within_bound, low_iy_nw);
    sw_within_bound      = select((int4)(0, 0, 0, 0), sw_within_bound, up_ix_nw);
    sw_within_bound      = select((int4)(0, 0, 0, 0), sw_within_bound, up_iy_nw);

    // get se_within_bound
    low_ix_nw            = select((int4)(0, 0, 0, 0), (int4)(-1, -1, -1, -1), ix_se >= 0);
    low_iy_nw            = select((int4)(0, 0, 0, 0), (int4)(-1, -1, -1, -1), iy_se >= 0);
    up_ix_nw             = select((int4)(0, 0, 0, 0), (int4)(-1, -1, -1, -1), ix_se < input_width);
    up_iy_nw             = select((int4)(0, 0, 0, 0), (int4)(-1, -1, -1, -1), iy_se < input_height);
    int4 se_within_bound = select((int4)(0, 0, 0, 0), (int4)(-1, -1, -1, -1), low_ix_nw);
    se_within_bound      = select((int4)(0, 0, 0, 0), se_within_bound, low_iy_nw);
    se_within_bound      = select((int4)(0, 0, 0, 0), se_within_bound, up_ix_nw);
    se_within_bound      = select((int4)(0, 0, 0, 0), se_within_bound, up_iy_nw);

    FLOAT4 nw = select((FLOAT)0, (ix_se - ix) * (iy_se - iy), nw_within_bound);
    FLOAT4 ne = select((FLOAT)0, (ix - ix_sw) * (iy_sw - iy), ne_within_bound);
    FLOAT4 sw = select((FLOAT)0, (ix_ne - ix) * (iy - iy_ne), sw_within_bound);
    FLOAT4 se = select((FLOAT)0, (ix - ix_nw) * (iy - iy_nw), se_within_bound);

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
