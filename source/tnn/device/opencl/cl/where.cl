#include "base.inc"

__kernel void WhereGeneral(GLOBAL_SIZE_2_DIMS
                           __read_only image2d_t input0,
                           __read_only image2d_t input1,
                           __read_only image2d_t input2,
                           int4 output_shape,
                           int4 input0_shape,
                           int4 input1_shape,
                           int4 input2_shape,
                           __private const int input0_c_4_blocks,
                           __private const int input1_c_4_blocks,
                           __private const int input2_c_4_blocks,
                           __write_only image2d_t output
                           ) {

    const int output_cw = get_global_id(0);
    const int output_bh = get_global_id(1);

    DEAL_NON_UNIFORM_DIM2(output_cw, output_bh);

    const int output_h_idx = output_bh % output_shape.z;
    const int output_b_idx = output_bh / output_shape.z;
    const int output_w_idx = output_cw % output_shape.w;
    const int output_c_4_idx = output_cw / output_shape.w;

    FLOAT4 in0, in1;
    int4 in2;
    const int input0_h_idx = select(output_h_idx, 0, input0_shape.z == 1);
    const int input0_b_idx = select(output_b_idx, 0, input0_shape.x == 1);
    const int input0_w_idx = select(output_w_idx, 0, input0_shape.w == 1);
    const int input0_c_4_idx = select(input0_c_4_blocks - 1, output_c_4_idx, output_c_4_idx < input0_c_4_blocks);
    in0 = RI_F(input0, SAMPLER, (int2)(input0_c_4_idx * input0_shape.w + input0_w_idx,
                                       input0_b_idx * input0_shape.z + input0_h_idx));
    if (input0_shape.y == 1) {
        in0.y = in0.x;
        in0.z = in0.x;
        in0.w = in0.x;
    }

    const int input1_h_idx = select(output_h_idx, 0, input1_shape.z == 1);
    const int input1_b_idx = select(output_b_idx, 0, input1_shape.x == 1);
    const int input1_w_idx = select(output_w_idx, 0, input1_shape.w == 1);
    const int input1_c_4_idx = select(input1_c_4_blocks - 1, output_c_4_idx, output_c_4_idx < input1_c_4_blocks);
    in1 = RI_F(input1, SAMPLER, (int2)(input1_c_4_idx * input1_shape.w + input1_w_idx,
                                       input1_b_idx * input1_shape.z + input1_h_idx));
    if (input1_shape.y == 1) {
        in1.y = in1.x;
        in1.z = in1.x;
        in1.w = in1.x;
    }

    const int input2_h_idx = select(output_h_idx, 0, input2_shape.z == 1);
    const int input2_b_idx = select(output_b_idx, 0, input2_shape.x == 1);
    const int input2_w_idx = select(output_w_idx, 0, input2_shape.w == 1);
    const int input2_c_4_idx = select(input2_c_4_blocks - 1, output_c_4_idx, output_c_4_idx < input2_c_4_blocks);
    in2 = read_imagei(input2, SAMPLER, (int2)(input2_c_4_idx * input2_shape.w + input2_w_idx,
                                              input2_b_idx * input2_shape.z + input2_h_idx));
    if (input2_shape.y == 1) {
        in2.y = in2.x;
        in2.z = in2.x;
        in2.w = in2.x;
    }

    // int4 out = {in0.x == in1.x, in0.y == in1.y, in0.z == in1.z, in0.w == in1.w};
    FLOAT4 out = {select(in1.x, in0.x, in2.x),
                  select(in1.y, in0.y, in2.y),
                  select(in1.z, in0.z, in2.z),
                  select(in1.w, in0.w, in2.w)};
    WI_F(output, (int2)(output_cw, output_bh), out);
}

__kernel void WhereGeneral5D(GLOBAL_SIZE_2_DIMS
                             __read_only image2d_t input0,
                             __read_only image2d_t input1,
                             __read_only image2d_t input2,
                             shape_5d output_shape,
                             shape_5d input0_shape,
                             shape_5d input1_shape,
                             shape_5d input2_shape,
                             __private const int input0_c_4_blocks,
                             __private const int input1_c_4_blocks,
                             __private const int input2_c_4_blocks,
                             __write_only image2d_t output) {
    const int output_c_d4    = get_global_id(0);
    const int output_b_d2_d3 = get_global_id(1);

    DEAL_NON_UNIFORM_DIM2(output_c_d4, output_b_d2_d3)

    const int output_d2xd3  = output_shape.data[2] * output_shape.data[3];
    const int output_d2_d3  = output_b_d2_d3 % output_d2xd3;
    const int output_b_idx  = output_b_d2_d3 / (output_d2xd3);
    const int output_d2_idx = output_d2_d3 / output_shape.data[3];
    const int output_d3_idx = output_d2_d3 % output_shape.data[3];

    const int output_d4_idx  = output_c_d4 % output_shape.data[4];
    const int output_c_4_idx = output_c_d4 / output_shape.data[4];

    FLOAT4 in0, in1;
    int4 in2;
    const int input0_b_idx   = select(output_b_idx, 0, input0_shape.data[0] == 1);
    const int input0_d2_idx  = select(output_d2_idx, 0, input0_shape.data[2] == 1);
    const int input0_d3_idx  = select(output_d3_idx, 0, input0_shape.data[3] == 1);
    const int input0_d4_idx  = select(output_d4_idx, 0, input0_shape.data[4] == 1);
    const int input0_c_4_idx = select(input0_c_4_blocks - 1, output_c_4_idx, output_c_4_idx < input0_c_4_blocks);

    in0 = RI_F(input0, SAMPLER,
               (int2)(input0_c_4_idx * input0_shape.data[4] + input0_d4_idx,
                      input0_b_idx * input0_shape.data[2] * input0_shape.data[3] +
                          input0_d2_idx * input0_shape.data[3] + input0_d3_idx));
    if (input0_shape.data[1] == 1) {
        in0.y = in0.x;
        in0.z = in0.x;
        in0.w = in0.x;
    }

    const int input1_b_idx   = select(output_b_idx, 0, input1_shape.data[0] == 1);
    const int input1_d2_idx  = select(output_d2_idx, 0, input1_shape.data[2] == 1);
    const int input1_d3_idx  = select(output_d3_idx, 0, input1_shape.data[3] == 1);
    const int input1_d4_idx  = select(output_d4_idx, 0, input1_shape.data[4] == 1);
    const int input1_c_4_idx = select(input1_c_4_blocks - 1, output_c_4_idx, output_c_4_idx < input1_c_4_blocks);

    in1 = RI_F(input1, SAMPLER,
               (int2)(input1_c_4_idx * input1_shape.data[4] + input1_d4_idx,
                      input1_b_idx * input1_shape.data[2] * input1_shape.data[3] +
                          input1_d2_idx * input1_shape.data[3] + input1_d3_idx));
    if (input1_shape.data[1] == 1) {
        in1.y = in1.x;
        in1.z = in1.x;
        in1.w = in1.x;
    }

    const int input2_b_idx   = select(output_b_idx, 0, input2_shape.data[0] == 1);
    const int input2_d2_idx  = select(output_d2_idx, 0, input2_shape.data[2] == 1);
    const int input2_d3_idx  = select(output_d3_idx, 0, input2_shape.data[3] == 1);
    const int input2_d4_idx  = select(output_d4_idx, 0, input2_shape.data[4] == 1);
    const int input2_c_4_idx = select(input2_c_4_blocks - 1, output_c_4_idx, output_c_4_idx < input2_c_4_blocks);

    in2 = read_imagei(input2, SAMPLER,
                      (int2)(input2_c_4_idx * input2_shape.data[4] + input2_d4_idx,
                             input2_b_idx * input2_shape.data[2] * input2_shape.data[3] +
                             input2_d2_idx * input2_shape.data[3] + input2_d3_idx));
    if (input2_shape.data[1] == 1) {
        in2.y = in2.x;
        in2.z = in2.x;
        in2.w = in2.x;
    }

    FLOAT4 out = {select(in1.x, in0.x, in2.x),
                  select(in1.y, in0.y, in2.y),
                  select(in1.z, in0.z, in2.z),
                  select(in1.w, in0.w, in2.w)};
    WI_F(output, (int2)(output_c_d4, output_b_d2_d3), out);
}
