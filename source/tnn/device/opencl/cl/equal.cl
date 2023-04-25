#include "base.inc"

__kernel void EqualGeneral(GLOBAL_SIZE_2_DIMS
                           __read_only image2d_t input0,
                           __read_only image2d_t input1,
                           int4 output_shape,
                           int4 input0_shape,
                           int4 input1_shape,
                           __private const int input0_c_4_blocks,
                           __private const int input1_c_4_blocks,
                           __write_only image2d_t output
                           ) {

    const int output_cw = get_global_id(0);
    const int output_bh = get_global_id(1);

    DEAL_NON_UNIFORM_DIM2(output_cw, output_bh);

    const int output_h_idx = output_bh % output_shape.z;
    const int output_b_idx = output_bh / output_shape.z;
    const int output_w_idx = output_cw % output_shape.w;
    const int output_c_4_idx = output_cw / output_shape.w;

    int4 in0, in1;
    const int input0_h_idx = select(output_h_idx, 0, input0_shape.z == 1);
    const int input0_b_idx = select(output_b_idx, 0, input0_shape.x == 1);
    const int input0_w_idx = select(output_w_idx, 0, input0_shape.w == 1);
    const int input0_c_4_idx = select(input0_c_4_blocks - 1, output_c_4_idx, output_c_4_idx < input0_c_4_blocks);
    in0 = read_imagei(input0, SAMPLER, (int2)(input0_c_4_idx * input0_shape.w + input0_w_idx,
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
    in1 = read_imagei(input1, SAMPLER, (int2)(input1_c_4_idx * input1_shape.w + input1_w_idx,
                                              input1_b_idx * input1_shape.z + input1_h_idx));
    if (input1_shape.y == 1) {
        in1.y = in1.x;
        in1.z = in1.x;
        in1.w = in1.x;
    }

    int4 out = {in0.x == in1.x, in0.y == in1.y, in0.z == in1.z, in0.w == in1.w};
    write_imagei(output, (int2)(output_cw, output_bh), out);
}
