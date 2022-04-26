#include "base.inc"

__kernel void BinaryElementWise(GLOBAL_SIZE_2_DIMS __read_only image2d_t input0,
                                __read_only image2d_t input1,
                                __write_only image2d_t output) {
    const int cw = get_global_id(0);
    const int bh = get_global_id(1);

    DEAL_NON_UNIFORM_DIM2(cw, bh);

    FLOAT4 in0 = RI_F(input0, SAMPLER, (int2)(cw, bh));
    FLOAT4 in1 = RI_F(input1, SAMPLER, (int2)(cw, bh));
    FLOAT4 out = OPERATOR;
    WI_F(output, (int2)(cw, bh), out);
}

__kernel void BinarySingle(GLOBAL_SIZE_2_DIMS __read_only image2d_t input,
                           __read_only image2d_t param,
                           __write_only image2d_t output) {
    const int cw = get_global_id(0);
    const int bh = get_global_id(1);

    DEAL_NON_UNIFORM_DIM2(cw, bh);

    FLOAT4 in0 = RI_F(input, SAMPLER, (int2)(cw, bh));
    FLOAT in1  = RI_F(param, SAMPLER, (int2)(0, 0)).x;
    FLOAT4 out = OPERATOR;
    WI_F(output, (int2)(cw, bh), out);
}

__kernel void BinaryChannel(GLOBAL_SIZE_2_DIMS __read_only image2d_t input,
                            __read_only image2d_t param,
                            __private const int height,
                            __private const int width,
                            __private const int param_batch,
                            __write_only image2d_t output) {
    const int cw = get_global_id(0);
    const int bh = get_global_id(1);

    DEAL_NON_UNIFORM_DIM2(cw, bh);

    const int c_idx = cw / width;
    const int b_idx = bh / height;
    const int param_b_idx = b_idx % param_batch;
    FLOAT4 in0      = RI_F(input, SAMPLER, (int2)(cw, bh));
    FLOAT4 in1      = RI_F(param, SAMPLER, (int2)(c_idx, param_b_idx));
    FLOAT4 out      = OPERATOR;
    WI_F(output, (int2)(cw, bh), out);
}

__kernel void BinaryWidth(GLOBAL_SIZE_2_DIMS __read_only image2d_t input,
                            __read_only image2d_t param,
                            __private const int height,
                            __private const int width,
                            __private const int param_batch,
                            __write_only image2d_t output) {
    const int cw = get_global_id(0);
    const int bh = get_global_id(1);

    DEAL_NON_UNIFORM_DIM2(cw, bh);

    const int w_idx = cw % width;
    const int b_idx = bh / height;
    const int param_b_idx = b_idx % param_batch;
    FLOAT4 in0      = RI_F(input, SAMPLER, (int2)(cw, bh));
    FLOAT in1       = RI_F(param, SAMPLER, (int2)(w_idx, param_b_idx)).x;
    FLOAT4 out      = OPERATOR;
    WI_F(output, (int2)(cw, bh), out);
}

__kernel void BinaryCHW(GLOBAL_SIZE_2_DIMS __read_only image2d_t input,
                        __read_only image2d_t param,
                        __private const int height,
                        __private const int width,
                        __private const int param_batch,
                        __write_only image2d_t output) {
    const int cw = get_global_id(0);
    const int bh = get_global_id(1);

    DEAL_NON_UNIFORM_DIM2(cw, bh);

    const int h_idx = bh % height;
    const int b_idx = bh / height;
    const int param_b_idx = b_idx % param_batch;
    FLOAT4 in0      = RI_F(input, SAMPLER, (int2)(cw, bh));
    FLOAT4 in1      = RI_F(param, SAMPLER, (int2)(cw, h_idx + param_b_idx * height));
    FLOAT4 out      = OPERATOR;
    WI_F(output, (int2)(cw, bh), out);
}

__kernel void BinaryHW(GLOBAL_SIZE_2_DIMS __read_only image2d_t input,
                       __read_only image2d_t param, __private const int height,
                       __private const int width, __private const int param_batch,
                       __write_only image2d_t output) {
    const int cw = get_global_id(0);
    const int bh = get_global_id(1);

    DEAL_NON_UNIFORM_DIM2(cw, bh);

    const int h_idx = bh % height;
    const int b_idx = bh / height;
    const int param_b_idx = b_idx % param_batch;
    const int w_idx = cw % width;
    FLOAT4 in0      = RI_F(input, SAMPLER, (int2)(cw, bh));
    FLOAT in1       = RI_F(param, SAMPLER, (int2)(w_idx, h_idx + param_b_idx * height)).x;
    FLOAT4 out      = OPERATOR;
    WI_F(output, (int2)(cw, bh), out);
}

__kernel void BinaryBroadcast(GLOBAL_SIZE_2_DIMS __read_only image2d_t input0,
                              __read_only image2d_t input1, int4 output_shape,
                              int4 input0_shape, int4 input1_shape,
                              __private const int input0_c_4_blocks,
                              __private const int input1_c_4_blocks,
                              __write_only image2d_t output) {
    const int output_cw = get_global_id(0);
    const int output_bh = get_global_id(1);

    DEAL_NON_UNIFORM_DIM2(output_cw, output_bh);

    const int output_h_idx = output_bh % output_shape.z;
    const int output_b_idx = output_bh / output_shape.z;
    const int output_w_idx = output_cw % output_shape.w;
    const int output_c_4_idx = output_cw / output_shape.w;

    FLOAT4 in0, in1;
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

    FLOAT4 out      = OPERATOR;
    WI_F(output, (int2)(output_cw, output_bh), out);
}

__kernel void BinaryBroadcast5D(GLOBAL_SIZE_2_DIMS __read_only image2d_t input0, __read_only image2d_t input1,
                                shape_5d output_shape, shape_5d input0_shape, shape_5d input1_shape,
                                __private const int input0_c_4_blocks, __private const int input1_c_4_blocks,
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

    FLOAT4 out = OPERATOR;
    WI_F(output, (int2)(output_c_d4, output_b_d2_d3), out);
}
