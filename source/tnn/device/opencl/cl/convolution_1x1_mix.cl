#include "base.inc"
#include "activation.inc"
#include "io.inc"

__kernel void Conv2D1x1_S1_MIX(GLOBAL_SIZE_2_DIMS __read_only image2d_t input, 
                          __global const FLOAT16 *weights_ptr,
                          __global const FLOAT4 *bias_ptr,
                          __write_only image2d_t output, __private const int2 wh,
                          __private const int input_c_blocks,
                          __private const int output_w_updiv_4,
                          __private const int activation_type) {

    const int output_cw_idx = get_global_id(0); //c/4 w/4
    const int bh_idx  = get_global_id(1); //b h

    DEAL_NON_UNIFORM_DIM2(output_cw_idx, bh_idx);

    const int output_c_block_idx = output_cw_idx / output_w_updiv_4;
    const int output_w_block_idx = output_cw_idx % output_w_updiv_4;

    FLOAT4 out0 = bias_ptr[output_c_block_idx];
    FLOAT4 out1 = out0;
    FLOAT4 out2 = out0;
    FLOAT4 out3 = out0;

    const int out_x_idx = output_w_block_idx << 2;

    int input_w_idx0 = out_x_idx;
    int input_w_idx1 = out_x_idx + 1;
    int input_w_idx2 = out_x_idx + 2;
    int input_w_idx3 = out_x_idx + 3;

    input_w_idx0 = select(input_w_idx0, INT_MIN, input_w_idx0 >= wh.x);
    input_w_idx1 = select(input_w_idx1, INT_MIN, input_w_idx1 >= wh.x);
    input_w_idx2 = select(input_w_idx2, INT_MIN, input_w_idx2 >= wh.x);
    input_w_idx3 = select(input_w_idx3, INT_MIN, input_w_idx3 >= wh.x);

    int input_w_base   = 0;
    int weights_offset = mul24(output_c_block_idx, input_c_blocks);
    for (int input_c_block_idx = 0; input_c_block_idx < input_c_blocks; ++input_c_block_idx) {
        FLOAT4 in0 = RI_F(input, SAMPLER, (int2)(input_w_base + input_w_idx0, bh_idx));
        FLOAT4 in1 = RI_F(input, SAMPLER, (int2)(input_w_base + input_w_idx1, bh_idx));
        FLOAT4 in2 = RI_F(input, SAMPLER, (int2)(input_w_base + input_w_idx2, bh_idx));
        FLOAT4 in3 = RI_F(input, SAMPLER, (int2)(input_w_base + input_w_idx3, bh_idx));

        FLOAT16 weights = weights_ptr[weights_offset];
        CALCULATE_VEC16_OUTPUT(0);
        CALCULATE_VEC16_OUTPUT(1);
        CALCULATE_VEC16_OUTPUT(2);
        CALCULATE_VEC16_OUTPUT(3);

        input_w_base   += wh.x;
        weights_offset++;
    }

    out0 = ActivationProcess(out0, activation_type);
    out1 = ActivationProcess(out1, activation_type);
    out2 = ActivationProcess(out2, activation_type);
    out3 = ActivationProcess(out3, activation_type);

    const int out_x_base = mul24(output_c_block_idx, wh.x);

    const int remain = wh.x - out_x_idx;
    int output_w_idx = out_x_base + out_x_idx;
    WriteOutputAntiOutOfBounds(output, out0, out1, out2, out3, output_w_idx,
                               bh_idx, remain);
}

__kernel void Conv2D1x1_S1_MIX_CB2(GLOBAL_SIZE_2_DIMS __read_only image2d_t input,
                                   __global const FLOAT *weights_ptr,
                                   __global const FLOAT *bias_ptr,
                                   __write_only image2d_t output, __private const int2 wh,
                                   __private const int input_c_blocks,
                                   __private const int out_channel_block_length,
                                   __private const int out_width_blocks,
                                   __private const int activation_type) {

    const int output_channel_slice_w_idx = get_global_id(0); //c/4 w/4
    const int output_bh_idx  = get_global_id(1); //b h

    DEAL_NON_UNIFORM_DIM2(output_channel_slice_w_idx, output_bh_idx);

    const int out_channel_slice_idx = output_channel_slice_w_idx / out_width_blocks;
    const int out_channel_block_idx = out_channel_slice_idx << 1;
    const int output_w_block_idx = output_channel_slice_w_idx % out_width_blocks;

    FLOAT4 out_w0_s0 = vload4(out_channel_block_idx, (__global FLOAT *)bias_ptr);
    FLOAT4 out_w1_s0 = out_w0_s0;
    FLOAT4 out_w2_s0 = out_w0_s0;
    FLOAT4 out_w3_s0 = out_w0_s0;

    bool is_s1_in_boundary = (out_channel_block_idx + 1 < out_channel_block_length);
    FLOAT4 out_w0_s1 = vload4(select(out_channel_block_idx, out_channel_block_idx + 1, is_s1_in_boundary), (__global FLOAT *)bias_ptr);
    FLOAT4 out_w1_s1 = out_w0_s1;
    FLOAT4 out_w2_s1 = out_w0_s1;
    FLOAT4 out_w3_s1 = out_w0_s1;

    FLOAT4 in0, in1, in2, in3;
    FLOAT4 weights_c0_s0, weights_c1_s0, weights_c2_s0, weights_c3_s0;
    FLOAT4 weights_c0_s1, weights_c1_s1, weights_c2_s1, weights_c3_s1;

    const int out_x_idx = output_w_block_idx << 2;

    int input_w_idx0 = out_x_idx;
    int input_w_idx1 = out_x_idx + 1;
    int input_w_idx2 = out_x_idx + 2;
    int input_w_idx3 = out_x_idx + 3;

    input_w_idx0 = select(input_w_idx0, INT_MIN, input_w_idx0 >= wh.x);
    input_w_idx1 = select(input_w_idx1, INT_MIN, input_w_idx1 >= wh.x);
    input_w_idx2 = select(input_w_idx2, INT_MIN, input_w_idx2 >= wh.x);
    input_w_idx3 = select(input_w_idx3, INT_MIN, input_w_idx3 >= wh.x);

    int4 input_w_idx = {input_w_idx0, input_w_idx1, input_w_idx2, input_w_idx3};

    const int input_channels = input_c_blocks << 2;
    int weights_offset_s0 = mul24(out_channel_block_idx, input_channels);
    int weights_offset_s1 = weights_offset_s0 + input_channels;
    int2 weights_offset = {weights_offset_s0, select(0, weights_offset_s1, is_s1_in_boundary)};
    for (int input_c_block_idx = 0; input_c_block_idx < input_c_blocks; ++input_c_block_idx) {
        in0 = RI_F(input, SAMPLER, (int2)(input_w_idx.x, output_bh_idx));
        in1 = RI_F(input, SAMPLER, (int2)(input_w_idx.y, output_bh_idx));
        in2 = RI_F(input, SAMPLER, (int2)(input_w_idx.z, output_bh_idx));
        in3 = RI_F(input, SAMPLER, (int2)(input_w_idx.w, output_bh_idx));

        weights_c0_s0 = vload4(weights_offset.x, (__global FLOAT *)weights_ptr);
        weights_c1_s0 = vload4(weights_offset.x + 1, (__global FLOAT *)weights_ptr);
        weights_c2_s0 = vload4(weights_offset.x + 2, (__global FLOAT *)weights_ptr);
        weights_c3_s0 = vload4(weights_offset.x + 3, (__global FLOAT *)weights_ptr);

        weights_c0_s1 = vload4(weights_offset.y, (__global FLOAT *)weights_ptr);
        weights_c1_s1 = vload4(weights_offset.y + 1, (__global FLOAT *)weights_ptr);
        weights_c2_s1 = vload4(weights_offset.y + 2, (__global FLOAT *)weights_ptr);
        weights_c3_s1 = vload4(weights_offset.y + 3, (__global FLOAT *)weights_ptr);

        CALCULATE_SLICE_OUTPUT(0);
        CALCULATE_SLICE_OUTPUT(1);

        input_w_idx   += wh.x;
        weights_offset += 4;
    }

    out_w0_s0 = ActivationProcess(out_w0_s0, activation_type);
    out_w1_s0 = ActivationProcess(out_w1_s0, activation_type);
    out_w2_s0 = ActivationProcess(out_w2_s0, activation_type);
    out_w3_s0 = ActivationProcess(out_w3_s0, activation_type);

    out_w0_s1 = ActivationProcess(out_w0_s1, activation_type);
    out_w1_s1 = ActivationProcess(out_w1_s1, activation_type);
    out_w2_s1 = ActivationProcess(out_w2_s1, activation_type);
    out_w3_s1 = ActivationProcess(out_w3_s1, activation_type);

    const int out_x_base = mul24(out_channel_block_idx, wh.x);

    const int remain = wh.x - out_x_idx;
    int output_w_idx_s0 = out_x_base + out_x_idx;
    WriteSliceOutputAntiOutOfBounds(output, out_w0_s0, out_w1_s0,
                                    out_w2_s0, out_w3_s0, output_w_idx_s0,
                                    output_bh_idx, remain);

    if (!is_s1_in_boundary) return;
    int output_w_idx_s1 = output_w_idx_s0 + wh.x;
    WriteSliceOutputAntiOutOfBounds(output, out_w0_s1, out_w1_s1,
                                    out_w2_s1, out_w3_s1, output_w_idx_s1,
                                    output_bh_idx, remain);
}

__kernel void Conv2D1x1_S1_MIX_WB1(GLOBAL_SIZE_2_DIMS __read_only image2d_t input,
                          __global const FLOAT16 *weights_ptr,
                          __global const FLOAT4 *bias_ptr,
                          __write_only image2d_t output, __private const int2 wh,
                          __private const int input_c_blocks,
                          __private const int activation_type) {

    const int output_cw_idx = get_global_id(0); //c/4 w
    const int bh_idx  = get_global_id(1); //b h

    DEAL_NON_UNIFORM_DIM2(output_cw_idx, bh_idx);

    const int output_c_block_idx = output_cw_idx / wh.x;
    const int out_x_idx = output_cw_idx % wh.x;

    FLOAT4 out0 = bias_ptr[output_c_block_idx];

    int input_w_idx0 = out_x_idx;

    input_w_idx0 = select(input_w_idx0, INT_MIN, input_w_idx0 >= wh.x);

    int input_w_base   = 0;
    int weights_offset = mul24(output_c_block_idx, input_c_blocks);
    for (int input_c_block_idx = 0; input_c_block_idx < input_c_blocks; ++input_c_block_idx) {
        FLOAT4 in0 = RI_F(input, SAMPLER, (int2)(input_w_base + input_w_idx0, bh_idx));

        FLOAT16 weights = weights_ptr[weights_offset];

        CALCULATE_VEC16_OUTPUT(0);

        input_w_base   += wh.x;
        weights_offset++;
    }

    out0 = ActivationProcess(out0, activation_type);

    const int out_x_base = mul24(output_c_block_idx, wh.x);

    int output_w_idx = out_x_base + out_x_idx;
    WI_F(output, (int2)(output_w_idx, bh_idx), out0);
}

__kernel void Conv2D1x1_S1_MIX_WB1_Local(GLOBAL_SIZE_2_DIMS __read_only image2d_t input,
                          __global const FLOAT16 *weights_ptr,
                          __global const FLOAT4 *bias_ptr,
                          __write_only image2d_t output, __private const int2 wh,
                          __private const int input_c_blocks,
                          __private const int local_block_size,
                          __local FLOAT4* local_output,
                          __private const int activation_type) {

    const int local_id = get_local_id(0);
    const int group_size = get_local_size(0);
    const int global_id = get_global_id(0);
    const int output_cw_idx = global_id / group_size; //c/4 w
    const int bh_idx  = get_global_id(1); //b h

    DEAL_NON_UNIFORM_DIM2(global_id, bh_idx);

    const int output_c_block_idx = output_cw_idx / wh.x;
    const int out_x_idx = output_cw_idx % wh.x;

    local_output[local_id] = (FLOAT4)0.f;

    int input_w_idx0 = out_x_idx;

    input_w_idx0 = select(input_w_idx0, INT_MIN, input_w_idx0 >= wh.x);

    int pos = local_id;
    int input_w_stride = mul24(group_size, wh.x);
    int weights_stride = group_size;
    int input_w_base   = mul24(pos, wh.x);
    int weights_offset = mad24(output_c_block_idx, input_c_blocks, pos);
    for (unsigned short i = 0; i < local_block_size; i++) {
        if (pos >= input_c_blocks) break;
        FLOAT4 in0 = RI_F(input, SAMPLER, (int2)(input_w_base + input_w_idx0, bh_idx));

        FLOAT16 weights = weights_ptr[weights_offset];

        local_output[local_id] += weights.s0123 * in0.x;
        local_output[local_id] += weights.s4567 * in0.y;
        local_output[local_id] += weights.s89ab * in0.z;
        local_output[local_id] += weights.scdef * in0.w;

        input_w_base   += input_w_stride;
        weights_offset += weights_stride;
        pos += group_size;
    }

    barrier(CLK_LOCAL_MEM_FENCE);
    for (unsigned short stride = (group_size >> 1); stride > 0; stride >>= 1) {
        if (local_id < stride) {
            local_output[local_id] += local_output[local_id + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (local_id == 0) {
        local_output[local_id] += bias_ptr[output_c_block_idx];
        local_output[local_id] = ActivationProcess(local_output[local_id], activation_type);

        const int out_x_base = mul24(output_c_block_idx, wh.x);

        int output_w_idx = out_x_base + out_x_idx;
        WI_F(output, (int2)(output_w_idx, bh_idx), local_output[local_id]);
    }
}
