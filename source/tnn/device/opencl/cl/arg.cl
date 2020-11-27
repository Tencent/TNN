#include "base.inc"

#define MinOp <
#define MaxOp >

#define CALCULATE_GUARD(in, guard, idx)     \
    if (in BINARY_OPERATOR guard) {         \
        guard = in;                         \
        out.x = idx;                        \
    }


__kernel void ArgOpN(GLOBAL_SIZE_2_DIMS __read_only image2d_t input,
                     __write_only image2d_t output,
                     __private const int output_channel,
                     __private const int output_height,
                     __private const int output_width,
                     __private const int input_batch) {
    const int out_cw_idx        = get_global_id(0);
    const int out_height_idx    = get_global_id(1);

    DEAL_NON_UNIFORM_DIM2(out_cw_idx, out_height_idx);

    FLOAT4 guard_val    = RI_F(input, SAMPLER, (int2)(out_cw_idx, out_height_idx));
    FLOAT4 out          = 0;

    for (int b = 1; b < input_batch; b++) {
        int2 pos    = (int2)(out_cw_idx, b * output_height + out_height_idx);
        FLOAT4 in   = RI_F(input, SAMPLER, pos);

        out.x = select(out.x, (FLOAT)b, CONVERT_INT(in.x BINARY_OPERATOR guard_val.x));
        out.y = select(out.y, (FLOAT)b, CONVERT_INT(in.y BINARY_OPERATOR guard_val.y));
        out.z = select(out.z, (FLOAT)b, CONVERT_INT(in.z BINARY_OPERATOR guard_val.z));
        out.w = select(out.w, (FLOAT)b, CONVERT_INT(in.w BINARY_OPERATOR guard_val.w));

        guard_val = OPERATOR(guard_val, in);
    }

    WI_F(output, (int2)(out_cw_idx, out_height_idx), out);
}

__kernel void ArgOpC(GLOBAL_SIZE_2_DIMS __read_only image2d_t input,
                     __write_only image2d_t output,
                     __private const int output_batch,
                     __private const int output_height,
                     __private const int output_width,
                     __private const int input_channel,
                     __private const int input_channel_blocks) {
    const int out_width_idx = get_global_id(0);
    const int out_bh_idx    = get_global_id(1);

    DEAL_NON_UNIFORM_DIM2(out_width_idx, out_bh_idx);

    FLOAT4 in       = RI_F(input, SAMPLER, (int2)(out_width_idx, out_bh_idx));
    FLOAT guard_val = in.x;
    FLOAT4 out      = 0;
    
    int input_c_base, cb;
    if (input_channel >= 4) {
        CALCULATE_GUARD(in.y, guard_val, 1);
        CALCULATE_GUARD(in.z, guard_val, 2);
        CALCULATE_GUARD(in.w, guard_val, 3);

        for (cb = 1; cb < input_channel_blocks - 1; cb++) {
            int2 pos        = (int2)(cb * output_width + out_width_idx, out_bh_idx);
            in              = RI_F(input, SAMPLER, pos);
            input_c_base    = cb << 2;

            CALCULATE_GUARD(in.x, guard_val, input_c_base);
            CALCULATE_GUARD(in.y, guard_val, input_c_base + 1);
            CALCULATE_GUARD(in.z, guard_val, input_c_base + 2);
            CALCULATE_GUARD(in.w, guard_val, input_c_base + 3);
        }

        if (input_channel > 4) {
            int2 pos        = (int2)(cb * output_width + out_width_idx, out_bh_idx);
            in              = RI_F(input, SAMPLER, pos);
            input_c_base    = cb << 2;
            int remain      = input_channel - input_c_base;
            if (remain >= 4) {
                CALCULATE_GUARD(in.x, guard_val, input_c_base);
                CALCULATE_GUARD(in.y, guard_val, input_c_base + 1);
                CALCULATE_GUARD(in.z, guard_val, input_c_base + 2);
                CALCULATE_GUARD(in.w, guard_val, input_c_base + 3);
            } else if (remain == 3) {
                CALCULATE_GUARD(in.x, guard_val, input_c_base);
                CALCULATE_GUARD(in.y, guard_val, input_c_base + 1);
                CALCULATE_GUARD(in.z, guard_val, input_c_base + 2);
            } else if (remain == 2) {
                CALCULATE_GUARD(in.x, guard_val, input_c_base);
                CALCULATE_GUARD(in.y, guard_val, input_c_base + 1);
            } else if (remain == 1) {
                CALCULATE_GUARD(in.x, guard_val, input_c_base);
            }
        }
    } else if (input_channel == 3) {
        CALCULATE_GUARD(in.y, guard_val, 1);
        CALCULATE_GUARD(in.z, guard_val, 2);
    } else if (input_channel == 2) {
        CALCULATE_GUARD(in.y, guard_val, 1);
    }

    WI_F(output, (int2)(out_width_idx, out_bh_idx), out);
}

__kernel void ArgOpH(GLOBAL_SIZE_2_DIMS __read_only image2d_t input,
                     __write_only image2d_t output,
                     __private const int output_batch,
                     __private const int output_channel,
                     __private const int output_width,
                     __private const int input_height) {
    const int out_cw_idx        = get_global_id(0);
    const int out_batch_idx     = get_global_id(1);

    DEAL_NON_UNIFORM_DIM2(out_cw_idx, out_batch_idx);

    int input_bh_idx    = out_batch_idx * input_height;
    FLOAT4 guard_val    = RI_F(input, SAMPLER, (int2)(out_cw_idx, input_bh_idx));
    FLOAT4 out          = 0;

    for (int h = 1; h < input_height; h++) {
        int2 pos    = (int2)(out_cw_idx, input_bh_idx + h);
        FLOAT4 in   = RI_F(input, SAMPLER, pos);

        out.x = select(out.x, (FLOAT)h, CONVERT_INT(in.x BINARY_OPERATOR guard_val.x));
        out.y = select(out.y, (FLOAT)h, CONVERT_INT(in.y BINARY_OPERATOR guard_val.y));
        out.z = select(out.z, (FLOAT)h, CONVERT_INT(in.z BINARY_OPERATOR guard_val.z));
        out.w = select(out.w, (FLOAT)h, CONVERT_INT(in.w BINARY_OPERATOR guard_val.w));

        guard_val = OPERATOR(guard_val, in);
    }

    WI_F(output, (int2)(out_cw_idx, out_batch_idx), out);
}

__kernel void ArgOpW(GLOBAL_SIZE_2_DIMS __read_only image2d_t input,
                     __write_only image2d_t output,
                     __private const int output_batch,
                     __private const int output_channel,
                     __private const int output_height,
                     __private const int input_width) {
    const int out_channel_blocks_idx    = get_global_id(0);
    const int out_bh_idx                = get_global_id(1);

    DEAL_NON_UNIFORM_DIM2(out_channel_blocks_idx, out_bh_idx);

    int input_cw_idx    = out_channel_blocks_idx * input_width;
    FLOAT4 guard_val    = RI_F(input, SAMPLER, (int2)(input_cw_idx, out_bh_idx));
    FLOAT4 out          = 0;

    for (int w = 1; w < input_width; w++) {
        int2 pos    = (int2)(input_cw_idx + w, out_bh_idx);
        FLOAT4 in   = RI_F(input, SAMPLER, pos);

        out.x = select(out.x, (FLOAT)w, CONVERT_INT(in.x BINARY_OPERATOR guard_val.x));
        out.y = select(out.y, (FLOAT)w, CONVERT_INT(in.y BINARY_OPERATOR guard_val.y));
        out.z = select(out.z, (FLOAT)w, CONVERT_INT(in.z BINARY_OPERATOR guard_val.z));
        out.w = select(out.w, (FLOAT)w, CONVERT_INT(in.w BINARY_OPERATOR guard_val.w));

        guard_val = OPERATOR(guard_val, in);
    }

    WI_F(output, (int2)(out_channel_blocks_idx, out_bh_idx), out);
}
