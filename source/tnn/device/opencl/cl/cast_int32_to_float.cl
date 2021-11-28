#include "base.inc"

__kernel void CastIntToFloat(GLOBAL_SIZE_2_DIMS  
                    __read_only image2d_t input, 
                    __write_only image2d_t output,
                    int4 input_offset,
                    int4 output_offset,
                    int2 input_wh,
                    int2 output_wh,
                    int2 wh
                    ) {
    int cw = get_global_id(0);
    int bh = get_global_id(1);
    DEAL_NON_UNIFORM_DIM2(cw, bh);

    //N, C, H, W
    int4 pos = (int4)(bh/wh.y, cw/wh.x, bh%wh.y, cw%wh.x);

    int4 pos_input = input_offset + pos;
    int4 pos_output = output_offset + pos;

    int2 output_pos = (int2)(pos_output.w + pos_output.y*output_wh.x, pos_output.x*output_wh.y + pos_output.z);
    int2 input_pos = (int2)(pos_input.w + pos_input.y*input_wh.x, pos_input.x*input_wh.y + pos_input.z);

    WI_F(output, output_pos, CONVERT(read_imagei(input, SAMPLER, input_pos)));
}