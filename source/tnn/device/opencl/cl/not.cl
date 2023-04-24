#include "base.inc"

__kernel void Not(GLOBAL_SIZE_2_DIMS
                  __read_only image2d_t input,
                  __write_only image2d_t output
                  ) {
    int cw = get_global_id(0);
    int bh = get_global_id(1);
    DEAL_NON_UNIFORM_DIM2(cw, bh);

    int4 val = read_imagei(input, SAMPLER, (int2)(cw, bh));

    int4 output_val = {val.x == 0, val.y == 0, val.z == 0, val.w == 0};

    write_imagei(output, (int2)(cw, bh), output_val);
}
