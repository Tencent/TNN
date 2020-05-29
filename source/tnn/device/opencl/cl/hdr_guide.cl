#include "base.inc"

__kernel void HdrGuide(GLOBAL_SIZE_2_DIMS __read_only image2d_t input,
                       __write_only image2d_t output, __read_only image2d_t ccm,
                       __read_only image2d_t shifts,
                       __read_only image2d_t slopes,
                       __read_only image2d_t projection) {
    const int w  = get_global_id(0);
    const int hb = get_global_id(1);

    DEAL_NON_UNIFORM_DIM2(w, hb);

    FLOAT4 in = RI_F(input, SAMPLER, (int2)(w, hb));
    in.w      = 1.0f;

    FLOAT4 new_rgb;
    new_rgb.x = dot(in, RI_F(ccm, SAMPLER, (int2)(0, 0)));
    new_rgb.y = dot(in, RI_F(ccm, SAMPLER, (int2)(1, 0)));
    new_rgb.z = dot(in, RI_F(ccm, SAMPLER, (int2)(2, 0)));
    new_rgb.w = 0.0f;

    FLOAT4 guide_rgb = (FLOAT4)(0.0f, 0.0f, 0.0f, 0.0f);
    guide_rgb += max(new_rgb - RI_F(shifts, SAMPLER, (int2)(0, 0)),
                     (FLOAT4)(0.0f, 0.0f, 0.0f, 0.0f)) *
                 RI_F(slopes, SAMPLER, (int2)(0, 0));
    guide_rgb += max(new_rgb - RI_F(shifts, SAMPLER, (int2)(1, 0)),
                     (FLOAT4)(0.0f, 0.0f, 0.0f, 0.0f)) *
                 RI_F(slopes, SAMPLER, (int2)(1, 0));
    guide_rgb += max(new_rgb - RI_F(shifts, SAMPLER, (int2)(2, 0)),
                     (FLOAT4)(0.0f, 0.0f, 0.0f, 0.0f)) *
                 RI_F(slopes, SAMPLER, (int2)(2, 0));
    guide_rgb += max(new_rgb - RI_F(shifts, SAMPLER, (int2)(3, 0)),
                     (FLOAT4)(0.0f, 0.0f, 0.0f, 0.0f)) *
                 RI_F(slopes, SAMPLER, (int2)(3, 0));
    guide_rgb.w = 1.0f;

    FLOAT out_val = dot(guide_rgb, RI_F(projection, SAMPLER, (int2)(0, 0)));
    out_val       = clamp(out_val, (FLOAT)(0.0f), (FLOAT)(1.0f));

    WI_F(output, (int2)(w, hb), (FLOAT4)(out_val, 0.0f, 0.0f, 0.0f));
}
