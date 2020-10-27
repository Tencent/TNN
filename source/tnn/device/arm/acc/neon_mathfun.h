/* NEON implementation of sin, cos, exp and log
 *
 *   Inspired by Intel Approximate Math library, and based on the
 *   corresponding algorithms of the cephes math library
 */

/* Copyright (C) 2011  Julien Pommier
 *
 *  This software is provided 'as-is', without any express or implied
 *  warranty.  In no event will the authors be held liable for any damages
 *  arising from the use of this software.
 *
 *  Permission is granted to anyone to use this software for any purpose,
 *  including commercial applications, and to alter it and redistribute it
 *  freely, subject to the following restrictions:
 *
 *  1. The origin of this software must not be misrepresented; you must not
 *     claim that you wrote the original software. If you use this software
 *     in a product, an acknowledgment in the product documentation would be
 *     appreciated but is not required.
 *  2. Altered source versions must be plainly marked as such, and must not be
 *     misrepresented as being the original software.
 *  3. This notice may not be removed or altered from any source distribution.
 *
 *  (this is the zlib license)
 */

#include <arm_neon.h>

#define c_inv_mant_mask ~0x7f800000u
#define c_cephes_SQRTHF 0.707106781186547524
#define c_cephes_log_p0 7.0376836292E-2
#define c_cephes_log_p1 - 1.1514610310E-1
#define c_cephes_log_p2 1.1676998740E-1
#define c_cephes_log_p3 - 1.2420140846E-1
#define c_cephes_log_p4 + 1.4249322787E-1
#define c_cephes_log_p5 - 1.6668057665E-1
#define c_cephes_log_p6 + 2.0000714765E-1
#define c_cephes_log_p7 - 2.4999993993E-1
#define c_cephes_log_p8 + 3.3333331174E-1
#define c_cephes_log_q1 -2.12194440e-4
#define c_cephes_log_q2 0.693359375

/* natural logarithm computed for 4 simultaneous float
 *   return NaN for x <= 0
 */
static inline float32x4_t log_ps(float32x4_t x)
{
    float32x4_t one = vdupq_n_f32(1);

    x = vmaxq_f32(x, vdupq_n_f32(0)); /* force flush to zero on denormal values */
    uint32x4_t invalid_mask = vcleq_f32(x, vdupq_n_f32(0));

    int32x4_t ux = vreinterpretq_s32_f32(x);

    int32x4_t emm0 = vshrq_n_s32(ux, 23);

    /* keep only the fractional part */
    ux = vandq_s32(ux, vdupq_n_s32(c_inv_mant_mask));
    ux = vorrq_s32(ux, vreinterpretq_s32_f32(vdupq_n_f32(0.5f)));
    x = vreinterpretq_f32_s32(ux);

    emm0 = vsubq_s32(emm0, vdupq_n_s32(0x7f));
    float32x4_t e = vcvtq_f32_s32(emm0);

    e = vaddq_f32(e, one);

    /* part2:
     *     if( x < SQRTHF ) {
     *       e -= 1;
     *       x = x + x - 1.0;
     *     } else { x = x - 1.0; }
     */
    uint32x4_t mask = vcltq_f32(x, vdupq_n_f32(c_cephes_SQRTHF));
    float32x4_t tmp = vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(x), mask));
    x = vsubq_f32(x, one);
    e = vsubq_f32(e, vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(one), mask)));
    x = vaddq_f32(x, tmp);

    float32x4_t z = vmulq_f32(x,x);

    float32x4_t y = vdupq_n_f32(c_cephes_log_p0);
    y = vmulq_f32(y, x);
    y = vaddq_f32(y, vdupq_n_f32(c_cephes_log_p1));
    y = vmulq_f32(y, x);
    y = vaddq_f32(y, vdupq_n_f32(c_cephes_log_p2));
    y = vmulq_f32(y, x);
    y = vaddq_f32(y, vdupq_n_f32(c_cephes_log_p3));
    y = vmulq_f32(y, x);
    y = vaddq_f32(y, vdupq_n_f32(c_cephes_log_p4));
    y = vmulq_f32(y, x);
    y = vaddq_f32(y, vdupq_n_f32(c_cephes_log_p5));
    y = vmulq_f32(y, x);
    y = vaddq_f32(y, vdupq_n_f32(c_cephes_log_p6));
    y = vmulq_f32(y, x);
    y = vaddq_f32(y, vdupq_n_f32(c_cephes_log_p7));
    y = vmulq_f32(y, x);
    y = vaddq_f32(y, vdupq_n_f32(c_cephes_log_p8));
    y = vmulq_f32(y, x);

    y = vmulq_f32(y, z);


    tmp = vmulq_f32(e, vdupq_n_f32(c_cephes_log_q1));
    y = vaddq_f32(y, tmp);


    tmp = vmulq_f32(z, vdupq_n_f32(0.5f));
    y = vsubq_f32(y, tmp);

    tmp = vmulq_f32(e, vdupq_n_f32(c_cephes_log_q2));
    x = vaddq_f32(x, y);
    x = vaddq_f32(x, tmp);
    x = vreinterpretq_f32_u32(vorrq_u32(vreinterpretq_u32_f32(x), invalid_mask)); // negative arg will be NAN
    return x;
}

#define c_exp_hi 88.3762626647949f
#define c_exp_lo -88.3762626647949f

#define c_cephes_LOG2EF 1.44269504088896341
#define c_cephes_exp_C1 0.693359375
#define c_cephes_exp_C2 -2.12194440e-4

#define c_cephes_exp_p0 1.9875691500E-4
#define c_cephes_exp_p1 1.3981999507E-3
#define c_cephes_exp_p2 8.3334519073E-3
#define c_cephes_exp_p3 4.1665795894E-2
#define c_cephes_exp_p4 1.6666665459E-1
#define c_cephes_exp_p5 5.0000001201E-1

/* exp() computed for 4 float at once */
static inline float32x4_t exp_ps(float32x4_t x)
{
    float32x4_t tmp, fx;

    float32x4_t one = vdupq_n_f32(1);
    x = vminq_f32(x, vdupq_n_f32(c_exp_hi));
    x = vmaxq_f32(x, vdupq_n_f32(c_exp_lo));

    /* express exp(x) as exp(g + n*log(2)) */
    fx = vmlaq_f32(vdupq_n_f32(0.5f), x, vdupq_n_f32(c_cephes_LOG2EF));

    /* perform a floorf */
    tmp = vcvtq_f32_s32(vcvtq_s32_f32(fx));

    /* if greater, substract 1 */
    uint32x4_t mask = vcgtq_f32(tmp, fx);
    mask = vandq_u32(mask, vreinterpretq_u32_f32(one));


    fx = vsubq_f32(tmp, vreinterpretq_f32_u32(mask));

    tmp = vmulq_f32(fx, vdupq_n_f32(c_cephes_exp_C1));
    float32x4_t z = vmulq_f32(fx, vdupq_n_f32(c_cephes_exp_C2));
    x = vsubq_f32(x, tmp);
    x = vsubq_f32(x, z);

    static const float cephes_exp_p[6] = { c_cephes_exp_p0, c_cephes_exp_p1, c_cephes_exp_p2, c_cephes_exp_p3, c_cephes_exp_p4, c_cephes_exp_p5 };
    float32x4_t y = vld1q_dup_f32(cephes_exp_p+0);
    float32x4_t c1 = vld1q_dup_f32(cephes_exp_p+1);
    float32x4_t c2 = vld1q_dup_f32(cephes_exp_p+2);
    float32x4_t c3 = vld1q_dup_f32(cephes_exp_p+3);
    float32x4_t c4 = vld1q_dup_f32(cephes_exp_p+4);
    float32x4_t c5 = vld1q_dup_f32(cephes_exp_p+5);

    y = vmulq_f32(y, x);
    z = vmulq_f32(x, x);

    y = vaddq_f32(y, c1);
    y = vmulq_f32(y, x);
    y = vaddq_f32(y, c2);
    y = vmulq_f32(y, x);
    y = vaddq_f32(y, c3);
    y = vmulq_f32(y, x);
    y = vaddq_f32(y, c4);
    y = vmulq_f32(y, x);
    y = vaddq_f32(y, c5);

    y = vmulq_f32(y, z);
    y = vaddq_f32(y, x);
    y = vaddq_f32(y, one);

    /* build 2^n */
    int32x4_t mm;
    mm = vcvtq_s32_f32(fx);
    mm = vaddq_s32(mm, vdupq_n_s32(0x7f));
    mm = vshlq_n_s32(mm, 23);
    float32x4_t pow2n = vreinterpretq_f32_s32(mm);

    y = vmulq_f32(y, pow2n);
    return y;
}

#define c_minus_cephes_DP1 -0.78515625
#define c_minus_cephes_DP2 -2.4187564849853515625e-4
#define c_minus_cephes_DP3 -3.77489497744594108e-8
#define c_sincof_p0 -1.9515295891E-4
#define c_sincof_p1  8.3321608736E-3
#define c_sincof_p2 -1.6666654611E-1
#define c_coscof_p0  2.443315711809948E-005
#define c_coscof_p1 -1.388731625493765E-003
#define c_coscof_p2  4.166664568298827E-002
#define c_cephes_FOPI 1.27323954473516 // 4 / M_PI

/* evaluation of 4 sines & cosines at once.
 *
 *   The code is the exact rewriting of the cephes sinf function.
 *   Precision is excellent as long as x < 8192 (I did not bother to
 *   take into account the special handling they have for greater values
 *   -- it does not return garbage for arguments over 8192, though, but
 *   the extra precision is missing).
 *
 *   Note that it is such that sinf((float)M_PI) = 8.74e-8, which is the
 *   surprising but correct result.
 *
 *   Note also that when you compute sin(x), cos(x) is available at
 *   almost no extra price so both sin_ps and cos_ps make use of
 *   sincos_ps..
 */
static inline void sincos_ps(float32x4_t x, float32x4_t *ysin, float32x4_t *ycos)
{
    // any x
    float32x4_t xmm1, xmm2, xmm3, y;

    uint32x4_t emm2;

    uint32x4_t sign_mask_sin, sign_mask_cos;
    sign_mask_sin = vcltq_f32(x, vdupq_n_f32(0));
    x = vabsq_f32(x);

    /* scale by 4/Pi */
    y = vmulq_f32(x, vdupq_n_f32(c_cephes_FOPI));

    /* store the integer part of y in mm0 */
    emm2 = vcvtq_u32_f32(y);
    /* j=(j+1) & (~1) (see the cephes sources) */
    emm2 = vaddq_u32(emm2, vdupq_n_u32(1));
    emm2 = vandq_u32(emm2, vdupq_n_u32(~1));
    y = vcvtq_f32_u32(emm2);

    /* get the polynom selection mask
     *     there is one polynom for 0 <= x <= Pi/4
     *     and another one for Pi/4<x<=Pi/2
     *
     *     Both branches will be computed.
     */
    uint32x4_t poly_mask = vtstq_u32(emm2, vdupq_n_u32(2));

    /* The magic pass: "Extended precision modular arithmetic"
     *     x = ((x - y * DP1) - y * DP2) - y * DP3; */
    xmm1 = vmulq_n_f32(y, c_minus_cephes_DP1);
    xmm2 = vmulq_n_f32(y, c_minus_cephes_DP2);
    xmm3 = vmulq_n_f32(y, c_minus_cephes_DP3);
    x = vaddq_f32(x, xmm1);
    x = vaddq_f32(x, xmm2);
    x = vaddq_f32(x, xmm3);

    sign_mask_sin = veorq_u32(sign_mask_sin, vtstq_u32(emm2, vdupq_n_u32(4)));
    sign_mask_cos = vtstq_u32(vsubq_u32(emm2, vdupq_n_u32(2)), vdupq_n_u32(4));

    /* Evaluate the first polynom  (0 <= x <= Pi/4) in y1,
     *     and the second polynom      (Pi/4 <= x <= 0) in y2 */
    float32x4_t z = vmulq_f32(x,x);
    float32x4_t y1, y2;

    y1 = vmulq_n_f32(z, c_coscof_p0);
    y2 = vmulq_n_f32(z, c_sincof_p0);
    y1 = vaddq_f32(y1, vdupq_n_f32(c_coscof_p1));
    y2 = vaddq_f32(y2, vdupq_n_f32(c_sincof_p1));
    y1 = vmulq_f32(y1, z);
    y2 = vmulq_f32(y2, z);
    y1 = vaddq_f32(y1, vdupq_n_f32(c_coscof_p2));
    y2 = vaddq_f32(y2, vdupq_n_f32(c_sincof_p2));
    y1 = vmulq_f32(y1, z);
    y2 = vmulq_f32(y2, z);
    y1 = vmulq_f32(y1, z);
    y2 = vmulq_f32(y2, x);
    y1 = vsubq_f32(y1, vmulq_f32(z, vdupq_n_f32(0.5f)));
    y2 = vaddq_f32(y2, x);
    y1 = vaddq_f32(y1, vdupq_n_f32(1));

    /* select the correct result from the two polynoms */
    float32x4_t ys = vbslq_f32(poly_mask, y1, y2);
    float32x4_t yc = vbslq_f32(poly_mask, y2, y1);
    *ysin = vbslq_f32(sign_mask_sin, vnegq_f32(ys), ys);
    *ycos = vbslq_f32(sign_mask_cos, yc, vnegq_f32(yc));
}

static inline float32x4_t sin_ps(float32x4_t x)
{
    float32x4_t ysin, ycos;
    sincos_ps(x, &ysin, &ycos);
    return ysin;
}

static inline float32x4_t cos_ps(float32x4_t x)
{
    float32x4_t ysin, ycos;
    sincos_ps(x, &ysin, &ycos);
    return ycos;
}

static inline float32x4_t div_ps(float32x4_t a, float32x4_t b)
{
    float32x4_t reciprocal = vrecpeq_f32(b);
    reciprocal = vmulq_f32(vrecpsq_f32(b, reciprocal), reciprocal);
    reciprocal = vmulq_f32(vrecpsq_f32(b, reciprocal), reciprocal);
    return vmulq_f32(a, reciprocal);
}

static inline float32x4_t pow_ps(float32x4_t a, float32x4_t b)
{
    // pow(x, m) = exp(m * log(x))
    return exp_ps(vmulq_f32(b, log_ps(a)));
}

// tanh neon vector version
// refer the scalar version from Cephes Math Library

#define c_cephes_HALFMAXLOGF 44.014845935754205f
#define c_cephes_tanh_C1 0.625f

#define c_cephes_tanh_p0 - 5.70498872745E-3
#define c_cephes_tanh_p1 + 2.06390887954E-2
#define c_cephes_tanh_p2 - 5.37397155531E-2
#define c_cephes_tanh_p3 + 1.33314422036E-1
#define c_cephes_tanh_p4 - 3.33332819422E-1

/* Single precision hyperbolic tangent computed for 4 simultaneous float */
static inline float32x4_t tanh_ps(float32x4_t x)
{
    float32x4_t x2 = vabsq_f32(x);

    uint32x4_t mask_l = vcgeq_f32(x2, vdupq_n_f32(c_cephes_tanh_C1));
    uint32x4_t mask_l2 = vcgtq_f32(x2, vdupq_n_f32(c_cephes_HALFMAXLOGF));

    // abs(x) >= 0.625
    // tanh(x) = 1 − 2 / (exp(2x) + 1)
    float32x4_t _one = vdupq_n_f32(1.f);
    float32x4_t _two = vdupq_n_f32(2.f);
    float32x4_t exp_x_x = exp_ps(vaddq_f32(x, x));
#if __aarch64__
    float32x4_t y0 = vsubq_f32(_one, vdivq_f32(_two, vaddq_f32(exp_x_x, _one)));
#else
    float32x4_t y0 = vsubq_f32(_one, div_ps(_two, vaddq_f32(exp_x_x, _one)));
#endif

    // abs(x) < 0.625
    /*
        z = x2 * x2;
        z =
        (((( -5.70498872745E-3 * z
        + 2.06390887954E-2) * z
        - 5.37397155531E-2) * z
        + 1.33314422036E-1) * z
        - 3.33332819422E-1) * z * x
        + x;
    */
    static const float cephes_tanh_p[5] = { c_cephes_tanh_p0, c_cephes_tanh_p1, c_cephes_tanh_p2, c_cephes_tanh_p3, c_cephes_tanh_p4 };
    float32x4_t y = vld1q_dup_f32(cephes_tanh_p+0);
    float32x4_t c1 = vld1q_dup_f32(cephes_tanh_p+1);
    float32x4_t c2 = vld1q_dup_f32(cephes_tanh_p+2);
    float32x4_t c3 = vld1q_dup_f32(cephes_tanh_p+3);
    float32x4_t c4 = vld1q_dup_f32(cephes_tanh_p+4);

    float32x4_t z = vmulq_f32(x, x);

    y = vmulq_f32(y, z);
    y = vaddq_f32(y, c1);
    y = vmulq_f32(y, z);
    y = vaddq_f32(y, c2);
    y = vmulq_f32(y, z);
    y = vaddq_f32(y, c3);
    y = vmulq_f32(y, z);
    y = vaddq_f32(y, c4);

    y = vmulq_f32(y, z);
    y = vmulq_f32(y, x);
    y = vaddq_f32(y, x);

    // abs(x) > HALFMAXLOGF
    // return 1.0 or -1.0
    uint32x4_t mask_pos = vcgtq_f32(x2, vdupq_n_f32(0.f));
    float32x4_t y1 = vreinterpretq_f32_u32(vbslq_u32(mask_pos, vreinterpretq_u32_f32(vdupq_n_f32(1.f)), vreinterpretq_u32_f32(vdupq_n_f32(-1.f))));

    y = vreinterpretq_f32_u32(vbslq_u32(mask_l, vreinterpretq_u32_f32(y0), vreinterpretq_u32_f32(y)));
    y = vreinterpretq_f32_u32(vbslq_u32(mask_l2, vreinterpretq_u32_f32(y1), vreinterpretq_u32_f32(y)));
    return y;
}

static inline float32x4_t sqrt_ps(float32x4_t v)
{
    // 1/sqrt(x)
    float32x4_t y;
    float32x4_t y1 = v;
    float32x4_t y0 = vrsqrteq_f32(v);
    float32x4_t y2 = vmulq_f32(y0, y1);
    float32x4_t y3 = vrsqrtsq_f32(y2, y0);
    y0 = vmulq_f32(y0, y3);
    y2 = vmulq_f32(y0, y1);
    y3 = vrsqrtsq_f32(y2, y0);
    y0 = vmulq_f32(y0, y3);

    // 1/x
    // y1 = vrecpeq_f32(y0);
    // y2 = vrecpsq_f32(y1, y0);
    // y1 = vmulq_f32(y1, y2);
    // y2 = vrecpsq_f32(y1, y0);
    // y0 = vmulq_f32(y1, y2);
    float32x4_t one = vdupq_n_f32(1.f);
    y = div_ps(one, y0);

    return y;
}

static inline float32x4_t sigmoid_ps(float32x4_t v)
{
    float32x4_t y;
    float32x4_t one = vdupq_n_f32(1.f);
    float32x4_t zero = vdupq_n_f32(0.f);

    // 1.0f/(1.0f + std::exp(-x))
    y = div_ps(one, vaddq_f32(one, exp_ps(vsubq_f32(zero, v))));

    return y;
}

//http://ybeernet.blogspot.com/2011/03/speeding-up-sigmoid-function-by.html
/* sigmoid() computed for 4 float at once: small error*/
static inline float32x4_t fast_sigmoid_ps(float32x4_t x)
{
    float32x4_t const16 = vdupq_n_f32(16.0f);
    uint32x4_t mask = vcgeq_f32(x, const16);
    
    float32x4_t const1 = vdupq_n_f32(1.0f);
    float32x4_t const025 = vdupq_n_f32(0.25f*0.25f);
    
    //y = 1-x/16
    float32x4_t temp = vmlsq_f32(const1, const025, x);
    //y2
    temp = vmulq_f32(temp, temp);
    //y4
    temp = vmulq_f32(temp, temp);
    //y8
    temp = vmulq_f32(temp, temp);
    //y16
    temp = vmulq_f32(temp, temp);
    
    temp = vaddq_f32(temp, const1);
    float32x4_t result = vrecpeq_f32(temp);
    result = vmulq_f32(vrecpsq_f32(temp, result), result);
//    result = vmulq_f32(vrecpsq_f32(temp, result), result);
    result = vbslq_f32(mask, const1, result);
    
    return result;
}
