
#if __ARM_NEON
#include <arm_neon.h>
#endif // __ARM_NEON

void rotate_3_c1(const unsigned char* src, int srcw, int srch, unsigned char* dst)
{
    int w = srcw;
    int h = srch;

    // point to the last dst pixel
    unsigned char* dstend = dst + w * h;

    const unsigned char* src0 = src;
    unsigned char* dst0 = dstend - 1;

    int y = 0;
    for (; y < srch; y++)
    {
#if 0//__ARM_NEON
        dst0 -= 7;

        int nn = srcw >> 4;
        int remain = srcw - (nn << 4);

#if __aarch64__
        for (; nn>0; nn--)
        {
            uint8x8_t _src = vld1_u8(src0);
            uint8x8_t _src2 = vld1_u8(src0 + 8);

            _src = vrev64_u8(_src);

            _src2 = vrev64_u8(_src2);

            vst1_u8(dst0, _src);
            vst1_u8(dst0 - 8, _src2);

            src0 += 16;
            dst0 -= 16;
        }
#else
        if (nn > 0)
        {
        asm volatile(
            "mov        r4, #-16            \n"
            "0:                             \n"
            "pld        [%1, #128]          \n"
            "vld2.u8    {d0-d1}, [%1]!      \n"
            "vrev64.u8  d0, d0              \n"
            "vrev64.u8  d1, d1              \n"
            "subs       %0, #1              \n"
            "vst2.u8    {d0-d1}, [%2], r4   \n"
            "bne        0b                  \n"
            : "=r"(nn),     // %0
              "=r"(src0),   // %1
              "=r"(dst0)    // %2
            : "0"(nn),
              "1"(src0),
              "2"(dst0)
            : "cc", "memory", "q0", "r4"
        );
        }
#endif // __aarch64__

        dst0 += 7;
#else
        int remain = srcw;
#endif // __ARM_NEON

        for (; remain>0; remain--)
        {
            *dst0 = *src0;

            src0 += 1;
            dst0 -= 1;
        }
    }
}

void rotate_3_c2(const unsigned char* src, int srcw, int srch, unsigned char* dst)
{
    int w = srcw;
    int h = srch;

    // point to the last dst pixel
    unsigned char* dstend = dst + w * h * 2;

    const unsigned char* src0 = src;
    unsigned char* dst0 = dstend - 2;

    int y = 0;
    for (; y < srch; y++)
    {
#if __ARM_NEON
        dst0 -= 7*2;

        int nn = srcw >> 4;
        int remain = srcw - (nn << 4);

#if __aarch64__
        for (; nn>0; nn--)
        {
            uint8x8x2_t _src = vld2_u8(src0);
            uint8x8x2_t _src2 = vld2_u8(src0 + 8*2);

            _src.val[0] = vrev64_u8(_src.val[0]);
            _src.val[1] = vrev64_u8(_src.val[1]);

            _src2.val[0] = vrev64_u8(_src2.val[0]);
            _src2.val[1] = vrev64_u8(_src2.val[1]);

            vst2_u8(dst0, _src);
            vst2_u8(dst0 - 8*2, _src2);

            src0 += 16*2;
            dst0 -= 16*2;
        }
#else
        if (nn > 0)
        {
        asm volatile(
            "mov        r4, #-16            \n"
            "0:                             \n"
            "pld        [%1, #128]          \n"
            "vld2.u8    {d0-d1}, [%1]!      \n"
            "vrev64.u8  d0, d0              \n"
            "pld        [%1, #128]          \n"
            "vld2.u8    {d2-d3}, [%1]!      \n"
            "vrev64.u8  d1, d1              \n"
            "vrev64.u8  d2, d2              \n"
            "vst2.u8    {d0-d1}, [%2], r4   \n"
            "vrev64.u8  d3, d3              \n"
            "subs       %0, #1              \n"
            "vst2.u8    {d2-d3}, [%2], r4   \n"
            "bne        0b                  \n"
            : "=r"(nn),     // %0
              "=r"(src0),   // %1
              "=r"(dst0)    // %2
            : "0"(nn),
              "1"(src0),
              "2"(dst0)
            : "cc", "memory", "q0", "q1", "r4"
        );
        }
#endif // __aarch64__

        dst0 += 7*2;
#else
        int remain = srcw;
#endif // __ARM_NEON

        for (; remain>0; remain--)
        {
            dst0[0] = src0[0];
            dst0[1] = src0[1];

            src0 += 2;
            dst0 -= 2;
        }
    }
}

void rotate_3_c3(const unsigned char* src, int srcw, int srch, unsigned char* dst)
{
    int w = srcw;
    int h = srch;

    // point to the last dst pixel
    unsigned char* dstend = dst + w * h * 3;

    const unsigned char* src0 = src;
    unsigned char* dst0 = dstend - 3;

    int y = 0;
    for (; y < srch; y++)
    {
#if __ARM_NEON
        dst0 -= 7*3;

        int nn = srcw >> 4;
        int remain = srcw - (nn << 4);

#if __aarch64__
        for (; nn>0; nn--)
        {
            uint8x8x3_t _src = vld3_u8(src0);
            uint8x8x3_t _src2 = vld3_u8(src0 + 8*3);

            _src.val[0] = vrev64_u8(_src.val[0]);
            _src.val[1] = vrev64_u8(_src.val[1]);
            _src.val[2] = vrev64_u8(_src.val[2]);

            _src2.val[0] = vrev64_u8(_src2.val[0]);
            _src2.val[1] = vrev64_u8(_src2.val[1]);
            _src2.val[2] = vrev64_u8(_src2.val[2]);

            vst3_u8(dst0, _src);
            vst3_u8(dst0 - 8*3, _src2);

            src0 += 16*3;
            dst0 -= 16*3;
        }
#else
        if (nn > 0)
        {
        asm volatile(
            "mov        r4, #-24            \n"
            "0:                             \n"
            "pld        [%1, #192]          \n"
            "vld3.u8    {d0-d2}, [%1]!      \n"
            "vrev64.u8  d0, d0              \n"
            "vrev64.u8  d1, d1              \n"
            "pld        [%1, #192]          \n"
            "vld3.u8    {d4-d6}, [%1]!      \n"
            "vrev64.u8  d2, d2              \n"
            "vrev64.u8  d4, d4              \n"
            "vst3.u8    {d0-d2}, [%2], r4   \n"
            "vrev64.u8  d5, d5              \n"
            "vrev64.u8  d6, d6              \n"
            "subs       %0, #1              \n"
            "vst3.u8    {d4-d6}, [%2], r4   \n"
            "bne        0b                  \n"
            : "=r"(nn),     // %0
              "=r"(src0),   // %1
              "=r"(dst0)    // %2
            : "0"(nn),
              "1"(src0),
              "2"(dst0)
            : "cc", "memory", "q0", "q1", "q2", "q3", "r4"
        );
        }
#endif // __aarch64__

        dst0 += 7*3;
#else
        int remain = srcw;
#endif // __ARM_NEON

        for (; remain>0; remain--)
        {
            dst0[0] = src0[0];
            dst0[1] = src0[1];
            dst0[2] = src0[2];

            src0 += 3;
            dst0 -= 3;
        }
    }
}

void rotate_3_c4(const unsigned char* src, int srcw, int srch, unsigned char* dst)
{
    int w = srcw;
    int h = srch;

    // point to the last dst pixel
    unsigned char* dstend = dst + w * h * 4;

    const unsigned char* src0 = src;
    unsigned char* dst0 = dstend - 4;

    int y = 0;
    for (; y < srch; y++)
    {
#if __ARM_NEON
        dst0 -= 7*4;

        int nn = srcw >> 4;
        int remain = srcw - (nn << 4);

#if __aarch64__
        for (; nn>0; nn--)
        {
            uint8x8x4_t _src = vld4_u8(src0);
            uint8x8x4_t _src2 = vld4_u8(src0 + 8*4);

            _src.val[0] = vrev64_u8(_src.val[0]);
            _src.val[1] = vrev64_u8(_src.val[1]);
            _src.val[2] = vrev64_u8(_src.val[2]);
            _src.val[3] = vrev64_u8(_src.val[3]);

            _src2.val[0] = vrev64_u8(_src2.val[0]);
            _src2.val[1] = vrev64_u8(_src2.val[1]);
            _src2.val[2] = vrev64_u8(_src2.val[2]);
            _src2.val[3] = vrev64_u8(_src2.val[3]);

            vst4_u8(dst0, _src);
            vst4_u8(dst0 - 8*4, _src2);

            src0 += 16*4;
            dst0 -= 16*4;
        }
#else
        if (nn > 0)
        {
        asm volatile(
            "mov        r4, #-32            \n"
            "0:                             \n"
            "pld        [%1, #256]          \n"
            "vld4.u8    {d0-d3}, [%1]!      \n"
            "vrev64.u8  d0, d0              \n"
            "vrev64.u8  d1, d1              \n"
            "vrev64.u8  d2, d2              \n"
            "pld        [%1, #256]          \n"
            "vld4.u8    {d4-d7}, [%1]!      \n"
            "vrev64.u8  d3, d3              \n"
            "vrev64.u8  d4, d4              \n"
            "vrev64.u8  d5, d5              \n"
            "vst4.u8    {d0-d3}, [%2], r4   \n"
            "vrev64.u8  d6, d6              \n"
            "vrev64.u8  d7, d7              \n"
            "subs       %0, #1              \n"
            "vst4.u8    {d4-d7}, [%2], r4   \n"
            "bne        0b                  \n"
            : "=r"(nn),     // %0
              "=r"(src0),   // %1
              "=r"(dst0)    // %2
            : "0"(nn),
              "1"(src0),
              "2"(dst0)
            : "cc", "memory", "q0", "q1", "q2", "q3", "r4"
        );
        }
#endif // __aarch64__

        dst0 += 7*4;
#else
        int remain = srcw;
#endif // __ARM_NEON

        for (; remain>0; remain--)
        {
            dst0[0] = src0[0];
            dst0[1] = src0[1];
            dst0[2] = src0[2];
            dst0[3] = src0[3];

            src0 += 4;
            dst0 -= 4;
        }
    }
}
