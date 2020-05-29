
#if __ARM_NEON
#include <arm_neon.h>
#endif // __ARM_NEON

void rotate_4_c1(const unsigned char* src, int srcw, int srch, unsigned char* dst)
{
    int w = srcw;
    int h = srch;

    // point to the last dst pixel row
    unsigned char* dstend = dst + w * (h-1);

    int size = srcw;

    const unsigned char* src0 = src;
    const unsigned char* src1 = src + size;
    unsigned char* dst0 = dstend;
    unsigned char* dst1 = dstend - size;

    int y = 0;
    for (; y+1 < srch; y+=2)
    {
#if __ARM_NEON
        int nn = size >> 5;
        int remain = size - (nn << 5);
#if __aarch64__
        for (; nn>0; nn--)
        {
            uint8x16_t _src0 = vld1q_u8(src0);
            uint8x16_t _src0n = vld1q_u8(src0 + 16);
            vst1q_u8(dst0, _src0);
            vst1q_u8(dst0 + 16, _src0n);

            uint8x16_t _src1 = vld1q_u8(src1);
            uint8x16_t _src1n = vld1q_u8(src1 + 16);
            vst1q_u8(dst1, _src1);
            vst1q_u8(dst1 + 16, _src1n);

            src0 += 32;
            src1 += 32;
            dst0 += 32;
            dst1 += 32;
        }
#else
        if (nn > 0)
        {
        asm volatile(
            "0:                             \n"
            "pld        [%1, #256]          \n"
            "vld1.u8    {d0-d3}, [%1]!      \n"
            "pld        [%2, #256]          \n"
            "vld1.u8    {d4-d7}, [%2]!      \n"
            "subs       %0, #1              \n"
            "vst1.u8    {d0-d3}, [%3]!      \n"
            "vst1.u8    {d4-d7}, [%4]!      \n"
            "bne        0b                  \n"
            : "=r"(nn),     // %0
              "=r"(src0),   // %1
              "=r"(src1),   // %2
              "=r"(dst0),   // %3
              "=r"(dst1)    // %4
            : "0"(nn),
              "1"(src0),
              "2"(src1),
              "3"(dst0),
              "4"(dst1)
            : "cc", "memory", "q0", "q1", "q2", "q3"
        );
        }
#endif // __aarch64__
#else
        int remain = size;
#endif // __ARM_NEON

        for (; remain>0; remain--)
        {
            *dst0++ = *src0++;
            *dst1++ = *src1++;
        }

        src0 += size;
        src1 += size;
        dst0 -= size*3;
        dst1 -= size*3;
    }

    for (; y < srch; y++)
    {
#if __ARM_NEON
        int nn = size >> 5;
        int remain = size - (nn << 5);
#if __aarch64__
        for (; nn>0; nn--)
        {
            uint8x16_t _src = vld1q_u8(src0);
            uint8x16_t _src2 = vld1q_u8(src0 + 16);
            vst1q_u8(dst0, _src);
            vst1q_u8(dst0 + 16, _src2);

            src0 += 32;
            dst0 += 32;
        }
#else
        if (nn > 0)
        {
        asm volatile(
            "0:                             \n"
            "pld        [%1, #256]          \n"
            "vld1.u8    {d0-d3}, [%1]!      \n"
            "subs       %0, #1              \n"
            "vst1.u8    {d0-d3}, [%2]!      \n"
            "bne        0b                  \n"
            : "=r"(nn),     // %0
              "=r"(src0),   // %1
              "=r"(dst0)    // %2
            : "0"(nn),
              "1"(src0),
              "2"(dst0)
            : "cc", "memory", "q0", "q1"
        );
        }
#endif // __aarch64__
#else
        int remain = size;
#endif // __ARM_NEON

        for (; remain>0; remain--)
        {
            *dst0++ = *src0++;
        }

        dst0 -= size*2;
    }
}

void rotate_4_c2(const unsigned char* src, int srcw, int srch, unsigned char* dst)
{
    int w = srcw;
    int h = srch;

    // point to the last dst pixel row
    unsigned char* dstend = dst + w * (h-1) * 2;

    int size = srcw * 2;

    const unsigned char* src0 = src;
    const unsigned char* src1 = src + size;
    unsigned char* dst0 = dstend;
    unsigned char* dst1 = dstend - size;

    int y = 0;
    for (; y+1 < srch; y+=2)
    {
#if __ARM_NEON
        int nn = size >> 5;
        int remain = size - (nn << 5);
#if __aarch64__
        for (; nn>0; nn--)
        {
            uint8x16_t _src0 = vld1q_u8(src0);
            uint8x16_t _src0n = vld1q_u8(src0 + 16);
            vst1q_u8(dst0, _src0);
            vst1q_u8(dst0 + 16, _src0n);

            uint8x16_t _src1 = vld1q_u8(src1);
            uint8x16_t _src1n = vld1q_u8(src1 + 16);
            vst1q_u8(dst1, _src1);
            vst1q_u8(dst1 + 16, _src1n);

            src0 += 32;
            src1 += 32;
            dst0 += 32;
            dst1 += 32;
        }
#else
        if (nn > 0)
        {
        asm volatile(
            "0:                             \n"
            "pld        [%1, #256]          \n"
            "vld1.u8    {d0-d3}, [%1]!      \n"
            "pld        [%2, #256]          \n"
            "vld1.u8    {d4-d7}, [%2]!      \n"
            "subs       %0, #1              \n"
            "vst1.u8    {d0-d3}, [%3]!      \n"
            "vst1.u8    {d4-d7}, [%4]!      \n"
            "bne        0b                  \n"
            : "=r"(nn),     // %0
              "=r"(src0),   // %1
              "=r"(src1),   // %2
              "=r"(dst0),   // %3
              "=r"(dst1)    // %4
            : "0"(nn),
              "1"(src0),
              "2"(src1),
              "3"(dst0),
              "4"(dst1)
            : "cc", "memory", "q0", "q1", "q2", "q3"
        );
        }
#endif // __aarch64__
#else
        int remain = size;
#endif // __ARM_NEON

        for (; remain>0; remain--)
        {
            *dst0++ = *src0++;
            *dst1++ = *src1++;
        }

        src0 += size;
        src1 += size;
        dst0 -= size*3;
        dst1 -= size*3;
    }

    for (; y < srch; y++)
    {
#if __ARM_NEON
        int nn = size >> 5;
        int remain = size - (nn << 5);
#if __aarch64__
        for (; nn>0; nn--)
        {
            uint8x16_t _src = vld1q_u8(src0);
            uint8x16_t _src2 = vld1q_u8(src0 + 16);
            vst1q_u8(dst0, _src);
            vst1q_u8(dst0 + 16, _src2);

            src0 += 32;
            dst0 += 32;
        }
#else
        if (nn > 0)
        {
        asm volatile(
            "0:                             \n"
            "pld        [%1, #256]          \n"
            "vld1.u8    {d0-d3}, [%1]!      \n"
            "subs       %0, #1              \n"
            "vst1.u8    {d0-d3}, [%2]!      \n"
            "bne        0b                  \n"
            : "=r"(nn),     // %0
              "=r"(src0),   // %1
              "=r"(dst0)    // %2
            : "0"(nn),
              "1"(src0),
              "2"(dst0)
            : "cc", "memory", "q0", "q1"
        );
        }
#endif // __aarch64__
#else
        int remain = size;
#endif // __ARM_NEON

        for (; remain>0; remain--)
        {
            *dst0++ = *src0++;
        }

        dst0 -= size*2;
    }
}

void rotate_4_c3(const unsigned char* src, int srcw, int srch, unsigned char* dst)
{
    int w = srcw;
    int h = srch;

    // point to the last dst pixel row
    unsigned char* dstend = dst + w * (h-1) * 3;

    int size = srcw * 3;

    const unsigned char* src0 = src;
    const unsigned char* src1 = src + size;
    unsigned char* dst0 = dstend;
    unsigned char* dst1 = dstend - size;

    int y = 0;
    for (; y+1 < srch; y+=2)
    {
#if __ARM_NEON
        int nn = size >> 5;
        int remain = size - (nn << 5);
#if __aarch64__
        for (; nn>0; nn--)
        {
            uint8x16_t _src0 = vld1q_u8(src0);
            uint8x16_t _src0n = vld1q_u8(src0 + 16);
            vst1q_u8(dst0, _src0);
            vst1q_u8(dst0 + 16, _src0n);

            uint8x16_t _src1 = vld1q_u8(src1);
            uint8x16_t _src1n = vld1q_u8(src1 + 16);
            vst1q_u8(dst1, _src1);
            vst1q_u8(dst1 + 16, _src1n);

            src0 += 32;
            src1 += 32;
            dst0 += 32;
            dst1 += 32;
        }
#else
        if (nn > 0)
        {
        asm volatile(
            "0:                             \n"
            "pld        [%1, #256]          \n"
            "vld1.u8    {d0-d3}, [%1]!      \n"
            "pld        [%2, #256]          \n"
            "vld1.u8    {d4-d7}, [%2]!      \n"
            "subs       %0, #1              \n"
            "vst1.u8    {d0-d3}, [%3]!      \n"
            "vst1.u8    {d4-d7}, [%4]!      \n"
            "bne        0b                  \n"
            : "=r"(nn),     // %0
              "=r"(src0),   // %1
              "=r"(src1),   // %2
              "=r"(dst0),   // %3
              "=r"(dst1)    // %4
            : "0"(nn),
              "1"(src0),
              "2"(src1),
              "3"(dst0),
              "4"(dst1)
            : "cc", "memory", "q0", "q1", "q2", "q3"
        );
        }
#endif // __aarch64__
#else
        int remain = size;
#endif // __ARM_NEON

        for (; remain>0; remain--)
        {
            *dst0++ = *src0++;
            *dst1++ = *src1++;
        }

        src0 += size;
        src1 += size;
        dst0 -= size*3;
        dst1 -= size*3;
    }

    for (; y < srch; y++)
    {
#if __ARM_NEON
        int nn = size >> 5;
        int remain = size - (nn << 5);
#if __aarch64__
        for (; nn>0; nn--)
        {
            uint8x16_t _src = vld1q_u8(src0);
            uint8x16_t _src2 = vld1q_u8(src0 + 16);
            vst1q_u8(dst0, _src);
            vst1q_u8(dst0 + 16, _src2);

            src0 += 32;
            dst0 += 32;
        }
#else
        if (nn > 0)
        {
        asm volatile(
            "0:                             \n"
            "pld        [%1, #256]          \n"
            "vld1.u8    {d0-d3}, [%1]!      \n"
            "subs       %0, #1              \n"
            "vst1.u8    {d0-d3}, [%2]!      \n"
            "bne        0b                  \n"
            : "=r"(nn),     // %0
              "=r"(src0),   // %1
              "=r"(dst0)    // %2
            : "0"(nn),
              "1"(src0),
              "2"(dst0)
            : "cc", "memory", "q0", "q1"
        );
        }
#endif // __aarch64__
#else
        int remain = size;
#endif // __ARM_NEON

        for (; remain>0; remain--)
        {
            *dst0++ = *src0++;
        }

        dst0 -= size*2;
    }
}

void rotate_4_c4(const unsigned char* src, int srcw, int srch, unsigned char* dst)
{
    int w = srcw;
    int h = srch;

    // point to the last dst pixel row
    unsigned char* dstend = dst + w * (h-1) * 4;

    int size = srcw * 4;

    const unsigned char* src0 = src;
    const unsigned char* src1 = src + size;
    unsigned char* dst0 = dstend;
    unsigned char* dst1 = dstend - size;

    int y = 0;
    for (; y+1 < srch; y+=2)
    {
#if __ARM_NEON
        int nn = size >> 5;
        int remain = size - (nn << 5);
#if __aarch64__
        for (; nn>0; nn--)
        {
            uint8x16_t _src0 = vld1q_u8(src0);
            uint8x16_t _src0n = vld1q_u8(src0 + 16);
            vst1q_u8(dst0, _src0);
            vst1q_u8(dst0 + 16, _src0n);

            uint8x16_t _src1 = vld1q_u8(src1);
            uint8x16_t _src1n = vld1q_u8(src1 + 16);
            vst1q_u8(dst1, _src1);
            vst1q_u8(dst1 + 16, _src1n);

            src0 += 32;
            src1 += 32;
            dst0 += 32;
            dst1 += 32;
        }
#else
        if (nn > 0)
        {
        asm volatile(
            "0:                             \n"
            "pld        [%1, #256]          \n"
            "vld1.u8    {d0-d3}, [%1]!      \n"
            "pld        [%2, #256]          \n"
            "vld1.u8    {d4-d7}, [%2]!      \n"
            "subs       %0, #1              \n"
            "vst1.u8    {d0-d3}, [%3]!      \n"
            "vst1.u8    {d4-d7}, [%4]!      \n"
            "bne        0b                  \n"
            : "=r"(nn),     // %0
              "=r"(src0),   // %1
              "=r"(src1),   // %2
              "=r"(dst0),   // %3
              "=r"(dst1)    // %4
            : "0"(nn),
              "1"(src0),
              "2"(src1),
              "3"(dst0),
              "4"(dst1)
            : "cc", "memory", "q0", "q1", "q2", "q3"
        );
        }
#endif // __aarch64__
#else
        int remain = size;
#endif // __ARM_NEON

        for (; remain>0; remain--)
        {
            *dst0++ = *src0++;
            *dst1++ = *src1++;
        }

        src0 += size;
        src1 += size;
        dst0 -= size*3;
        dst1 -= size*3;
    }

    for (; y < srch; y++)
    {
#if __ARM_NEON
        int nn = size >> 5;
        int remain = size - (nn << 5);
#if __aarch64__
        for (; nn>0; nn--)
        {
            uint8x16_t _src = vld1q_u8(src0);
            uint8x16_t _src2 = vld1q_u8(src0 + 16);
            vst1q_u8(dst0, _src);
            vst1q_u8(dst0 + 16, _src2);

            src0 += 32;
            dst0 += 32;
        }
#else
        if (nn > 0)
        {
        asm volatile(
            "0:                             \n"
            "pld        [%1, #256]          \n"
            "vld1.u8    {d0-d3}, [%1]!      \n"
            "subs       %0, #1              \n"
            "vst1.u8    {d0-d3}, [%2]!      \n"
            "bne        0b                  \n"
            : "=r"(nn),     // %0
              "=r"(src0),   // %1
              "=r"(dst0)    // %2
            : "0"(nn),
              "1"(src0),
              "2"(dst0)
            : "cc", "memory", "q0", "q1"
        );
        }
#endif // __aarch64__
#else
        int remain = size;
#endif // __ARM_NEON

        for (; remain>0; remain--)
        {
            *dst0++ = *src0++;
        }

        dst0 -= size*2;
    }
}
