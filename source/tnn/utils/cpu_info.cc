// Tencent is pleased to support the open source community by making TNN available.
//
// Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include "cpu_info.h"

#if defined(__ANDROID__) || defined(__linux__)

#include <alloca.h>
#include <fcntl.h>
#include <string.h>
#include <unistd.h>

#ifdef __ANDROID__
#include <sys/system_properties.h>
#endif

#define BUFFER_SIZE 1024

typedef bool (*cpuinfo_line_callback)(const char *, const char *, void *, uint64_t);

/* Only contain hardware and midr now. */
struct proc_cpuinfo_parser_state {
    char *hardware;
    struct cpuinfo_arm_linux_processor *processor;
};

inline static uint32_t midr_set_part(uint32_t midr, uint32_t part) {
    return (midr & ~CPUINFO_ARM_MIDR_PART_MASK) | ((part << CPUINFO_ARM_MIDR_PART_OFFSET) & CPUINFO_ARM_MIDR_PART_MASK);
}

inline static uint32_t midr_set_implementer(uint32_t midr, uint32_t implementer) {
    return (midr & ~CPUINFO_ARM_MIDR_IMPLEMENTER_MASK) |
           ((implementer << CPUINFO_ARM_MIDR_IMPLEMENTER_OFFSET) & CPUINFO_ARM_MIDR_IMPLEMENTER_MASK);
}

static void parse_cpu_part(const char *cpu_part_start, const char *cpu_part_end,
                           struct cpuinfo_arm_linux_processor *processor) {
    const size_t cpu_part_length = (size_t)(cpu_part_end - cpu_part_start);

    /*
     * CPU part should contain hex prefix (0x) and one to three hex digits.
     * I have never seen less than three digits as a value of this field,
     * but I don't think it is impossible to see such values in future.
     * Value can not contain more than three hex digits since
     * Main ID Register (MIDR) assigns only a 12-bit value for CPU part.
     */
    if (cpu_part_length < 3 || cpu_part_length > 5) {
        return;
    }

    /* Verify the presence of hex prefix */
    if (cpu_part_start[0] != '0' || cpu_part_start[1] != 'x') {
        return;
    }

    /* Verify that characters after hex prefix are hexadecimal digits and decode them */
    uint32_t cpu_part = 0;
    for (const char *digit_ptr = cpu_part_start + 2; digit_ptr != cpu_part_end; digit_ptr++) {
        const char digit_char = *digit_ptr;
        uint32_t digit;
        if (digit_char >= '0' && digit_char <= '9') {
            digit = digit_char - '0';
        } else if ((uint32_t)(digit_char - 'A') < 6) {
            digit = 10 + (digit_char - 'A');
        } else if ((uint32_t)(digit_char - 'a') < 6) {
            digit = 10 + (digit_char - 'a');
        } else {
            return;
        }
        cpu_part = cpu_part * 16 + digit;
    }

    processor->midr = midr_set_part(processor->midr, cpu_part);
}

static void parse_cpu_implementer(const char *cpu_implementer_start, const char *cpu_implementer_end,
                                  struct cpuinfo_arm_linux_processor *processor) {
    const size_t cpu_implementer_length = cpu_implementer_end - cpu_implementer_start;

    /*
     * Value should contain hex prefix (0x) and one or two hex digits.
     * I have never seen single hex digit as a value of this field,
     * but I don't think it is impossible in future.
     * Value can not contain more than two hex digits since
     * Main ID Register (MIDR) assigns only an 8-bit value for CPU implementer.
     */
    switch (cpu_implementer_length) {
        case 3:
        case 4:
            break;
        default:
            return;
    }

    /* Verify the presence of hex prefix */
    if (cpu_implementer_start[0] != '0' || cpu_implementer_start[1] != 'x') {
        return;
    }

    /* Verify that characters after hex prefix are hexadecimal digits and decode them */
    uint32_t cpu_implementer = 0;
    for (const char *digit_ptr = cpu_implementer_start + 2; digit_ptr != cpu_implementer_end; digit_ptr++) {
        const char digit_char = *digit_ptr;
        uint32_t digit;
        if (digit_char >= '0' && digit_char <= '9') {
            digit = digit_char - '0';
        } else if ((uint32_t)(digit_char - 'A') < 6) {
            digit = 10 + (digit_char - 'A');
        } else if ((uint32_t)(digit_char - 'a') < 6) {
            digit = 10 + (digit_char - 'a');
        } else {
            return;
        }
        cpu_implementer = cpu_implementer * 16 + digit;
    }

    processor->midr = midr_set_implementer(processor->midr, cpu_implementer);
}

/* Decode a single line of /proc/cpuinfo information. */
/* Only decode Hardware now. */
static bool parse_line(const char *line_start, const char *line_end, struct proc_cpuinfo_parser_state *state,
                       uint64_t line_number) {
    /* Empty line. Skip. */
    if (line_start == line_end) {
        return true;
    }

    /* Search for ':' on the line. */
    const char *separator = line_start;
    for (; separator != line_end; separator++) {
        if (*separator == ':') {
            break;
        }
    }
    /* Skip line if no ':' separator was found. */
    if (separator == line_end) {
        return true;
    }

    /* Skip trailing spaces in key part. */
    const char *key_end = separator;
    for (; key_end != line_start; key_end--) {
        if (key_end[-1] != ' ' && key_end[-1] != '\t') {
            break;
        }
    }
    /* Skip line if key contains nothing but spaces. */
    if (key_end == line_start) {
        return true;
    }

    /* Skip leading spaces in value part. */
    const char *value_start = separator + 1;
    for (; value_start != line_end; value_start++) {
        if (*value_start != ' ') {
            break;
        }
    }
    /* Value part contains nothing but spaces. Skip line. */
    if (value_start == line_end) {
        return true;
    }

    /* Skip trailing spaces in value part (if any) */
    const char *value_end = line_end;
    for (; value_end != value_start; value_end--) {
        if (value_end[-1] != ' ') {
            break;
        }
    }

    const size_t key_length = key_end - line_start;
    switch (key_length) {
        case 8:
            if (memcmp(line_start, "CPU part", key_length) == 0) {
                parse_cpu_part(value_start, value_end, state->processor);
            } else if (memcmp(line_start, "Hardware", key_length) == 0) {
                size_t value_length = value_end - value_start;
                if (value_length > CPUINFO_HARDWARE_VALUE_MAX) {
                    value_length = CPUINFO_HARDWARE_VALUE_MAX;
                } else {
                    state->hardware[value_length] = '\0';
                }
                memcpy(state->hardware, value_start, value_length);
            }
            break;
        case 15:
            if (memcmp(line_start, "CPU implementer", key_length) == 0) {
                parse_cpu_implementer(value_start, value_end, state->processor);
            } else if (memcmp(line_start, "CPU implementor", key_length) == 0) {
                parse_cpu_implementer(value_start, value_end, state->processor);
            }
        default:
            break;
    }
    return true;
}

#define CLEAN_UP                                                                                                       \
    if (file != -1) {                                                                                                  \
        close(file);                                                                                                   \
        file = -1;                                                                                                     \
    }                                                                                                                  \
    return status;

bool cpuinfo_linux_parse_multiline_file(const char *filename, size_t buffer_size, cpuinfo_line_callback callback,
                                        void *context) {
    int file     = -1;
    bool status  = false;
    char *buffer = (char *)alloca(buffer_size);

    file = open(filename, O_RDONLY);
    if (file == -1) {
        CLEAN_UP;
    }

    /* Only used for error reporting */
    size_t position        = 0;
    uint64_t line_number   = 1;
    const char *buffer_end = &buffer[buffer_size];
    char *data_start       = buffer;
    ssize_t bytes_read;
    do {
        bytes_read = read(file, data_start, (size_t)(buffer_end - data_start));
        if (bytes_read < 0) {
            CLEAN_UP;
        }

        position += (size_t)bytes_read;
        const char *data_end   = data_start + (size_t)bytes_read;
        const char *line_start = buffer;

        if (bytes_read == 0) {
            /* No more data in the file: process the remaining text in the buffer as a single entry */
            const char *line_end = data_end;
            if (!callback(line_start, line_end, context, line_number)) {
                CLEAN_UP;
            }
        } else {
            const char *line_end;
            do {
                /* Find the end of the entry, as indicated by newline character ('\n') */
                for (line_end = line_start; line_end != data_end; line_end++) {
                    if (*line_end == '\n') {
                        break;
                    }
                }

                /*
                 * If we located separator at the end of the entry, parse it.
                 * Otherwise, there may be more data at the end; read the file once again.
                 */
                if (line_end != data_end) {
                    if (!callback(line_start, line_end, context, line_number++)) {
                        CLEAN_UP;
                    }
                    line_start = line_end + 1;
                }
            } while (line_end != data_end);

            /* Move remaining partial line data at the end to the beginning of the buffer */
            const size_t line_length = (size_t)(line_end - line_start);
            memmove(buffer, line_start, line_length);
            data_start = &buffer[line_length];
        }
    } while (bytes_read != 0);

    /* Commit */
    status = true;

    CLEAN_UP;
}

#undef CLEAN_UP

/* Only get hardware and midr now*/
bool cpuinfo_arm_linux_parse_proc_cpuinfo(char *hardware, struct cpuinfo_arm_linux_processor *processor) {
    struct proc_cpuinfo_parser_state state = {
        .hardware  = hardware,
        .processor = processor,
    };
    return cpuinfo_linux_parse_multiline_file("/proc/cpuinfo", BUFFER_SIZE, (cpuinfo_line_callback)parse_line, &state);
}

#ifdef __ANDROID__
void cpuinfo_arm_android_parse_properties(struct cpuinfo_android_properties *properties) {
    __system_property_get("ro.product.board", properties->ro_product_board);
    __system_property_get("ro.board.platform", properties->ro_board_platform);
    __system_property_get("ro.mediatek.platform", properties->ro_mediatek_platform);
    __system_property_get("ro.arch", properties->ro_arch);
    __system_property_get("ro.chipname", properties->ro_chipname);
    __system_property_get("ro.hardware.chipname", properties->ro_hardware_chipname);
}
#endif

enum cpuinfo_android_chipset_property {
    cpuinfo_android_chipset_property_proc_cpuinfo_hardware = 0,
    cpuinfo_android_chipset_property_ro_product_board,
    cpuinfo_android_chipset_property_ro_board_platform,
    cpuinfo_android_chipset_property_ro_mediatek_platform,
    cpuinfo_android_chipset_property_ro_arch,
    cpuinfo_android_chipset_property_ro_chipname,
    cpuinfo_android_chipset_property_ro_hardware_chipname,
    cpuinfo_android_chipset_property_max,
};

static inline uint32_t load_u32le(const void *ptr) {
    return *((const uint32_t *)ptr);
}

static inline uint16_t load_u16le(const void *ptr) {
    return *((const uint16_t *)ptr);
}

/**
 * Tries to match /Samsung Exynos\d{4}$/ signature (case-insensitive) for Samsung Exynos chipsets.
 * If match successful, extracts model information into \p chipset argument.
 *
 * @param start - start of the /proc/cpuinfo Hardware string to match.
 * @param end - end of the /proc/cpuinfo Hardware string to match.
 * @param[out] chipset - location where chipset information will be stored upon a successful match.
 *
 * @returns true if signature matched, false otherwise.
 */
static bool match_samsung_exynos(const char *start, const char *end, struct cpuinfo_arm_chipset *chipset) {
    /*
     * Expect at 18-19 symbols:
     * - "Samsung" (7 symbols) + space + "Exynos" (6 symbols) + optional space 4-digit model number
     */
    const size_t length = end - start;
    switch (length) {
        case 18:
        case 19:
            break;
        default:
            return false;
    }

    /*
     * Check that the string starts with "samsung exynos", case-insensitive.
     * Blocks of 4 characters are loaded and compared as little-endian 32-bit word.
     * Case-insensitive characters are binary ORed with 0x20 to convert them to lowercase.
     */
    const uint32_t expected_sams = uint32_t(0x20202000) | load_u32le(start);
    if (expected_sams != uint32_t(0x736D6153) /* "smaS" = reverse("Sams") */) {
        return false;
    }
    const uint32_t expected_ung = uint32_t(0x00202020) | load_u32le(start + 4);
    if (expected_ung != uint32_t(0x20676E75) /* " ung" = reverse("ung ") */) {
        return false;
    }
    const uint32_t expected_exyn = uint32_t(0x20202000) | load_u32le(start + 8);
    if (expected_exyn != uint32_t(0x6E797845) /* "nyxE" = reverse("Exyn") */) {
        return false;
    }
    const uint16_t expected_os = uint16_t(0x2020) | load_u16le(start + 12);
    if (expected_os != uint16_t(0x736F) /* "so" = reverse("os") */) {
        return false;
    }

    const char *pos = start + 14;

    /* There can be a space ' ' following the "Exynos" string */
    if (*pos == ' ') {
        pos++;

        /* If optional space if present, we expect exactly 19 characters */
        if (length != 19) {
            return false;
        }
    }

    /* Validate and parse 4-digit model number */
    uint32_t model = 0;
    for (uint32_t i = 0; i < 4; i++) {
        const uint32_t digit = (uint32_t)(uint8_t)(*pos++) - '0';
        if (digit >= 10) {
            /* Not really a digit */
            return false;
        }
        model = model * 10 + digit;
    }

    /* Return parsed chipset */
    *chipset = (struct cpuinfo_arm_chipset){
        .vendor = cpuinfo_arm_chipset_vendor_samsung,
        .series = cpuinfo_arm_chipset_series_samsung_exynos,
        .model  = model,
    };
    return true;
}

/**
 * Tries to match /exynos\d{4}$/ signature for Samsung Exynos chipsets.
 * If match successful, extracts model information into \p chipset argument.
 *
 * @param start - start of the platform identifier (ro.board.platform or ro.chipname) to match.
 * @param end - end of the platform identifier (ro.board.platform or ro.chipname) to match.
 * @param[out] chipset - location where chipset information will be stored upon a successful match.
 *
 * @returns true if signature matched, false otherwise.
 */
static bool match_exynos(const char *start, const char *end, struct cpuinfo_arm_chipset *chipset) {
    /* Expect exactly 10 symbols: "exynos" (6 symbols) + 4-digit model number */
    if (start + 10 != end) {
        return false;
    }

    /* Load first 4 bytes as little endian 32-bit word */
    const uint32_t expected_exyn = load_u32le(start);
    if (expected_exyn != uint32_t(0x6E797865) /* "nyxe" = reverse("exyn") */) {
        return false;
    }

    /* Load next 2 bytes as little endian 16-bit word */
    const uint16_t expected_os = load_u16le(start + 4);
    if (expected_os != uint16_t(0x736F) /* "so" = reverse("os") */) {
        return false;
    }

    /* Check and parse 4-digit model number */
    uint32_t model = 0;
    for (uint32_t i = 6; i < 10; i++) {
        const uint32_t digit = (uint32_t)(uint8_t)start[i] - '0';
        if (digit >= 10) {
            /* Not really a digit */
            return false;
        }
        model = model * 10 + digit;
    }

    /* Return parsed chipset. */
    *chipset = (struct cpuinfo_arm_chipset){
        .vendor = cpuinfo_arm_chipset_vendor_samsung,
        .series = cpuinfo_arm_chipset_series_samsung_exynos,
        .model  = model,
    };
    return true;
}

/**
 * Tries to match /universal\d{4}$/ signature for Samsung Exynos chipsets.
 * If match successful, extracts model information into \p chipset argument.
 *
 * @param start - start of the platform identifier (/proc/cpuinfo Hardware string, ro.product.board or ro.chipname)
 *                to match.
 * @param end - end of the platform identifier (/proc/cpuinfo Hardware string, ro.product.board or ro.chipname)
 *              to match.
 * @param[out] chipset - location where chipset information will be stored upon a successful match.
 *
 * @returns true if signature matched, false otherwise.
 */
static bool match_universal(const char *start, const char *end, struct cpuinfo_arm_chipset *chipset) {
    /* Expect exactly 13 symbols: "universal" (9 symbols) + 4-digit model number */
    if (start + 13 != end) {
        return false;
    }

    /*
     * Check that the string starts with "universal".
     * Blocks of 4 characters are loaded and compared as little-endian 32-bit word.
     * Case-insensitive characters are binary ORed with 0x20 to convert them to lowercase.
     */
    const uint8_t expected_u = uint8_t(0x20) | (uint8_t)start[0];
    if (expected_u != uint8_t(0x75) /* "u" */) {
        return false;
    }
    const uint32_t expected_nive = uint32_t(0x20202020) | load_u32le(start + 1);
    if (expected_nive != uint32_t(0x6576696E) /* "evin" = reverse("nive") */) {
        return false;
    }
    const uint32_t expected_ersa = uint32_t(0x20202020) | load_u32le(start + 5);
    if (expected_ersa != uint32_t(0x6C617372) /* "lasr" = reverse("rsal") */) {
        return false;
    }

    /* Validate and parse 4-digit model number */
    uint32_t model = 0;
    for (uint32_t i = 9; i < 13; i++) {
        const uint32_t digit = (uint32_t)(uint8_t)start[i] - '0';
        if (digit >= 10) {
            /* Not really a digit */
            return false;
        }
        model = model * 10 + digit;
    }

    /* Return parsed chipset. */
    *chipset = (struct cpuinfo_arm_chipset){
        .vendor = cpuinfo_arm_chipset_vendor_samsung,
        .series = cpuinfo_arm_chipset_series_samsung_exynos,
        .model  = model,
    };
    return true;
}

struct cpuinfo_arm_chipset cpuinfo_arm_linux_decode_chipset_from_proc_cpuinfo_hardware(const char *hardware) {
    struct cpuinfo_arm_chipset chipset;
    const size_t hardware_length = strnlen(hardware, CPUINFO_HARDWARE_VALUE_MAX);
    const char *hardware_end     = hardware + hardware_length;

    /* Check Samsung Exynos signature */
    if (match_samsung_exynos(hardware, hardware_end, &chipset)) {
        return chipset;
    }

    /* Check universalXXXX (Samsung Exynos) signature */
    if (match_universal(hardware, hardware_end, &chipset)) {
        return chipset;
    }

    return (struct cpuinfo_arm_chipset){
        .vendor = cpuinfo_arm_chipset_vendor_unknown,
        .series = cpuinfo_arm_chipset_series_unknown,
    };
}

struct cpuinfo_arm_chipset cpuinfo_arm_android_decode_chipset_from_ro_product_board(const char *ro_product_board) {
    struct cpuinfo_arm_chipset chipset;
    const char *board         = ro_product_board;
    const size_t board_length = strnlen(ro_product_board, CPUINFO_BUILD_PROP_VALUE_MAX);
    const char *board_end     = ro_product_board + board_length;

    /* Check universaXXXX (Samsung Exynos) signature */
    if (match_universal(board, board_end, &chipset)) {
        return chipset;
    }

    return (struct cpuinfo_arm_chipset){
        .vendor = cpuinfo_arm_chipset_vendor_unknown,
        .series = cpuinfo_arm_chipset_series_unknown,
    };
}

struct cpuinfo_arm_chipset cpuinfo_arm_android_decode_chipset_from_ro_board_platform(const char *platform) {
    struct cpuinfo_arm_chipset chipset;
    const size_t platform_length = strnlen(platform, CPUINFO_BUILD_PROP_VALUE_MAX);
    const char *platform_end     = platform + platform_length;

    /* Check exynosXXXX (Samsung Exynos) signature */
    if (match_exynos(platform, platform_end, &chipset)) {
        return chipset;
    }

    /* None of the ro.board.platform signatures matched, indicate unknown chipset */
    return (struct cpuinfo_arm_chipset){
        .vendor = cpuinfo_arm_chipset_vendor_unknown,
        .series = cpuinfo_arm_chipset_series_unknown,
    };
}

struct cpuinfo_arm_chipset cpuinfo_arm_android_decode_chipset_from_ro_mediatek_platform(const char *platform) {
    struct cpuinfo_arm_chipset chipset;
    const char *platform_end = platform + strnlen(platform, CPUINFO_BUILD_PROP_VALUE_MAX);

    return (struct cpuinfo_arm_chipset){
        .vendor = cpuinfo_arm_chipset_vendor_unknown,
        .series = cpuinfo_arm_chipset_series_unknown,
    };
}

struct cpuinfo_arm_chipset cpuinfo_arm_android_decode_chipset_from_ro_arch(const char *arch) {
    struct cpuinfo_arm_chipset chipset;
    const char *arch_end = arch + strnlen(arch, CPUINFO_BUILD_PROP_VALUE_MAX);

    /* Check Samsung exynosXXXX signature */
    if (match_exynos(arch, arch_end, &chipset)) {
        return chipset;
    }

    return (struct cpuinfo_arm_chipset){
        .vendor = cpuinfo_arm_chipset_vendor_unknown,
        .series = cpuinfo_arm_chipset_series_unknown,
    };
}

struct cpuinfo_arm_chipset cpuinfo_arm_android_decode_chipset_from_ro_chipname(const char *chipname) {
    struct cpuinfo_arm_chipset chipset;
    const size_t chipname_length = strnlen(chipname, CPUINFO_BUILD_PROP_VALUE_MAX);
    const char *chipname_end     = chipname + chipname_length;

    /* Check exynosXXXX (Samsung Exynos) signature */
    if (match_exynos(chipname, chipname_end, &chipset)) {
        return chipset;
    }

    /* Check universalXXXX (Samsung Exynos) signature */
    if (match_universal(chipname, chipname_end, &chipset)) {
        return chipset;
    }

    return (struct cpuinfo_arm_chipset){
        .vendor = cpuinfo_arm_chipset_vendor_unknown,
        .series = cpuinfo_arm_chipset_series_unknown,
    };
}

/* Only detect Samsung Exynos chipsets now*/
struct cpuinfo_arm_chipset cpuinfo_arm_android_decode_chipset(const struct cpuinfo_android_properties *properties) {
    struct cpuinfo_arm_chipset chipset = {
        .vendor = cpuinfo_arm_chipset_vendor_unknown,
        .series = cpuinfo_arm_chipset_series_unknown,
    };

    struct cpuinfo_arm_chipset chipsets[cpuinfo_android_chipset_property_max] = {
        [cpuinfo_android_chipset_property_proc_cpuinfo_hardware] =
            cpuinfo_arm_linux_decode_chipset_from_proc_cpuinfo_hardware(properties->proc_cpuinfo_hardware),
        [cpuinfo_android_chipset_property_ro_product_board] =
            cpuinfo_arm_android_decode_chipset_from_ro_product_board(properties->ro_product_board),
        [cpuinfo_android_chipset_property_ro_board_platform] =
            cpuinfo_arm_android_decode_chipset_from_ro_board_platform(properties->ro_board_platform),
        [cpuinfo_android_chipset_property_ro_mediatek_platform] =
            cpuinfo_arm_android_decode_chipset_from_ro_mediatek_platform(properties->ro_mediatek_platform),
        [cpuinfo_android_chipset_property_ro_arch] =
            cpuinfo_arm_android_decode_chipset_from_ro_arch(properties->ro_arch),
        [cpuinfo_android_chipset_property_ro_chipname] =
            cpuinfo_arm_android_decode_chipset_from_ro_chipname(properties->ro_chipname),
        [cpuinfo_android_chipset_property_ro_hardware_chipname] =
            cpuinfo_arm_android_decode_chipset_from_ro_chipname(properties->ro_hardware_chipname),
    };

    enum cpuinfo_arm_chipset_vendor vendor = cpuinfo_arm_chipset_vendor_unknown;
    for (size_t i = 0; i < cpuinfo_android_chipset_property_max; i++) {
        const enum cpuinfo_arm_chipset_vendor decoded_vendor = chipsets[i].vendor;
        if (decoded_vendor != cpuinfo_arm_chipset_vendor_unknown) {
            if (vendor == cpuinfo_arm_chipset_vendor_unknown) {
                vendor = decoded_vendor;
            } else if (vendor != decoded_vendor) {
                /* Parsing different system properties produces different chipset vendors. This situation is rare. */
                return chipset;
            }
        }
    }
    if (vendor == cpuinfo_arm_chipset_vendor_unknown) {
        return chipset;
    }

    for (size_t i = 0; i < cpuinfo_android_chipset_property_max; i++) {
        if (chipsets[i].series != cpuinfo_arm_chipset_series_unknown) {
            chipset = chipsets[i];
            break;
        }
    }

    return chipset;
}

#endif  // __ANDROID__ || __linux__
