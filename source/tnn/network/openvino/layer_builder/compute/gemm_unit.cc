#include "gemm_unit.h"


namespace TNN
{
namespace openvino {
    GemmUnit::map_mem_t      GemmUnit::g_memory       = std::unordered_map<std::string, std::shared_ptr<memory>> ();
    GemmUnit::map_ip_primd_t GemmUnit::g_ip_prim_desc = std::unordered_map<std::string, std::shared_ptr<inner_product_forward::primitive_desc>>();
    GemmUnit::map_mm_primd_t GemmUnit::g_mm_prim_desc = std::unordered_map<std::string, std::shared_ptr<matmul::primitive_desc>>();
    GemmUnit::map_prim_t     GemmUnit::g_prim         = std::unordered_map<std::string, std::shared_ptr<primitive>>();
}
}