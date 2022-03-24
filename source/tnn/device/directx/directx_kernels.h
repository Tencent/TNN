#include <map> 
#include <string> 

#include "tnn/core/macro.h"

namespace TNN_NS {
namespace directx {

std::map<std::string, const unsigned char *> & get_kernel_map();

std::map<std::string, size_t> & get_kernel_size_map();

} // directX
} // TNN_NS

