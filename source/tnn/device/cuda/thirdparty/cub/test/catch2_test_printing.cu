#include <sstream>

#include "test_util.h"
#include <c2h/catch2_test_helper.cuh>

template <typename T>
std::string print(T val)
{
  std::stringstream ss;
  ss << val;
  return ss.str();
}

#if CUB_IS_INT128_ENABLED
TEST_CASE("Test utils can print __int128", "[test][utils]")
{
  REQUIRE(print(__int128_t{0}) == "0");
  REQUIRE(print(__int128_t{42}) == "42");
  REQUIRE(print(__int128_t{-1}) == "-1");
  REQUIRE(print(__int128_t{-42}) == "-42");
  REQUIRE(print(-1 * (__int128_t{1} << 120)) == "-1329227995784915872903807060280344576");
}

TEST_CASE("Test utils can print __uint128", "[test][utils]")
{
  REQUIRE(print(__uint128_t{0}) == "0");
  REQUIRE(print(__uint128_t{1}) == "1");
  REQUIRE(print(__uint128_t{42}) == "42");
  REQUIRE(print(__uint128_t{1} << 120) == "1329227995784915872903807060280344576");
}
#endif

TEST_CASE("Test utils can print KeyValuePair", "[test][utils]")
{
  REQUIRE(print(cub::KeyValuePair<int, int>{42, -42}) == "(42,-42)");
}
