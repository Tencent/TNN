// %PARAM% TEST_ERR err 0:1

int main()
{
#if TEST_ERR == 0
  static_assert(false, "fail one"); // expected-error-0 {{"fail one"}}
#elif TEST_ERR == 1
  static_assert(false, "fail two"); // expected-error-1 {{"fail two"}}
#endif
}
