int main()
{
  static_assert(false, "fail one"); // expected-error {{"fail one"}}
}
