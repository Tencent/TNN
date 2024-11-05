# Test Parametrization

Some of CUB's tests are very slow to build and are capable of exhausting RAM
during compilation/linking. To avoid such issues, large tests are split into
multiple executables to take advantage of parallel computation and reduce memory
usage.

CUB facilitates this by checking for special `%PARAM%` comments in each test's
source code, and then uses this information to generate multiple executables
with different configurations.

## Using `%PARAM%`

The `%PARAM%` hint provides an automated method of generating multiple test
executables from a single source file. To use it, add one or more special
comments to the test source file:

```cpp
// %PARAM% [definition] [label] [values]
```

CMake will parse the source file and extract these comments, using them to
generate multiple test executables for the full cartesian product of values.

- `definition` will be used as a preprocessor definition name. By convention,
  these begin with `TEST_`.
- `label` is a short, human-readable label that will be used in the test
  executable's name to identify the test variant.
- `values` is a colon-separated list of values used during test generation. Only
  numeric values have been tested.

## Special Labels

### Testing Different Launchers

If a `label` is `lid`, it is assumed that the parameter is used to explicitly
test variants built with different launchers. The `values` for such a
parameter must be `0:1:2`, with `0` indicating host launch and CDP disabled (RDC off),
`1` indicating device launch and CDP enabled (RDC on),
`2` indicating graph capture launch and CDP disabled (RDC off).

Tests that do not contain a variant labeled `lid` will only enable RDC if
the CMake config enables them.

## Example

For example, if `test_baz.cu` contains the following lines:

```cpp
// %PARAM% TEST_FOO foo 0:1:2
// %PARAM% TEST_LAUNCH lid 0:1
```

Six executables and CTest targets will be generated with unique definitions
(only c++17 targets shown):

| Executable Name                  | Preprocessor Definitions       | Launcher  |
|----------------------------------|--------------------------------|-----------|
| `cub.cpp17.test.baz.foo_0.lid_0` | `-DTEST_FOO=0 -DTEST_LAUNCH=0` | Host      |
| `cub.cpp17.test.baz.foo_0.lid_1` | `-DTEST_FOO=0 -DTEST_LAUNCH=1` | Device    |
| `cub.cpp17.test.baz.foo_1.lid_0` | `-DTEST_FOO=1 -DTEST_LAUNCH=0` | Host      |
| `cub.cpp17.test.baz.foo_1.lid_1` | `-DTEST_FOO=1 -DTEST_LAUNCH=1` | Device    |
| `cub.cpp17.test.baz.foo_2.lid_0` | `-DTEST_FOO=2 -DTEST_LAUNCH=0` | Host      |
| `cub.cpp17.test.baz.foo_2.lid_1` | `-DTEST_FOO=2 -DTEST_LAUNCH=1` | Device    |

## Changing `%PARAM%` Hints

Since CMake does not automatically reconfigure the build when source files are
modified, CMake will need to be rerun manually whenever the `%PARAM%` comments
change.

## Building and Running Split Tests

CMake will generate individual build and test targets for each test variant, and
also provides build "metatargets" that compile all variants of a given test.

The variants follow the usual naming convention for CUB's tests, but include a
suffix that differentiates them (e.g. `.foo_X.bar_Y` in the example above).

### Individual Test Variants

Continuing with the `test_baz.cu` example, the test variant that uses
`-DTEST_FOO=1 -DTEST_BAR=4` can be built and run alone:

```bash
# Build a single variant:
make cub.cpp17.test.baz.foo_1.bar_4

# Run a single variant
bin/cub.cpp17.test.baz.foo_1.bar_4

# Run a single variant using CTest regex:
ctest -R cub\.cpp17\.test\.baz\.foo_1\.bar_4
```

### All Variants of a Test

Using a metatarget and the proper regex, all variants of a test can be built and
executed without listing all variants explicitly:

```bash
# Build all variants using the `.all` metatarget
make cub.cpp17.test.baz.all

# Run all variants:
ctest -R cub\.cpp17\.test\.baz\.
```

## Debugging

Running CMake with `--log-level=VERBOSE` will print out extra information about
all detected test variants.

## Additional Info

Ideally, only parameters that directly influence kernel template instantiations
should be split out in this way. If changing a parameter doesn't change the
kernel template type, the same kernel will be compiled into multiple
executables. This defeats the purpose of splitting up the test since the
compiler will generate redundant code across the new split executables.

The best candidate parameters for splitting are input value types, rather than
integral parameters like BLOCK_THREADS, etc. Splitting by value type allows more
infrastructure (data generation, validation) to be reused. Splitting other
parameters can cause build times to increase since type-related infrastructure
has to be rebuilt for each test variant.
