# Table of Contents

1. [Contributing to CUB](#contributing-to-cub)
1. [CMake Options](#cmake-options)
1. [Development Model](#development-model)

# Contributing to CUB

CUB uses Github to manage all open-source development, including bug tracking,
pull requests, and design discussions. CUB is tightly coupled to the Thrust
project, and a compatible version of Thrust is required when working on the
development version of CUB.

To setup a CUB development branch, it is recommended to recursively clone the
Thrust repository and use the CUB submodule at `dependencies/cub` to stage
changes. CUB's tests and examples can be built by configuring Thrust with the
CMake option `THRUST_INCLUDE_CUB_CMAKE=ON`.

This process is described in more detail in Thrust's
[CONTRIBUTING.md](https://nvidia.github.io/cccl/thrust/contributing.html).

The CMake options in the following section may be used to customize CUB's build
process. Note that some of these are controlled by Thrust for compatibility and
may not have an effect when building CUB through the Thrust build system. This
is pointed out in the documentation below where applicable.

# CMake Options

A CUB build is configured using CMake options. These may be passed to CMake
using

```
cmake -D<option_name>=<value> [Thrust or CUB project source root]
```

or configured interactively with the `ccmake` or `cmake-gui` interfaces.

The configuration options for CUB are:

- `CMAKE_BUILD_TYPE={Release, Debug, RelWithDebInfo, MinSizeRel}`
  - Standard CMake build option. Default: `RelWithDebInfo`
- `CUB_ENABLE_INSTALL_RULES={ON, OFF}`
  - If true, installation rules will be generated for CUB. Default is `ON` when
    building CUB alone, and `OFF` when CUB is a subproject added via CMake's
    `add_subdirectory`.
- `CUB_ENABLE_HEADER_TESTING={ON, OFF}`
  - Whether to test compile public headers. Default is `ON`.
- `CUB_ENABLE_TESTING={ON, OFF}`
  - Whether to build unit tests. Default is `ON`.
- `CUB_ENABLE_EXAMPLES={ON, OFF}`
  - Whether to build examples. Default is `ON`.
- `CUB_ENABLE_DIALECT_CPPXX={ON, OFF}`
  - Setting this has no effect when building CUB as a component of Thrust.
    See Thrust's dialect options, which CUB will inherit.
  - Toggle whether a specific C++ dialect will be targeted.
  - Multiple dialects may be targeted in a single build.
  - Possible values of `XX` are `{11, 14, 17}`.
  - By default, only C++14 is enabled.
- `CUB_ENABLE_RDC_TESTS={ON, OFF}`
  - Enable tests that require separable compilation.
  - Default is `ON`.
- `CUB_FORCE_RDC={ON, OFF}`
  - Enable separable compilation on all targets that are agnostic of RDC.
  - Targets that explicitly require RDC to be enabled or disabled will ignore this setting.
  - Default is `OFF`.
