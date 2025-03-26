#!/usr/bin/env python

import hpccm

hpccm.config.set_container_format('docker')

Stage0 += hpccm.primitives.baseimage(image='nvidia/cuda:12.2.0-devel-ubuntu22.04')
Stage0 += hpccm.building_blocks.apt_get(ospackages=['git', 'tmux', 'gcc', 'g++', 'vim', 'python3', 'python-is-python3', 'ninja-build'])
# Stage0 += hpccm.building_blocks.llvm(version='15', extra_tools=True, toolset=True)
Stage0 += hpccm.building_blocks.cmake(eula=True, version='3.26.3')
# Stage0 += hpccm.building_blocks.nsight_compute(eula=True, version='2023.1.1')
Stage0 += hpccm.building_blocks.pip(packages=['fpzip', 'numpy', 'pandas', 'pynvml'], pip='pip3')
Stage0 += hpccm.primitives.environment(variables={'CUDA_MODULE_LOADING': 'EAGER'})
