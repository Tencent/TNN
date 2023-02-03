from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext
from distutils.version import LooseVersion
import subprocess
import platform
from shutil import copy2, rmtree, copytree
import os
import re
import sys


old_dir = os.getcwd()
os.chdir(os.path.dirname(os.path.abspath(__file__)) + "/..")
p = 'package/onnx2tnn/'


def ignore_files(dir, names):
    return [name for name in names
            if name == '__pycache__' or name.endswith('.pyc')]

class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=''):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class Env:

    @classmethod
    def _find_command(self):
        env = os.environ.copy()

        self._cmake = env.get('CMAKE', 'cmake')
        self._cpp   = env.get('CPP_COMPILER', 'g++')
        self._c     = env.get('C_COMPILER', 'gcc')

    @classmethod
    def get_update_env(self):
        return {
            'CMAKE': self._cmake,
            'CPP_COMPILER': self._cpp,
            'C_COMPILER': self._c,
        }


class CMakeBuild(build_ext):

    def run(self):

        env = Env.get_update_env()
        self._cmake = env['CMAKE']
        self._cpp   = env['CPP_COMPILER']
        self._c     = env['C_COMPILER']

        try:
            out = subprocess.check_output([self._cmake, '--version'])
        except OSError:
            raise RuntimeError("CMake must be installed to build the following extensions: " +
                               ", ".join(e.name for e in self.extensions))

        if platform.system() == "Windows":
            raise RuntimeError('Windows not supported yet')
        else:
            cmake_version = LooseVersion(
                re.search(r'version\s*([\d.]+)', out.decode())
                .group(1)
            )
            if cmake_version < '3.5.0':
                raise RuntimeError("CMake >= 3.5.0 is required")

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extdir = os.path.abspath(
            os.path.dirname(self.get_ext_fullpath(ext.name))
        )
        cmake_args = ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + extdir,
                      '-DPYTHON_EXECUTABLE=' + sys.executable,
                      '-DCMAKE_C_COMPILER=' + self._c,
                      '-DCMAKE_CXX_COMPILER=' + self._cpp]

        cfg = 'Debug' if self.debug else 'Release'
        build_args = ['--config', cfg]

        cmake_args += ['-DCMAKE_BUILD_TYPE=' + cfg]
        build_args += ['--', '-j']

        env = os.environ.copy()
        env['CXXFLAGS'] = '{} -DVERSION_INFO=\\"{}\\"'\
            .format(env.get('CXXFLAGS', ''),
                    self.distribution.get_version())

        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
        subprocess.check_call([self._cmake, ext.sourcedir] + cmake_args, cwd=self.build_temp, env=env)
        subprocess.check_call([self._cmake, '--build', '.'] + build_args, cwd=self.build_temp)


try:
    from detect_dependency import detect_dependency

    Env._find_command()
    os.environ.update(Env.get_update_env())
    detect_dependency()

    os.makedirs(p, exist_ok=True)
    copy2('onnx2tnn.py', p + '__main__.py')
    copy2('version.py', p)
    copy2('onnx_model_cheker.py', p)
    copy2('__init__.py', p)
    rmtree(p + 'onnx_optimizer', True)
    copytree('onnx_optimizer', p + 'onnx_optimizer', ignore=ignore_files)

    setup(name='onnx2tnn',
          version='0.0.1',
          description='tools to convert onnx model to tnn model',
          long_description='tools to convert onnx model to tnn model',
          author='darrenyao, dandiding, lucas, wingzygan(contributer)',
          package_dir={'': 'package'},
          packages=find_packages('package'),
          ext_modules=[CMakeExtension('onnx2tnn.onnx2tnn', '.')],
          cmdclass=dict(build_ext=CMakeBuild),
          install_requires=[
              'onnx==1.13.0',
              'onnxruntime==1.1',
              'onnx-simplifier==0.2.4',
          ])

finally:
    os.chdir(old_dir)
    rmtree('package', True)
