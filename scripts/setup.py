import os
import sys
import glob
import setuptools
from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext
from setuptools.command.develop import develop
from setuptools.command.install import install
from distutils.cmd import Command
from wheel.bdist_wheel import bdist_wheel

from shutil import copyfile, rmtree

import subprocess

dir_path = os.path.dirname(os.path.realpath(__file__))

def build_pytnn():
    cmd = [os.path.join(dir_path, "build_tnntorch_linux.sh")]
    status_code = subprocess.run(cmd).returncode

    if status_code != 0:
        sys.exit(status_code)

class InstallCommand(install):
    description = "Builds the package"

    def initialize_options(self):
        install.initialize_options(self)

    def finalize_options(self):
        install.finalize_options(self)

    def run(self):
        build_pytnn()
        install.run(self)


class BdistCommand(bdist_wheel):
    description = "Builds the package"

    def initialize_options(self):
        bdist_wheel.initialize_options(self)

    def finalize_options(self):
        bdist_wheel.finalize_options(self)

    def run(self):
        build_pytnn()
        bdist_wheel.run(self)

setup(name='pytnn',
      setup_requires=[],
      version='0.3.0',
      cmdclass={
          'install': InstallCommand,
          'bdist_wheel': BdistCommand,
      },
      zip_safe=False,
      packages=['pytnn'],
      package_dir={'pytnn': 'tnntorch_linux_release/lib'},
      package_data={'pytnn': ['*.so*', '*.py']}
      )
