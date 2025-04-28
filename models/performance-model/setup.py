from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import sys
import os
import subprocess

class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=''):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)

class CMakeBuild(build_ext):
    def run(self):
        try:
            subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError("CMake must be installed")

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        
        cmake_args = [
            f'-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}',
            f'-DPYTHON_EXECUTABLE={sys.executable}',
            '-GNinja',
            '-DCMAKE_BUILD_TYPE=Release'
        ]

        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
            
        subprocess.check_call(['cmake', ext.sourcedir] + cmake_args, cwd=self.build_temp)
        subprocess.check_call(['ninja'], cwd=self.build_temp)

setup(
    name='panda',
    version='0.1',
    author='Jacob Peake',
    description='Panda Python Bindings',
    ext_modules=[CMakeExtension('panda._panda')],
    cmdclass=dict(build_ext=CMakeBuild),
    packages=['panda'],
    package_dir={'panda': '.'},
    py_modules=['interface'],
    entry_points={
        'console_scripts': [
            'panda = interface:main'
        ]
    },
    zip_safe=False,
) 