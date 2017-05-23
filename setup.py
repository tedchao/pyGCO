try:
    from setuptools import setup, Extension, Command
    from setuptools.command.build_ext import build_ext
except ImportError:
    from distutils.core import setup, Extension, Command
    from distutils.command.build_ext import build_ext

import os

from pip.req import parse_requirements
import pkg_resources
import pip
import subprocess

source_directory = "gco_source"

if subprocess.call(["git", "rev-parse", "--resolve-git-dir",
                   os.path.join(source_directory, ".git")],
                   stderr=subprocess.STDOUT,
                   stdout=open(os.devnull, 'w')) != 0:
    print("Download the code")
    subprocess.call(['./git_download.sh'])
else:
    print("Assume you have the code")





pygco_directory = "pygco"


files = ['GCoptimization.cpp', 'graph.cpp', 'LinkedBlockList.cpp',
         'maxflow.cpp']

files = [os.path.join(source_directory, f) for f in files]
files.insert(0, os.path.join(pygco_directory, 'cgco.cpp'))

def _parse_requirements(filepath):
    pip_version = list(map(int, pkg_resources.get_distribution('pip').version.split('.')[:2]))
    if pip_version >= [6, 0]:
        raw = parse_requirements(filepath, session=pip.download.PipSession())
    else:
        raw = parse_requirements(filepath)

    return [str(i.req) for i in raw]


# parse_requirements() returns generator of pip.req.InstallRequirement objects
install_reqs = _parse_requirements("requirements.txt")


setup(name='pygco',
    packages=['pygco'],
    version='0.0.1',
    description="A python wrapper for gco-v3.0 package, used for graph cuts based MRF optimization.",
    author="yujiali, amueller",
    url='https://github.com/yujiali/pygco',
    include_package_data=True,
    install_requires=install_reqs,
    test_suite='nose.collector',
    tests_require=['nose'],
    license="MIT",
    zip_safe=False,
    keywords='pygco',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License (MIT)',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.6'
    ],
    cmdclass={'build_ext': build_ext},
      ext_modules=[Extension("cgco", files, language="c++",
                             include_dirs=[source_directory, pygco_directory],
                             libraries=[],
                             library_dirs=[source_directory],
                             #extra_compile_args=["-fpermissive"]
                             )]
    )
