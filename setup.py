"""

>>> python setup.py build_ext --inplace
>>> python setup.py install
"""

import os
import numpy
import urllib3
import zipfile
import shutil
import pkg_resources
import pip
try:
    from setuptools import setup, Extension, Command, find_packages
    from setuptools.command.build_ext import build_ext
except ImportError:
    from distutils.core import setup, Extension, Command, find_packages
    from distutils.command.build_ext import build_ext

PACKAGE_NAME = 'gco-v3.0.zip'
GCO_LIB = 'http://vision.csd.uwo.ca/code/' + PACKAGE_NAME
pack_name = os.path.basename(GCO_LIB)
LOCAL_SOURCE = 'gco_source'
if not os.path.exists(LOCAL_SOURCE):
    try:
        os.mkdir(LOCAL_SOURCE)
    except:
        print('no permission to create a directory')

# download code
if not os.path.exists(PACKAGE_NAME):
    http = urllib3.PoolManager()
    with http.request('GET', GCO_LIB, preload_content=False) as resp, \
            open(pack_name, 'wb') as out_file:
        shutil.copyfileobj(resp, out_file)
    resp.release_conn()

# unzip the package
with zipfile.ZipFile(pack_name, 'r') as zip_ref:
    zip_ref.extractall(LOCAL_SOURCE)

# if subprocess.call(["git", "rev-parse", "--resolve-git-dir",
#                     os.path.join(LOCAL_SOURCE, ".git")],
#                    stderr=subprocess.STDOUT,
#                    stdout=open(os.devnull, 'w')) != 0:
#     print("Download the code")
#     subprocess.call(['./git_download.sh'])
# else:
#     print("Assume you have the code")

def _parse_requirements(filepath):
    pip_version = list(map(int, pkg_resources.get_distribution('pip').version.split('.')[:2]))
    if pip_version >= [6, 0]:
        raw = pip.req.parse_requirements(filepath, session=pip.download.PipSession())
    else:
        raw = pip.req.parse_requirements(filepath)

    return [str(i.req) for i in raw]


gco_files = [os.path.join(LOCAL_SOURCE, f) for f in ['LinkedBlockList.cpp',
                                                     'graph.cpp',
                                                     'maxflow.cpp',
                                                     'GCoptimization.cpp']]
# gco_files = [os.path.join(LOCAL_SOURCE, f) for f in gco_files]
# gco_files.insert(0, os.path.join(pygco_directory, 'cgco.cpp'))
gco_files += [os.path.join('pygco', 'cgco.cpp')]

# parse_requirements() returns generator of pip.req.InstallRequirement objects
install_reqs = _parse_requirements("requirements.txt")

setup(name='gco-wrapper',
      url='http://vision.csd.uwo.ca/code/',
      packages=['pygco'],
      author='yujiali, amueller',
      maintainer='Jiri Borovec',
      maintainer_email='jiri.borovec@fel.cvut.cz',
      license='MIT',
      platforms=['Linux'],
      version='3.0.1',
      test_suite='nose.collector',
      tests_require=['nose'],
      description='pygco: a python wrapper for the graph cuts package',
      download_url='https://github.com/Borda/pyGCO',
      cmdclass={'build_ext': build_ext},
      ext_modules=[Extension('libcgco', gco_files, language='c++',
                             include_dirs=[LOCAL_SOURCE, numpy.get_include()],
                             library_dirs=[LOCAL_SOURCE],
                             # extra_compile_args=["-fpermissive"]
                   )],
      py_modules=['cgco', 'pygco'],
      # packages=find_packages(),
      install_requires=install_reqs,
      long_description='This is a python wrapper for gco package '
                       '(http://vision.csd.uwo.ca/code/), '
                       'which implements a graph cuts based move-making '
                       'algorithm for optimization in Markov Random Fields.',
      # See https://PyPI.python.org/PyPI?%3Aaction=list_classifiers
      classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License (MIT)',
        'Natural Language :: English',
        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
      ],
)
