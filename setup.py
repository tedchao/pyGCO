"""

>>> python setup.py build_ext --inplace
>>> python setup.py install
"""

import os
import numpy
import urllib3
import zipfile
import shutil
from setuptools import setup, find_packages, Extension
# from distutils.core import setup
# from distutils.extension import Extension
from Cython.Distutils import build_ext

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

gco_files = [os.path.join(LOCAL_SOURCE, f) for f in ['LinkedBlockList.cpp',
                                                     'graph.cpp',
                                                     'maxflow.cpp',
                                                     'GCoptimization.cpp']]
gco_files += ['cgco.cpp']

setup(name='gco-wrapper',
      url='http://vision.csd.uwo.ca/code/',
      author='Yujia',
      maintainer='J. Borovec',
      maintainer_email='jiri.borovec@fel.cvut.cz',
      license='MIT',
      platforms=['Linux'],
      version='3.0.0',
      description='pygco: a python wrapper for the graph cuts package',
      download_url='https://github.com/Borda/pyGCO',
      cmdclass={'build_ext': build_ext},
      ext_modules=[Extension('libcgco', gco_files, language='c++',
                             include_dirs=[LOCAL_SOURCE, numpy.get_include()],
                             library_dirs=[LOCAL_SOURCE],
                             extra_compile_args=["-fpermissive"])],
      py_modules=['cgco', 'pygco'],
      packages=find_packages(),
      install_requires=['numpy>=1.8.2',
                        'Cython'],
      long_description='This is a python wrapper for gco package '
                       '(http://vision.csd.uwo.ca/code/), '
                       'which implements a graph cuts based move-making '
                       'algorithm for optimization in Markov Random Fields.',
      # See https://PyPI.python.org/PyPI?%3Aaction=list_classifiers
      classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 4 - Beta',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 2.7',
      ],
)
