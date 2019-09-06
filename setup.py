from setuptools import setup, find_packages

setup(name='pget',
  version='0.1.0',
  install_requires = [
    'ml_utils @ git+https://git@github.com/tehzevo/ml-utils@master#egg=ml_utils',
  ],
  packages=find_packages())
