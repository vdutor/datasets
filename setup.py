#!/usr/bin/env python
# -*- coding: utf-8 -*-

# pylint: skip-file

from setuptools import setup
from setuptools import find_packages

# Dependencies of GPflow
requirements = [
    'numpy>=1.10.0',
    'tensorflow',
]

setup(name='datasets',
      version=0.1,
      author="Vincent Dutordoir",
      author_email="dutordoirv@gmail.com",
      description=("Datasets"),
      license="Apache License 2.0",
      install_requires=requirements)
