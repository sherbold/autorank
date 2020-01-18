#!/usr/bin/env python

from setuptools import setup, find_packages
from os import path

import autorank

# read the contents of your README file

this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='autorank',
    version=autorank.__version__,
    description='Automated ranking of populations in a repeated measures experiment, e.g., to rank different machine learning approaches tested on the same data.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    install_requires=['numpy', 'pandas', 'statsmodels', 'scipy', 'matplotlib'],
    author='sherbold',
    author_email='herbold@cs.uni.goettingen.de',
    url='https://github.com/sherbold/autorank',
    download_url='https://github.com/sherbold/autorank/zipball/master',
    packages=find_packages(),
    zip_safe=False,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Development Status :: 4 - Beta",
        "Environment :: Console",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: POSIX :: Linux",
        "Topic :: Scientific/Engineering",
    ],
)
