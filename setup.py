#!/usr/bin/env python

from setuptools import setup, find_packages
from os import path

this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='autorank',
    version='1.3.0',
    description='Automated ranking of populations in a repeated measures experiment, e.g., to rank different machine '
                'learning approaches tested on the same data.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    install_requires=['numpy', 'pandas>=2.0.3', 'statsmodels>=0.14.0', 'scipy>=1.10.1', 'matplotlib>=3.7.2',
                      'baycomp>=1.0.3', 'jinja2>=3.1.2'],
    author='sherbold',
    author_email='steffen.herbold@uni-passau.de',
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
