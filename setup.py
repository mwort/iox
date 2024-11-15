#!/usr/bin/env python3
import os
from setuptools import setup, find_packages
import subprocess

# try:
#     rev = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip()
#     subprocess.check_output(["sed", "-i", f"s/$dev/{rev}/g", "iox.py"])
# except subprocess.CalledProcessError as e:
#     pass

from iox import __version__

setup(
    name="iox",
    version=__version__,
    package_dir={"": "."},
    entry_points={'console_scripts': ["iox=iox:__main__"]},
    # Additional metadata
    author="Michel Wortmann",
    author_email="michel.wortmann@ecmwf.int",
    long_description="Check input and output to conditionally execute commands in parallel",
    url="https://github.com/mwort/iox",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)