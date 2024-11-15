from setuptools import setup, find_packages

setup(
    name="iox",
    version="0.1.0",
    package_dir={"": "."},
    entry_points={'console_scripts': ["iox=iox:__main__"]},
    # Additional metadata
    author="Michel Wortmann",
    author_email="michel.wortmann@ecmwf.int",
    description="Check input and output to conditionally execute commands in parallel",
    url="https://github.com/mwort/iox",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)