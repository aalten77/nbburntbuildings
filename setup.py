import os
from setuptools import setup

# Utility function to read the README file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...

def read_reqs(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read().splitlines()

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name = "nbburntbuildings",
    version = "0.1.0",
    author = "Ai-Linh Alten",
    author_email = "ai-linh.alten@sjsu.edu",
    description = ("Functions for the burnt buildings GBDX Notebook"),
    license = "BSD",
    keywords = "example documentation tutorial",
    packages=['nbburntbuildings'],
    long_description=read('README.md'),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Topic :: Utilities",
        "License :: OSI Approved :: BSD License",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 2",
        "Programming Language :: Python :: 2.7",
    ],
    install_requirements=read_reqs('requirements.txt')
)