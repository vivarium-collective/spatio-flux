import re
from setuptools import setup, find_packages


VERSION = '0.0.1'

with open("README.md", "r") as readme:
    description = readme.read()


setup(
    name="spatio-flux",
    version=VERSION,
    author="Ryan Spangler, Eran Agmon",
    author_email="ryan.spangler@gmail.com, agmon.eran@gmail.com",
    description="",
    long_description=description,
    long_description_content_type="text/markdown",
    url="https://github.com/vivarium-collective/spatio-flux",
    packages=[
        'spatio_flux.processes',
        'spatio_flux.viz',
        'spatio_flux',
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.6",
    install_requires=[
        "process-bigraph",
        "bigraph-viz",
        "scipy",
        "cobra",
        "imageio",
        "matplotlib",
        "numpy",
    ]
)
