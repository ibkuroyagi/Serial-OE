#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Setup Anomaly Sound Detection library."""

import os

from setuptools import find_packages
from setuptools import setup

requirements = {
    "install": [],
    "setup": [
        "numpy",
        "pytest-runner",
    ],
    "test": ["pytest>=3.3.0", "hacking>=3.0.0", "flake8-docstrings>=1.3.1"],
}

install_requires = requirements["install"]
setup_requires = requirements["setup"]
extras_require = {
    k: v for k, v in requirements.items() if k not in ["install", "setup"]
}

dirname = os.path.dirname(__file__)
setup(
    name="serial_oe",
    version="0.0.0",
    url="https://github.com/ibkuroyagi/Serial-OE.git",
    author="Ibuki Kuroyanagi",
    author_email="kuroyanagi.ibuki@g.sp.m.is.nagoya-u.ac.jp",
    description="Anomalous Sound Detection using Serial method with Outlier Exposure",
    long_description=open(os.path.join(dirname, "README.md"), encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    license="Apache License 2.0",
    packages=find_packages(include=["serial_oe*"]),
    install_requires=install_requires,
    setup_requires=setup_requires,
    extras_require=extras_require,
)
