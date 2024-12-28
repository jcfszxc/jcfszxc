#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time          : 2024/12/28 21:57
# @Author        : jcfszxc
# @Email         : jcfszxc.ai@gmail.com
# @File          : setup.py
# @Description   : 

from setuptools import setup, find_packages

setup(
    name="task-manager",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        'PyQt6>=6.4.0',
    ],
    entry_points={
        'console_scripts': [
            'task-manager=src.main:main',
        ],
    },
    author="Your Name",
    author_email="your.email@example.com",
    description="A personal task management application",
    license="MIT",
    python_requires='>=3.8',
)