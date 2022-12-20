#!/usr/bin/env python

import setuptools

with open('README.md', 'r') as f:
    readme = f.read()

with open('requirements.txt', 'r') as f:
    required = f.read().splitlines()

setuptools.setup(
    name='zuko',
    version='0.0.7',
    packages=setuptools.find_packages(),
    description='Normalizing flows in PyTorch',
    keywords=[
        'normalizing flows',
        'probability',
        'density',
        'generative',
        'deep learning',
        'torch',
    ],
    long_description=readme,
    long_description_content_type='text/markdown',
    author='François Rozet',
    author_email='francois.rozet@outlook.com',
    license='MIT license',
    url='https://github.com/francois-rozet/zuko',
    project_urls={
        'Documentation': 'https://francois-rozet.github.io/zuko',
        'Source': 'https://github.com/francois-rozet/zuko',
        'Tracker': 'https://github.com/francois-rozet/zuko/issues',
    },
    classifiers=[
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
    ],
    install_requires=required,
    python_requires='>=3.8',
)
