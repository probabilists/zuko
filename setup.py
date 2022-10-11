#!/usr/bin/env python

import setuptools

with open('README.md', 'r') as f:
    readme = f.read()

with open('requirements.txt', 'r') as f:
    required = f.read().splitlines()

setuptools.setup(
    name='backfire',
    version='0.0.0',
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
    url='https://github.com/francois-rozet/backfire',
    project_urls={
        'Documentation': 'https://francois-rozet.github.io/backfire',
        'Source': 'https://github.com/francois-rozet/backfire',
        'Tracker': 'https://github.com/francois-rozet/backfire/issues',
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
    extras_require={
        'docs': ['furo', 'sphinx'],
        'test': ['pytest'],
    },
    python_requires='>=3.8',
)
