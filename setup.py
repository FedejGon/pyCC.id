from setuptools import setup, find_packages

setup(
    name='pycc',
    version='0.0',
    packages=find_packages(),
    install_requires=[
        'torch',
        'numpy'
    ],
    author='Federico J. Gonzalez',
    description='Library to train and use NN models for physical systems',
    python_requires='>=3.7',
)

