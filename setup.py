from setuptools import setup, find_packages

setup(
    name='pycc.id',
    version='0.1.45',
    packages=find_packages(),
    install_requires=[
        'torch',
        'numpy'
    ],
    author='Federico J. Gonzalez',
    description='Library to system identification using CC',
    python_requires='>=3.7',
)


