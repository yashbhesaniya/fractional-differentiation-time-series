from setuptools import setup, find_packages
import codecs
import os

VERSION = '0.0.4'
DESCRIPTION = 'Application of Machine Learning in Finances'
LONG_DESCRIPTION = 'A package that allows you to build pipelines and evaluate trading strategies, resulting in instructions to operate in the market.'


with open("README.md", "r") as f:
    long_description = f.read()


# Setting up
setup(
    name="finance_ml",
    version=VERSION,
    author="Fabio Maia",
    author_email="<fabio.masaracchia@gmail.com>",
    description=DESCRIPTION,
    long_description_content_type="text/markdown",
    long_description=long_description,
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=['pandas', 'numpy', 'matplotlib','pyarrow'],
    keywords=['python', 'finance', 'mlops'],
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Operating System :: Unix",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ]
)