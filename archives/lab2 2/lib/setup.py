from setuptools import setup, find_packages

VERSION = "0.0.1"

setup(
    name="wsilib",
    version=VERSION,
    packages=find_packages(),
    install_requires=["numpy", "autograd"],
)
