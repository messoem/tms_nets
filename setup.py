from setuptools import setup, find_packages

setup(
    name="tms_nets",
    version="0.1.0",
    packages=find_packages(),
    install_requires=["numpy", "seaborn", "pandas", "matplotlib", "galois"],
)
