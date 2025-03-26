from setuptools import setup, find_packages

setup(
    name="tms_nets",
    version="0.1.0",
    author="Arsenii Sychev",
    description="Niederreiter algo",
    url="https://github.com/messoem/tms_nets",
    packages=find_packages(),
    install_requires=["numpy", "seaborn", "pandas", "matplotlib", "galois"],
    
)
