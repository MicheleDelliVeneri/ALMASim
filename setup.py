import os
import subprocess
import setuptools
from setuptools.command.install import install as _install


class InstallWithSubmodule(_install):
    def run(self):
        # Check if we are in a git repository
        if os.path.isdir(".git"):
            # Initialize and update submodule
            subprocess.check_call(
                ["git", "submodule", "update", "--init", "--recursive"]
            )

            # Install the submodule
            subprocess.check_call(["pip", "install", "./illustris_python"])
        else:
            print("Skipping submodule installation: not in a git repository")

        # Continue with the normal installation
        _install.run(self)


setuptools.setup(
    name="almasim",
    version="2.0",
    author="Michele Delli Veneri",
    author_email="micheledelliveneri@gmail.com",
    description="An ALMA Simulation package for a more civilized era.",
    long_description=open("README.rst").read(),
    long_description_content_type="text/x-rst",
    url="https://github.com/MicheleDelliVeneri/ALMASim",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.12.4",
    install_requires=[
        "astropy",
        "pyvo",
        "numpy",
        "pandas",
        "matplotlib",
        "seaborn",
        "pyQt6",
        "tqdm",
        "scipy",
        "h5py",
        "kaggle",
        "dask",
        "dask-expr",
        "dask_jobqueue",
        "distributed",
        "astromartini",
        "Hdecompose",
        "paramiko",
        "pysftp",
        "setuptools",
        "tenacity",
        "nifty8",
    ],
    cmdclass={
        "install": InstallWithSubmodule,
    },
)
