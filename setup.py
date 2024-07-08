import setuptools
import subprocess
import sys
import os


class CustomInstallCommand(setuptools.Command):
    description = (
        "Custom install command that clones and installs the illustris_python package."
    )
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        # Clone the illustris_python repository
        subprocess.check_call(
            ["git", "clone", "https://github.com/illustristng/illustris_python.git"]
        )

        # Change to the cloned directory
        os.chdir("illustris_python")

        # Install the cloned package
        subprocess.check_call([sys.executable, "-m", "pip", "install", "."])

        # Change back to the original directory
        os.chdir("..")

        # Run the standard install command
        setuptools.Command.run(self)


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
        "setuptools",
    ],
    cmdclass={
        "install": CustomInstallCommand,
    },
)
