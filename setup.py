import setuptools

APP = ['almasim/ui.py']  # Replace with the path to your main script
DATA_FILES = []  # Include any additional data files needed
OPTIONS = {
    'argv_emulation': True,
    'iconfile': 'pictures/almasim.icns',  # Path to your .icns icon file
    'packages': [],  # List any additional packages your app needs
    'plist': {
        'CFBundleName': 'ALMASim',
        'CFBundleDisplayName': 'ALMASim',
        'CFBundleGetInfoString': 'ALMASim Application',
        'CFBundleIdentifier': 'The University of the Street',
        'CFBundleVersion': '2.1.10',
        'CFBundleShortVersionString': '2.1.10',
    },
}


setuptools.setup(
    name="almasim",
    version="2.1.10",
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
        "asyncssh",
        "bokeh",
        "pyvo",
        "numpy",
        "pandas",
        "matplotlib",
        "seaborn",
        "pyQt6",
        "qtrangeslider",
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
        "scikit-image",
        "imagecodecs",
    ],
)
