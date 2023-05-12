from setuptools import setup, find_namespace_packages

setup(
    name='ALMASim',
    version='1.1.1',
    packages=find_namespace_packages(),
    install_requires=["numpy", "h5py", "six", 
                      "astropy==5.1.1", "scipy", "matplotlib", "pandas", "tqdm",
                      "h5py", "hdecompose", "spectral_cube", "astromartini", 
                      "dask", "dask[distributed]", "natsort", "distributed", "casatools==6.5.5.21", 
                      "casatasks==6.5.5.21", "casadata==2023.4.10",
                      
                      ],
    tests_require=["nose","coverage"],
)
