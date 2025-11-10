.. image:: https://github.com/MicheleDelliVeneri/ALMASim/raw/main/pictures/ALMASimBanner.jpeg

|Python version| |PyPI| |Repostatus| |Zenodo| |Tests| |Documentation Status| |CodeCov|

.. |Tests| image:: https://github.com/MicheleDelliVeneri/ALMASim/actions/workflows/lint_and_test.yml/badge.svg?branch=main
   :target: https://github.com/MicheleDelliVeneri/ALMASim/actions/workflows/lint_and_test.yml
.. |PyPI| image:: https://img.shields.io/pypi/v/ALMASim?color=green&label=PyPI
   :target: https://pypi.org/project/ALMASim/
.. |Documentation Status| image:: https://readthedocs.org/projects/almasim/badge/?version=latest
   :target: https://almasim.readthedocs.io
.. |Python Version| image:: https://img.shields.io/pypi/pyversions/ALMASim?color=green&label=Python%20Version
   :target: https://pypi.org/project/ALMASim/
.. |Upload Python Package| image:: https://github.com/MicheleDelliVeneri/ALMASim/actions/workflows/python-publish.yml/badge.svg
   :target: https://github.com/MicheleDelliVeneri/ALMASim/actions/workflows/python-publish.yml
.. |Zenodo| image:: https://zenodo.org/badge/501944702.svg
   :target: https://zenodo.org/doi/10.5281/zenodo.12684237
.. |Repostatus| image:: https://www.repostatus.org/badges/latest/active.svg
.. |CodeCov| image:: https://codecov.io/github/MicheleDelliVeneri/ALMASim/graph/badge.svg?token=9SZVW78DR2
   :target: https://codecov.io/github/MicheleDelliVeneri/ALMASim

.. INTRO_START_LABEL

Overview
--------

ALMASim is a package to generate mock observations of radio sources
as observed by the Atacama Large Millimetre/Submillimetre Array (ALMA).
ALMASim primary goal is to allow users to generate simulated datasets on
which to test deconvolution and source detection models. ALMASim is
intended to leverage MPI parallel computing (Dask, Slurm, PBS) on modern HPC clusters to
generate thousands of ALMA data cubes, but can also work on laptopts.
The legacy PyQt desktop interface has been retired; ALMASim now exposes a
set of headless services that can be orchestrated from CLIs, notebooks,
or web backends (e.g. FastAPI + React).
ALMA database or a prepared catalogue is queried to sample observational
metadata such as band, bandwidth, integration time, antenna
configurations and so on. ALMASim employs the MARTINI Package
(https://github.com/kyleaoman/martini), and the Illustris Python Package
(https://github.com/illustristng/illustris_python).

.. INTRO_END_LABEL

.. CITING_START_LABEL

Citing ALMASim
--------------

If you use ALMASim in your research, please cite the following paper:

::

   @ARTICLE{10.1093/mnras/stac3314,
   author = {Delli Veneri, Michele and Tychoniec, Łukasz and Guglielmetti, Fabrizia and Longo, Giuseppe and Villard, Eric},
   title = "{3D Detection and Characterisation of ALMA Sources through Deep Learning}",
   journal = {Monthly Notices of the Royal Astronomical Society},
   year = {2022},
   month = {11},
   issn = {0035-8711}, 
   doi = {10.1093/mnras/stac3314},
   url = {https://doi.org/10.1093/mnras/stac3314},
   note = {stac3314},
   eprint = {https://academic.oup.com/mnras/advance-article-pdf/doi/10.1093/mnras/stac3314/47014718/stac3314.pdf}
   }

ALMASim entry: https://doi.org/10.1093/mnras/stac3314

.. CITING_END_LABEL

.. INSTALLATION_NOTES_START_LABEL

Installation Notes
------------------

ALMASim works with ``python3`` (version ``3.12``), and does not support
``python2``. First create a virtual environment with ``python3`` and
activate it. Then install the required packages with ``pip``:

-  Create the Python Environment:``python3.12 -m venv astro-env``
-  Activate it: ``source astro-env/bin/activate`` (in case of your shell
   is Bash, otherwise check the other activations scripts within the bin
   folders)

.. INSTALLATION_NOTES_END_LABEL

Installing with pip
-------------------
- pip install almasim

.. GITHUB_INSTALLATION_NOTES_START_LABEL

Installing from GitHub 
----------------------
-  Clone the ALMASim Repository:
   ``git clone https://github.com/MicheleDelliVeneri/ALMASim.git``
-  Install packages from the requirements file:
   ``pip install -e .``

.. GITHUB_INSTALLATION_NOTES_END_LABEL

Adding Kaggle API
-----------------
-  Login into Kaggle and go to: ``https://www.kaggle.com/settings``
-  Click on ``create new token``, this will produce a kaggle.json file
   which must be saved in your home folder: ``~/.kaggle/kaggle.json``

.. QUICKSTART_START_LABEL

Getting started
---------------

The refactor removes the PyQt desktop shell in favour of three headless
services (metadata, datasets, simulation). A typical workflow is now:

1. Query or load ALMA metadata (``almasim.services.metadata`` contains helpers).
2. Convert one of the metadata rows into :class:`SimulationParams`.
3. Call :func:`run_simulation` to generate skymodels and visibilities.

Below is a minimal end-to-end example that uses the bundled
``qso_metadata.csv`` catalogue:

.. code-block:: python

   from pathlib import Path

   import almasim.astro as astro
   from almasim.services import metadata as metadata_service
   from almasim.services import simulation as sim
   from almasim import SimulationParams, run_simulation

   repo_root = Path(__file__).resolve().parents[1]
   main_dir = repo_root / "almasim"
   metadata = metadata_service.load_metadata(main_dir / "metadata" / "qso_metadata.csv")
   rest_frequency, _ = astro.get_line_info(main_dir)
   sample = sim.sample_given_redshift(metadata, 1, rest_frequency, False, None)
   target = sample.iloc[0]

   params = SimulationParams.from_metadata_row(
       target,
       idx=0,
       project_name="demo",
       main_dir=main_dir,
       output_dir=Path("./outputs"),
       tng_dir=Path("/data/TNG100-1"),
       galaxy_zoo_dir=Path("/data/galaxy_zoo"),
       hubble_dir=Path("/data/hubble"),
       source_type="point",
       save_mode="npz",
   )

   run_simulation(params)

``SimulationParams.from_metadata_row`` normalises an ALMA record,
expands/absolutises all directories, and accepts overrides for knobs
such as ``snr``, ``source_type``, ``n_pix``, or ``save_mode``. You can
still instantiate the dataclass manually if you already have fully
prepared values.

Notes
-----

Cube size will dictate simulation speed and RAM usage. To gauge what you
can affort to run, we advice to start with a single simulation of a 256 x
256 x 256 cube.

.. QUICKSTART_END_LABEL


Parameters and Configuration
----------------------------

- The robustness parameter, often referred to as the Briggs robustness parameter or robfac in some implementations, is typically taken within a specific range that allows for a meaningful balance between sensitivity and resolution. The common range for this parameter is:

	-	robust = -2 to robust = +2

Interpretation of Values

	-	robust = -2:
	-	This setting heavily favors natural weighting.
	-	It emphasizes sensitivity, meaning that the weighting scheme gives more emphasis to areas of the UV plane that are more densely sampled.
	-	The resulting image will generally have lower noise but may have lower resolution and broader synthesized beams.
	-	robust = 0:
	-	This is a balanced setting that attempts to compromise between natural and uniform weighting.
	-	It offers a good balance between resolution and sensitivity.
	-	This is often considered a default or starting point in many imaging processes.
	-	robust = +2:
	-	This setting favors uniform weighting.
	-	It prioritizes higher resolution by giving more uniform weight across the UV plane, even in less densely sampled areas.
	-	The resulting image will typically have a higher resolution and a narrower synthesized beam, but with increased noise.

Usage in Practice

	-	Default Values: Depending on the imaging task, astronomers often start with robust = 0 as a default and then adjust based on the specific needs (e.g., higher resolution or lower noise).
	-	Adjustment: The choice of robust is influenced by the science goals. For example:
	-	If detecting faint structures is the priority, a lower robust value (towards -2) might be chosen.
	-	If resolving fine details in an image is more critical, a higher robust value (towards +2) might be preferred.
	-	Exploration: It’s common to generate images using several different robust values to see how the balance of resolution and noise affects the final image.
