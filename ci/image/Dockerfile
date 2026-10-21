# CI image for the MADIV-28 pipeline regression jobs.
#
# Pins the imaging and calibration stack to the versions used on zwicky
# (/SKAO_data/spack/.../wsclean-3.7.20260424): WSClean at the same commit built
# against casacore 3.8.0, and the casatools/casatasks/python-casacore wheels
# from uv.lock. Products made in this image are compared bit for bit with
# references made in the same image, so every version here is pinned.

ARG UBUNTU_VERSION=24.04

FROM ubuntu:${UBUNTU_VERSION} AS build

ARG CASACORE_VERSION=3.8.0
ARG WSCLEAN_COMMIT=c5314d942913d1502e2518b675c5605f234cdb75

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        bison build-essential ca-certificates cmake flex gfortran git \
        libblas-dev libboost-date-time-dev libboost-filesystem-dev \
        libboost-program-options-dev libboost-system-dev libboost-test-dev \
        libcfitsio-dev libfftw3-dev libgsl-dev libhdf5-dev liblapack-dev \
        libncurses-dev libreadline-dev pkg-config pybind11-dev python3 python3-dev \
        wcslib-dev \
    && rm -rf /var/lib/apt/lists/*

RUN git clone --depth 1 --branch "v${CASACORE_VERSION}" \
        https://github.com/casacore/casacore.git /src/casacore \
    && cmake -S /src/casacore -B /src/casacore/build \
        -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/opt/stack \
        -DBUILD_PYTHON3=OFF -DBUILD_TESTING=OFF -DUSE_OPENMP=OFF \
        -DDATA_DIR=/opt/casa-data/casacore \
    && cmake --build /src/casacore/build -j "$(nproc)" \
    && cmake --install /src/casacore/build

RUN git clone https://gitlab.com/aroffringa/wsclean.git /src/wsclean \
    && git -C /src/wsclean checkout "${WSCLEAN_COMMIT}" \
    && git -C /src/wsclean submodule update --init --recursive \
    && cmake -S /src/wsclean -B /src/wsclean/build \
        -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/opt/stack \
        -DCMAKE_PREFIX_PATH=/opt/stack -DPORTABLE=ON \
    && cmake --build /src/wsclean/build -j "$(nproc)" \
    && cmake --install /src/wsclean/build


FROM ubuntu:${UBUNTU_VERSION}

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl git libblas3 libboost-date-time1.83.0 \
        libboost-filesystem1.83.0 libboost-program-options1.83.0 \
        libboost-system1.83.0 libcfitsio10t64 libfftw3-double3 \
        libfftw3-single3 libgfortran5 libgomp1 libgsl27 libhdf5-103-1t64 \
        libhdf5-cpp-103-1t64 \
        liblapack3 libpython3.12t64 libreadline8t64 libwcs8 xz-utils \
    && rm -rf /var/lib/apt/lists/*

COPY --from=build /opt/stack /opt/stack
COPY --from=ghcr.io/astral-sh/uv:0.11.14 /uv /uvx /usr/local/bin/

ENV PATH=/opt/stack/bin:/opt/venv/bin:$PATH \
    LD_LIBRARY_PATH=/opt/stack/lib \
    OPENBLAS_NUM_THREADS=1 \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_PYTHON_INSTALL_DIR=/opt/python \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1

# Python environment from the lockfile only (no project code): CI jobs mount
# the checkout and run `uv sync --frozen`, which reuses this environment.
WORKDIR /opt/almasim-lock
COPY pyproject.toml uv.lock README.md ./
RUN uv python install 3.12 \
    && uv sync --frozen --no-install-project \
        --no-default-groups --group test --extra casa --extra ms-casacore

# CASA runtime data (measures, geodetic), read-only at /opt/casa-data.
# casasiteconfig points casatools at it and turns off the auto-updates that
# would otherwise try to write to ~/.casa/data at import time.
RUN python -c "from casaconfig import pull_data; pull_data('/opt/casa-data')" \
    && chmod -R a+rX /opt/casa-data \
    && printf '%s\n' \
        'measurespath = "/opt/casa-data"' \
        'measures_auto_update = False' \
        'data_auto_update = False' \
        > "$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')/casasiteconfig.py"

# Fail the build if any shared library is missing (a pipe into head would hide it).
RUN if ldd /opt/stack/bin/wsclean | grep "not found"; then exit 1; fi \
    && wsclean --version > /tmp/wsclean-version && head -2 /tmp/wsclean-version \
    && python -c "import casatools, casatasks, casacore.tables, ducc0; print(casatools.version_string())"

WORKDIR /work
