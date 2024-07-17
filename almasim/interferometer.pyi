from PyQt6.QtCore import QObject
from _typeshed import Incomplete

def showError(message) -> None: ...

class Interferometer(QObject):
    progress_signal: Incomplete
    idx: Incomplete
    terminal: Incomplete
    skymodel: Incomplete
    antenna_array: Incomplete
    noise: Incomplete
    int_time: Incomplete
    obs_date: Incomplete
    c_ms: Incomplete
    main_dir: Incomplete
    output_dir: Incomplete
    plot_dir: Incomplete
    Hfac: Incomplete
    deg2rad: Incomplete
    rad2deg: Incomplete
    deg2arcsec: float
    arcsec2deg: Incomplete
    second2hour: Incomplete
    curzoom: Incomplete
    robust: Incomplete
    deltaAng: Incomplete
    gamma: float
    lfac: float
    header: Incomplete
    Hmax: Incomplete
    lat: Incomplete
    trlat: Incomplete
    Diameters: Incomplete
    ra: Incomplete
    dec: Incomplete
    trdec: Incomplete
    central_freq: Incomplete
    band_range: Incomplete
    imsize: Incomplete
    Xaxmax: Incomplete
    Npix: Incomplete
    Np4: Incomplete
    Nchan: Incomplete
    Nphf: Incomplete
    pixsize: Incomplete
    xx: Incomplete
    yy: Incomplete
    distmat: Incomplete
    robfac: float
    W2W1: int
    currcmap: Incomplete
    zooming: int
    save_mode: Incomplete
    def __init__(
        self,
        idx,
        skymodel,
        main_dir,
        output_dir,
        ra,
        dec,
        central_freq,
        band_range,
        fov,
        antenna_array,
        noise,
        int_time,
        obs_date,
        header,
        save_mode,
        terminal,
        robust: float = 0.5,
    ) -> None: ...
    def run_interferometric_sim(self): ...
    def set_noise(self) -> None: ...
