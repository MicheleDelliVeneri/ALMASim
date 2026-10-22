"""Corruptions of model visibilities, beyond thermal noise.

Each model takes the main table's columns for a chunk of rows and returns
per-row, per-channel factors or additive terms; ``corrupt`` applies them:

    V = g_i g_j* B_i B_j* V_model + n

- thermal noise ``n``: per visibility, from the real WEIGHT (scaled to the
  scatter measured in the real data), optionally correlated between
  adjacent channels as ALMA's spectral (Hanning) response makes it;
- antenna gains ``g``: residual amplitude and phase errors after
  calibration, constant per scan plus a slow drift within it;
- atmosphere: a Kolmogorov phase screen blown over the array (frozen
  flow), what phase referencing leaves of it (a high-pass over the
  calibration cycle), phase proportional to frequency;
- bandpass ``B``: residual smooth ripples across the window, per antenna.

Every random draw comes from one numpy Generator seeded per window and
iteration, so a counterpart can be remade exactly.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

C = 299_792_458.0


@dataclass
class ErrorModel:
    """Which corruptions to apply, and how strong."""

    thermal: bool = True
    thermal_scale: float = 1.0  # times the noise the real WEIGHT and scatter imply
    spectral_kernel: str = "none"  # "none" | "hanning" | "measured" (from the real data)
    gain_amp_rms: float = 0.0  # fractional, per antenna, per scan
    gain_phase_rms_deg: float = 0.0  # per antenna, per scan
    gain_drift_rms: float = 0.0  # fraction of the above, a random walk within a scan
    gain_drift_s: float = 60.0  # correlation time of the drift
    screen_path_um_1km: float = 0.0  # residual path rms between antennas 1 km apart
    screen_wind_mps: float = 10.0
    screen_cycle_s: float = 0.0  # phase-referencing cycle (0: no referencing)
    screen_outer_km: float = 3.0
    bandpass_amp_rms: float = 0.0  # fractional ripple across the window, per antenna
    bandpass_phase_rms_deg: float = 0.0
    bandpass_modes: int = 4  # Fourier modes of the ripple
    extra: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, values: dict) -> ErrorModel:
        known = {k: v for k, v in values.items() if k in cls.__dataclass_fields__}
        return cls(**known)


# Spectral response: white noise convolved with a Hanning window correlates
# adjacent channels by 2/3 and the next ones by 1/6, as ALMA's correlator
# leaves native channels; normalised so the per-channel variance is unchanged.
_HANNING = np.array([0.25, 0.5, 0.25])
_HANNING = _HANNING / np.sqrt(np.sum(_HANNING**2))


def spectral_kernel(rho1: float, rho2: float) -> np.ndarray:
    """A 3-tap filter whose output's channel autocorrelation is (1, rho1, rho2).

    White noise filtered by [h0, h1, h2] (unit norm) has lag-1 correlation
    h0 h1 + h1 h2 and lag-2 correlation h0 h2: solved from the measured ones.
    """
    from scipy.optimize import fsolve

    if rho1 < 0.05 and abs(rho2) < 0.05:
        return np.array([0.0, 1.0, 0.0])

    def equations(h):
        return [h @ h - 1, h[0] * h[1] + h[1] * h[2] - rho1, h[0] * h[2] - rho2]

    start = _HANNING if rho2 > 0.05 else np.array([0.0, 0.85, 0.5])
    h = fsolve(equations, start)
    solved = np.max(np.abs(equations(h))) < 1e-4  # fsolve flags a start on the root as "no progress"
    return h if solved else np.array([0.0, 1.0, 0.0])


def thermal_noise(
    rng: np.random.Generator, sigma: np.ndarray, nchan: int, kernel: str | np.ndarray
) -> np.ndarray:
    """Complex noise (nrow, nchan, ncorr) with per-(row, corr) sigma per component.

    ``kernel``: "none" (independent channels), "hanning", or a 3-tap filter.
    """
    nrow, ncorr = sigma.shape
    shape = (nrow, nchan + 2, ncorr)
    noise = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    taps = _HANNING if isinstance(kernel, str) and kernel == "hanning" else kernel
    if isinstance(taps, np.ndarray):
        noise = taps[0] * noise[:, :-2] + taps[1] * noise[:, 1:-1] + taps[2] * noise[:, 2:]
    else:
        noise = noise[:, 1:-1]
    return (noise * sigma[:, None, :]).astype(np.complex64)


@dataclass
class AntennaGains:
    """Residual antenna gains: per (antenna, pol), constant per scan plus a drift."""

    amp: dict  # (ant, scan) -> (npol,) fractional amplitude error
    phase: dict  # (ant, scan) -> (npol,) phase error, radians
    drift_amp: float
    drift_phase: float
    drift_s: float
    seed: int

    @classmethod
    def draw(cls, rng, model: ErrorModel, antennas, scans, npol: int) -> AntennaGains:
        amp, phase = {}, {}
        for a in antennas:
            for s in scans:
                amp[(a, s)] = rng.normal(0, model.gain_amp_rms, npol)
                common = rng.normal(0, np.radians(model.gain_phase_rms_deg))
                phase[(a, s)] = common + rng.normal(0, np.radians(model.gain_phase_rms_deg) / 4, npol)
        return cls(
            amp,
            phase,
            model.gain_drift_rms * model.gain_amp_rms,
            model.gain_drift_rms * np.radians(model.gain_phase_rms_deg),
            model.gain_drift_s,
            int(rng.integers(2**31)),
        )

    def _drift(self, ant: np.ndarray, time: np.ndarray, which: int) -> np.ndarray:
        """A smooth zero-mean drift per antenna: sum of a few random sinusoids."""
        out = np.zeros(len(ant))
        for a in np.unique(ant):
            sel = ant == a
            rng = np.random.default_rng([self.seed, int(a), which])
            periods = self.drift_s * rng.uniform(0.5, 4, 3)
            phases = rng.uniform(0, 2 * np.pi, 3)
            out[sel] = np.sum(
                [np.sin(2 * np.pi * time[sel] / p + f) for p, f in zip(periods, phases, strict=True)],
                axis=0,
            ) / np.sqrt(1.5)
        return out

    def factors(self, ant: np.ndarray, scan: np.ndarray, time: np.ndarray, npol: int) -> np.ndarray:
        """g (nrow, npol) for these rows' antenna."""
        g = np.ones((len(ant), npol), dtype=np.complex128)
        da = self._drift(ant, time, 0) * self.drift_amp
        dp = self._drift(ant, time, 1) * self.drift_phase
        for i, (a, s) in enumerate(zip(ant, scan, strict=True)):
            amp = self.amp.get((int(a), int(s)))
            if amp is None:
                continue
            g[i] = (1 + amp + da[i]) * np.exp(1j * (self.phase[(int(a), int(s))] + dp[i]))
        return g


class PhaseScreen:
    """A frozen Kolmogorov screen of excess path, blown over the array."""

    def __init__(self, rng, model: ErrorModel, positions_m: np.ndarray, duration_s: float):
        self.wind = model.screen_wind_mps
        self.cycle = model.screen_cycle_s
        extent = np.ptp(positions_m[:, 0]) + self.wind * duration_s + 2_000.0
        extent_y = np.ptp(positions_m[:, 1]) + 2_000.0
        self.step = 20.0  # metres per screen pixel
        nx = int(2 ** np.ceil(np.log2(extent / self.step)))
        ny = int(2 ** np.ceil(np.log2(extent_y / self.step)))
        kx = np.fft.fftfreq(nx, self.step)
        ky = np.fft.fftfreq(ny, self.step)
        k2 = kx[None, :] ** 2 + ky[:, None] ** 2 + (1 / (model.screen_outer_km * 1e3)) ** 2
        spectrum = k2 ** (-11 / 12)  # amplitude of a k^-11/3 power spectrum
        screen = np.real(
            np.fft.ifft2(spectrum * (rng.standard_normal((ny, nx)) + 1j * rng.standard_normal((ny, nx))))
        )
        self.origin = positions_m[:, :2].min(axis=0) - 1_000.0
        self.screen = screen
        # Scale so that the rms path difference 1 km apart is the model's.
        i0 = ny // 2
        sep = int(round(1_000 / self.step))
        diff = screen[i0, : nx - sep] - screen[i0, sep:]
        self.screen *= (model.screen_path_um_1km * 1e-6) / np.std(diff)

    def path_m(self, xy: np.ndarray, time: np.ndarray) -> np.ndarray:
        """Excess path (m) over antennas at ``xy`` (n, 2) at ``time`` (n,), frozen flow."""
        x = (xy[:, 0] - self.origin[0] + self.wind * (time - time.min())) / self.step
        y = (xy[:, 1] - self.origin[1]) / self.step
        ix = np.clip(x.astype(int), 0, self.screen.shape[1] - 1)
        iy = np.clip(y.astype(int), 0, self.screen.shape[0] - 1)
        return self.screen[iy, ix]

    def residual_path_m(self, xy: np.ndarray, time: np.ndarray) -> np.ndarray:
        """What phase referencing leaves: the path minus its mean over each cycle."""
        path = self.path_m(xy, time)
        if self.cycle <= 0:
            return path
        # Referencing follows the screen at the cycle's pace: the residual is the
        # path minus the path a cycle-length later in the flow, halved (interpolated).
        later = self.path_m(xy, time + self.cycle)
        return (path - later) / 2


def bandpass_factors(rng, model: ErrorModel, antennas, nchan: int, npol: int) -> dict:
    """B per antenna (nchan, npol): smooth complex ripples across the window."""
    x = np.linspace(0, 1, nchan)
    out = {}
    for a in antennas:
        amp = np.zeros((nchan, npol))
        phase = np.zeros((nchan, npol))
        for m in range(1, model.bandpass_modes + 1):
            weight = 1 / m  # smoother: lower modes dominate
            for p in range(npol):
                amp[:, p] += weight * rng.normal() * np.sin(2 * np.pi * m * x + rng.uniform(0, 2 * np.pi))
                phase[:, p] += weight * rng.normal() * np.sin(2 * np.pi * m * x + rng.uniform(0, 2 * np.pi))
        norm = np.sqrt(np.sum([(1 / m) ** 2 for m in range(1, model.bandpass_modes + 1)]) / 2)
        out[a] = (1 + model.bandpass_amp_rms * amp / norm) * np.exp(
            1j * np.radians(model.bandpass_phase_rms_deg) * phase / norm
        )
    return out
