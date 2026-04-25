"""Row-wise visibility helpers for MS-style export."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
from astropy.time import Time


@dataclass
class VisibilityTable:
    """Internal row-wise visibility representation."""

    uvw_m: np.ndarray
    antenna1: np.ndarray
    antenna2: np.ndarray
    time_mjd_s: np.ndarray
    interval_s: np.ndarray
    exposure_s: np.ndarray
    data: np.ndarray
    model_data: np.ndarray
    flag: np.ndarray
    weight: np.ndarray
    sigma: np.ndarray
    channel_freq_hz: np.ndarray
    antenna_names: list[str]
    antenna_positions_m: np.ndarray
    source_name: str
    field_ra_rad: float
    field_dec_rad: float
    observation_date: str
    config_name: str | None = None
    array_type: str | None = None

    def as_dict(self) -> dict[str, Any]:
        """Convert to a plain dict for serialization."""
        return {
            "uvw_m": self.uvw_m,
            "antenna1": self.antenna1,
            "antenna2": self.antenna2,
            "time_mjd_s": self.time_mjd_s,
            "interval_s": self.interval_s,
            "exposure_s": self.exposure_s,
            "data": self.data,
            "model_data": self.model_data,
            "flag": self.flag,
            "weight": self.weight,
            "sigma": self.sigma,
            "channel_freq_hz": self.channel_freq_hz,
            "antenna_names": self.antenna_names,
            "antenna_positions_m": self.antenna_positions_m,
            "source_name": self.source_name,
            "field_ra_rad": self.field_ra_rad,
            "field_dec_rad": self.field_dec_rad,
            "observation_date": self.observation_date,
            "config_name": self.config_name,
            "array_type": self.array_type,
        }


def build_channel_visibility_rows(
    *,
    modelfft: np.ndarray,
    gains: np.ndarray,
    noise_samples: np.ndarray,
    u_waves: np.ndarray,
    v_waves: np.ndarray,
    w_waves: np.ndarray,
    antnum: np.ndarray,
    nH: int,
    Nphf: int,
    uv_pixsize: float,
    noise_std: float,
    mean_wavelength_m: float,
) -> dict[str, np.ndarray]:
    """Build per-channel row-wise visibility samples before UV gridding."""
    nbas = int(antnum.shape[0])
    nrows = int(nbas * nH)
    antenna1 = np.repeat(antnum[:, 0].astype(np.int32), nH)
    antenna2 = np.repeat(antnum[:, 1].astype(np.int32), nH)
    time_index = np.tile(np.arange(nH, dtype=np.int32), nbas)
    u_flat = np.asarray(u_waves, dtype=np.float32).reshape(nrows)
    v_flat = np.asarray(v_waves, dtype=np.float32).reshape(nrows)
    w_flat = np.asarray(w_waves, dtype=np.float32).reshape(nrows)
    gains_flat = np.asarray(gains, dtype=np.complex64).reshape(nrows)
    noise_flat = np.asarray(noise_samples, dtype=np.complex64).reshape(nrows)

    pix_u = np.rint(u_flat / uv_pixsize).astype(np.int32)
    pix_v = np.rint(v_flat / uv_pixsize).astype(np.int32)
    valid = (np.abs(pix_u) < Nphf) & (np.abs(pix_v) < Nphf)

    model_data = np.zeros(nrows, dtype=np.complex64)
    data = np.zeros(nrows, dtype=np.complex64)
    if np.any(valid):
        p_v = pix_v[valid] + Nphf
        m_u = -pix_u[valid] + Nphf
        # The gridding path samples model visibilities at (v, -u).
        model_vis_grid = np.fft.fftshift(modelfft)
        sampled = model_vis_grid[p_v, m_u].astype(np.complex64)
        model_data[valid] = sampled
        data[valid] = sampled * gains_flat[valid] + noise_flat[valid] * np.abs(gains_flat[valid])

    sigma = np.full(nrows, max(float(noise_std), 1e-12), dtype=np.float32)
    weight = np.full(nrows, 1.0 / np.square(sigma[0]), dtype=np.float32)
    uvw_m = np.stack(
        [
            u_flat * mean_wavelength_m,
            v_flat * mean_wavelength_m,
            w_flat * mean_wavelength_m,
        ],
        axis=1,
    ).astype(np.float64)

    return {
        "uvw_m": uvw_m,
        "antenna1": antenna1,
        "antenna2": antenna2,
        "time_index": time_index,
        "valid": valid.astype(np.bool_),
        "model_data": model_data,
        "data": data,
        "weight": weight,
        "sigma": sigma,
    }


def assemble_visibility_table(
    *,
    channel_rows: list[dict[str, np.ndarray]],
    channel_freq_hz: np.ndarray,
    scan_time_s: float,
    observation_date: str,
    antenna_names: list[str],
    antenna_positions_m: np.ndarray,
    source_name: str,
    field_ra_rad: float,
    field_dec_rad: float,
    config_name: str | None = None,
    array_type: str | None = None,
) -> VisibilityTable:
    """Assemble channel-wise row samples into an MS-like visibility table."""
    if not channel_rows:
        raise ValueError("channel_rows must not be empty")

    reference = channel_rows[0]
    nrows = int(reference["uvw_m"].shape[0])
    nchan = int(len(channel_rows))
    ncorr = 1
    data = np.zeros((nrows, ncorr, nchan), dtype=np.complex64)
    model_data = np.zeros_like(data)
    flag = np.ones((nrows, ncorr, nchan), dtype=np.bool_)
    weight_spectrum = np.zeros((nrows, ncorr, nchan), dtype=np.float32)
    sigma_spectrum = np.zeros((nrows, ncorr, nchan), dtype=np.float32)

    for chan_idx, rows in enumerate(channel_rows):
        data[:, 0, chan_idx] = rows["data"]
        model_data[:, 0, chan_idx] = rows["model_data"]
        flag[:, 0, chan_idx] = ~rows["valid"]
        weight_spectrum[:, 0, chan_idx] = rows["weight"]
        sigma_spectrum[:, 0, chan_idx] = rows["sigma"]

    start = Time(f"{observation_date}T00:00:00", format="isot", scale="utc")
    time_offsets_s = reference["time_index"].astype(np.float64) * float(scan_time_s)
    times_mjd_s = start.mjd * 86400.0 + time_offsets_s
    interval_s = np.full(nrows, float(scan_time_s), dtype=np.float64)
    exposure_s = np.full(nrows, float(scan_time_s), dtype=np.float64)

    # Collapse per-channel noise metadata to row-level MS WEIGHT/SIGMA.
    weight = np.mean(weight_spectrum, axis=2).astype(np.float32)
    sigma = np.mean(sigma_spectrum, axis=2).astype(np.float32)

    return VisibilityTable(
        uvw_m=np.asarray(reference["uvw_m"], dtype=np.float64),
        antenna1=np.asarray(reference["antenna1"], dtype=np.int32),
        antenna2=np.asarray(reference["antenna2"], dtype=np.int32),
        time_mjd_s=times_mjd_s,
        interval_s=interval_s,
        exposure_s=exposure_s,
        data=data,
        model_data=model_data,
        flag=flag,
        weight=weight,
        sigma=sigma,
        channel_freq_hz=np.asarray(channel_freq_hz, dtype=np.float64),
        antenna_names=list(antenna_names),
        antenna_positions_m=np.asarray(antenna_positions_m, dtype=np.float64),
        source_name=str(source_name),
        field_ra_rad=float(field_ra_rad),
        field_dec_rad=float(field_dec_rad),
        observation_date=str(observation_date),
        config_name=config_name,
        array_type=array_type,
    )


def concatenate_visibility_tables(tables: list[Mapping[str, Any]]) -> dict[str, Any]:
    """Concatenate multiple visibility tables across rows."""
    if not tables:
        raise ValueError("tables must not be empty")
    if len(tables) == 1:
        return dict(tables[0])

    first = tables[0]
    return {
        "uvw_m": np.concatenate([np.asarray(t["uvw_m"]) for t in tables], axis=0),
        "antenna1": np.concatenate([np.asarray(t["antenna1"]) for t in tables], axis=0),
        "antenna2": np.concatenate([np.asarray(t["antenna2"]) for t in tables], axis=0),
        "time_mjd_s": np.concatenate([np.asarray(t["time_mjd_s"]) for t in tables], axis=0),
        "interval_s": np.concatenate([np.asarray(t["interval_s"]) for t in tables], axis=0),
        "exposure_s": np.concatenate([np.asarray(t["exposure_s"]) for t in tables], axis=0),
        "data": np.concatenate([np.asarray(t["data"]) for t in tables], axis=0),
        "model_data": np.concatenate([np.asarray(t["model_data"]) for t in tables], axis=0),
        "flag": np.concatenate([np.asarray(t["flag"]) for t in tables], axis=0),
        "weight": np.concatenate([np.asarray(t["weight"]) for t in tables], axis=0),
        "sigma": np.concatenate([np.asarray(t["sigma"]) for t in tables], axis=0),
        "channel_freq_hz": np.asarray(first["channel_freq_hz"]),
        "antenna_names": list(first["antenna_names"]),
        "antenna_positions_m": np.asarray(first["antenna_positions_m"]),
        "source_name": first["source_name"],
        "field_ra_rad": float(first["field_ra_rad"]),
        "field_dec_rad": float(first["field_dec_rad"]),
        "observation_date": first["observation_date"],
        "config_name": first.get("config_name"),
        "array_type": first.get("array_type"),
    }
