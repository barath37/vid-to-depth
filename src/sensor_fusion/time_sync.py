import numpy as np
from scipy import interpolate, signal


def estimate_time_offset(
    t_vo: np.ndarray,
    v_vo: np.ndarray,
    t_gnss: np.ndarray,
    v_gnss: np.ndarray,
    max_window_sec: float = 2.0,
    resample_hz: float = 50.0
) -> float:
    """Estimates sub-second clock offset between Visual Odometry and GNSS telemetry.
    Calculates Δt* = argmax_τ ∫ v_VO(t) * v_GNSS(t + τ) dt (§5 of Spec).
    Returns Δt in seconds (positive means GNSS is ahead of VO).
    """
    if len(t_vo) < 5 or len(t_gnss) < 5:
        return 0.0

    # Determine common overlapping time span
    t_min = max(t_vo[0], t_gnss[0])
    t_max = min(t_vo[-1], t_gnss[-1])
    if t_max - t_min <= max_window_sec:
        return 0.0

    dt = 1.0 / resample_hz
    t_grid = np.arange(t_min, t_max, dt)

    # Interpolate both speed signals onto regular grid
    f_vo = interpolate.interp1d(t_vo, v_vo, kind="linear", fill_value="extrapolate")
    f_gnss = interpolate.interp1d(t_gnss, v_gnss, kind="linear", fill_value="extrapolate")

    s_vo = f_vo(t_grid)
    s_gnss = f_gnss(t_grid)

    # Normalize signals (zero mean)
    s_vo_norm = s_vo - np.mean(s_vo)
    s_gnss_norm = s_gnss - np.mean(s_gnss)

    if np.std(s_vo_norm) < 1e-4 or np.std(s_gnss_norm) < 1e-4:
        return 0.0

    # Cross-correlation
    corr = signal.correlate(s_gnss_norm, s_vo_norm, mode="full")
    lags = signal.correlation_lags(len(s_gnss_norm), len(s_vo_norm), mode="full")

    # Limit search to [-max_window_sec, +max_window_sec]
    max_lags = int(max_window_sec * resample_hz)
    valid_mask = (lags >= -max_lags) & (lags <= max_lags)

    corr_windowed = corr[valid_mask]
    lags_windowed = lags[valid_mask]

    best_lag = lags_windowed[np.argmax(corr_windowed)]
    offset_sec = float(-best_lag * dt)

    return offset_sec
