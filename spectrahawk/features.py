"""RF Feature Extraction.

Extracts spectral and temporal features from complex IQ baseband data.
"""
import numpy as np
from scipy import signal as sps

def stft_rf(x, fs, n_fft=512, hop=256):
    """Complex STFT of RF data."""
    f, t, Zxx = sps.stft(x, fs=fs, nperseg=n_fft, noverlap=n_fft-hop, return_onesided=False)
    # Sort frequencies to be -fs/2 to fs/2
    sort_idx = np.argsort(f)
    return f[sort_idx], t, Zxx[sort_idx, :]

def psd_rf(x, fs, n_fft=1024):
    """Welch's PSD for RF data."""
    f, Pxx = sps.welch(x, fs=fs, nperseg=n_fft, return_onesided=False)
    sort_idx = np.argsort(f)
    return f[sort_idx], Pxx[sort_idx]

def feature_vector(x, fs):
    """Compact RF spectral feature vector.
    
    Includes:
    - Band power
    - Spectral centroid
    - Spectral flatness
    - Bandwidth occupancy (99% power bandwidth)
    """
    f, Pxx = psd_rf(x, fs)
    
    # Power
    total_power = np.sum(Pxx) + 1e-12
    
    # Centroid
    centroid = np.sum(f * Pxx) / total_power
    
    # Flatness (geometric mean / arithmetic mean)
    gmean = np.exp(np.mean(np.log(Pxx + 1e-12)))
    amean = np.mean(Pxx)
    flatness = gmean / (amean + 1e-12)
    
    # 99% bandwidth
    cum_power = np.cumsum(Pxx) / total_power
    idx_low = np.searchsorted(cum_power, 0.005)
    idx_high = np.searchsorted(cum_power, 0.995)
    bw_99 = f[idx_high] - f[idx_low] if idx_high < len(f) else fs
    
    # Simple band powers (sub-divide band into 4 quadrants)
    quads = np.array_split(Pxx, 4)
    band_powers = [np.sum(q) / total_power for q in quads]
    
    vec = [float(total_power), float(centroid), float(flatness), float(bw_99)] + band_powers
    return np.array(vec)


def feature_matrix_windows(X, fs, n_fft=1024):
    """Vectorized version of ``feature_vector`` for a 2D window matrix."""
    X = np.asarray(X)
    f, Pxx = sps.welch(X, fs=fs, nperseg=n_fft, return_onesided=False, axis=-1)
    sort_idx = np.argsort(f)
    f = f[sort_idx]
    Pxx = Pxx[:, sort_idx]

    total_power = np.sum(Pxx, axis=1) + 1e-12
    centroid = np.sum(Pxx * f[None, :], axis=1) / total_power

    gmean = np.exp(np.mean(np.log(Pxx + 1e-12), axis=1))
    amean = np.mean(Pxx, axis=1)
    flatness = gmean / (amean + 1e-12)

    cum_power = np.cumsum(Pxx, axis=1) / total_power[:, None]
    idx_low = np.argmax(cum_power >= 0.005, axis=1)
    has_high = cum_power[:, -1] >= 0.995
    idx_high = np.argmax(cum_power >= 0.995, axis=1)
    bw_99 = np.full(len(Pxx), fs, dtype=float)
    bw_99[has_high] = f[idx_high[has_high]] - f[idx_low[has_high]]

    band_powers = [
        np.sum(q, axis=1) / total_power
        for q in np.array_split(Pxx, 4, axis=1)
    ]

    return np.column_stack([total_power, centroid, flatness, bw_99, *band_powers])
