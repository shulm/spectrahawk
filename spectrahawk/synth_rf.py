"""Synthetic RF data generator.

Simulates narrowband RF bursts (drone control/telemetry) with frequency hopping,
as well as background noise (Wi-Fi/Bluetooth/thermal).
"""
import numpy as np

def _generate_burst(fs, dur, f_center, bw, snr_db, rng):
    t = np.arange(int(fs * dur)) / fs
    # Baseband narrowband signal
    bb = rng.standard_normal(len(t)) + 1j * rng.standard_normal(len(t))
    # Lowpass filter to bw
    # Simple spectral shaping
    F = np.fft.fft(bb)
    freqs = np.fft.fftfreq(len(t), 1/fs)
    F[np.abs(freqs) > bw/2] = 0
    bb = np.fft.ifft(F)
    # Upconvert to f_center
    sig = bb * np.exp(1j * 2 * np.pi * f_center * t)
    
    # Add noise
    sig = sig / (np.std(sig) + 1e-12)
    noise_power = 10 ** (-snr_db / 10.0)
    noise = np.sqrt(noise_power/2) * (rng.standard_normal(len(t)) + 1j * rng.standard_normal(len(t)))
    return sig + noise

def generate_drone_rf(fs=20e6, dur=0.01, f_centers=[1e6, -2e6, 5e6], bw=500e3, snr_db=10, seed=None):
    """Simulates a drone RF signature (e.g. frequency hopping bursts)."""
    rng = np.random.default_rng(seed)
    n_samples = int(fs * dur)
    sig = np.zeros(n_samples, dtype=np.complex64)
    # Hop through centers
    hop_dur = dur / len(f_centers)
    for i, fc in enumerate(f_centers):
        start = int(i * hop_dur * fs)
        end = int((i+1) * hop_dur * fs)
        dur_actual = (end - start) / fs
        burst = _generate_burst(fs, dur_actual, fc, bw, snr_db, rng)
        sig[start:end] = burst
    return sig

def generate_background_rf(fs=20e6, dur=0.01, snr_db=10, seed=None):
    """Simulates background RF noise (thermal + wideband interferers)."""
    rng = np.random.default_rng(seed)
    # Thermal noise
    noise = rng.standard_normal(int(fs*dur)) + 1j * rng.standard_normal(int(fs*dur))
    # Wideband interferer (e.g. Wi-Fi)
    interf = _generate_burst(fs, dur, f_center=-5e6, bw=10e6, snr_db=0, rng=rng)
    sig = noise + 0.5 * interf
    sig = sig / (np.std(sig) + 1e-12)
    noise_power = 10 ** (-snr_db / 10.0)
    final_noise = np.sqrt(noise_power/2) * (rng.standard_normal(len(sig)) + 1j * rng.standard_normal(len(sig)))
    return sig + final_noise

def make_synthetic_dataset(n_per_class=50, fs=20e6, dur=0.01, n_classes=3, seed=0):
    """Generates synthetic RF data for testing.
    
    Returns
    -------
    X : list of complex ndarrays
    y : ndarray (0 = background, 1..N = drone types)
    groups : ndarray (source recording IDs to test leakage-aware splitting)
    fs : float
    """
    rng = np.random.default_rng(seed)
    X = []
    y = []
    groups = []
    
    group_id = 0
    # Class 0: Background
    for _ in range(n_per_class):
        snr = rng.uniform(-5, 15)
        # Simulate multiple windows from the same 'recording'
        n_windows = rng.integers(2, 5)
        for _ in range(n_windows):
            X.append(generate_background_rf(fs, dur, snr, seed=int(rng.integers(1e9))))
            y.append(0)
            groups.append(group_id)
        group_id += 1
        
    # Class 1..N: Drone types (overlapping frequencies to harden the problem)
    # Background has noise and a wideband interferer at -5e6.
    drone_profiles = [
        ([1e6, -2e6, 5e6], 500e3),     # Type 1
        ([1e6, -1.5e6, 4e6], 600e3),   # Type 2 (overlaps Type 1)
        ([4e6, -2e6], 800e3),          # Type 3 (overlaps Types 1 and 2)
        ([0], 5e6)                     # Type 4
    ]
    
    for c in range(1, min(n_classes, len(drone_profiles) + 1)):
        fcs, bw = drone_profiles[c-1]
        for _ in range(n_per_class):
            snr = rng.uniform(-10, 5)
            n_windows = rng.integers(2, 5)
            for _ in range(n_windows):
                X.append(generate_drone_rf(fs, dur, fcs, bw, snr, seed=int(rng.integers(1e9))))
                y.append(c)
                groups.append(group_id)
            group_id += 1
            
    return X, np.array(y), np.array(groups), fs
