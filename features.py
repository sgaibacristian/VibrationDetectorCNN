import numpy as np

def _skew(x: np.ndarray) -> float:
    x = x.astype(np.float32)
    mu = float(np.mean(x))
    s = float(np.std(x) + 1e-8)
    return float(np.mean(((x - mu) / s) ** 3))

def _kurtosis(x: np.ndarray) -> float:
    x = x.astype(np.float32)
    mu = float(np.mean(x))
    s = float(np.std(x) + 1e-8)
    # kurtosis "excess" (minus 3) este comună în vibrații
    return float(np.mean(((x - mu) / s) ** 4) - 3.0)

def _zero_crossing_rate(x: np.ndarray) -> float:
    x = x.astype(np.float32)
    return float(np.mean(x[:-1] * x[1:] < 0))

def _band_energy(mag: np.ndarray, f0: float, f1: float) -> float:
    # mag: magnitudine FFT (rfft), f0/f1 în fracții din Nyquist (0..1)
    n = len(mag)
    i0 = int(np.clip(np.floor(f0 * (n - 1)), 0, n - 1))
    i1 = int(np.clip(np.ceil (f1 * (n - 1)), 0, n - 1))
    if i1 <= i0:
        return 0.0
    return float(np.sum(mag[i0:i1] ** 2))

def extract_features(window: np.ndarray) -> np.ndarray:
    """
    window: (256,) sau (256,1) – semnal normalizat sau brut; merge în ambele cazuri.
    Returnează un vector de features (float32).
    """
    x = window.squeeze().astype(np.float32)

    # --- time domain ---
    mean = float(np.mean(x))
    std = float(np.std(x))
    rms = float(np.sqrt(np.mean(x**2)))
    p2p = float(np.max(x) - np.min(x))
    abs_mean = float(np.mean(np.abs(x)))
    zcr = _zero_crossing_rate(x)
    sk = _skew(x)
    ku = _kurtosis(x)

    # --- frequency domain (FFT) ---
    # folosim rfft (semnal real) -> magnitudine
    fft = np.fft.rfft(x)
    mag = np.abs(fft).astype(np.float32)

    # evităm div0
    mag_sum = float(np.sum(mag) + 1e-8)

    # centroid spectral (index-based, fără frecvențe reale)
    idx = np.arange(len(mag), dtype=np.float32)
    centroid = float(np.sum(idx * mag) / mag_sum)

    # band energies pe benzi fracționale din Nyquist (0..1)
    e_low  = _band_energy(mag, 0.00, 0.10)
    e_mid  = _band_energy(mag, 0.10, 0.30)
    e_high = _band_energy(mag, 0.30, 0.60)
    e_vhi  = _band_energy(mag, 0.60, 1.00)

    # raport energie high/low (util pt defecte cu armonici)
    hl_ratio = float((e_high + e_vhi) / (e_low + 1e-8))

    feats = np.array([
        mean, std, rms, p2p, abs_mean, zcr, sk, ku,
        centroid, e_low, e_mid, e_high, e_vhi, hl_ratio
    ], dtype=np.float32)

    return feats

def featurize_dataset(X: np.ndarray) -> np.ndarray:
    """
    X: (N,256,1) sau (N,256)
    Return: (N, F)
    """
    return np.stack([extract_features(X[i]) for i in range(len(X))], axis=0)
