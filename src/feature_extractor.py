import numpy as np
import scipy.signal as signal
import pywt
from numba import njit

# Algoritmo de Entropía de Muestra compilado Just-In-Time (JIT) a C
@njit(fastmath=True)
def _fast_sampen_kernel(window: np.ndarray, m: int, r: float) -> float:
    N = len(window)
    B = 0
    A = 0
    
    for i in range(N - m):
        for j in range(i + 1, N - m):
            # Distancia de Chebyshev para longitud m
            dist_m = 0.0
            for k in range(m):
                diff = abs(window[i + k] - window[j + k])
                if diff > dist_m:
                    dist_m = diff
            
            if dist_m <= r:
                B += 1
                # Evaluar para m + 1
                diff_m1 = abs(window[i + m] - window[j + m])
                if max(dist_m, diff_m1) <= r:
                    A += 1
                    
    if B == 0 or A == 0:
        return 0.0
        
    return -np.log(A / B)


class MultidomainFeatureExtractor:
    """
    Extractor de características optimizado en C mediante Numba (STFT + DWT db4 + SampEn JIT).
    """
    def __init__(self, fs: int = 4000):
        self.fs = fs

    def extract_stft_features(self, window: np.ndarray) -> list:
        nperseg = min(256, len(window))
        f, t, Zxx = signal.stft(window, fs=self.fs, window='hamming', nperseg=nperseg, noverlap=nperseg//2)
        psd = np.abs(Zxx) ** 2
        
        psd_mean = np.mean(psd, axis=1)
        total_power = np.sum(psd_mean) + 1e-12
        
        f_mean = np.sum(f * psd_mean) / total_power
        
        idx_band = np.where((f >= 50.0) & (f <= 180.0))[0]
        band_power_ratio = np.sum(psd_mean[idx_band]) / total_power
        
        return [f_mean, band_power_ratio]

    def extract_dwt_features(self, window: np.ndarray) -> list:
        coeffs = pywt.wavedec(window, wavelet='db4', level=4)
        dwt_features = []
        for c in coeffs:
            energy = np.sum(c ** 2) / (len(c) + 1e-12)
            std_dev = np.std(c)
            dwt_features.extend([energy, std_dev])
            
        return dwt_features

    def extract_sample_entropy(self, window: np.ndarray, m: int = 2, r_factor: float = 0.2) -> float:
        std_val = np.std(window)
        r = r_factor * std_val
        if r == 0:
            return 0.0
        # Llamada a la función acelerada por Numba
        return float(_fast_sampen_kernel(window, m, r))

    def extract_full_vector(self, window: np.ndarray) -> np.ndarray:
        stft_feat = self.extract_stft_features(window)
        dwt_feat = self.extract_dwt_features(window)
        samp_en = [self.extract_sample_entropy(window)]
        
        return np.concatenate([stft_feat, dwt_feat, samp_en])