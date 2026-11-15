import numpy as np
import scipy.signal as signal

class IEMGPreprocessor:
    """
    Módulo de filtrado y acondicionamiento digital para señales iEMG estandarizadas a 4000 Hz.
    """
    def __init__(self, fs: int = 4000):
        self.fs = fs
        # Diseño de Filtro Butterworth Pasabanda (20 - 500 Hz, Orden 4)
        self.b_band, self.a_band = signal.butter(
            N=4, 
            Wn=[20.0, 500.0], 
            btype='bandpass', 
            fs=self.fs
        )
        # Diseño de Filtro Muesca IIR Notch (50 Hz, Q = 30)
        self.b_notch, self.a_notch = signal.iirnotch(
            w0=50.0, 
            Q=30.0, 
            fs=self.fs
        )

    def apply_butterworth_bandpass(self, x: np.ndarray) -> np.ndarray:
        """
        Filtro Butterworth de fase cero (filtfilt) para eliminar artefactos de baja frecuencia 
        y ruido de alta frecuencia sin distorsión de fase.
        """
        return signal.filtfilt(self.b_band, self.a_band, x)

    def apply_notch_filter(self, x: np.ndarray) -> np.ndarray:
        """
        Filtro Notch IIR de fase cero (filtfilt) para atenuar la interferencia de 50 Hz.
        """
        return signal.filtfilt(self.b_notch, self.a_notch, x)

    def preprocess_window(self, raw_window: np.ndarray) -> np.ndarray:
        """
        Cadena completa de acondicionamiento digital sobre la ventana temporal.
        """
        # 1. Remoción de la componente continua (offset DC)
        detrended = raw_window - np.mean(raw_window)
        # 2. Filtrado pasabanda
        bandpassed = self.apply_butterworth_bandpass(detrended)
        # 3. Filtrado muesca (Notch)
        conditioned_signal = self.apply_notch_filter(bandpassed)
        return conditioned_signal