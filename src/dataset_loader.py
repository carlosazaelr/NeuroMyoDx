import os
import glob
import re
import numpy as np
import scipy.interpolate as interpolate

# Intento de importación opcional de wfdb para registros de PhysioNet
try:
    import wfdb
    HAS_WFDB = True
except ImportError:
    HAS_WFDB = False


class MendeleyIEMGDatasetLoader:
    """
    Cargador especializado para el dataset de EMG de Mendeley Data (543xpjycj9.1),
    PhysioNet emgdb y EMGLAB N2001. Soporta formatos .asc, .dat y .hea.
    Estandariza las bioseñales a 4,000 Hz manteniendo la trazabilidad 
    del ID de paciente para validación LOSO.
    """
    def __init__(self, target_fs: int = 4000, orig_fs: int = 32768):
        self.target_fs = target_fs
        self.orig_fs = orig_fs

    def parse_filename_metadata(self, filepath: str):
        """
        Extrae el ID del paciente, músculo y etiqueta clínica desde el nombre del archivo.
        Ejemplo: 'EMG004_02_LD_Neu.asc' -> Patient: '02', Muscle: 'LD', Label: 1 (Neuropatía)
        """
        filename = os.path.basename(filepath)
        name_no_ext = os.path.splitext(filename)[0]
        parts = name_no_ext.split('_')
        
        # Extracción por patrones o posición
        patient_id = parts[1] if len(parts) >= 2 else "unk"
        muscle = parts[2] if len(parts) >= 3 else "unknown"
        class_str = parts[-1].lower() if len(parts) >= 4 else name_no_ext.lower()
        
        # Mapeo de Clases: 0 -> Sano (Hea), 1 -> Neuropatía (Neu), 2 -> Miopatía (Myo)
        if "hea" in class_str or "healthy" in class_str:
            label = 0
            label_name = "Control Sano"
        elif "neu" in class_str or "neuropathy" in class_str:
            label = 1
            label_name = "Neuropatía"
        elif "myo" in class_str or "myopathy" in class_str:
            label = 2
            label_name = "Miopatía"
        else:
            label = 1
            label_name = "Patológico"
            
        return {
            'subject_id': f"Patient_{patient_id}",
            'muscle': muscle,
            'label': label,
            'label_name': label_name,
            'filename': filename
        }

    def load_signal(self, filepath: str):
        """
        Carga unificada de bioseñales reconociendo automáticamente la extensión:
        - .asc / .txt: Archivos de texto plano con valores de amplitud.
        - .dat / .hea: Registros binarios/cabeceras PhysioNet WFDB o texto numérico.
        
        Retorna:
            tuple: (signal: np.ndarray, detected_fs: int)
        """
        ext = os.path.splitext(filepath)[1].lower()
        record_path = os.path.splitext(filepath)[0]
        
        signal = None
        detected_fs = self.orig_fs

        # 1. Intentar cargar mediante wfdb si es formato WFDB (.hea / .dat)
        if ext in ['.dat', '.hea'] and HAS_WFDB:
            try:
                record = wfdb.rdrecord(record_path)
                signal = record.p_signal[:, 0] if record.p_signal.ndim > 1 else record.p_signal.flatten()
                detected_fs = int(record.fs)
                return signal, detected_fs
            except Exception:
                pass  # Si falla wfdb (ej. si el .dat es texto plano), continuar al fallback

        # 2. Carga directa para texto plano (ASCII / np.loadtxt / parsing por columnas)
        try:
            signal = np.loadtxt(filepath)
            if signal.ndim > 1:
                signal = signal[:, 0]  # Si hay múltiples columnas, tomar la primera
        except Exception:
            # Lectura tolerante a fallos línea por línea (omite cabeceras de texto)
            data = []
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith(('#', '%', 'S', 'T', 'N', 'P', 'R')):
                        parts = line.split()
                        try:
                            data.append(float(parts[-1]))
                        except ValueError:
                            continue
            signal = np.array(data, dtype=float)

        if signal is None or len(signal) == 0:
            raise IOError(f"Error al leer la bioseñal desde {filepath}: No se encontraron datos numéricos válidos.")

        return signal, detected_fs

    def load_asc_signal(self, filepath: str) -> np.ndarray:
        """
        Mantiene compatibilidad hacia atrás con la interfaz existente y llamadas de la GUI.
        Retorna únicamente la señal como np.ndarray.
        """
        signal, _ = self.load_signal(filepath)
        return signal

    def resample_cubic_spline(self, signal: np.ndarray, orig_fs: int = None) -> np.ndarray:
        """
        Remuestra la señal a target_fs (4000 Hz por defecto) mediante Splines Cúbicos.
        Permite recibir orig_fs dinámicamente si difiere de self.orig_fs.
        """
        fs_in = orig_fs if orig_fs is not None else self.orig_fs
        
        if fs_in == self.target_fs:
            return signal
        
        duration = len(signal) / fs_in
        t_orig = np.linspace(0, duration, len(signal), endpoint=False)
        
        num_target_samples = int(np.round(duration * self.target_fs))
        t_target = np.linspace(0, duration, num_target_samples, endpoint=False)
        
        cs = interpolate.CubicSpline(t_orig, signal)
        return cs(t_target)

    def build_structured_dataset(self, data_root_dir: str, binary_mode: bool = True):
        """
        Escanea recursivamente el directorio buscando archivos .asc, .dat o .hea.
        Desduplica automáticamente pares .hea/.dat de PhysioNet.
        
        Parameters:
            data_root_dir (str): Ruta raíz del dataset descomprimido.
            binary_mode (bool): Si es True, filtra solo clases Control Sano (0) y Neuropatía (1).
        """
        dataset = []
        all_files = glob.glob(os.path.join(data_root_dir, "**", "*.*"), recursive=True)
        
        valid_files = []
        seen_records = set()

        for f in all_files:
            ext = os.path.splitext(f)[1].lower()
            base = os.path.splitext(f)[0]
            
            if ext == '.asc':
                valid_files.append(f)
            elif ext in ['.dat', '.hea']:
                if base not in seen_records:
                    seen_records.add(base)
                    # Preferir la ruta sin extensión o el .hea si existe
                    target_f = base + '.hea' if os.path.exists(base + '.hea') else base + '.dat'
                    valid_files.append(target_f)

        if not valid_files:
            print(f"[ADVERTENCIA] No se encontraron archivos de bioseñal (.asc, .dat, .hea) en {data_root_dir}")
            return dataset

        print(f"[INFO] Procesando {len(valid_files)} registros de bioseñal desde {data_root_dir}...")

        for filepath in valid_files:
            try:
                meta = self.parse_filename_metadata(filepath)
                
                # Filtrar para enfoque binario (Sano vs Neuropatía) si se requiere
                if binary_mode and meta['label'] == 2:
                    continue  # Omite miopatías para centrarse en Neuropatía
                
                raw_signal, detected_fs = self.load_signal(filepath)
                signal_4000hz = self.resample_cubic_spline(raw_signal, orig_fs=detected_fs)
                
                dataset.append({
                    'subject_id': meta['subject_id'],
                    'muscle': meta['muscle'],
                    'label': meta['label'],
                    'label_name': meta['label_name'],
                    'filename': meta['filename'],
                    'orig_fs': detected_fs,
                    'target_fs': self.target_fs,
                    'signal': signal_4000hz
                })
                print(f"[OK] {meta['filename']} | {meta['subject_id']} ({meta['label_name']}) | "
                      f"{len(raw_signal)} mstr ({detected_fs} Hz) -> {len(signal_4000hz)} mstr ({self.target_fs} Hz)")
            except Exception as e:
                print(f"[ERROR] Falló en {filepath}: {str(e)}")
                
        return dataset


# Test de verificación local
if __name__ == "__main__":
    mendeley_dir = "./Electromyographic data recordings during isometric contractions of the biceps brachii and deltoid collected from patients and healthy subjects/Dataset Reposistory/EMG Dataset Reposistory"
    
    loader = MendeleyIEMGDatasetLoader(target_fs=4000, orig_fs=32768)
    dataset = loader.build_structured_dataset(mendeley_dir, binary_mode=True)
    
    print(f"\n--- Resumen del Banco de Datos ---")
    print(f"Total de registros cargados: {len(dataset)}")