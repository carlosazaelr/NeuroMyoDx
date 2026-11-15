import os
import time
import numpy as np
import joblib

from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score

from preprocessor import IEMGPreprocessor
from feature_extractor import MultidomainFeatureExtractor
from clustering import UnsupervisedAnalyzer

class ALSInferenceEngine:
    """
    Motor principal: entrenamiento offline con validación LOSO, exportación del modelo
    e inferencia optimizada sin advertencias de Scikit-Learn.
    """
    def __init__(self, model_path: str = "./models/model_svm_neu.joblib"):
        self.model_path = model_path
        self.preprocessor = IEMGPreprocessor(fs=4000)
        self.extractor = MultidomainFeatureExtractor(fs=4000)
        self.analyzer = None
        self.model = None

    def build_feature_matrix_from_dataset(self, dataset: list, window_len_ms: int = 1000, overlap_ms: int = 500):
        X_list, y_list, groups_list = [], [], []
        win_samples = int((window_len_ms / 1000.0) * 4000)
        step_samples = int(((window_len_ms - overlap_ms) / 1000.0) * 4000)

        print("[INFO] Extrayendo características por ventanas temporales...")
        for sample in dataset:
            sig = sample['signal']
            label = sample['label']
            subj = sample['subject_id']

            num_wins = (len(sig) - win_samples) // step_samples + 1
            for i in range(num_wins):
                start = i * step_samples
                raw_win = sig[start : start + win_samples]
                
                proc_win = self.preprocessor.preprocess_window(raw_win)
                feat_vec = self.extractor.extract_full_vector(proc_win)
                
                X_list.append(feat_vec)
                y_list.append(label)
                groups_list.append(subj)

        return np.array(X_list), np.array(y_list), np.array(groups_list)

    def _create_calibrated_svm(self):
        """
        Crea un clasificador SVM calibrado evitando la depreciación de probability=True.
        """
        base_svm = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, class_weight='balanced')
        return CalibratedClassifierCV(estimator=base_svm, ensemble=False)

    def train_and_validate_loso(self, dataset: list):
        X, y, groups = self.build_feature_matrix_from_dataset(dataset)
        logo = LeaveOneGroupOut()
        
        y_true_all, y_pred_all, y_prob_all = [], [], []
        unique_groups = np.unique(groups)

        print(f"\n--- Ejecutando Validación LOSO sobre {len(unique_groups)} Sujetos ---")
        
        for fold_idx, (train_idx, test_idx) in enumerate(logo.split(X, y, groups=groups)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Reducción PCA silenciada durante las iteraciones
            analyzer_fold = UnsupervisedAnalyzer(variance_threshold=0.95)
            X_train_pca = analyzer_fold.fit_transform_pca(X_train, verbose=False)
            X_test_pca = analyzer_fold.transform_pca(X_test)
            
            svm_fold = self._create_calibrated_svm()
            svm_fold.fit(X_train_pca, y_train)
            
            preds = svm_fold.predict(X_test_pca)
            probs = svm_fold.predict_proba(X_test_pca)[:, 1]
            
            y_true_all.extend(y_test)
            y_pred_all.extend(preds)
            y_prob_all.extend(probs)

        # Cálculo de Métricas Diagnósticas
        cm = confusion_matrix(y_true_all, y_pred_all)
        TN, FP, FN, TP = cm.ravel()
        
        sens = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        spec = TN / (TN + FP) if (TN + FP) > 0 else 0.0
        vpp = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        vpn = TN / (TN + FN) if (TN + FN) > 0 else 0.0
        acc = accuracy_score(y_true_all, y_pred_all)
        auc = roc_auc_score(y_true_all, y_prob_all)

        print("\n================ RESULTADOS PRUEBA DIAGNÓSTICA (LOSO) ================")
        print(f"Matriz de Confusión:\n{cm}")
        print(f"Sensibilidad (Se):       {sens*100:.2f}%")
        print(f"Especificidad (Sp):      {spec*100:.2f}%")
        print(f"Valor Predictivo Pos.:  {vpp*100:.2f}%")
        print(f"Valor Predictivo Neg.:  {vpn*100:.2f}%")
        print(f"Precisión Global (Acc): {acc*100:.2f}%")
        print(f"Área Bajo la Curva ROC: {auc:.4f}")
        print("======================================================================")

        # Entrenamiento final con la totalidad de los datos y guardado
        print("\n[INFO] Entrenando modelo final calibrado...")
        self.analyzer = UnsupervisedAnalyzer(variance_threshold=0.95)
        X_pca_final = self.analyzer.fit_transform_pca(X, verbose=True)
        
        self.model = self._create_calibrated_svm()
        self.model.fit(X_pca_final, y)
        
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        joblib.dump({'model': self.model, 'analyzer': self.analyzer}, self.model_path)
        print(f"[OK] Modelo serializado con éxito en '{self.model_path}'.")

    def load_pretrained_model(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"No se encontró el modelo en '{self.model_path}'.")
        
        saved_objects = joblib.load(self.model_path)
        self.model = saved_objects['model']
        self.analyzer = saved_objects['analyzer']
        print(f"[OK] Modelo pre-entrenado cargado desde '{self.model_path}'.")

    def predict_window(self, raw_window: np.ndarray):
        t_start = time.time()
        
        proc_win = self.preprocessor.preprocess_window(raw_window)
        feat_vec = self.extractor.extract_full_vector(proc_win).reshape(1, -1)
        feat_pca = self.analyzer.transform_pca(feat_vec)
        
        prob_als = self.model.predict_proba(feat_pca)[0, 1]
        prediction_label = int(prob_als >= 0.38)
        
        # Confianza diagnóstica de la clase predicha
        confidence = prob_als if prediction_label == 1 else (1.0 - prob_als)
        
        elapsed_ms = (time.time() - t_start) * 1000.0
        
        return {
            'als_probability': prob_als,
            'prediction': prediction_label,
            'class_name': "NEUROPATÍA DETECTADA" if prediction_label == 1 else "CONTROL SANO",
            'confidence': confidence,
            'latency_ms': elapsed_ms
        }
    
    # Agrega este método dentro de la clase ALSInferenceEngine en src/inference_engine.py
    def predict_patient_signal(
        self,
        full_signal: np.ndarray,
        preprocessor,
        fs: int = 4000,
        window_ms: int = 1000,
        overlap_pct: int = 50,
        threshold: float = 0.40,
        max_windows: int = 15
    ) -> dict:
        """
        Evalúa múltiples ventanas solapadas de un mismo paciente y agrega sus probabilidades.
        
        Parameters:
            full_signal: Señal remuestreada completa (ej. 4000 Hz).
            preprocessor: Instancia de IEMGPreprocessor.
            fs: Frecuencia de muestreo (4000 Hz).
            window_ms: Tamaño de cada ventana en milisegundos.
            overlap_pct: Porcentaje de traslape entre ventanas (0 a 90%).
            threshold: Umbral de decisión calibrado (ej. 0.40 para mayor sensibilidad).
            max_windows: Límite máximo de ventanas distribuidas para mantener baja latencia.
        """
        t_start = time.perf_counter()
    
        win_len = int((window_ms / 1000.0) * fs)
        overlap_pct = max(0, min(overlap_pct, 90))
        step = max(1, int(win_len * (1.0 - overlap_pct / 100.0)))
    
        # Generar índices de inicio para todas las ventanas posibles
        if len(full_signal) <= win_len:
            start_indices = [0]
        else:
            start_indices = list(range(0, len(full_signal) - win_len + 1, step))
    
        # Muestrear uniformemente a lo largo de toda la grabación si hay demasiadas ventanas
        if len(start_indices) > max_windows:
            sampled_pos = np.linspace(0, len(start_indices) - 1, max_windows, dtype=int)
            start_indices = [start_indices[i] for i in sampled_pos]
    
        window_probs = []
        for start in start_indices:
            end = min(start + win_len, len(full_signal))
            raw_win = full_signal[start:end]
            
            if len(raw_win) < win_len:
                raw_win = np.pad(raw_win, (0, win_len - len(raw_win)), mode='constant')
    
            filt_win = preprocessor.preprocess_window(raw_win)
            res_win = self.predict_window(filt_win)
            window_probs.append(res_win['als_probability'])
    
        window_probs = np.array(window_probs)
    
        # 1. Agregación por Promedio de Probabilidades
        mean_prob = float(np.mean(window_probs))
    
        # 2. Votación Mayoritaria (porcentaje de ventanas que superan el umbral)
        positive_votes = int(np.sum(window_probs >= threshold))
        total_windows = len(window_probs)
        vote_ratio = positive_votes / total_windows
    
        # Decisión final combinada
        prediction = 1 if mean_prob >= threshold else 0
        confidence = mean_prob if prediction == 1 else (1.0 - mean_prob)
    
        latency_ms = (time.perf_counter() - t_start) * 1000.0
    
        # Mantiene las mismas llaves que predict_window para compatibilidad con Screen3 y el PDF
        return {
            'prediction': prediction,
            'als_probability': mean_prob,
            'confidence': float(confidence),
            'latency_ms': float(latency_ms),
            'windows_analyzed': total_windows,
            'positive_votes': positive_votes,
            'vote_ratio': float(vote_ratio)
        }