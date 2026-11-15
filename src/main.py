import os
import sys

# Importación del cargador desde dataset_loader.py
from dataset_loader import MendeleyIEMGDatasetLoader
from inference_engine import ALSInferenceEngine
from collections import defaultdict
from sklearn.metrics import confusion_matrix, roc_auc_score
import numpy as np

if __name__ == "__main__":
    # Ruta directa al repositorio descomprimido de Mendeley Data (usando r"" para evitar errores de escape en Windows)
    mendeley_dir = r"./Electromyographic data recordings during isometric contractions of the biceps brachii and deltoid collected from patients and healthy subjects/Dataset Reposistory/EMG Dataset Reposistory"

    print("======================================================================")
    print("      NEUROMYO DX - MÓDULO PRINCIPAL DE INTEGRACIÓN Y PRUEBA     ")
    print("======================================================================\n")

    # 1. Ingestión del banco de datos usando dataset_loader.py
    loader = MendeleyIEMGDatasetLoader(target_fs=4000, orig_fs=32768)
    dataset = loader.build_structured_dataset(mendeley_dir, binary_mode=True)

    print(f"\n[RESUMEN] Total de registros biopotenciales cargados: {len(dataset)}")

    if len(dataset) == 0:
        print("[ERROR] No se encontraron registros para procesar. Verifica la ruta del directorio.")
        sys.exit(1)

    # 2. Inicialización del Motor de Inferencia y Entrenamiento
    model_path = "./models/model_svm_neu.joblib"
    engine = ALSInferenceEngine(model_path=model_path)

    # 3. Entrenamiento Offline y Validación Cruzada Disjunta por Sujeto (LOSO)
    print("\n--- Paso 1: Ejecutando Entrenamiento y Validación Cruzada Disjunta (LOSO) ---")
    engine.train_and_validate_loso(dataset)
    def evaluate_patient_level_loso(subject_ids, y_true_windows, y_prob_windows, threshold=0.40):
        patient_probs = defaultdict(list)
        patient_labels = {}
    
        for subj, y_t, y_p in zip(subject_ids, y_true_windows, y_prob_windows):
            patient_probs[subj].append(y_p)
            patient_labels[subj] = y_t
    
        y_true_pat = []
        y_prob_pat = []
        y_pred_pat = []
    
        for subj, probs in patient_probs.items():
            mean_p = np.mean(probs)
            y_true_pat.append(patient_labels[subj])
            y_prob_pat.append(mean_p)
            y_pred_pat.append(1 if mean_p >= threshold else 0)
    
        tn, fp, fn, tp = confusion_matrix(y_true_pat, y_pred_pat).ravel()
        se = tp / (tp + fn)
        sp = tn / (tn + fp)
        acc = (tp + tn) / (tp + tn + fp + fn)
        auc = roc_auc_score(y_true_pat, y_prob_pat)
    
        print(f"\n=== RESULTADOS A NIVEL PACIENTE (AGREGACIÓN MULTIVENTANA) ===")
        print(f"Matriz de Confusión:\n[[{tn}  {fp}]\n [{fn}  {tp}]]")
        print(f"Sensibilidad (Se): {se*100:.2f}% | Especificidad (Sp): {sp*100:.2f}%")
        print(f"Precisión (Acc):   {acc*100:.2f}% | AUC-ROC: {auc:.4f}")

    # 4. Prueba del Modo Inferencia en Tiempo Real sobre el Modelo Serializado (.joblib)
    print("\n--- Paso 2: Verificación de Inferencia Directa (< 200 ms) ---")
    engine.load_pretrained_model()

    # Selección de una ventana de prueba (1000 ms = 4000 muestras) del primer sujeto
    sample_signal = dataset[0]['signal']
    test_window = sample_signal[0:4000]

   # Inferencia diagnóstica instantánea
    result = engine.predict_window(test_window)

    print("\n[RESULTADO DE INFERENCIA EN TIEMPO REAL]")
    print(f" • Paciente de Prueba:    {dataset[0]['subject_id']} ({dataset[0]['label_name']})")
    print(f" • Diagnóstico Emitido:   {result['class_name']}")
    print(f" • Confianza Diagnóstico: {result['confidence'] * 100:.2f}%")
    print(f" • Probabilidad de Neuropatía:   {result['als_probability'] * 100:.2f}%")
    print(f" • Latencia de Respuesta: {result['latency_ms']:.2f} ms")
    print("======================================================================")