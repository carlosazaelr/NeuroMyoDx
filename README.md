# NeuroMyoDx v2.0 — Sistema CADx para Diagnóstico de Neuropatías mediante iEMG

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-SVM%20%7C%20PCA-orange?logo=scikit-learn&logoColor=white)
![Signal Processing](https://img.shields.io/badge/DSP-DWT%20%7C%20STFT%20%7C%20SampEn-teal)
![Platform](https://img.shields.io/badge/Platform-Windows%2064--bit-lightgrey?logo=windows)
![Status](https://img.shields.io/badge/Version-2.0-brightgreen)

**NeuroMyoDx v2.0** es una plataforma computacional de diagnóstico asistido por computadora (*Computer-Aided Diagnosis*, CADx) desarrollada en Python para la detección objetiva y automatizada de patrones neuropáticos compatibles con **Esclerosis Lateral Amiotrófica (ELA)** a partir de señales de **electromiografía intramuscular (iEMG)**.

El sistema transforma registros bioeléctricos complejos en indicadores cuantitativos y reproducibles mediante procesamiento digital de señales (DSP), extracción híbrida de características tiempo-frecuencia y aprendizaje automático supervisado, reduciendo la subjetividad y variabilidad interobservador propias de la inspección visual clínica tradicional.

---

## Características Principales

* **Ingestión Multiformato Unificada:** Soporte nativo para archivos electrofisiológicos `.asc`, `.dat` y `.hea`, con detección automática de metadatos y frecuencia de muestreo original.
* **Acondicionamiento Digital Determinista:** Filtrado pasabanda *Butterworth* ($20\text{--}500\text{ Hz}$), supresión de interferencia de red mediante filtro *Notch* ($50\text{ Hz}$) y estandarización temporal a $f_s = 4000\text{ Hz}$ por interpolación de *splines* cúbicos.
* **Análisis Tiempo-Frecuencia y Complejidad:**
  * Descomposición multirresolución mediante **Transformada Wavelet Discreta (DWT `db4`)** en subbandas de detalle (`D1: 1000-2000 Hz`, `D2: 500-1000 Hz`, `D3: 250-500 Hz`, `D4: 125-250 Hz`).
  * Espectrogramas 2D mediante **Transformada de Fourier de Tiempo Corto (STFT)**.
  * Cuantificación no lineal mediante **Entropía de Muestra (`SampEn`)**, frecuencia media ($F_{\text{mean}}$) y energía espectral total.
* **Inferencia Probabilística Multiventana:** Reducción de dimensionalidad mediante **PCA** ($95\%$ de varianza explicada) acoplada a un clasificador **SVM (*Kernel* RBF)** calibrado clínicamente ($\theta = 0.40$) que consolida las probabilidades de todas las ventanas de un registro para emitir un dictamen integral por paciente en menos de $200\text{ ms}$ por ventana.
* **Interfaz Gráfica y Reportes Clínicos PDF:** Aplicación de escritorio desarrollada en `Tkinter` con visualización interactiva de señales, espectrogramas, subbandas Wavelet y exportación automatizada de informes médicos en formato PDF.

---

## Estructura del Repositorio

```text
NeuroMyoDx/
├── assets/
│   └── icono.ico                 # Identidad gráfica multi-resolución de la aplicación
├── models/
│   └── model_svm_neu.joblib      # Pipeline entrenado y calibrado (Scaler + PCA + SVM)
├── reports/                      # Directorio de salida para informes clínicos generados en PDF
├── src/
│   ├── app_gui.py                # Interfaz gráfica de usuario (GUI) y flujo de navegación
│   ├── dataset_loader.py         # Cargador unificado de registros iEMG (.asc, .dat, .hea)
│   ├── preprocessor.py           # Filtrado digital, remuestreo a 4000 Hz y ventaneo (1000 ms)
│   ├── feature_extractor.py      # Extracción de descriptores DWT, STFT, SampEn y PCA
│   └── inference_engine.py       # Motor de predicción, calibración (theta=0.40) y agregación por sujeto
└── README.md
```

---

## Flujo de Procesamiento (*Pipeline* CADx)

1. **Carga y Estandarización:** Lectura de la serie temporal de voltaje ($\mu\text{V}$), remuestreo uniforme a $4000\text{ Hz}$ y segmentación en ventanas de $1000\text{ ms}$ con traslape (*overlap*) del $50\%$.
2. **Filtrado Digital:** Aplicación de filtro pasabanda *Butterworth* ($20\text{--}500\text{ Hz}$) y filtro muesca ($50\text{ Hz}$) para preservar la morfología de los Potenciales de Acción de Unidad Motora (MUAP).
3. **Extracción y Reducción de Rasgos:** Cálculo de coeficientes estadísticos/energéticos en subbandas DWT (`db4`), densidad espectral STFT y complejidad `SampEn`, proyectados mediante PCA reteniendo $\ge 95\%$ de varianza explicada.
4. **Clasificación y Dictamen:** Evaluación mediante SVM con estimación probabilística y umbral ajustado a $\theta = 0.40$, integrando el riesgo promedio a nivel de sujeto y generando el reporte diagnóstico en PDF.

---

## Validación y Desempeño Diagnóstico

El modelo fue evaluado bajo un esquema estricto de validación cruzada disjunto por sujeto (**Leave-One-Subject-Out — LOSO**), garantizando cero fuga de datos (*data leakage*) entre entrenamiento y prueba sobre una muestra clínica de **$N = 143$ sujetos** ($50$ controles sanos y $93$ pacientes con neuropatía) y **$n = 2,718$ ventanas** de análisis:

| Métrica Diagnóstica | Modelo Base ($\theta = 0.50$) | Modelo Calibrado ($\theta = 0.40$) | Variación ($\Delta$) |
| :--- | :---: | :---: | :---: |
| **Precisión Global ($Acc$)** | $88.74\%$ | **$89.22\%$** | $+0.48\%$ |
| **Sensibilidad ($S_e$)** | $76.25\%$ | **$79.08\%$** | $+2.83\%$ |
| **Especificidad ($S_p$)** | $95.11\%$ | **$94.39\%$** | $-0.72\%$ |
| **Valor Predictivo Positivo (VPP)** | $88.83\%$ | **$87.79\%$** | $-1.04\%$ |
| **Valor Predictivo Negativo (VPN)** | $88.70\%$ | **$89.85\%$** | $+1.15\%$ |
| **Área Bajo la Curva ROC (AUC)** | $0.9259$ | **$0.9244\%$** | $-0.0015$ |

---

## Instalación y Ejecución

### 1. Clonar el repositorio
```bash
git clone https://github.com/carlosazaelr/NeuroMyoDx.git
cd NeuroMyoDx
```

### 2. Crear entorno virtual e instalar dependencias
```bash
python -m venv venv
# En Windows:
venv\Scripts\activate
# En Linux/macOS:
source venv/bin/activate

pip install numpy scipy scikit-learn PyWavelets joblib matplotlib fpdf2
```

### 3. Ejecutar la aplicación desde el código fuente
```bash
python src/app_gui.py
```

### 4. Compilar ejecutable independiente para Windows (Opcional)
Para generar el paquete binario autónomo (`NeuroMyoDx_v2.exe`) mediante `PyInstaller`:

```bash
pyinstaller --clean --noconfirm --onedir --windowed --name "NeuroMyoDx_v2" ^
  --icon="assets\icono.ico" ^
  --paths "src" ^
  --add-data "models;models" ^
  --add-data "assets;assets" ^
  --exclude-module PyQt5 --exclude-module PyQt6 ^
  --exclude-module PySide2 --exclude-module PySide6 ^
  --collect-submodules sklearn --collect-submodules joblib ^
  --hidden-import="scipy.special._cdflib" ^
  --hidden-import="pywt._extensions._cwt" ^
  --hidden-import="matplotlib.backends.backend_tkagg" ^
  src/app_gui.py
```

---

## Conjunto de Datos Utilizado

El entrenamiento y validación experimental se realizaron utilizando registros de electromiografía intramuscular de contracciones isométricas en *biceps brachii* y *deltoides* procedentes del repositorio público:

> Messekher, A., Mekaoui, S., Kedir-Talha, M., Kediha, M. I., & Mostefaoui, F. (2023). *Electromyographic data recordings during isometric contractions of the biceps brachii and deltoid collected from patients and healthy subjects* (Version 1) [Data set]. Mendeley Data. https://doi.org/10.17632/543xpjycj9.1

---

## Citación

Si utilizas este código o metodología en tu investigación, por favor cítalo como:

```bibtex
@software{NeuroMyoDx_2026,
  author       = {Ramírez, Carlos Azael},
  title        = {NeuroMyoDx v2.0: Sistema CADx para Diagnóstico de Neuropatías mediante Electromiografía Intramuscular},
  year         = {2026},
  url          = {https://github.com/carlosazaelr/NeuroMyoDx}
}
```

---

## Aviso Clínico (*Disclaimer*)

**NeuroMyoDx v2.0** es un prototipo de investigación y desarrollo tecnológico diseñado estrictamente como **herramienta de apoyo a la decisión clínica (segunda opinión diagnóstica)**. Los resultados probabilísticos generados por el sistema no constituyen un diagnóstico médico definitivo y deben ser interpretados exclusivamente por un médico especialista en neurofisiología clínica en conjunto con la valoración integral del paciente.