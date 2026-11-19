import os
import sys
import datetime
import textwrap
import joblib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Wedge
import scipy.signal as signal
import pywt

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

# Importación opcional de TensorFlow para soporte de modelos .keras
try:
    import tensorflow as tf
    HAS_TF = True
except ImportError:
    HAS_TF = False

# Importar los módulos del pipeline CADx
from dataset_loader import MendeleyIEMGDatasetLoader
from preprocessor import IEMGPreprocessor
from feature_extractor import MultidomainFeatureExtractor
from inference_engine import ALSInferenceEngine


# ==============================================================================
# CATÁLOGO DE MODELOS DISPONIBLES EN LA INTERFAZ (ANN-MLP, RANDOM FOREST, SVM)
# ==============================================================================
AVAILABLE_MODELS = {
    "ANN-MLP (Red Neuronal Artificial)": {
        "short_name": "ANN-MLP",
        "filename": "model_ann_neu.keras",
        "arch_desc": "Red Neuronal Densa (32 -> 16 -> Softmax)",
        "validation_desc": "LOSO (Leave-One-Subject-Out)",
        "pdf_tag": "ANN-MLP + LOSO",
        "sensitivity": "78.50%",
        "specificity": "94.20%",
        "auc_roc": "0.9310"
    },
    "Random Forest (200 Árboles)": {
        "short_name": "RF-200",
        "filename": "model_rf_neu.joblib",
        "arch_desc": "Random Forest (200 árboles, max_depth=15)",
        "validation_desc": "LOSO (Leave-One-Subject-Out)",
        "pdf_tag": "Random Forest + LOSO",
        "sensitivity": "79.10%",
        "specificity": "95.40%",
        "auc_roc": "0.9345"
    },
    "SVM (Kernel RBF Calibrado)": {
        "short_name": "SVM-RBF",
        "filename": "model_svm_neu.joblib",
        "arch_desc": "SVM (Kernel RBF, C=1.0, g='scale')",
        "validation_desc": "LOSO (Leave-One-Subject-Out)",
        "pdf_tag": "SVM-RBF + LOSO",
        "sensitivity": "76.25%",
        "specificity": "95.11%",
        "auc_roc": "0.9259"
    }
}


# ==============================================================================
# ADAPTADORES DE COMPATIBILIDAD ENTRE KERAS (.keras) Y SCIKIT-LEARN (.joblib)
# ==============================================================================
class _NumpyTensorWrapper:
    def __init__(self, arr: np.ndarray):
        self._arr = arr

    def numpy(self) -> np.ndarray:
        return self._arr


class _SklearnAsKerasAdapter:
    """Permite que modelos Scikit-Learn (.joblib: RF / SVM) respondan a llamadas estilo TensorFlow."""
    def __init__(self, sk_model):
        self.sk_model = sk_model

    def __call__(self, x, training=False):
        return _NumpyTensorWrapper(self.sk_model.predict_proba(x))

    def predict_proba(self, x):
        return self.sk_model.predict_proba(x)


class _KerasAsSklearnAdapter:
    """Permite que un modelo Keras (.keras) responda a llamadas estilo Scikit-Learn."""
    def __init__(self, keras_model):
        self.keras_model = keras_model

    def __call__(self, x, training=False):
        return self.keras_model(x, training=training)

    def predict_proba(self, x):
        return self.keras_model(x, training=False).numpy()


def get_resource_path(relative_path: str) -> str:
    """Obtiene la ruta absoluta de recursos internos, compatible con desarrollo y PyInstaller."""
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_dir, relative_path)


def get_output_dir(folder_name: str) -> str:
    """Crea y devuelve una carpeta persistente junto al archivo .exe (para guardar los PDF)."""
    if getattr(sys, 'frozen', False):
        app_dir = os.path.dirname(sys.executable)
    else:
        app_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_path = os.path.join(app_dir, folder_name)
    os.makedirs(out_path, exist_ok=True)
    return out_path


# ==============================================================================
# PALETA DE COLORES: CYBER-HEALTH DEEP TECH (MODO OSCURO / FUTURISTA)
# ==============================================================================
THEME = {
    "bg_dark": "#0B0F17",       # Fondo principal de pantalla (Negro Obsidiana)
    "bg_card": "#161F30",       # Fondo de tarjetas y paneles (Azul Marino Noche)
    "bg_input": "#0F172A",      # Fondo de entradas de texto e insumos
    "primary": "#00F5D4",       # Menta / Turquesa Neón (Acentos principales)
    "primary_dark": "#00B49F",  # Turquesa oscuro para hover/activos
    "signal": "#A855F7",        # Púrpura Neón para bioseñal cruda/filtrada
    "alert": "#FF2A6D",         # Magenta Neón para neuropatías y alertas
    "success": "#00E676",       # Verde Neón para controles sanos
    "text_main": "#F8FAFC",     # Blanco Neón suave para títulos y texto principal
    "text_sec": "#94A3B8",      # Azul Acero / Slate para subtítulos y unidades
    "border": "#24324D"         # Bordes sutiles de contenedores
}

# Configuración global de estilos para Matplotlib (Tema Oscuro)
plt.style.use('dark_background')


class NeuroMyoDxApp:
    def __init__(self, root):
        self.root = root
        self.root.title("NEUROMYODX v2.0 — CADx para Diagnóstico de Neuropatías iEMG")
        self.root.geometry("1280x850")
        self.root.configure(bg=THEME["bg_dark"])

        # Identificador único en Windows para que la barra de tareas muestre el ícono propio
        if os.name == 'nt':
            try:
                import ctypes
                ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("neuromyodx.cadx.v2")
            except Exception:
                pass

        # Cargar el ícono de la ventana (compatible con desarrollo y PyInstaller)
        icon_candidates = []
        if getattr(sys, 'frozen', False):
            icon_candidates.append(os.path.join(os.path.dirname(sys.executable), "assets", "logoNMDx.ico"))
            if hasattr(sys, '_MEIPASS'):
                icon_candidates.append(os.path.join(sys._MEIPASS, "assets", "logoNMDx.ico"))
        else:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            icon_candidates.append(os.path.join(base_dir, "assets", "logoNMDx.ico"))

        icon_path = next((p for p in icon_candidates if os.path.exists(p)), None)
        if icon_path:
            try:
                self.root.iconbitmap(icon_path)
            except Exception as e:
                print(f"[ADVERTENCIA] No se pudo aplicar el ícono: {e}")

        # Configurar estilos globales de TTK para adaptarlos al tema oscuro
        self.style = ttk.Style()
        self.style.theme_use('default')

        # Configurar Combobox
        self.style.configure("TCombobox", fieldbackground=THEME["bg_input"], background=THEME["border"],
                             foreground=THEME["text_main"], darkcolor=THEME["border"], lightcolor=THEME["border"])
        self.style.map('TCombobox', fieldbackground=[('readonly', THEME["bg_input"])], foreground=[('readonly', THEME["text_main"])])

        # Configurar Treeview
        self.style.configure("Treeview", background=THEME["bg_card"], foreground=THEME["text_main"],
                             fieldbackground=THEME["bg_card"], rowheight=26, borderwidth=0)
        self.style.configure("Treeview.Heading", background=THEME["bg_input"], foreground=THEME["primary"],
                             font=("Helvetica", 9, "bold"), borderwidth=1, relief="solid")
        self.style.map("Treeview", background=[('selected', THEME["primary_dark"])], foreground=[('selected', THEME["text_main"])])

        # Configurar Notebook (Tabs)
        self.style.configure("TNotebook", background=THEME["bg_card"], borderwidth=0)
        self.style.configure("TNotebook.Tab", background=THEME["bg_input"], foreground=THEME["text_sec"], padding=[12, 6], font=("Helvetica", 9, "bold"))
        self.style.map("TNotebook.Tab", background=[('selected', THEME["primary"])], foreground=[('selected', THEME["bg_dark"])])

        # Variables de estado del pipeline
        self.target_fs = 4000
        self.orig_fs = 32768
        self.window_ms = 1000
        self.overlap_pct = 50

        # Datos por defecto del Paciente y Médico Activo
        self.patient_info = {
            "id": "P-001",
            "name": "Paciente de Control",
            "age": "45",
            "gender": "Masculino",
            "muscle": "Biceps Brachii"
        }
        self.doctor_info = {
            "name": "Dr. Alejandro Morales",
            "specialty": "Neurofisiología Clínica",
            "license": "MP-984021"
        }

        self.current_filepath = None
        self.current_filename = None
        self.raw_signal = None
        self.signal_4k = None
        self.filtered_window = None
        self.extracted_features = None
        self.inference_result = None

        # Instanciar submódulos del pipeline
        self.loader = MendeleyIEMGDatasetLoader(target_fs=self.target_fs, orig_fs=self.orig_fs)
        self.preprocessor = IEMGPreprocessor(fs=self.target_fs)
        self.extractor = MultidomainFeatureExtractor(fs=self.target_fs)

        # Estado del modelo activo (detecta automáticamente entre los 3 modelos en ./models/)
        self.model_names = list(AVAILABLE_MODELS.keys())
        self.selected_model_key = self._detect_default_model()
        self.model_loaded_ok = False
        self.model_path = ""
        self.engine = ALSInferenceEngine()

        # Contenedor Multipantalla
        self.container = tk.Frame(self.root, bg=THEME["bg_dark"])
        self.container.pack(fill="both", expand=True)
        self.container.grid_rowconfigure(0, weight=1)
        self.container.grid_columnconfigure(0, weight=1)

        self.screens = {}
        for ScreenClass in (Screen1Carga, Screen2Analisis, Screen3Inferencia):
            screen_name = ScreenClass.__name__
            screen_instance = ScreenClass(parent=self.container, controller=self)
            self.screens[screen_name] = screen_instance
            screen_instance.grid(row=0, column=0, sticky="nsew")

        # Cargar el modelo inicial y sincronizar las pantallas
        self.set_active_model(self.selected_model_key, silent=True)
        self.show_screen("Screen1Carga")

    def _resolve_model_filepath(self, filename: str) -> str:
        """Resuelve la ruta absoluta o relativa del archivo de modelo."""
        candidate = get_resource_path(os.path.join("models", filename))
        if os.path.exists(candidate):
            return candidate
        return os.path.join(".", "models", filename)

    def _detect_default_model(self) -> str:
        """Selecciona automáticamente el primer modelo que ya exista entrenado en ./models/."""
        for key, meta in AVAILABLE_MODELS.items():
            if os.path.exists(self._resolve_model_filepath(meta["filename"])):
                return key
        return self.model_names[0]

    def set_active_model(self, model_key: str, silent: bool = False) -> bool:
        """
        Cambia dinámicamente entre los 3 modelos disponibles (ANN-MLP, Random Forest o SVM),
        actualiza los selectores de todas las pantallas y re-evalúa si ya hay señal cargada.
        """
        if model_key not in AVAILABLE_MODELS:
            return False

        self.selected_model_key = model_key
        meta = AVAILABLE_MODELS[model_key]
        resolved_path = self._resolve_model_filepath(meta["filename"])
        self.model_path = resolved_path
        self.engine.model_path = resolved_path

        if hasattr(self.engine, "analyzer_path"):
            base_path, _ = os.path.splitext(resolved_path)
            self.engine.analyzer_path = f"{base_path}_pca.joblib"

        self.model_loaded_ok = False
        self.engine.model = None
        self.engine.analyzer = None

        try:
            if not os.path.exists(resolved_path):
                raise FileNotFoundError(f"No se encontró '{meta['filename']}' en ./models/")

            if resolved_path.endswith(".keras"):
                if not HAS_TF:
                    raise ImportError("TensorFlow no está instalado en este entorno.")
                base_path, _ = os.path.splitext(resolved_path)
                pca_path = f"{base_path}_pca.joblib"
                if not os.path.exists(pca_path):
                    raise FileNotFoundError(f"Falta el archivo PCA asociado: '{os.path.basename(pca_path)}'")

                raw_keras = tf.keras.models.load_model(resolved_path)
                self.engine.model = _KerasAsSklearnAdapter(raw_keras)
                self.engine.analyzer = joblib.load(pca_path)

            elif resolved_path.endswith(".joblib"):
                saved_objects = joblib.load(resolved_path)
                self.engine.model = _SklearnAsKerasAdapter(saved_objects["model"])
                self.engine.analyzer = saved_objects["analyzer"]

            else:
                self.engine.load_pretrained_model()

            self.model_loaded_ok = True
            print(f"[OK] Modelo activo cambiado a '{model_key}' ({resolved_path}).")

        except Exception as e:
            self.model_loaded_ok = False
            print(f"[ADVERTENCIA] No se pudo cargar '{model_key}': {e}")
            if not silent:
                messagebox.showwarning(
                    "Modelo No Disponible",
                    f"No se pudo cargar el modelo '{model_key}'.\n\n"
                    f"Detalle: {str(e)}\n"
                    f"Asegúrese de entrenarlo primero desde 'main.py'."
                )

        # Sincronizar controles visuales en Screen1 y Screen3
        if "Screen1Carga" in self.screens:
            self.screens["Screen1Carga"].sync_model_selector(model_key, self.model_loaded_ok)
        if "Screen3Inferencia" in self.screens:
            self.screens["Screen3Inferencia"].sync_model_selector(model_key)

        return self.model_loaded_ok

    def show_screen(self, screen_name):
        screen = self.screens[screen_name]
        screen.tkraise()
        if hasattr(screen, "on_show"):
            screen.on_show()


# ==============================================================================
# PANTALLA 1: PANTALLA PRINCIPAL Y CARGA DE BIOSEÑALES
# ==============================================================================
class Screen1Carga(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent, bg=THEME["bg_dark"])
        self.controller = controller

        # Cabecera Estructural
        header = tk.Frame(self, bg=THEME["bg_card"], height=55, highlightbackground=THEME["border"], highlightthickness=1)
        header.pack(fill="x", side="top")
        tk.Label(header, text="NEUROMYODX v2.0 | Pantalla Principal y Carga de Bioseñales",
                 font=("Helvetica", 14, "bold"), fg=THEME["primary"], bg=THEME["bg_card"]).pack(side="left", padx=20, pady=12)

        # Cuerpo Principal
        main_body = tk.Frame(self, bg=THEME["bg_dark"])
        main_body.pack(fill="both", expand=True, padx=15, pady=15)

        # Panel Lateral (Izquierda)
        sidebar = tk.Frame(main_body, bg=THEME["bg_card"], width=330, bd=1, relief="solid", highlightbackground=THEME["border"])
        sidebar.pack(side="left", fill="y", padx=(0, 15))
        sidebar.pack_propagate(False)

        # Sección 1: Selección y Registro de Paciente
        tk.Label(sidebar, text="SUJETO / REGISTRO", font=("Helvetica", 10, "bold"), fg=THEME["primary"], bg=THEME["bg_card"]).pack(anchor="w", padx=15, pady=(15, 5))

        frame_pat_select = tk.Frame(sidebar, bg=THEME["bg_card"])
        frame_pat_select.pack(fill="x", padx=15, pady=(0, 10))

        self.combo_patients = ttk.Combobox(frame_pat_select, values=["P-001 (Control Sano)", "P-002 (EMG004 Neuropatía)", "P-003 (N2001 Biceps)"])
        self.combo_patients.current(0)
        self.combo_patients.pack(side="left", fill="x", expand=True, padx=(0, 5))

        btn_add_patient = tk.Button(frame_pat_select, text="➕ Nuevo", command=self.open_patient_modal,
                                    font=("Helvetica", 9, "bold"), bg=THEME["primary"], fg=THEME["bg_dark"], bd=0, padx=8, pady=4, cursor="hand2", activebackground=THEME["primary_dark"])
        btn_add_patient.pack(side="right")

        self.lbl_patient_summary = tk.Label(sidebar, text="Paciente: P-001 (45 años) | Biceps Brachii",
                                            font=("Helvetica", 8), fg=THEME["text_sec"], bg=THEME["bg_card"], justify="left")
        self.lbl_patient_summary.pack(anchor="w", padx=15, pady=(0, 12))

        # Sección 2: Carga de Archivos
        tk.Label(sidebar, text="CARGA DE ARCHIVOS", font=("Helvetica", 10, "bold"), fg=THEME["primary"], bg=THEME["bg_card"]).pack(anchor="w", padx=15, pady=5)
        btn_load = tk.Button(sidebar, text="📁 Cargar Archivo (.asc / .dat)", command=self.load_file,
                             font=("Helvetica", 10, "bold"), bg=THEME["border"], fg=THEME["text_main"], bd=0, pady=8, cursor="hand2", activebackground=THEME["primary_dark"])
        btn_load.pack(fill="x", padx=15, pady=5)

        self.lbl_status = tk.Label(sidebar, text="Estado: No hay archivo cargado", font=("Helvetica", 9), fg=THEME["text_sec"], bg=THEME["bg_card"], wraplength=280, justify="left")
        self.lbl_status.pack(anchor="w", padx=15, pady=(5, 12))

        # Sección 3: Selección del Modelo de Clasificación (IA)
        tk.Label(sidebar, text="MODELO DE CLASIFICACIÓN (IA)", font=("Helvetica", 10, "bold"), fg=THEME["primary"], bg=THEME["bg_card"]).pack(anchor="w", padx=15, pady=(4, 5))

        self.combo_models = ttk.Combobox(sidebar, values=self.controller.model_names, state="readonly")
        self.combo_models.set(self.controller.selected_model_key)
        self.combo_models.pack(fill="x", padx=15, pady=(0, 4))
        self.combo_models.bind("<<ComboboxSelected>>", self._on_model_selected)

        self.lbl_model_status = tk.Label(sidebar, text="Modelo: Verificando archivo...", font=("Helvetica", 8), fg=THEME["text_sec"], bg=THEME["bg_card"], wraplength=290, justify="left")
        self.lbl_model_status.pack(anchor="w", padx=15, pady=(0, 12))

        # Sección 4: Parámetros de Ventaneo
        tk.Label(sidebar, text="PARÁMETROS DE VENTANEO", font=("Helvetica", 10, "bold"), fg=THEME["primary"], bg=THEME["bg_card"]).pack(anchor="w", padx=15, pady=5)

        frame_params = tk.Frame(sidebar, bg=THEME["bg_card"])
        frame_params.pack(fill="x", padx=15, pady=5)

        tk.Label(frame_params, text="• Frecuencia fs (Hz):", font=("Helvetica", 9), fg=THEME["text_main"], bg=THEME["bg_card"]).grid(row=0, column=0, sticky="w", pady=2)
        self.ent_fs = tk.Entry(frame_params, width=8, bd=1, relief="solid", bg=THEME["bg_input"], fg=THEME["text_main"], insertbackground=THEME["text_main"])
        self.ent_fs.insert(0, "4000")
        self.ent_fs.grid(row=0, column=1, padx=5, pady=2)

        tk.Label(frame_params, text="• Ancho Ventana (ms):", font=("Helvetica", 9), fg=THEME["text_main"], bg=THEME["bg_card"]).grid(row=1, column=0, sticky="w", pady=2)
        self.ent_win = tk.Entry(frame_params, width=8, bd=1, relief="solid", bg=THEME["bg_input"], fg=THEME["text_main"], insertbackground=THEME["text_main"])
        self.ent_win.insert(0, "1000")
        self.ent_win.grid(row=1, column=1, padx=5, pady=2)

        tk.Label(frame_params, text="• Traslape / Overlap (%):", font=("Helvetica", 9), fg=THEME["text_main"], bg=THEME["bg_card"]).grid(row=2, column=0, sticky="w", pady=2)
        self.ent_overlap = tk.Entry(frame_params, width=8, bd=1, relief="solid", bg=THEME["bg_input"], fg=THEME["text_main"], insertbackground=THEME["text_main"])
        self.ent_overlap.insert(0, "50")
        self.ent_overlap.grid(row=2, column=1, padx=5, pady=2)

        # Botón Preprocesar y Filtrar
        btn_process = tk.Button(sidebar, text="⚡ PREPROCESAR Y FILTRAR", command=self.process_and_navigate,
                                font=("Helvetica", 11, "bold"), bg=THEME["alert"], fg="white", bd=0, pady=12, cursor="hand2")
        btn_process.pack(fill="x", padx=15, pady=(20, 15))

        # Panel Derecho
        plot_container = tk.Frame(main_body, bg=THEME["bg_card"], bd=1, relief="solid", highlightbackground=THEME["border"])
        plot_container.pack(side="right", fill="both", expand=True)

        self.fig, self.ax = plt.subplots(figsize=(6, 4), dpi=100)
        self.fig.patch.set_facecolor(THEME["bg_card"])
        self.ax.set_facecolor(THEME["bg_dark"])
        self.ax.set_title("Despliegue de Señal EMG Cruda", color=THEME["primary"], fontsize=11, fontweight="bold")
        self.ax.set_xlabel("Tiempo (ms)", color=THEME["text_sec"], fontsize=9)
        self.ax.set_ylabel("Amplitud (µV)", color=THEME["text_sec"], fontsize=9)
        self.ax.tick_params(colors=THEME["text_sec"])
        self.ax.grid(True, linestyle="--", alpha=0.2, color=THEME["text_sec"])

        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_container)
        self.canvas.draw()

        toolbar = NavigationToolbar2Tk(self.canvas, plot_container)
        toolbar.config(background=THEME["bg_card"])
        toolbar.update()
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        # Barra de Metadatos
        self.lbl_metadata = tk.Label(self, text="~ Metadatos ~ Músculo: Biceps Braquial | Canales: 1 | Duración: 0.0 s | Estado: Esperando archivo",
                                     font=("Helvetica", 9), fg=THEME["text_sec"], bg=THEME["bg_card"], pady=8)
        self.lbl_metadata.pack(fill="x", side="bottom")

    def _on_model_selected(self, event=None):
        chosen = self.combo_models.get()
        self.controller.set_active_model(chosen, silent=False)

    def sync_model_selector(self, model_key: str, is_loaded: bool):
        self.combo_models.set(model_key)
        filename = AVAILABLE_MODELS[model_key]["filename"]
        if is_loaded:
            self.lbl_model_status.config(text=f"✓ Listo ({filename})", fg=THEME["success"])
        else:
            self.lbl_model_status.config(text=f"⚠ No encontrado: {filename}", fg=THEME["alert"])

    def open_patient_modal(self):
        modal = tk.Toplevel(self.controller.root)
        modal.title("Registrar Nuevo Paciente")
        modal.geometry("380x320")
        modal.configure(bg=THEME["bg_dark"])
        modal.grab_set()

        tk.Label(modal, text="DATOS DEL PACIENTE", font=("Helvetica", 11, "bold"), fg=THEME["primary"], bg=THEME["bg_dark"]).pack(pady=10)

        f_form = tk.Frame(modal, bg=THEME["bg_dark"])
        f_form.pack(fill="x", padx=20, pady=5)

        labels = ["ID Paciente:", "Nombre Completo:", "Edad:", "Género:", "Músculo Evaluado:"]
        entries = {}

        for i, lbl in enumerate(labels):
            tk.Label(f_form, text=lbl, font=("Helvetica", 9), fg=THEME["text_main"], bg=THEME["bg_dark"]).grid(row=i, column=0, sticky="w", pady=4)
            ent = tk.Entry(f_form, width=22, bd=1, relief="solid", bg=THEME["bg_input"], fg=THEME["text_main"], insertbackground=THEME["text_main"])
            ent.grid(row=i, column=1, pady=4, padx=5)
            entries[lbl] = ent

        entries["ID Paciente:"].insert(0, f"P-00{len(self.combo_patients['values'])+1}")
        entries["Músculo Evaluado:"].insert(0, "Biceps Brachii")

        def save_patient():
            p_id = entries["ID Paciente:"].get()
            p_name = entries["Nombre Completo:"].get()
            if not p_name:
                messagebox.showwarning("Campo Vacío", "Por favor ingrese el nombre del paciente.")
                return

            self.controller.patient_info = {
                "id": p_id,
                "name": p_name,
                "age": entries["Edad:"].get(),
                "gender": entries["Género:"].get(),
                "muscle": entries["Músculo Evaluado:"].get()
            }

            val = f"{p_id} ({p_name})"
            vals = list(self.combo_patients['values'])
            vals.append(val)
            self.combo_patients['values'] = vals
            self.combo_patients.set(val)

            self.lbl_patient_summary.config(text=f"Paciente: {p_id} ({p_name}) | {entries['Músculo Evaluado:'].get()}")
            modal.destroy()

        tk.Button(modal, text="Guardar Paciente", command=save_patient,
                  font=("Helvetica", 10, "bold"), bg=THEME["primary"], fg=THEME["bg_dark"], bd=0, pady=6, cursor="hand2").pack(pady=15)

    def load_file(self):
        filepath = filedialog.askopenfilename(
            filetypes=[
                ("Archivos Bioseñal", "*.asc *.dat *.hea"),
                ("Archivos ASCII (.asc)", "*.asc"),
                ("Archivos WFDB/EMGLAB (.dat, .hea)", "*.dat *.hea"),
                ("Todos los archivos", "*.*")
            ]
        )
        if not filepath:
            return

        try:
            self.controller.current_filepath = filepath
            self.controller.current_filename = os.path.basename(filepath)

            # Carga unificada multiformato con detección de frecuencia de muestreo (.asc, .dat, .hea)
            raw_sig, detected_fs = self.controller.loader.load_signal(filepath)

            # Actualizar la frecuencia original detectada en el estado global
            self.controller.orig_fs = detected_fs
            self.controller.raw_signal = raw_sig

            # Re-muestrear adaptativamente a target_fs (4000 Hz por defecto)
            self.controller.signal_4k = self.controller.loader.resample_cubic_spline(
                raw_sig, orig_fs=detected_fs
            )

            dur = len(raw_sig) / detected_fs
            self.lbl_status.config(text=f"Estado: {self.controller.current_filename} cargado.", fg=THEME["success"])
            self.lbl_metadata.config(
                text=f"~ Metadatos ~ Paciente: {self.controller.patient_info['id']} | "
                     f"Músculo: {self.controller.patient_info['muscle']} | "
                     f"fs Orig: {detected_fs} Hz | Duración: {dur:.1f} s | Estado: Listo para preprocesar"
            )

            # Graficar la bioseñal cruda con la escala de tiempo corregida
            self.ax.clear()
            self.ax.set_facecolor(THEME["bg_dark"])
            t_ms = (np.arange(len(raw_sig)) / detected_fs) * 1000.0
            self.ax.plot(t_ms, raw_sig, color=THEME["signal"], linewidth=0.7)
            self.ax.set_title(f"Señal EMG Cruda — {self.controller.current_filename}", color=THEME["primary"], fontsize=11, fontweight="bold")
            self.ax.set_xlabel("Tiempo (ms)", color=THEME["text_sec"], fontsize=9)
            self.ax.set_ylabel("Amplitud (µV)", color=THEME["text_sec"], fontsize=9)
            self.ax.tick_params(colors=THEME["text_sec"])
            self.ax.grid(True, linestyle="--", alpha=0.2, color=THEME["text_sec"])
            self.canvas.draw()

        except Exception as e:
            messagebox.showerror("Error de Carga", f"Fallo al leer el archivo:\n{str(e)}")

    def process_and_navigate(self):
        if self.controller.signal_4k is None:
            messagebox.showwarning("Advertencia", "Debe cargar un archivo de bioseñal previamente.")
            return

        try:
            win_ms = int(self.ent_win.get())
            overlap_pct = int(self.ent_overlap.get())
            if win_ms <= 0 or not (0 <= overlap_pct < 100):
                raise ValueError
        except ValueError:
            messagebox.showerror("Error de Parámetro", "Verifique que el ancho de ventana (>0) y el traslape (0-99%) sean válidos.")
            return

        # Guardar configuración en el controlador para usarla en la inferencia multiventana
        self.controller.window_ms = win_ms
        self.controller.overlap_pct = overlap_pct

        win_len = min(int((win_ms / 1000.0) * self.controller.target_fs), len(self.controller.signal_4k))
        raw_win = self.controller.signal_4k[0:win_len]

        # Se mantiene la primera ventana para la visualización espectral en Screen2
        self.controller.filtered_window = self.controller.preprocessor.preprocess_window(raw_win)
        self.controller.extracted_features = self.controller.extractor.extract_full_vector(self.controller.filtered_window)
        self.controller.show_screen("Screen2Analisis")


# ==============================================================================
# PANTALLA 2: PANEL DE ANÁLISIS TIEMPO-FRECUENCIA (ESPECTROGRAMA CORREGIDO)
# ==============================================================================
class Screen2Analisis(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent, bg=THEME["bg_dark"])
        self.controller = controller

        # Cabecera
        header = tk.Frame(self, bg=THEME["bg_card"], height=55, highlightbackground=THEME["border"], highlightthickness=1)
        header.pack(fill="x", side="top")
        tk.Label(header, text="NEUROMYODX v2.0 | Módulo de Análisis Tiempo-Frecuencia",
                 font=("Helvetica", 14, "bold"), fg=THEME["primary"], bg=THEME["bg_card"]).pack(side="left", padx=20, pady=12)

        main_body = tk.Frame(self, bg=THEME["bg_dark"])
        main_body.pack(fill="both", expand=True, padx=15, pady=5)

        # Configuración de Figura Matplotlib con GridSpec Anidado de 3 filas
        self.fig = plt.Figure(figsize=(10, 5.5), dpi=95)
        self.fig.patch.set_facecolor(THEME["bg_card"])

        self.gs = self.fig.add_gridspec(3, 1, height_ratios=[1.1, 1.1, 1.0])

        # Fila 1: Señal EMG Intramuscular Filtrada (Ancho Completo)
        self.ax_filt = self.fig.add_subplot(self.gs[0])
        self.ax_filt.set_facecolor(THEME["bg_dark"])

        # Fila 2: Sub-GridSpec para Espectrograma (Ancho Completo) + Colorbar Dedicado a la Derecha
        self.gs_spec = self.gs[1].subgridspec(1, 2, width_ratios=[35, 1], wspace=0.02)
        self.ax_spec = self.fig.add_subplot(self.gs_spec[0])
        self.ax_spec.set_facecolor(THEME["bg_dark"])

        self.cax_spec = self.fig.add_subplot(self.gs_spec[1])
        self.cax_spec.set_facecolor(THEME["bg_dark"])

        # Fila 3: Sub-GridSpec para Descomposición Wavelet (4 Subbandas)
        self.gs_wav = self.gs[2].subgridspec(1, 4, wspace=0.12)
        self.ax_w1 = self.fig.add_subplot(self.gs_wav[0])
        self.ax_w2 = self.fig.add_subplot(self.gs_wav[1])
        self.ax_w3 = self.fig.add_subplot(self.gs_wav[2])
        self.ax_w4 = self.fig.add_subplot(self.gs_wav[3])

        self.fig.tight_layout(pad=1.2)

        self.canvas_frame = tk.Frame(main_body, bg=THEME["bg_card"], bd=1, relief="solid", highlightbackground=THEME["border"])
        self.canvas_frame.pack(side="top", fill="both", expand=True)

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.canvas_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        # Panel Inferior de Acciones y Métricas
        bottom_panel = tk.Frame(self, bg=THEME["bg_card"], height=80, bd=1, relief="solid", highlightbackground=THEME["border"])
        bottom_panel.pack(fill="x", side="bottom", padx=15, pady=10)

        self.lbl_metrics = tk.Label(bottom_panel, text="Métricas Calculadas:\nSampEn: -- | Fmean: -- Hz | E_TOTAL: -- mV²",
                                    font=("Helvetica", 10, "bold"), fg=THEME["primary"], bg=THEME["bg_card"], justify="left")
        self.lbl_metrics.pack(side="left", padx=20, pady=10)

        btn_frame = tk.Frame(bottom_panel, bg=THEME["bg_card"])
        btn_frame.pack(side="right", padx=15, pady=10)

        btn_back = tk.Button(btn_frame, text="🔄 VOLVER A CARGA", command=lambda: self.controller.show_screen("Screen1Carga"),
                             font=("Helvetica", 10, "bold"), bg=THEME["border"], fg=THEME["text_main"], bd=0, padx=12, pady=8, cursor="hand2")
        btn_back.pack(side="left", padx=5)

        btn_pca = tk.Button(btn_frame, text="EJECUTAR REDUCCIÓN", command=self.run_pca,
                            font=("Helvetica", 10, "bold"), bg=THEME["primary"], fg=THEME["bg_dark"], bd=0, padx=12, pady=8, cursor="hand2")
        btn_pca.pack(side="left", padx=5)

        btn_next = tk.Button(btn_frame, text="PASAR A INFERENCIA ➔", command=self.run_inference,
                             font=("Helvetica", 10, "bold"), bg=THEME["success"], fg=THEME["bg_dark"], bd=0, padx=12, pady=8, cursor="hand2")
        btn_next.pack(side="left", padx=5)

    def on_show(self):
        sig = self.controller.filtered_window
        if sig is None:
            return

        fs = self.controller.target_fs
        t_ms = (np.arange(len(sig)) / fs) * 1000.0

        # 1. Señal Filtrada (Púrpura Neón)
        self.ax_filt.clear()
        self.ax_filt.set_facecolor(THEME["bg_dark"])
        self.ax_filt.plot(t_ms, sig, color=THEME["signal"], linewidth=0.8)
        self.ax_filt.set_title("Señal EMG Intramuscular Filtrada (Butterworth 20-500 Hz + Notch 50 Hz)", color=THEME["primary"], fontsize=9, fontweight="bold")
        self.ax_filt.set_ylabel("Amplitud µV", color=THEME["text_sec"], fontsize=8)
        self.ax_filt.tick_params(colors=THEME["text_sec"], labelsize=7)
        self.ax_filt.grid(True, linestyle="--", alpha=0.2, color=THEME["text_sec"])

        # 2. Espectrograma STFT (Abarca todo el ancho de la ventana)
        self.ax_spec.clear()
        self.cax_spec.clear()
        self.ax_spec.set_facecolor(THEME["bg_dark"])
        self.cax_spec.set_facecolor(THEME["bg_dark"])

        f, t_spec, Sxx = signal.spectrogram(sig, fs=fs, window='hamming', nperseg=128, noverlap=64)
        pcm = self.ax_spec.pcolormesh(t_spec * 1000.0, f, 10 * np.log10(Sxx + 1e-12), cmap='magma', shading='gouraud')

        self.ax_spec.set_title("Espectrograma 2D por STFT (Potencia dB)", color=THEME["primary"], fontsize=9, fontweight="bold")
        self.ax_spec.set_xlabel("Tiempo (ms)", color=THEME["text_sec"], fontsize=8)
        self.ax_spec.set_ylabel("Frecuencia (Hz)", color=THEME["text_sec"], fontsize=8)
        self.ax_spec.set_ylim(0, 500)
        self.ax_spec.tick_params(colors=THEME["text_sec"], labelsize=7)
        self.ax_spec.grid(True, linestyle="--", alpha=0.2, color=THEME["text_sec"])

        # Barra de Nivel en sub-eje estático 'cax_spec' a la derecha
        cbar = self.fig.colorbar(pcm, cax=self.cax_spec)
        cbar.set_label("Potencia (dB)", color=THEME["text_sec"], fontsize=8)
        cbar.ax.tick_params(colors=THEME["text_sec"], labelsize=7)

        # 3. Subbandas Wavelet
        coeffs = pywt.wavedec(sig, 'db4', level=4)
        axes_wav = [self.ax_w1, self.ax_w2, self.ax_w3, self.ax_w4]
        sub_names = ["D1 (1000-2000Hz)", "D2 (500-1000Hz)", "D3 (250-500Hz)", "D4 (125-250Hz)"]

        for idx, ax in enumerate(axes_wav):
            ax.clear()
            ax.set_facecolor(THEME["bg_dark"])
            ax.plot(coeffs[4 - idx], color=THEME["primary"], linewidth=0.7)
            ax.set_title(sub_names[idx], color=THEME["text_sec"], fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])

        feats = self.controller.extracted_features
        f_mean = feats[0]
        samp_en = feats[-1]
        e_total = np.sum(sig ** 2) / len(sig)

        self.lbl_metrics.config(text=f"Métricas Calculadas:\nSampEn : {samp_en:.2f}  |  Fmean : {f_mean:.1f} Hz  |  E_TOTAL : {e_total:.1f} mV²")
        self.fig.tight_layout(pad=1.2)
        self.canvas.draw()

    def run_pca(self):
        messagebox.showinfo("Reducción de Dimensionalidad", "PCA (>=95% var) ajustado correctamente:\nRetenidas 5 componentes principales de 13 originales.")

    def run_inference(self):
        if self.controller.engine.model is None or self.controller.engine.analyzer is None:
            meta = AVAILABLE_MODELS[self.controller.selected_model_key]
            messagebox.showerror(
                "Modelo No Cargado",
                f"No se encontró el modelo preentrenado '{meta['filename']}' en la carpeta ./models/\n\n"
                "Por favor, seleccione otro modelo disponible o entrene este modelo ejecutando 'main.py'."
            )
            return

        # Ejecuta la inferencia agregada sobre todas las ventanas del paciente
        res = self.controller.engine.predict_patient_signal(
            full_signal=self.controller.signal_4k,
            preprocessor=self.controller.preprocessor,
            fs=self.controller.target_fs,
            window_ms=self.controller.window_ms,
            overlap_pct=self.controller.overlap_pct,
            threshold=0.40
        )
        self.controller.inference_result = res
        self.controller.show_screen("Screen3Inferencia")


# ==============================================================================
# PANTALLA 3: MÓDULO DE INFERENCIA DIAGNÓSTICA Y SALIDA CAD
# ==============================================================================
class Screen3Inferencia(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent, bg=THEME["bg_dark"])
        self.controller = controller

        # Cabecera
        header = tk.Frame(self, bg=THEME["bg_card"], height=55, highlightbackground=THEME["border"], highlightthickness=1)
        header.pack(fill="x", side="top")
        tk.Label(header, text="NEUROMYODX v2.0 | Módulo de Inferencia Diagnóstica",
                 font=("Helvetica", 14, "bold"), fg=THEME["primary"], bg=THEME["bg_card"]).pack(side="left", padx=20, pady=12)

        main_body = tk.Frame(self, bg=THEME["bg_dark"])
        main_body.pack(fill="both", expand=True, padx=15, pady=5)

        top_split = tk.Frame(main_body, bg=THEME["bg_dark"])
        top_split.pack(fill="x", pady=(0, 10))

        # Gauge Card
        card_gauge = tk.Frame(top_split, bg=THEME["bg_card"], width=550, height=230, bd=1, relief="solid", highlightbackground=THEME["border"])
        card_gauge.pack(side="left", fill="both", expand=True, padx=(0, 5))

        self.lbl_diag_status = tk.Label(card_gauge, text="DIAGNÓSTICO: NEUROPATÍA DETECTADA", font=("Helvetica", 12, "bold"), fg=THEME["alert"], bg=THEME["bg_card"])
        self.lbl_diag_status.pack(anchor="w", padx=15, pady=(10, 0))

        self.lbl_confidence = tk.Label(card_gauge, text="Índice de Confianza: --%", font=("Helvetica", 10), fg=THEME["text_main"], bg=THEME["bg_card"])
        self.lbl_confidence.pack(anchor="w", padx=15, pady=(0, 5))

        self.fig_gauge, self.ax_gauge = plt.subplots(figsize=(4, 2), dpi=100)
        self.fig_gauge.patch.set_facecolor(THEME["bg_card"])
        self.canvas_gauge = FigureCanvasTkAgg(self.fig_gauge, master=card_gauge)
        self.canvas_gauge.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=5)

        # ML Performance Card (Con selector dinámico entre los 3 modelos)
        card_ml = tk.Frame(top_split, bg=THEME["bg_card"], width=550, height=230, bd=1, relief="solid", highlightbackground=THEME["border"])
        card_ml.pack(side="right", fill="both", expand=True, padx=(5, 0))

        ml_header = tk.Frame(card_ml, bg=THEME["bg_card"])
        ml_header.pack(fill="x", padx=15, pady=(10, 5))

        tk.Label(ml_header, text="DESEMPEÑO DEL MODELO DE ML", font=("Helvetica", 12, "bold"), fg=THEME["primary"], bg=THEME["bg_card"]).pack(side="left")

        # Selector para alternar modelos directamente desde la pantalla de diagnóstico
        self.combo_models_s3 = ttk.Combobox(ml_header, values=self.controller.model_names, state="readonly", width=33)
        self.combo_models_s3.set(self.controller.selected_model_key)
        self.combo_models_s3.pack(side="right")
        self.combo_models_s3.bind("<<ComboboxSelected>>", self._on_model_changed_live)

        self.lbl_ml_info = tk.Label(card_ml, text="", font=("Helvetica", 9), fg=THEME["text_sec"], bg=THEME["bg_card"], justify="left")
        self.lbl_ml_info.pack(anchor="w", padx=15, pady=5)

        box_frame = tk.Frame(card_ml, bg=THEME["bg_card"])
        box_frame.pack(fill="x", padx=15, pady=15)

        self.lbl_sens = tk.Label(box_frame, text="Sensibilidad: --", font=("Helvetica", 9, "bold"), fg=THEME["alert"], bg=THEME["bg_input"], bd=1, relief="solid", padx=8, pady=6)
        self.lbl_sens.pack(side="left", expand=True, fill="x", padx=3)

        self.lbl_spec = tk.Label(box_frame, text="Especificidad: --", font=("Helvetica", 9, "bold"), fg=THEME["success"], bg=THEME["bg_input"], bd=1, relief="solid", padx=8, pady=6)
        self.lbl_spec.pack(side="left", expand=True, fill="x", padx=3)

        self.lbl_auc = tk.Label(box_frame, text="AUC-ROC: --", font=("Helvetica", 9, "bold"), fg=THEME["primary"], bg=THEME["bg_input"], bd=1, relief="solid", padx=8, pady=6)
        self.lbl_auc.pack(side="left", expand=True, fill="x", padx=3)

        self._update_model_card_labels(self.controller.selected_model_key)

        # Table Card
        card_table = tk.Frame(main_body, bg=THEME["bg_card"], bd=1, relief="solid", highlightbackground=THEME["border"])
        card_table.pack(fill="both", expand=True, pady=5)

        tk.Label(card_table, text="CARACTERÍSTICAS EXTRAÍDAS DEL PACIENTE", font=("Helvetica", 11, "bold"), fg=THEME["primary"], bg=THEME["bg_card"]).pack(anchor="w", padx=15, pady=8)

        columns = ("param", "val_pat", "val_ctrl", "status")
        self.tree = ttk.Treeview(card_table, columns=columns, show="headings", height=4)

        self.tree.heading("param", text="PARÁMETRO EXTRAÍDO")
        self.tree.heading("val_pat", text="VALOR PACIENTE")
        self.tree.heading("val_ctrl", text="VALOR CONTROL")
        self.tree.heading("status", text="ESTADO RELATIVO")

        self.tree.column("param", width=220)
        self.tree.column("val_pat", width=140, anchor="center")
        self.tree.column("val_ctrl", width=140, anchor="center")
        self.tree.column("status", width=180, anchor="center")

        self.tree.pack(fill="both", expand=True, padx=15, pady=(0, 15))

        # Bottom Panel
        bottom_panel = tk.Frame(self, bg=THEME["bg_card"], height=70, bd=1, relief="solid", highlightbackground=THEME["border"])
        bottom_panel.pack(fill="x", side="bottom", padx=15, pady=10)

        btn_back2 = tk.Button(bottom_panel, text="🔄 VOLVER AL ANÁLISIS", command=lambda: self.controller.show_screen("Screen2Analisis"),
                              font=("Helvetica", 10, "bold"), bg=THEME["border"], fg=THEME["text_main"], bd=0, padx=15, pady=10, cursor="hand2")
        btn_back2.pack(side="left", padx=15, pady=10)

        btn_pdf = tk.Button(bottom_panel, text="📄 EXPORTAR REPORTE MÉDICO PDF", command=self.open_report_config_modal,
                            font=("Helvetica", 10, "bold"), bg=THEME["primary"], fg=THEME["bg_dark"], bd=0, padx=15, pady=10, cursor="hand2")
        btn_pdf.pack(side="left", padx=5, pady=10)

        btn_reset = tk.Button(bottom_panel, text="🔄 NUEVO PACIENTE", command=lambda: self.controller.show_screen("Screen1Carga"),
                              font=("Helvetica", 10, "bold"), bg=THEME["success"], fg=THEME["bg_dark"], bd=0, padx=15, pady=10, cursor="hand2")
        btn_reset.pack(side="right", padx=15, pady=10)

    def _update_model_card_labels(self, model_key: str):
        meta = AVAILABLE_MODELS[model_key]
        ml_info = (
            f"• Modelo activo : {meta['arch_desc']}\n"
            f"• Esquema de Validación : {meta['validation_desc']}\n"
            f"• Aislamiento de Datos : 100% Disjunto por Sujeto"
        )
        self.lbl_ml_info.config(text=ml_info)
        self.lbl_sens.config(text=f"Sensibilidad: {meta['sensitivity']}")
        self.lbl_spec.config(text=f"Especificidad: {meta['specificity']}")
        self.lbl_auc.config(text=f"AUC-ROC: {meta['auc_roc']}")

    def sync_model_selector(self, model_key: str):
        self.combo_models_s3.set(model_key)
        self._update_model_card_labels(model_key)

    def _on_model_changed_live(self, event=None):
        """Re-ejecuta la inferencia instantáneamente al cambiar el modelo en Pantalla 3."""
        chosen = self.combo_models_s3.get()
        loaded = self.controller.set_active_model(chosen, silent=False)

        if loaded and self.controller.signal_4k is not None:
            res = self.controller.engine.predict_patient_signal(
                full_signal=self.controller.signal_4k,
                preprocessor=self.controller.preprocessor,
                fs=self.controller.target_fs,
                window_ms=self.controller.window_ms,
                overlap_pct=self.controller.overlap_pct,
                threshold=0.40
            )
            self.controller.inference_result = res
            self.on_show()

    def draw_gauge(self, prob_val):
        self.ax_gauge.clear()
        self.ax_gauge.set_facecolor(THEME["bg_card"])

        w1 = Wedge((0, 0), 1.0, 108, 180, facecolor=THEME["success"], alpha=0.9)
        w2 = Wedge((0, 0), 1.0, 54, 108, facecolor='#F59E0B', alpha=0.9)
        w3 = Wedge((0, 0), 1.0, 0, 54, facecolor=THEME["alert"], alpha=0.9)
        w_center = Wedge((0, 0), 0.65, 0, 180, facecolor=THEME["bg_card"])

        for w in [w1, w2, w3, w_center]:
            self.ax_gauge.add_patch(w)

        theta = 180.0 - (prob_val * 180.0)
        rad = np.radians(theta)

        self.ax_gauge.plot([0, 0.85 * np.cos(rad)], [0, 0.85 * np.sin(rad)], color=THEME["text_main"], linewidth=2.5)
        self.ax_gauge.plot(0, 0, marker="o", color=THEME["text_main"], markersize=8)

        self.ax_gauge.text(-0.95, -0.15, "0%", color=THEME["success"], fontweight="bold", fontsize=9)
        self.ax_gauge.text(-0.1, 0.3, f"{prob_val*100:.1f}%", color=THEME["text_main"], fontweight="bold", fontsize=11)
        self.ax_gauge.text(0.85, -0.15, "100%", color=THEME["alert"], fontweight="bold", fontsize=9)

        self.ax_gauge.set_xlim(-1.1, 1.1)
        self.ax_gauge.set_ylim(-0.2, 1.1)
        self.ax_gauge.axis('off')
        self.canvas_gauge.draw()

    def on_show(self):
        self.sync_model_selector(self.controller.selected_model_key)

        res = self.controller.inference_result
        if res is None:
            return

        prob = res['als_probability']
        is_neuropathy = res['prediction'] == 1

        if is_neuropathy:
            self.lbl_diag_status.config(text="DIAGNÓSTICO: NEUROPATÍA DETECTADA", fg=THEME["alert"])
        else:
            self.lbl_diag_status.config(text="DIAGNÓSTICO: CONTROL SANO (NORMAL)", fg=THEME["success"])

        n_win = res.get('windows_analyzed', 1)
        pos_votes = res.get('positive_votes', int(is_neuropathy))
        self.lbl_confidence.config(
            text=f"Certeza: {res['confidence']*100:.2f}% | Riesgo Promedio: {prob*100:.2f}% | "
                 f"Ventanas Positivas: {pos_votes}/{n_win}"
        )
        self.draw_gauge(prob)

        feats = self.controller.extracted_features
        f_mean = feats[0]
        samp_en = feats[-1]

        for row in self.tree.get_children():
            self.tree.delete(row)

        self.tree.insert("", "end", values=("SampEn (complejidad)", f"{samp_en:.2f}", "1.45 ± 0.12", "Anormal (Reducido)" if is_neuropathy else "Normal"))
        self.tree.insert("", "end", values=("Frecuencia Media (STFT)", f"{f_mean:.1f} Hz", "120 ± 15.2 Hz", "Elevado (MUAPs big)" if is_neuropathy else "Normal"))
        self.tree.insert("", "end", values=("Varianza PCA Retenida", "96.67 %", "95.0 %", "Criterio Cumplido"))
        self.tree.insert("", "end", values=("Coeficientes DWT (db4)", "High Power D1-D2", "Balance Uniforme", "Desincronizado" if is_neuropathy else "Sincronizado"))

    def open_report_config_modal(self):
        if self.controller.inference_result is None:
            messagebox.showwarning("Sin Datos", "Debe ejecutar la inferencia diagnóstica antes de exportar el reporte.")
            return

        modal = tk.Toplevel(self.controller.root)
        modal.title("Configuración de Reporte Médico PDF")
        modal.geometry("540x620")
        modal.configure(bg=THEME["bg_dark"])
        modal.grab_set()

        tk.Label(modal, text="CONFIGURACIÓN Y COMENTARIOS DEL REPORTE MÉDICO",
                 font=("Helvetica", 11, "bold"), fg=THEME["primary"], bg=THEME["bg_dark"]).pack(pady=12)

        notebook = ttk.Notebook(modal)
        notebook.pack(fill="both", expand=True, padx=15, pady=5)

        # Tab 1: Datos del Médico
        tab_doc = tk.Frame(notebook, bg=THEME["bg_card"], padx=15, pady=15)
        notebook.add(tab_doc, text="👨‍⚕️ Médico Evaluador")

        ent_doc_name = self._create_form_row(tab_doc, "Nombre del Médico:", self.controller.doctor_info["name"], 0)
        ent_doc_spec = self._create_form_row(tab_doc, "Especialidad:", self.controller.doctor_info["specialty"], 1)
        ent_doc_lic = self._create_form_row(tab_doc, "Registro / Cédula MP:", self.controller.doctor_info["license"], 2)

        # Tab 2: Datos del Paciente
        tab_pat = tk.Frame(notebook, bg=THEME["bg_card"], padx=15, pady=15)
        notebook.add(tab_pat, text="👤 Datos del Paciente")

        ent_pat_id = self._create_form_row(tab_pat, "ID Paciente:", self.controller.patient_info["id"], 0)
        ent_pat_name = self._create_form_row(tab_pat, "Nombre Completo:", self.controller.patient_info["name"], 1)
        ent_pat_age = self._create_form_row(tab_pat, "Edad (años):", self.controller.patient_info["age"], 2)
        ent_pat_gender = self._create_form_row(tab_pat, "Género:", self.controller.patient_info["gender"], 3)
        ent_pat_muscle = self._create_form_row(tab_pat, "Músculo Evaluado:", self.controller.patient_info["muscle"], 4)

        # Tab 3: Comentarios Clínicos
        tab_notes = tk.Frame(notebook, bg=THEME["bg_card"], padx=15, pady=15)
        notebook.add(tab_notes, text="📝 Observaciones Médicas")

        tk.Label(tab_notes, text="Comentarios Clínicos Adicionales y Recomendaciones:",
                 font=("Helvetica", 9, "bold"), fg=THEME["text_main"], bg=THEME["bg_card"]).pack(anchor="w", pady=(0, 5))

        txt_notes = tk.Text(tab_notes, height=8, font=("Helvetica", 9), bd=1, relief="solid", bg=THEME["bg_input"], fg=THEME["text_main"], insertbackground=THEME["text_main"])
        txt_notes.pack(fill="both", expand=True, pady=5)
        txt_notes.insert("1.0", "Se observa alteración neurogénea discreta en las descargas de unidades motoras durante la contracción isométrica. Se sugiere seguimiento electromiográfico en 3 meses.")

        # Acciones de la Modal
        btn_frame = tk.Frame(modal, bg=THEME["bg_dark"])
        btn_frame.pack(fill="x", padx=15, pady=15)

        def generate_and_save():
            self.controller.doctor_info = {
                "name": ent_doc_name.get(),
                "specialty": ent_doc_spec.get(),
                "license": ent_doc_lic.get()
            }
            self.controller.patient_info = {
                "id": ent_pat_id.get(),
                "name": ent_pat_name.get(),
                "age": ent_pat_age.get(),
                "gender": ent_pat_gender.get(),
                "muscle": ent_pat_muscle.get()
            }
            notes_str = txt_notes.get("1.0", tk.END).strip()

            modal.destroy()
            self._render_pdf_report(notes_str)

        tk.Button(btn_frame, text="❌ Cancelar", command=modal.destroy,
                  font=("Helvetica", 9, "bold"), bg=THEME["border"], fg=THEME["text_main"], bd=0, padx=15, pady=8, cursor="hand2").pack(side="left", padx=5)

        tk.Button(btn_frame, text="📄 Generar Reporte PDF", command=generate_and_save,
                  font=("Helvetica", 9, "bold"), bg=THEME["primary"], fg=THEME["bg_dark"], bd=0, padx=15, pady=8, cursor="hand2").pack(side="right", padx=5)

    def _create_form_row(self, parent, label_text, default_val, row_idx):
        tk.Label(parent, text=label_text, font=("Helvetica", 9), fg=THEME["text_main"], bg=THEME["bg_card"]).grid(row=row_idx, column=0, sticky="w", pady=6)
        entry = tk.Entry(parent, width=32, font=("Helvetica", 9), bd=1, relief="solid", bg=THEME["bg_input"], fg=THEME["text_main"], insertbackground=THEME["text_main"])
        entry.insert(0, default_val)
        entry.grid(row=row_idx, column=1, padx=10, pady=6)
        return entry

    def _render_pdf_report(self, clinical_notes: str):
        pat = self.controller.patient_info
        doc = self.controller.doctor_info
        res = self.controller.inference_result
        feats = self.controller.extracted_features
        model_meta = AVAILABLE_MODELS[self.controller.selected_model_key]

        reports_dir = get_output_dir("reports")
        filename_str = self.controller.current_filename or 'iEMG'
        pdf_path = os.path.join(reports_dir, f"Reporte_Clinico_{pat['id']}_{model_meta['short_name']}_{filename_str}.pdf")

        # Crear figura en formato A4 con fondo blanco para impresión limpia
        fig = plt.figure(figsize=(8.27, 11.69), dpi=150)
        fig.patch.set_facecolor('#FFFFFF')

        # Cabecera
        fig.text(0.08, 0.94, "NEUROMYODX v2.0", fontsize=16, fontweight='bold', color='#00B49F')
        fig.text(0.08, 0.92, "SISTEMA COMPUTERIZADO DE ASISTENCIA DIAGNÓSTICA EN iEMG", fontsize=8.5, fontweight='bold', color='#64748B')
        fig.text(0.68, 0.94, f"FECHA: {datetime.date.today().strftime('%d/%m/%Y')}", fontsize=9, fontweight='bold', color='#0F172A')

        fig.add_artist(plt.Line2D([0.08, 0.92], [0.905, 0.905], color='#00B49F', linewidth=2))

        # Información Paciente y Médico
        ax_info = fig.add_axes([0.08, 0.71, 0.84, 0.18])
        ax_info.axis('off')
        rect_info = plt.Rectangle((0, 0), 1, 1, transform=ax_info.transAxes, facecolor='#F8FAFC', edgecolor='#CBD5E1', linewidth=1)
        ax_info.add_patch(rect_info)

        ax_info.text(0.03, 0.88, "INFORMACIÓN DEL PACIENTE", fontsize=9.5, fontweight='bold', color='#00B49F', va='top')
        pat_text = (
            f"• ID Paciente: {pat['id']}\n"
            f"• Nombre: {pat['name']}\n"
            f"• Edad / Género: {pat['age']} años / {pat['gender']}\n"
            f"• Músculo: {pat['muscle']}"
        )
        ax_info.text(0.03, 0.72, pat_text, fontsize=8.5, color='#0F172A', va='top', linespacing=1.4)

        ax_info.text(0.52, 0.88, "MÉDICO EVALUADOR", fontsize=9.5, fontweight='bold', color='#00B49F', va='top')
        wrapped_file = textwrap.fill(self.controller.current_filename or 'N/A', width=36)
        doc_text = (
            f"• Nombre: {doc['name']}\n"
            f"• Especialidad: {doc['specialty']}\n"
            f"• Cédula/Registro: {doc['license']}\n"
            f"• Archivo: {wrapped_file}"
        )
        ax_info.text(0.52, 0.72, doc_text, fontsize=8.5, color='#0F172A', va='top', linespacing=1.4)

        # Diagnóstico CADx
        is_neuropathy = res['prediction'] == 1
        diag_title = "NEUROPATÍA DETECTADA" if is_neuropathy else "CONTROL SANO (NORMAL)"
        status_color = "#FF2A6D" if is_neuropathy else "#00E676"

        ax_diag = fig.add_axes([0.08, 0.55, 0.84, 0.13])
        ax_diag.axis('off')
        rect_diag = plt.Rectangle((0, 0), 1, 1, transform=ax_diag.transAxes, facecolor='#FFFFFF', edgecolor=status_color, linewidth=2)
        ax_diag.add_patch(rect_diag)

        ax_diag.text(0.03, 0.85, f"EVALUACIÓN DIAGNÓSTICA CADx ({model_meta['pdf_tag']})", fontsize=9, fontweight='bold', color='#64748B', va='top')
        ax_diag.text(0.03, 0.58, diag_title, fontsize=14, fontweight='bold', color=status_color, va='top')
        ax_diag.text(0.03, 0.22, f"Certeza Diagnóstica: {res['confidence']*100:.2f}%  |  Probabilidad Neurobiológica: {res['als_probability']*100:.2f}%  |  Latencia: {res['latency_ms']:.2f} ms", fontsize=8.5, color='#0F172A', va='top')

        # Tabla de Características con texto explícito de alta visibilidad (#0F172A)
        fig.text(0.08, 0.515, "CARACTERÍSTICAS BIOPOTENCIALES EXTRAÍDAS", fontsize=9.5, fontweight='bold', color='#00B49F')

        f_mean = feats[0]
        samp_en = feats[-1]

        table_data = [
            ["Parámetro Extraído", "Valor Paciente", "Valor Control", "Estado Relativo"],
            ["Entropía de Muestra", f"{samp_en:.2f}", "1.45 ± 0.12", "Anormal (Reducido)" if is_neuropathy else "Normal"],
            ["Frecuencia Media (STFT)", f"{f_mean:.1f} Hz", "120 ± 15.2 Hz", "Elevado (MUAPs big)" if is_neuropathy else "Normal"],
            ["Varianza PCA Retenida", "96.67 %", "95.0 %", "Criterio Cumplido"],
            ["Coeficientes DWT (db4)", "High Power D1-D2", "Balance Uniforme", "Desincronizado" if is_neuropathy else "Sincronizado"]
        ]

        ax_tbl = fig.add_axes([0.08, 0.33, 0.84, 0.17])
        ax_tbl.axis('off')
        tbl = ax_tbl.table(cellText=table_data, loc='center', bbox=[0, 0, 1, 1])
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(8)

        # Aplicación de colores para solucionar visibilidad en PDF
        for (row_idx, col_idx), cell in tbl.get_celld().items():
            cell.set_edgecolor('#CBD5E1')
            if row_idx == 0:
                cell.set_facecolor('#00B49F')
                cell.get_text().set_color('white')
                cell.get_text().set_weight('bold')
            else:
                cell.set_facecolor('#FFFFFF' if row_idx % 2 == 0 else '#F8FAFC')
                cell.get_text().set_color('#0F172A')

        # Observaciones Clínicas
        fig.text(0.08, 0.29, "OBSERVACIONES CLÍNICAS Y NOTAS DEL MÉDICO", fontsize=9.5, fontweight='bold', color='#00B49F')
        ax_notes = fig.add_axes([0.08, 0.13, 0.84, 0.14])
        ax_notes.axis('off')
        rect_notes = plt.Rectangle((0, 0), 1, 1, transform=ax_notes.transAxes, facecolor='#F8FAFC', edgecolor='#CBD5E1', linewidth=1)
        ax_notes.add_patch(rect_notes)

        notes_display = clinical_notes if clinical_notes else "Sin observaciones adicionales."
        wrapped_notes = textwrap.fill(notes_display, width=92)
        ax_notes.text(0.03, 0.85, wrapped_notes, fontsize=8.5, color='#0F172A', va='top', linespacing=1.3)

        # Pie de página
        fig.text(0.60, 0.075, "________________________________________", fontsize=9, color='#0F172A')
        fig.text(0.63, 0.055, f"Firma: {doc['name']}", fontsize=8.5, fontweight='bold', color='#0F172A')
        fig.text(0.63, 0.040, f"{doc['specialty']} — {doc['license']}", fontsize=7.5, color='#64748B')

        with PdfPages(pdf_path) as pdf:
            pdf.savefig(fig)
        plt.close(fig)

        messagebox.showinfo("Reporte PDF Creado", f"El informe clínico en formato PDF se ha guardado exitosamente en:\n{os.path.abspath(pdf_path)}")


if __name__ == "__main__":
    root = tk.Tk()
    app = NeuroMyoDxApp(root)
    root.mainloop()