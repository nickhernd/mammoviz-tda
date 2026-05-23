#!/usr/bin/env python3
"""
create_github_issues_mammoviz.py
---------------------------------
Crea issues en GitHub a partir del TODO del proyecto MammoViz-TDA.

Uso:
    export GITHUB_TOKEN="ghp_xxxxxxxxxxxx"
    python create_github_issues_mammoviz.py --repo "nickhernd/mammoviz-tda" [--dry-run]

Requisitos:
    pip install requests
"""

import argparse
import os
import time
import requests

# ---------------------------------------------------------------------------
# LABELS
# ---------------------------------------------------------------------------
LABELS = [
    {"name": "fase-0",          "color": "ededed", "description": "Fase 0 — Infraestructura y dataset"},
    {"name": "fase-1",          "color": "ededed", "description": "Fase 1 — Carga de datos médicos"},
    {"name": "fase-2",          "color": "ededed", "description": "Fase 2 — TDA / Homología persistente"},
    {"name": "fase-3",          "color": "ededed", "description": "Fase 3 — Red neuronal e inferencia ONNX"},
    {"name": "fase-4",          "color": "ededed", "description": "Fase 4 — XAI / Explicabilidad"},
    {"name": "fase-5",          "color": "ededed", "description": "Fase 5 — Render OpenGL 4.6"},
    {"name": "fase-6",          "color": "ededed", "description": "Fase 6 — Dear ImGui UI"},
    {"name": "fase-7",          "color": "ededed", "description": "Fase 7 — Enlace entre vistas"},
    {"name": "fase-8",          "color": "ededed", "description": "Fase 8 — Análisis comparativo y validación"},
    {"name": "opcional",        "color": "ededed", "description": "Mejora opcional sin fecha fija"},
    {"name": "dataset",         "color": "0075ca", "description": "Adquisición y gestión de datos CBIS-DDSM"},
    {"name": "infraestructura", "color": "bfd4f2", "description": "CMake, entorno, dependencias"},
    {"name": "cpp",             "color": "f9d0c4", "description": "Código C++ / módulo de producción"},
    {"name": "python",          "color": "c2e0c6", "description": "Script Python auxiliar"},
    {"name": "tda",             "color": "5319e7", "description": "Análisis Topológico de Datos"},
    {"name": "dicom",           "color": "0052cc", "description": "Carga y procesado DICOM/ITK"},
    {"name": "opengl",          "color": "d93f0b", "description": "Render OpenGL / shaders GLSL"},
    {"name": "onnx",            "color": "e4e669", "description": "Inferencia ONNX Runtime"},
    {"name": "xai",             "color": "006b75", "description": "Explicabilidad / GradCAM / IG"},
    {"name": "ui",              "color": "84b6eb", "description": "Interfaz Dear ImGui"},
    {"name": "testing",         "color": "d4c5f9", "description": "Tests unitarios / integración"},
    {"name": "documentación",   "color": "1d76db", "description": "Documentación y análisis"},
    {"name": "rendimiento",     "color": "fbca04", "description": "Optimización y benchmarks"},
    {"name": "bloqueante",      "color": "b60205", "description": "Otro task depende de este"},
    {"name": "investigación",   "color": "fef2c0", "description": "Exploración científica / lectura"},
]

# ---------------------------------------------------------------------------
# MILESTONES
# ---------------------------------------------------------------------------
MILESTONES = [
    {"title": "Fase 0 — Infraestructura + Dataset real",        "description": "Semanas 1-3: DICOMs reales en disco, entorno compilando sin warnings"},
    {"title": "Fase 1 — Carga de datos médicos",                "description": "Semanas 4-7: 20 casos CBIS-DDSM cargados, histograma HU documentado"},
    {"title": "Fase 2 — TDA / Homología persistente",           "description": "Semanas 8-14: diagramas de persistencia de 20 casos reales"},
    {"title": "Fase 3 — Red neuronal e inferencia ONNX",        "description": "Semanas 15-20: modelo real en ONNX con AUC > 0.5"},
    {"title": "Fase 4 — XAI / Explicabilidad",                  "description": "Semanas 21-27: GradCAM destaca calcificaciones visualmente"},
    {"title": "Fase 5 — Render OpenGL 4.6",                     "description": "Semanas 28-39: ventana 3 paneles con caso real renderizado"},
    {"title": "Fase 6 — Dear ImGui UI",                         "description": "Semanas 40-45: 20 casos navegables desde la UI"},
    {"title": "Fase 7 — Enlace entre vistas",                   "description": "Semanas 46-49: click en diagrama ilumina vóxeles en volumen"},
    {"title": "Fase 8 — Análisis comparativo y validación",     "description": "Semanas 50-52: análisis estadístico 70 casos, GradCAM IoU > 0.2"},
    {"title": "Mejoras opcionales",                             "description": "Sin fecha fija: rendimiento, algoritmos TDA adicionales, rendering avanzado"},
]

# ---------------------------------------------------------------------------
# ISSUES
# ---------------------------------------------------------------------------
ISSUES = [

    # ── FASE 0 · Dataset CBIS-DDSM ────────────────────────────────────────
    {
        "title": "[F0] Registrarse en TCIA y explorar CBIS-DDSM",
        "body": (
            "- Registrarse en `cancerimagingarchive.net` (gratuito)\n"
            "- Leer la descripción del dataset: contenido, organización de series, metadatos\n"
            "- Descargar el cliente TCIA (`tcia-utils` pip o cliente Java oficial)\n\n"
            "**Por qué importa hacerlo ya:** si esperas a la Fase 8, habrás calibrado todo "
            "el código contra blobs gaussianos perfectos. Un DICOM real tiene artefactos, "
            "valores HU distintos, slices desordenados y resoluciones variables."
        ),
        "labels": ["fase-0", "dataset", "investigación"],
        "milestone": "Fase 0 — Infraestructura + Dataset real",
    },
    {
        "title": "[F0] Descargar 20 casos CBIS-DDSM (10 calc_benign + 10 calc_malignant)",
        "body": (
            "- Descargar al menos **20 casos**: 10 calcificación benigna + 10 calcificación maligna\n"
            "- Priorizar calcificaciones sobre masas (estructuras más discretas, más útiles para TDA)\n"
            "- Crear `data/real/calc_benign/` y `data/real/calc_malignant/`\n"
            "- Añadir `data/real/` a `.gitignore`\n"
            "- Documentar en `data/real/README.md`: IDs de casos, fecha de descarga, cómo reproducir"
        ),
        "labels": ["fase-0", "dataset", "bloqueante"],
        "milestone": "Fase 0 — Infraestructura + Dataset real",
    },
    {
        "title": "[F0] Explorar DICOMs descargados antes de escribir código",
        "body": (
            "Con un visor externo (3D Slicer, Horos, o `pydicom` + `matplotlib`):\n"
            "- Anotar rango de HU típico en mamografías FFDM de CBIS-DDSM\n"
            "- Anotar resolución espacial (spacing en mm), tamaño en píxeles\n"
            "- Entender la diferencia visual entre una calcificación benigna y una maligna\n"
            "- Documentar hallazgos en `data/real/README.md`\n\n"
            "**Recurso recomendado:** \"Digital Mammography: A Practical Approach\""
        ),
        "labels": ["fase-0", "dataset", "investigación"],
        "milestone": "Fase 0 — Infraestructura + Dataset real",
    },

    # ── FASE 0 · Entorno de compilación ───────────────────────────────────
    {
        "title": "[F0] Crear CMakeLists.txt raíz con soporte CMake 4.x y GCC 16",
        "body": (
            "- Políticas `CMP0167`/`CMP0144` para CMake 4.x\n"
            "- Workaround tabla de features GCC 16: inyectar `CMAKE_CXX_COMPILE_FEATURES` "
            "manualmente antes de cualquier `find_package` (sin esto Eigen3 e ITK fallan)\n"
            "- Preset Release (`-O3 -march=native`) y Debug (`-g -fsanitize=address,undefined`)\n"
            "- Ninja como generador por defecto\n"
            "- Verificar: `cmake -B build -G Ninja && ninja -C build` compila `main.cpp` vacío sin warnings"
        ),
        "labels": ["fase-0", "infraestructura", "bloqueante"],
        "milestone": "Fase 0 — Infraestructura + Dataset real",
    },
    {
        "title": "[F0] Configurar clang-format, clang-tidy y .gitignore",
        "body": (
            "- `.clang-format`: LLVM base, `ColumnLimit=100`, `IndentWidth=4`\n"
            "- `.clang-tidy`: checks `modernize-*`, `readability-*`, `performance-*`\n"
            "- `.gitignore`: `build/`, `*.o`, `*.a`, `data/real/`, `data/models/*.onnx`"
        ),
        "labels": ["fase-0", "infraestructura"],
        "milestone": "Fase 0 — Infraestructura + Dataset real",
    },
    {
        "title": "[F0] Integrar dependencias: ITK, Eigen, GLFW, GLEW, GTest, GUDHI, ONNX RT, ImGui",
        "body": (
            "- Documentar en `DEPS.md` los paquetes Arch exactos: `itk`, `eigen`, `glfw`, `glew`, `gtest`\n"
            "- `install_deps.sh`: descargar ONNX Runtime 1.17 precompilado, clonar Dear ImGui en `third_party/`\n"
            "- GUDHI 3.10 via `FetchContent_Populate` (no `MakeAvailable`) — solo headers\n"
            "- Target `INTERFACE` manual para GUDHI apuntando a `${gudhi_SOURCE_DIR}/include`\n"
            "- Verificar: `#include <gudhi/Simplex_tree.h>` compila correctamente"
        ),
        "labels": ["fase-0", "infraestructura", "bloqueante"],
        "milestone": "Fase 0 — Infraestructura + Dataset real",
    },
    {
        "title": "[F0] Crear estructura de directorios y headers vacíos",
        "body": (
            "- Árbol completo: `include/`, `src/`, `shaders/glsl/`, `tests/unit/`, `scripts/`, `data/`\n"
            "- `CMakeLists.txt` por subdirectorio: `src/`, `tests/`\n"
            "- Headers vacíos con guard en: `io/`, `tda/`, `nn/`, `xai/`, `render/`, `utils/`\n"
            "- `config.toml` inicial con todos los parámetros"
        ),
        "labels": ["fase-0", "infraestructura"],
        "milestone": "Fase 0 — Infraestructura + Dataset real",
    },
    {
        "title": "[F0] Implementar utilidades base: Logger, Timer, Config",
        "body": (
            "- `utils/Logger.h/.cpp`: niveles INFO/WARN/ERROR, timestamps, salida a `stderr`\n"
            "- `utils/Timer.h`: wrapper `std::chrono::high_resolution_clock`, método `elapsed_ms()`\n"
            "- `utils/Config.h/.cpp`: parser TOML mínimo — secciones `[x]`, claves `k=v`, tipos string/float/int/bool\n"
            "- Test manual: leer `config.toml`, imprimir todos los valores, verificar tipos\n\n"
            "**Criterio de éxito Fase 0:** DICOMs reales en disco explorados. `ninja -C build` sin warnings."
        ),
        "labels": ["fase-0", "cpp"],
        "milestone": "Fase 0 — Infraestructura + Dataset real",
    },

    # ── FASE 1 · VolumeData ───────────────────────────────────────────────
    {
        "title": "[F1] Implementar VolumeData — tensor 4D con métodos de acceso",
        "body": (
            "```cpp\n"
            "struct VolumeData {\n"
            "    std::vector<float> data;\n"
            "    size_t dims[4];       // X,Y,Z,T\n"
            "    float  spacing[3];    // mm/voxel\n"
            "    float  origin[3];\n"
            "    std::string patient_id, series_uid;\n"
            "};\n"
            "```\n\n"
            "- `at(x,y,z,t)`: bounds checking en Debug, sin overhead en Release (`[[likely]]`)\n"
            "- `slice(z)`: puntero al plano Z como `const float*`\n"
            "- `normalize()`: escala a [0,1] in-place, guarda min/max original\n"
            "- `histogramHU(bins=256)`: distribución de intensidades"
        ),
        "labels": ["fase-1", "cpp", "bloqueante"],
        "milestone": "Fase 1 — Carga de datos médicos",
    },
    {
        "title": "[F1] Implementar DicomLoader con ITK/GDCM",
        "body": (
            "- `load(path)`: detecta directorio DICOM o archivo `.mha`\n"
            "- **Caso directorio:** `itk::GDCMSeriesFileNames` para ordenar slices por `ImagePositionPatient`\n"
            "  - Fallback por `InstanceNumber` cuando `ImagePositionPatient` sea inconsistente (ocurre en CBIS-DDSM)\n"
            "- Leer `itk::Image<short,3>`, aplicar rescale slope/intercept, convertir a float\n"
            "- Normalizar HU a [0,1] — explorar histograma de casos reales antes de hardcodear el rango\n"
            "- **Caso `.mha`:** `itk::ImageFileReader` con `MetaImageIO`\n"
            "- Extraer `spacing` en mm, `patient_id`, `series_uid` desde metadatos DICOM\n"
            "- Loggear siempre: ruta, dimensiones, spacing, rango HU encontrado\n\n"
            "**Lectura:** ITK Software Guide Vol.1 Cap. 7, DICOM standard PS 3.3 Image Module."
        ),
        "labels": ["fase-1", "cpp", "dicom", "bloqueante"],
        "milestone": "Fase 1 — Carga de datos médicos",
    },
    {
        "title": "[F1] Validar carga de los 20 casos CBIS-DDSM reales",
        "body": (
            "- Cargar los 20 casos con `DicomLoader`\n"
            "- Imprimir histograma de HU de cada caso\n"
            "- Ajustar rango de normalización en `config.toml`:\n"
            "  - Tejido glandular → ~0.4-0.6\n"
            "  - Calcificaciones → > 0.8\n"
            "- Verificar que ningún caso falla en carga\n"
            "- Documentar casos problemáticos en `data/real/README.md`\n\n"
            "**Criterio de éxito Fase 1:** `./mammoviz data/real/calc_benign/case001` imprime dimensiones y spacing correctos. Los 20 casos sin crash."
        ),
        "labels": ["fase-1", "dataset", "dicom"],
        "milestone": "Fase 1 — Carga de datos médicos",
    },
    {
        "title": "[F1] Script gen_dicom.py — datos sintéticos para tests offline",
        "body": (
            "- Genera 32 slices 64×64 con blob gaussiano + metadatos DICOM mínimos válidos\n"
            "- Parametrizable: `-size`, `-snr`, `-blob-intensity`\n"
            "- **Solo para `tests/unit/`** — nunca como sustituto del pipeline real\n\n"
            "> Los datos sintéticos son el fallback para CI/tests offline, no el objetivo."
        ),
        "labels": ["fase-1", "python", "testing"],
        "milestone": "Fase 1 — Carga de datos médicos",
    },

    # ── FASE 2 · TDA ──────────────────────────────────────────────────────
    {
        "title": "[F2] Lectura teórica de TDA y homología persistente",
        "body": (
            "- \"A User's Guide to Topological Data Analysis\" — Lum et al. (2013), primeras 15 pp.\n"
            "- \"An Introduction to Topological Data Analysis\" — Chazal & Michel (2021), primeras 20 pp.\n"
            "- Documentación GUDHI 3.10: `Rips_complex`, `Simplex_tree`, `persistent_cohomology`\n"
            "- Entender qué significan β₀, β₁, β₂ en microcalcificaciones de mamografía:\n"
            "  - β₀: clusters separados · β₁: distribuciones en anillo/ductos · β₂: cavidades\n"
            "- Dibujar a mano un diagrama de persistencia de 5-6 puntos en R² (calcular H0 manualmente)\n"
            "- **Exploración cualitativa:** abrir un caso benigno y uno maligno en Python, dibujar píxeles de alta intensidad. ¿Tienen formas distintas a simple vista?"
        ),
        "labels": ["fase-2", "tda", "investigación"],
        "milestone": "Fase 2 — TDA / Homología persistente",
    },
    {
        "title": "[F2] Implementar PointCloud — extracción de nube de puntos desde VolumeData",
        "body": (
            "```cpp\n"
            "struct Point3D { float x, y, z, intensity; };\n"
            "```\n\n"
            "- `fromVolume(VolumeData&, float threshold)`: vóxeles sobre umbral, coordenadas en mm\n"
            "- `subsample(size_t max_points)`: muestreo aleatorio con `std::mt19937` + semilla fija\n"
            "- `exportOFF(path)`: formato OFF estándar para GUDHI\n"
            "- `stats()`: número de puntos, bbox en mm, densidad media\n\n"
            "**Calibración con datos reales:** ejecutar con threshold 0.7 / 0.8 / 0.85 en 5 casos. "
            "Objetivo: ~5000-20000 puntos. Documentar threshold elegido en `config.toml`."
        ),
        "labels": ["fase-2", "cpp", "tda", "bloqueante"],
        "milestone": "Fase 2 — TDA / Homología persistente",
    },
    {
        "title": "[F2] Implementar VietorisRips — homología persistente con GUDHI",
        "body": (
            "- `Config { float max_edge_length; int max_dimension; bool sparse; }`\n"
            "- Dense si n < 500 (`Gudhi::Rips_complex`), Sparse si n ≥ 500 (`Gudhi::Sparse_rips_complex`, aprox. 0.5)\n"
            "- Homología con `Standard_persistence_algorithm`, coeficientes Z/2Z\n"
            "- Devolver `std::vector<PersistencePair>` con `{birth, death, dimension}`\n"
            "- Loggear: nº simplices, tiempo de cómputo, nº pares por dimensión\n"
            "- Soporte DCE-MRI temporal: `computeTemporal(VolumeData& 4D)` itera sobre T\n\n"
            "**Calibración:** ejecutar en 5 casos reales. Si > 60s, ajustar `max_points` o `max_edge_length`. Objetivo: < 30s/caso."
        ),
        "labels": ["fase-2", "cpp", "tda", "bloqueante"],
        "milestone": "Fase 2 — TDA / Homología persistente",
    },
    {
        "title": "[F2] Implementar PersistenceDiagram — análisis y exportación",
        "body": (
            "- `filtered(float min_persistence)`: elimina ruido topológico\n"
            "- `featureVector(int bins=64)` → 192 floats (3 dims × 64 bins, L2-normalizado)\n"
            "- `bottleneckDistance(PersistenceDiagram&)`: aproximación O(n²)\n"
            "- `wassersteinDistance(PersistenceDiagram&)`: asignación greedy p=2\n"
            "- `saveCSV(path)` / `loadCSV(path)`: round-trip sin pérdida"
        ),
        "labels": ["fase-2", "cpp", "tda"],
        "milestone": "Fase 2 — TDA / Homología persistente",
    },
    {
        "title": "[F2] Script explore_tda.py — exploración topológica de datos reales",
        "body": (
            "Script Python para iterar rápido antes de fijar parámetros en C++:\n"
            "- Carga 5 casos benignos + 5 malignos con `pydicom`\n"
            "- Extrae nube de puntos con el threshold calibrado\n"
            "- Computa Vietoris-Rips con GUDHI Python bindings\n"
            "- Dibuja diagramas de persistencia de todos los casos juntos\n"
            "- Anotar: ¿se separan visualmente benignos y malignos? ¿qué dimensión (H0/H1/H2) diferencia mejor?\n"
            "- Usar hallazgos para decidir `max_dimension` y `min_persistence` en `config.toml`\n\n"
            "**Criterio de éxito Fase 2:** diagramas de 20 casos reales en CSV. Diferencia visual observable entre clases. `ctest` pasa los 6 tests unitarios."
        ),
        "labels": ["fase-2", "python", "tda", "investigación"],
        "milestone": "Fase 2 — TDA / Homología persistente",
    },
    {
        "title": "[F2] Tests unitarios TDA (con datos sintéticos como fallback)",
        "body": (
            "Workaround GCC 16 en `tests/CMakeLists.txt`: limpiar `INTERFACE_SYSTEM_INCLUDE_DIRECTORIES` y `INTERFACE_INCLUDE_DIRECTORIES` de targets GTest.\n\n"
            "Tests requeridos:\n"
            "- `FeatureVectorSize`: 192 valores, norma ≈ 1.0 (±1e-5)\n"
            "- `FilteredRemovesNoise`: pares cortos eliminados, largos conservados\n"
            "- `CSVRoundtrip`: save+load preserva campos (±1e-6)\n"
            "- `BottleneckDistanceSelf`: 0.0 (±1e-10)\n"
            "- `WassersteinDistanceSelf`: 0.0\n"
            "- `EmptyDiagram`: featureVector de vacío → ceros sin crash"
        ),
        "labels": ["fase-2", "testing"],
        "milestone": "Fase 2 — TDA / Homología persistente",
    },

    # ── FASE 3 · ONNX / Red neuronal ─────────────────────────────────────
    {
        "title": "[F3] Lectura teórica CNNs en imagen médica + revisión del estado del arte CBIS-DDSM",
        "body": (
            "- \"Deep Learning\" Goodfellow cap. 9 (CNNs), primeras 30 páginas\n"
            "- Entender qué capas exporta una CNN estándar en ONNX\n"
            "- Leer ≥2 papers de clasificación en CBIS-DDSM:\n"
            "  - Shen et al. (2019) \"Deep Learning for Mammography\" — buen punto de partida\n"
            "  - Buscar en Papers with Code \"CBIS-DDSM\" para ver estado del arte actual\n"
            "- Anotar: arquitectura usada, AUC reportado, preprocesado de imágenes"
        ),
        "labels": ["fase-3", "onnx", "investigación"],
        "milestone": "Fase 3 — Red neuronal e inferencia ONNX",
    },
    {
        "title": "[F3] Obtener modelo pre-entrenado en CBIS-DDSM y exportar a ONNX",
        "body": (
            "Opción 1 — modelo público:\n"
            "- Buscar en Hugging Face Hub: `mammography classification CBIS-DDSM`\n\n"
            "Opción 2 — fine-tuning propio (si no hay modelo público adecuado):\n"
            "- `scripts/train_model.py`: PyTorch, 10 épocas, batch 8\n"
            "- Input: parche 224×224 de la ROI anotada en CBIS-DDSM\n"
            "- Output: 2 clases (benign/malignant)\n"
            "- Guardar mejor checkpoint por AUC en validación\n\n"
            "En ambos casos:\n"
            "- Exportar con `torch.onnx.export()`, opset 17\n"
            "- Verificar con `netron.app`: arquitectura y nombres de capas intermedias\n"
            "- Validar con `onnx.checker.check_model()`\n"
            "- Guardar como `data/models/breast_cnn.onnx`"
        ),
        "labels": ["fase-3", "onnx", "python", "bloqueante"],
        "milestone": "Fase 3 — Red neuronal e inferencia ONNX",
    },
    {
        "title": "[F3] Script gen_onnx_model.py — modelo dummy para CI/tests",
        "body": (
            "- MLP mínimo: inputs `image_features [1,512]` + `tda_features [1,192]`\n"
            "- **Solo para CI y tests unitarios** — no representa el modelo real\n"
            "- Guardar como `data/models/dummy_model.onnx` (nombre distinto del modelo real)"
        ),
        "labels": ["fase-3", "python", "testing"],
        "milestone": "Fase 3 — Red neuronal e inferencia ONNX",
    },
    {
        "title": "[F3] Implementar ModelInference — wrapper ONNX Runtime",
        "body": (
            "- Constructor: carga `.onnx`, `Ort::Env` con WARNING, `SessionOptions` con 4 threads\n"
            "- **CRÍTICO:** leer nombres de inputs con `session.GetInputNameAllocated()` — no hardcodear (el modelo real tendrá nombres distintos al dummy)\n"
            "- `run(image_features, tda_features)` → `Prediction { int class_id; float confidence; float probs[2]; }`\n"
            "- Post-proceso: softmax sobre logits si el modelo los devuelve (verificar con `netron`)\n"
            "- Test de humo: tensores de ceros → predicción válida\n\n"
            "**Criterio de éxito Fase 3:** predicciones con el modelo real en 20 casos. AUC > 0.5."
        ),
        "labels": ["fase-3", "cpp", "onnx", "bloqueante"],
        "milestone": "Fase 3 — Red neuronal e inferencia ONNX",
    },
    {
        "title": "[F3] Implementar FeatureExtractor — activaciones de capas intermedias",
        "body": (
            "- `scripts/add_intermediate_outputs.py`: reexportar el modelo con capas `layer3` y `layer4` como outputs adicionales (ResNet-50)\n"
            "- Guardar como `breast_cnn_with_features.onnx`\n"
            "- `FeatureExtractor.h/.cpp`: cargar modelo extendido, extraer activaciones de la capa solicitada\n"
            "- Devolver `std::vector<float>` con activaciones aplanadas"
        ),
        "labels": ["fase-3", "cpp", "onnx", "xai"],
        "milestone": "Fase 3 — Red neuronal e inferencia ONNX",
    },

    # ── FASE 4 · XAI ──────────────────────────────────────────────────────
    {
        "title": "[F4] Lectura teórica XAI: Grad-CAM, Integrated Gradients, UMAP",
        "body": (
            "- \"Grad-CAM\" — Selvaraju et al. (2017)\n"
            "- \"Axiomatic Attribution for Deep Networks\" (Integrated Gradients) — Sundararajan et al. (2017)\n"
            "- Entender diferencia entre saliencia por perturbación, por gradiente y por IG\n"
            "- \"UMAP\" — McInnes et al. (2018), secciones 1 y 3"
        ),
        "labels": ["fase-4", "xai", "investigación"],
        "milestone": "Fase 4 — XAI / Explicabilidad",
    },
    {
        "title": "[F4] Implementar GradCAM por perturbación",
        "body": (
            "Método `compute(VolumeData&, tda_features, block_size=8)`:\n"
            "1. Baseline: `confidence_0 = model.run(image_features, tda).confidence`\n"
            "2. Para cada bloque 3D `block_size³`: zeroing de features, reejecutar, `Δconf = conf_0 - conf_new`\n"
            "3. Asignar `Δconf` a los vóxeles del bloque\n"
            "4. Normalizar a [0,1]\n"
            "5. Devolver `SaliencyMap` con las mismas dimensiones que VolumeData\n\n"
            "**Validación cualitativa:** exportar slice central de un caso maligno real como PNG. ¿El highlight cae sobre las calcificaciones visibles?"
        ),
        "labels": ["fase-4", "cpp", "xai", "bloqueante"],
        "milestone": "Fase 4 — XAI / Explicabilidad",
    },
    {
        "title": "[F4] Implementar Integrated Gradients como método alternativo de saliencia",
        "body": (
            "Método `computeIG(steps=50)`:\n"
            "1. Baseline = volumen de ceros (o ruido uniforme bajo)\n"
            "2. N interpolaciones lineales baseline→input\n"
            "3. Para cada interpolación: evaluar modelo, medir variación de score\n"
            "4. Integración trapezoidal\n\n"
            "- Comparar visualmente IG vs perturbación en el mismo caso real\n"
            "- Documentar cuál produce mapas más coherentes con la anatomía visible"
        ),
        "labels": ["fase-4", "cpp", "xai"],
        "milestone": "Fase 4 — XAI / Explicabilidad",
    },
    {
        "title": "[F4] Implementar ManifoldProjector — PCA y UMAP del espacio latente",
        "body": (
            "- `fitPCA(features_matrix)`: SVD Jacobi con Eigen3\n"
            "- `projectPCA(features_matrix)` → `std::vector<ProjectedPoint>`\n"
            "- `fitUMAP` / `projectUMAP`: llamar a script Python via `subprocess`, leer CSV resultado\n\n"
            "**Exploración con datos reales:** proyectar feature vectors de los 20 casos. ¿Se separan benignos de malignos? ¿Qué componente principal tiene más poder discriminativo?\n"
            "Comparar PCA vs UMAP documentando resultados.\n\n"
            "**Criterio de éxito Fase 4:** GradCAM de ≥1 caso maligno real destaca visualmente calcificaciones. PCA de 20 casos muestra alguna separación entre clases."
        ),
        "labels": ["fase-4", "cpp", "xai", "investigación"],
        "milestone": "Fase 4 — XAI / Explicabilidad",
    },

    # ── FASE 5 · OpenGL ───────────────────────────────────────────────────
    {
        "title": "[F5] Lectura teórica OpenGL 4.6 y volume rendering",
        "body": (
            "- Completar learnopengl.com caps. 1-5 — escribir los ejemplos a mano\n"
            "- Entender: VAO/VBO, pipeline de rasterización, uniform buffers\n"
            "- Entender: texturas 3D (`GL_TEXTURE_3D`), formatos `GL_R32F`, `GL_RGBA8`\n"
            "- Leer \"Real-Time Volume Graphics\" cap. 1-2 (ray casting DVR)"
        ),
        "labels": ["fase-5", "opengl", "investigación"],
        "milestone": "Fase 5 — Render OpenGL 4.6",
    },
    {
        "title": "[F5] Implementar RenderEngine — motor GLFW + layout 3 paneles",
        "body": (
            "- `init()`: ventana 1920×1080, OpenGL 4.6 Core Profile, MSAA 4x\n"
            "- Bucle: `pollEvents → updateState → render → swapBuffers`\n"
            "- Layout 3 paneles con `glViewport()`:\n"
            "  - Izquierdo (50%): volumen 3D\n"
            "  - Central (25%): diagrama de persistencia\n"
            "  - Derecho (25%): espacio latente\n"
            "- Callbacks: Escape cierra; W/A/S/D orbitan cámara; click en diagrama → selección\n"
            "- Sistema de cámara azimutal + elevación + distancia → matriz view 4×4"
        ),
        "labels": ["fase-5", "cpp", "opengl", "bloqueante"],
        "milestone": "Fase 5 — Render OpenGL 4.6",
    },
    {
        "title": "[F5] Implementar VolumeRenderer — ray casting con shaders GLSL",
        "body": (
            "- GPU upload: `GL_TEXTURE_3D` con `GL_R32F` para VolumeData real y SaliencyMap\n"
            "- Textura 1D para transfer function calibrada con HU reales de CBIS-DDSM:\n"
            "  - Tejido graso → transparente/amarillo tenue\n"
            "  - Tejido glandular → blanco semi-transparente\n"
            "  - Calcificaciones → cian brillante opaco\n"
            "- VAO/VBO cubo unitario [0,1]³, 12 triángulos\n"
            "- Shader `volume_raycast.vert`: transformar con MVP\n"
            "- Shader `volume_raycast.frag`: ray marching, compositing front-to-back, early termination (α > 0.95), overlay GradCAM\n"
            "- Uniforms: `u_mvp`, `u_model_inv`, `u_cam_pos`, `u_volume`, `u_saliency`, `u_transfer`, `u_step_size`, `u_show_gradcam`\n\n"
            "**Validación:** calcificaciones visibles como puntos cian sobre tejido. Si todo blanco/negro → rango de normalización mal calibrado."
        ),
        "labels": ["fase-5", "cpp", "opengl", "bloqueante"],
        "milestone": "Fase 5 — Render OpenGL 4.6",
    },
    {
        "title": "[F5] Implementar DiagramRenderer — scatter plot del diagrama de persistencia",
        "body": (
            "- Shader `diagram.vert/.frag`: puntos 2D en [0,1]²\n"
            "- VBO con `vec4(birth, death, dimension, persistence)` por punto\n"
            "- Color por dimensión: H0=azul `#4477CC`, H1=verde `#44BB88`, H2=rojo `#CC4444`\n"
            "- `gl_PointSize` proporcional a persistencia\n"
            "- Línea diagonal `y=x` con `GL_LINES`\n"
            "- Escala automática de ejes (bbox + 10% margen)\n"
            "- Highlight del punto seleccionado en amarillo"
        ),
        "labels": ["fase-5", "cpp", "opengl"],
        "milestone": "Fase 5 — Render OpenGL 4.6",
    },
    {
        "title": "[F5] Implementar ManifoldRenderer — scatter 3D del espacio latente",
        "body": (
            "- Shader `manifold.vert/.frag`: puntos 3D orbitables\n"
            "- VBO con `vec3(x,y,z)` + `float label` + `float confidence`\n"
            "- Color: benigno→azul, maligno→rojo, interpolado por confianza\n"
            "- Cámara propia: botón derecho + drag para orbitar\n"
            "- Hit test para hover: punto más cercano al cursor en 2D proyectado\n"
            "- Tooltip con label y confianza al hover\n\n"
            "**Criterio de éxito Fase 5:** ventana con 3 paneles funcionando. Caso real renderizado con calcificaciones visibles. Manifold con 20 casos con colores por clase."
        ),
        "labels": ["fase-5", "cpp", "opengl"],
        "milestone": "Fase 5 — Render OpenGL 4.6",
    },

    # ── FASE 6 · ImGui ────────────────────────────────────────────────────
    {
        "title": "[F6] Integrar Dear ImGui con GLFW + OpenGL 4.6",
        "body": (
            "- Integrar `imgui_impl_glfw.cpp` + `imgui_impl_opengl3.cpp`\n"
            "- Inicializar en `RenderEngine::init()`: `CreateContext()`, `InitForOpenGL()`, `Init(\"#version 460\")`\n"
            "- Bucle: `NewFrame()` → lógica UI → `Render()` → `RenderDrawData()`\n"
            "- Verificar que la demo window funciona"
        ),
        "labels": ["fase-6", "cpp", "ui", "bloqueante"],
        "milestone": "Fase 6 — Dear ImGui UI",
    },
    {
        "title": "[F6] Implementar panel de control del volumen",
        "body": (
            "- Slider \"Intensity Threshold\" [0,1] → regenera PointCloud al soltar\n"
            "- Slider \"Ray Step Size\" [0.001, 0.01] → calidad en tiempo real\n"
            "- Dropdown \"Transfer Function\": breast_default / calcification / custom\n"
            "- Toggle \"Show GradCAM\" → activa/desactiva overlay de saliencia\n"
            "- Botón \"Recompute TDA\" → relanza VietorisRips en thread separado con barra de progreso\n"
            "- Mostrar `patient_id` y `series_uid` del caso cargado"
        ),
        "labels": ["fase-6", "cpp", "ui"],
        "milestone": "Fase 6 — Dear ImGui UI",
    },
    {
        "title": "[F6] Implementar panel del diagrama, panel del manifold y ventana principal",
        "body": (
            "**Panel diagrama:**\n"
            "- Estadísticas: nº pares H0/H1/H2, rango de persistencia\n"
            "- Slider \"Min Persistence\" → filtra en tiempo real\n"
            "- Info del punto seleccionado: birth, death, dimensión, persistencia\n\n"
            "**Panel manifold:**\n"
            "- Radio buttons: PCA / UMAP (recalcula al cambiar)\n"
            "- Resultado de clasificación: clase, confianza, barra de probabilidad\n"
            "- Botón \"Export CSV\" a `output/`\n"
            "- Colores por ground truth de CBIS-DDSM\n\n"
            "**Ventana principal:**\n"
            "- Menu bar: File > Open DICOM / Open MHA / Open Case Folder / Exit\n"
            "- Status bar: FPS, tiempo último TDA, nº casos cargados\n\n"
            "**Criterio de éxito Fase 6:** 20 casos cargados y navegables desde la UI sin tocar el terminal."
        ),
        "labels": ["fase-6", "cpp", "ui"],
        "milestone": "Fase 6 — Dear ImGui UI",
    },

    # ── FASE 7 · Enlace entre vistas ──────────────────────────────────────
    {
        "title": "[F7] Implementar mapeo simplex→vóxel durante VietorisRips",
        "body": (
            "- Durante `compute()`: guardar `point_idx → voxel_idx` para cada punto\n"
            "- Identificar simplice crítico de cada par `(birth, death, dim)`\n"
            "- `map<PersistencePair, vector<int> voxel_indices>` — mapeo completo\n"
            "- `voxelsForPair(birth, death, dim)` → `vector<int>`"
        ),
        "labels": ["fase-7", "cpp", "tda", "bloqueante"],
        "milestone": "Fase 7 — Enlace entre vistas",
    },
    {
        "title": "[F7] Implementar selección diagrama → highlight en volumen",
        "body": (
            "- Click en panel central → par `(birth,death)` más cercano al cursor\n"
            "- `voxelsForPair()` → índices de vóxeles\n"
            "- Textura `GL_R8` adicional para máscara de selección en VolumeRenderer\n"
            "- Shader: si `texture(u_selection).r > 0.5` → mezclar con amarillo brillante\n"
            "- Animación: pulso sinusoidal para hacer el highlight visible"
        ),
        "labels": ["fase-7", "cpp", "opengl"],
        "milestone": "Fase 7 — Enlace entre vistas",
    },
    {
        "title": "[F7] Implementar hover volumen → resaltado en diagrama (bidireccional)",
        "body": (
            "- Vóxel bajo el rayo del cursor → buscar en mapa inverso los pares que lo contienen\n"
            "- Resaltar esos pares en DiagramRenderer\n\n"
            "**Criterio de éxito Fase 7:** click en un punto del diagrama de un caso real ilumina la región de tejido correspondiente en el volumen."
        ),
        "labels": ["fase-7", "cpp", "opengl"],
        "milestone": "Fase 7 — Enlace entre vistas",
    },

    # ── FASE 8 · Análisis y validación ────────────────────────────────────
    {
        "title": "[F8] Ampliar dataset a ~70 casos (incluir masas)",
        "body": (
            "- Descargar 50 casos adicionales de CBIS-DDSM (total ~70)\n"
            "- Incluir casos de masa para comparar topología vs calcificaciones\n"
            "- Organizar: `calc_benign`, `calc_malignant`, `mass_benign`, `mass_malignant`"
        ),
        "labels": ["fase-8", "dataset"],
        "milestone": "Fase 8 — Análisis comparativo y validación",
    },
    {
        "title": "[F8] Análisis estadístico de diagramas de persistencia (70 casos)",
        "body": (
            "Script Python `scripts/analyze_features.py`:\n"
            "- Computar feature vectors de los 70 casos → `output/features.csv`\n"
            "- t-SNE / PCA: ¿se separan las clases?\n"
            "- Test Mann-Whitney U entre benignos y malignos para cada bin del histograma\n"
            "- Identificar qué features topológicos (rango de persistencia, dimensión) tienen mayor poder discriminativo\n"
            "- Documentar hallazgos en `ANALYSIS.md`"
        ),
        "labels": ["fase-8", "python", "investigación", "documentación"],
        "milestone": "Fase 8 — Análisis comparativo y validación",
    },
    {
        "title": "[F8] Benchmarks de rendimiento del pipeline completo",
        "body": (
            "- Medir tiempo de cada etapa en los 70 casos: carga, PointCloud, Rips, ONNX, GradCAM\n"
            "- Tabular en `BENCHMARKS.md`: hardware, tamaño de volumen, tiempos por etapa\n"
            "- Identificar el cuello de botella principal y proponer optimización"
        ),
        "labels": ["fase-8", "rendimiento", "documentación"],
        "milestone": "Fase 8 — Análisis comparativo y validación",
    },
    {
        "title": "[F8] Validación cualitativa del GradCAM contra ROIs anotadas de CBIS-DDSM",
        "body": (
            "- Para 10 casos malignos: comparar mapa GradCAM con la segmentación ROI incluida en CBIS-DDSM\n"
            "- Calcular IoU entre máscara GradCAM (threshold > 0.5) y la ROI anotada\n"
            "- Si IoU < 0.2: revisar el modelo o el método GradCAM\n\n"
            "**Criterio de éxito Fase 8:** análisis estadístico documentado. GradCAM con IoU > 0.2 en al menos 7/10 casos malignos."
        ),
        "labels": ["fase-8", "xai", "investigación"],
        "milestone": "Fase 8 — Análisis comparativo y validación",
    },

    # ── MEJORAS OPCIONALES ────────────────────────────────────────────────
    {
        "title": "[OPC] Paralelizar GradCAM con std::async / thread pool",
        "body": "- Perfilar con `perf` / `gprof` para identificar hotspots\n- Paralelizar GradCAM con `std::async` o thread pool\n- Cachear feature vectors TDA en binario entre ejecuciones\n- Investigar GUDHI GPU si hay soporte experimental",
        "labels": ["opcional", "rendimiento", "cpp"],
        "milestone": "Mejoras opcionales",
    },
    {
        "title": "[OPC] Algoritmos TDA adicionales: Mapper, Persistent Entropy, Alpha Complex",
        "body": "- TDA Mapper para visualizar estructura global del dataset de 70 casos\n- Persistent Entropy como feature adicional en `featureVector`\n- Alpha Complexes como alternativa a Vietoris-Rips (más eficiente en puntos uniformes)\n- Wasserstein distance completa via librería Hera",
        "labels": ["opcional", "tda", "investigación"],
        "milestone": "Mejoras opcionales",
    },
    {
        "title": "[OPC] Rendering avanzado: SSAO, brush tool, exportar a PNG/EXR, overlay ROI",
        "body": "- SSAO en el ray caster del volumen\n- Brush tool interactivo para pintar vóxeles en OpenGL\n- Exportar render a PNG/EXR con `glReadPixels` para figuras de publicación\n- Superposición de la ROI anotada de CBIS-DDSM sobre el volumen renderizado",
        "labels": ["opcional", "opengl", "cpp"],
        "milestone": "Mejoras opcionales",
    },
    {
        "title": "[OPC] CI/CD: GitHub Actions, cobertura gcov/lcov, Doxygen, serialización JSON",
        "body": "- GitHub Actions: compilar en Ubuntu + Arch, correr `ctest`\n- Cobertura de código con `gcov` / `lcov`\n- Doxygen para API pública\n- Serialización completa de sesión (JSON): parámetros, diagrama, proyección, caso cargado",
        "labels": ["opcional", "infraestructura", "documentación"],
        "milestone": "Mejoras opcionales",
    },
    {
        "title": "[OPC] C++20: Ranges, Concepts, Coroutines, Regla de los 5",
        "body": "- Ranges y views: `std::views::filter`, `std::views::transform` — usar en pipeline de PointCloud\n- Concepts: definir `concept Volumetric` que requiera `at()` y `dims()`\n- Coroutines: explorar para pipeline asíncrono de GradCAM\n- Regla de los 5 en clases con recursos GPU\n- RAII consistente: todos los recursos OpenGL liberados en destructores",
        "labels": ["opcional", "cpp", "investigación"],
        "milestone": "Mejoras opcionales",
    },
]

# ---------------------------------------------------------------------------
# Funciones auxiliares (idénticas al script cv-lit)
# ---------------------------------------------------------------------------

def gh_headers(token: str) -> dict:
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }


def create_labels(repo: str, token: str, dry_run: bool) -> None:
    url = f"https://api.github.com/repos/{repo}/labels"
    existing = {l["name"] for l in requests.get(url, headers=gh_headers(token)).json() if isinstance(l, dict)}
    for label in LABELS:
        if label["name"] in existing:
            print(f"  [SKIP]  label '{label['name']}' ya existe")
            continue
        if dry_run:
            print(f"  [DRY]   crearía label '{label['name']}'")
            continue
        r = requests.post(url, headers=gh_headers(token), json=label)
        if r.status_code == 201:
            print(f"  [OK]    label '{label['name']}' creado")
        else:
            print(f"  [ERROR] label '{label['name']}': {r.status_code} {r.text}")
        time.sleep(0.3)


def create_milestones(repo: str, token: str, dry_run: bool) -> dict:
    url = f"https://api.github.com/repos/{repo}/milestones"
    existing = {m["title"]: m["number"] for m in requests.get(url, headers=gh_headers(token)).json() if isinstance(m, dict)}
    result = dict(existing)
    for ms in MILESTONES:
        if ms["title"] in existing:
            print(f"  [SKIP]  milestone '{ms['title']}' ya existe (#{existing[ms['title']]})")
            continue
        if dry_run:
            print(f"  [DRY]   crearía milestone '{ms['title']}'")
            result[ms["title"]] = 0
            continue
        r = requests.post(url, headers=gh_headers(token), json=ms)
        if r.status_code == 201:
            number = r.json()["number"]
            result[ms["title"]] = number
            print(f"  [OK]    milestone '{ms['title']}' creado (#{number})")
        else:
            print(f"  [ERROR] milestone '{ms['title']}': {r.status_code} {r.text}")
        time.sleep(0.3)
    return result


def create_issues(repo: str, token: str, dry_run: bool, delay: float, milestone_map: dict) -> None:
    url = f"https://api.github.com/repos/{repo}/issues"
    existing_titles: set = set()
    page = 1
    while True:
        r = requests.get(url, headers=gh_headers(token), params={"state": "all", "per_page": 100, "page": page})
        data = r.json()
        if not data or not isinstance(data, list):
            break
        existing_titles.update(i["title"] for i in data if isinstance(i, dict))
        if len(data) < 100:
            break
        page += 1

    for issue in ISSUES:
        if issue["title"] in existing_titles:
            print(f"  [SKIP]  '{issue['title']}' ya existe")
            continue
        if dry_run:
            print(f"  [DRY]   crearía: {issue['title']}")
            continue
        payload: dict = {
            "title": issue["title"],
            "body": issue["body"],
            "labels": issue["labels"],
        }
        if issue.get("milestone") and issue["milestone"] in milestone_map:
            payload["milestone"] = milestone_map[issue["milestone"]]
        r = requests.post(url, headers=gh_headers(token), json=payload)
        if r.status_code == 201:
            print(f"  [OK]    #{r.json()['number']} — {issue['title']}")
        else:
            print(f"  [ERROR] {issue['title']}: {r.status_code} {r.text}")
        time.sleep(delay)


def main() -> None:
    parser = argparse.ArgumentParser(description="Crear issues de MammoViz-TDA en GitHub")
    parser.add_argument("--repo",    required=True, help="owner/repo, e.g. nickhernd/mammoviz-tda")
    parser.add_argument("--dry-run", action="store_true", help="No crea nada, solo imprime lo que haría")
    parser.add_argument("--delay",   type=float, default=1.0, help="Segundos entre peticiones (default: 1)")
    args = parser.parse_args()

    token = os.environ.get("GITHUB_TOKEN")
    if not token and not args.dry_run:
        raise SystemExit("ERROR: define la variable de entorno GITHUB_TOKEN.")

    print(f"\n{'[DRY RUN] ' if args.dry_run else ''}Repositorio: {args.repo}\n")
    print(f"  Issues a crear: {len(ISSUES)}")
    print(f"  Labels:         {len(LABELS)}")
    print(f"  Milestones:     {len(MILESTONES)}\n")

    print("── Creando labels ──────────────────────────────────────────")
    create_labels(args.repo, token or "", args.dry_run)

    print("\n── Creando milestones ──────────────────────────────────────")
    milestone_map = create_milestones(args.repo, token or "", args.dry_run)

    print("\n── Creando issues ──────────────────────────────────────────")
    create_issues(args.repo, token or "", args.dry_run, args.delay, milestone_map)

    print("\nListo.")


if __name__ == "__main__":
    main()
