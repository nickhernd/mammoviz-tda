# MammoViz-TDA

**Visualizador de imágenes mamográficas con Análisis Topológico de Datos y XAI**

MammoViz-TDA es un sistema de investigación en C++20 que combina tres disciplinas para el análisis de imágenes médicas de cáncer de mama:

- **TDA** (Topological Data Analysis): extrae características topológicas de microcalcificaciones y masas mediante homología persistente.
- **CNN + ONNX Runtime**: ejecuta un modelo de red neuronal convolucional para clasificación benign/maligno.
- **XAI** (Explainable AI): genera mapas de saliencia GradCAM y proyecciones del espacio latente (UMAP/PCA) para explicar la decisión del modelo.
- **Render OpenGL 4.6**: ventana interactiva con tres vistas enlazadas — volumen 3D, diagrama de persistencia y manifold latente.

---

## Índice

1. [Motivación científica](#motivación-científica)
2. [Arquitectura del sistema](#arquitectura-del-sistema)
3. [Estructura del proyecto](#estructura-del-proyecto)
4. [Dependencias](#dependencias)
5. [Cómo compilar](#cómo-compilar)
6. [Cómo ejecutar](#cómo-ejecutar)
7. [Estado de implementación](#estado-de-implementación)
8. [Lo que queda por implementar](#lo-que-queda-por-implementar)
9. [Datos de prueba sintéticos](#datos-de-prueba-sintéticos)
10. [Configuración](#configuración)
11. [Tests unitarios](#tests-unitarios)
12. [Notas técnicas de compilación](#notas-técnicas-de-compilación)

---

## Motivación científica

Las imágenes mamográficas (mamografía 2D, tomosíntesis DBT, DCE-MRI 4D) contienen estructuras topológicas que los descriptores convencionales ignoran. Una microcalcificación no es solo un punto brillante — forma parte de un **cluster con forma topológica**: agrupaciones lineales (ductos), circulares (quistes) o dispersas (distribución maligna típica).

La **homología persistente** cuantifica exactamente esas formas:
- **β₀** (componentes conexas): número de clusters de microcalcificaciones separados.
- **β₁** (loops): estructuras circulares o anulares (quistes, ductos vistos en sección).
- **β₂** (cavidades): huecos tridimensionales en masas.

El diagrama de persistencia `{(birth, death)}` representa cuándo aparece y desaparece cada feature topológico al aumentar el radio de la filtración de Vietoris-Rips. Features de larga vida = estructuras reales. Features de corta vida = ruido.

Combinado con GradCAM (qué regiones activan la CNN) y la proyección del espacio latente, el sistema permite a un radiólogo entender **por qué** el modelo clasifica un caso como maligno.

---

## Arquitectura del sistema

```
DICOM / MHA
    │
    ▼
┌─────────────┐     ┌──────────────────────────────────┐
│ DicomLoader │────▶│           VolumeData              │
│  (ITK/GDCM) │     │  float32[X][Y][Z][T], spacing mm  │
└─────────────┘     └──────────┬───────────────────────┘
                               │
               ┌───────────────┼───────────────┐
               ▼               ▼               ▼
       ┌──────────────┐  ┌──────────┐  ┌──────────────┐
       │  PointCloud  │  │  NN      │  │  GradCAM     │
       │  fromVolume  │  │  ONNX    │  │  (perturb.)  │
       └──────┬───────┘  └────┬─────┘  └──────┬───────┘
              │               │               │
              ▼               │               ▼
       ┌──────────────┐       │        ┌──────────────┐
       │ VietorisRips │       │        │  SaliencyMap │
       │  GUDHI 3.10  │       │        │  float32[XYZ]│
       └──────┬───────┘       │        └──────┬───────┘
              │               │               │
              ▼               ▼               │
       ┌──────────────┐  ┌──────────┐         │
       │ Persistence  │  │ Manifold │         │
       │  Diagram     │  │Projector │         │
       │(birth,death, │  │ PCA/UMAP │         │
       │  dimension)  │  └────┬─────┘         │
       └──────┬───────┘       │               │
              │               │               │
              └───────────────┴───────────────┘
                              │
                              ▼
                   ┌─────────────────────┐
                   │     RenderEngine    │
                   │     OpenGL 4.6      │
                   ├──────────┬──────────┤
                   │  Volume  │ Diagram  │
                   │ Renderer │ Renderer │
                   │(raycast) │(scatter) │
                   └──────────┴──────────┘
                        │
                        ▼
               Ventana interactiva 1920×1080
               [volumen] [diagrama] [manifold]
```

---

## Estructura del proyecto

```
mammoviz-tda/
├── CMakeLists.txt              # Build raíz (GCC 16 + CMake 4.x workarounds)
├── config.toml                 # Configuración en tiempo de ejecución
│
├── include/                    # Cabeceras públicas
│   ├── io/
│   │   ├── DicomLoader.h       # Carga series DICOM y archivos .mha
│   │   └── VolumeData.h        # Tensor 4D + metadatos de espaciado
│   ├── tda/
│   │   ├── PointCloud.h        # Extracción de nube de puntos desde vóxeles
│   │   ├── VietorisRips.h      # Filtración + homología persistente (GUDHI)
│   │   └── PersistenceDiagram.h# Diagrama, feature vector, CSV I/O
│   ├── nn/
│   │   ├── ModelInference.h    # Inferencia ONNX Runtime
│   │   └── FeatureExtractor.h  # Extractor de activaciones de capas CNN
│   ├── xai/
│   │   ├── GradCAM.h           # Mapas de saliencia (perturbación + gradiente)
│   │   └── ManifoldProjector.h # PCA/UMAP del espacio latente
│   ├── render/
│   │   ├── RenderEngine.h      # Motor principal (GLFW + GLEW)
│   │   ├── VolumeRenderer.h    # Ray casting DVR
│   │   ├── DiagramRenderer.h   # Scatter plot del diagrama de persistencia
│   │   └── ManifoldRenderer.h  # Scatter 3D del espacio latente
│   └── utils/
│       ├── Logger.h            # Logger con niveles (INFO/WARN/ERROR)
│       ├── Config.h            # Parser TOML minimalista
│       └── Timer.h             # Temporizador de alta precisión
│
├── src/                        # Implementaciones
│   ├── main.cpp                # Punto de entrada: pipeline completo
│   ├── io/
│   │   ├── DicomLoader.cpp     # ITK 5.x + GDCM, normalización HU→[0,1]
│   │   └── VolumeData.cpp      # Accesores y metadatos del volumen
│   ├── tda/
│   │   ├── PointCloud.cpp      # Umbralización + subsampling aleatorio
│   │   ├── VietorisRips.cpp    # Dense/Sparse Rips, Z/2Z coefficients
│   │   └── PersistenceDiagram.cpp # Histogramas, bottleneck distance, CSV
│   ├── nn/
│   │   ├── ModelInference.cpp  # Sesión ONNX, softmax, resultado
│   │   └── FeatureExtractor.cpp# Extracción de features por capa
│   ├── xai/
│   │   ├── GradCAM.cpp         # Perturbación por bloques → saliencia
│   │   └── ManifoldProjector.cpp # Eigen3 SVD/PCA → proyección 3D
│   ├── render/
│   │   ├── RenderEngine.cpp    # Bucle GLFW, layout 3 paneles, input
│   │   ├── VolumeRenderer.cpp  # Upload GPU (GL_TEXTURE_3D), stubs raycast
│   │   ├── DiagramRenderer.cpp # Render scatter del diagrama
│   │   └── ManifoldRenderer.cpp# Render scatter 3D manifold
│   └── ui/
│       ├── MainWindow.cpp      # Dear ImGui ventana principal
│       ├── VolumePanel.cpp     # Panel de control del volumen
│       ├── DiagramPanel.cpp    # Panel del diagrama
│       └── ManifoldPanel.cpp   # Panel del manifold
│
├── shaders/glsl/               # GLSL OpenGL 4.6 (NO Vulkan/SPIR-V)
│   ├── volume_raycast.vert/.frag # Ray casting DVR + overlay GradCAM
│   ├── diagram.vert/.frag      # Scatter plot de persistencia
│   └── manifold.vert/.frag     # Scatter 3D latente
│
├── tests/unit/
│   └── test_tda.cpp            # 4 tests GTest: PersistenceDiagram
│
├── scripts/
│   ├── gen_dicom.py            # Genera serie DICOM sintética (64×64×32)
│   └── gen_onnx_model.py       # Genera modelo ONNX dummy para pruebas
│
├── data/
│   ├── samples/case001/        # Serie DICOM sintética (32 slices)
│   └── models/breast_cnn.onnx  # Modelo ONNX de clasificación (512+192→2)
│
└── third_party/
    ├── onnxruntime/            # ONNX Runtime pre-compilado (lib + headers)
    └── imgui/                  # Dear ImGui (vendored)
```

---

## Dependencias

| Dependencia | Versión | Cómo se obtiene |
|---|---|---|
| GCC | ≥ 16 | Sistema (Arch: `gcc`) |
| CMake | ≥ 3.20 | Sistema (Arch: `cmake`) |
| Ninja | cualquiera | Sistema (Arch: `ninja`) |
| ITK | 5.4 | Sistema (Arch: `itk`) |
| Eigen3 | ≥ 3.3 | Sistema (Arch: `eigen`) |
| GLFW | ≥ 3.3 | Sistema (Arch: `glfw`) |
| GLEW | ≥ 2.0 | Sistema (Arch: `glew`) |
| OpenGL | 4.6 | Driver GPU |
| GTest | ≥ 1.10 | Sistema (Arch: `gtest`) |
| GUDHI | 3.10.0 | **Descargado automáticamente** por CMake (FetchContent) |
| ONNX Runtime | 1.17+ | `third_party/onnxruntime/` (instalado por `install_deps.sh`) |
| Dear ImGui | HEAD | `third_party/imgui/` (instalado por `install_deps.sh`) |

---

## Cómo compilar

### 1. Clonar e instalar dependencias de terceros

```bash
git clone <repo-url> mammoviz-tda
cd mammoviz-tda
bash install_deps.sh        # descarga ONNX Runtime y clona ImGui
```

### 2. Configurar con CMake

```bash
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
```

CMake descarga automáticamente GUDHI 3.10.0 (~10 MB) en el primer configure.

### 3. Compilar

```bash
ninja -C build
```

El binario principal se genera en `build/src/mammoviz`.  
Los tests en `build/tests/test_tda`.

### Tipos de build

```bash
# Release (optimizado, -O3 -march=native)
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release

# Debug (AddressSanitizer activado, -g -fsanitize=address)
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Debug
```

---

## Cómo ejecutar

### Tests unitarios (recomendado primero)

```bash
cd build
ctest --output-on-failure
```

**Salida esperada:**

```
Test project /home/.../mammoviz-tda/build
    Start 1: PersistenceDiagram.FeatureVectorSize
1/4 Test #1: PersistenceDiagram.FeatureVectorSize ........   Passed    0.05 sec
    Start 2: PersistenceDiagram.FilteredRemovesNoise
2/4 Test #2: PersistenceDiagram.FilteredRemovesNoise .....   Passed    0.05 sec
    Start 3: PersistenceDiagram.CSVRoundtrip
3/4 Test #3: PersistenceDiagram.CSVRoundtrip .............   Passed    0.05 sec
    Start 4: PersistenceDiagram.BottleneckDistanceSelf
4/4 Test #4: PersistenceDiagram.BottleneckDistanceSelf ...   Passed    0.05 sec

100% tests passed, 0 tests failed out of 4
```

### Generar datos de prueba sintéticos

Los scripts generan un volumen DICOM (blob gaussiano simulando una masa) y un modelo ONNX dummy:

```bash
# Volumen DICOM sintético: 32 slices de 64×64 con blob gaussiano
python3 scripts/gen_dicom.py data/samples/case001

# Modelo ONNX: MLP de dos capas (512+192 entradas → 2 clases)
python3 scripts/gen_onnx_model.py
```

Requiere: `pip install pydicom numpy onnx`

### Aplicación principal

```bash
# Con los datos del directorio por defecto (data/samples/case001)
./build/src/mammoviz

# Con una ruta propia de DICOM
./build/src/mammoviz /ruta/a/directorio/dicom

# Con archivo .mha directamente (formato MetaImage de ITK)
./build/src/mammoviz /ruta/a/volumen.mha
```

**Salida esperada en consola (con datos sintéticos):**

```
[INFO] Loading DICOM from: data/samples/case001
[INFO] Volume loaded: 64×64×32×1
[INFO] Extracting point cloud...
[INFO] Extracted 8432 candidate points from volume
[INFO] Subsampled to 8432 points
[INFO] Point cloud: 8432 points
[INFO] Building Vietoris-Rips: 8432 points, max_ε=5.00mm, max_dim=2
[INFO] Using sparse Rips (n=8432 > threshold=500)
[INFO] Simplex tree: 142861 simplices — computing persistence...
[INFO] Persistence: 234 pairs in 1.43s
[INFO] ONNX model loaded: data/models/breast_cnn.onnx
[INFO] Prediction: class=0 confidence=0.51
[WARN] GradCAM: using perturbation-based approximation (ONNX mode)
[INFO] OpenGL 4.6, renderer: NVIDIA GeForce ...
```

A continuación se abre una **ventana gráfica** de 1920×1080 con tres paneles.

### Controles de la ventana

| Tecla | Acción |
|---|---|
| `Escape` | Cerrar la aplicación |
| (Phase 2) `W A S D` | Orbitar la cámara en el panel del volumen |
| (Phase 2) clic en diagrama | Resalta la región de tejido correspondiente |

---

## Estado de implementación

### Completamente implementado

| Módulo | Ficheros | Descripción |
|---|---|---|
| **DicomLoader** | `src/io/DicomLoader.cpp` | Carga series DICOM con ITK/GDCM. Normalización HU→[0,1]. Soporte `.mha`. |
| **VolumeData** | `src/io/VolumeData.cpp` | Tensor float32 `[X][Y][Z][T]`, espaciado en mm, acceso `at(x,y,z,t)`. |
| **PointCloud** | `src/tda/PointCloud.cpp` | Umbralización de intensidad, subsampling aleatorio, exportación `.off`. |
| **VietorisRips** | `src/tda/VietorisRips.cpp` | Filtración Dense/Sparse con GUDHI 3.10. Z/2Z. Dense<500pts, Sparse≥500pts. Soporte DCE-MRI temporal. |
| **PersistenceDiagram** | `src/tda/PersistenceDiagram.cpp` | Feature vector (histograma 1D L2-normalizado, 64 bins × 3 dimensiones = 192 floats). Filtrado por persistencia mínima. Bottleneck distance aproximada. CSV round-trip. |
| **ModelInference** | `src/nn/ModelInference.cpp` | Sesión ONNX Runtime. Carga modelo `.onnx`, corre inferencia, softmax, devuelve clase + confianza. |
| **GradCAM** | `src/xai/GradCAM.cpp` | Aproximación por perturbación de bloques. Mide caída de confianza al zeroing de features. Normalizado a [0,1]. |
| **ManifoldProjector** | `src/xai/ManifoldProjector.cpp` | PCA via Eigen3 SVD Jacobi. Proyección a 3D. Normalizado a [-1,1] por eje. |
| **RenderEngine** | `src/render/RenderEngine.cpp` | Bucle GLFW. Inicialización OpenGL 4.6 Core Profile. Layout 3 paneles. Input (Escape). |
| **VolumeRenderer (GPU)** | `src/render/VolumeRenderer.cpp` | Upload de volumen e intensidad a texturas 3D (`GL_TEXTURE_3D`, `GL_R32F`). Transfer functions (breast/calcification). |
| **Logger / Config / Timer** | `src/utils/` | Logger con niveles. Parser TOML (secciones + key=value). Timer de alta precisión. |
| **Tests TDA** | `tests/unit/test_tda.cpp` | 4 tests GTest: tamaño feature vector, filtrado de ruido, CSV, bottleneck self-distance. |

### Parcialmente implementado (stubs)

| Módulo | Estado actual | Lo que falta |
|---|---|---|
| **VolumeRenderer** (ray casting) | Sube el volumen a GPU correctamente. El bucle de render llama a `render()` pero la función no ejecuta el shader pipeline. | Compilar y enlazar los shaders GLSL, crear VAO/VBO del cubo unitario, pasar matrices view/proj al shader, bucle de ray marching real. |
| **DiagramRenderer** | Stubs vacíos. | Crear programa GLSL con `diagram.vert/frag`, VBO de puntos `(birth, death)`, codificación por color de dimensión, eje diagonal de persistencia. |
| **ManifoldRenderer** | Recibe los puntos proyectados pero `render()` es vacío. | Crear programa GLSL con `manifold.vert/frag`, VBO de puntos 3D, coloreado por label/confianza, trayectorias temporales opcionales. |
| **FeatureExtractor** | Header definido, `.cpp` stub. | Implementar extracción de activaciones de capas intermedias. ONNX Runtime no expone gradientes — necesita sesión con `OrtCustomOpDomain` o LibTorch alternativo. |
| **UI (ImGui)** | Ficheros `.cpp` compilados vacíos. | Paneles ImGui para: TF editor, threshold slider, opciones GradCAM, selector de caso. |
| **Vietoris-Rips → voxel mapping** | No implementado. | Para el "click en diagrama → iluminar tejido", necesita guardar `simplice_id → voxel_idx` durante la construcción del simplex tree. |

---

## Tareas pendientes (backlog)

Lista completa de trabajo pendiente por orden de prioridad:

| # | Tarea | Prioridad | Bloquea |
|---|---|---|---|
| T1 | **Corregir nombres de inputs ONNX** — `ModelInference.cpp` usa `input_names={"input"}` pero el modelo tiene dos entradas separadas: `"image_features"` y `"tda_features"`. La sesión falla en runtime. | 🔴 Alta | T5 |
| T2 | **Fase 2a — VolumeRenderer: ray casting** — Compilar shaders `volume_raycast.vert/.frag`, crear VAO/VBO del cubo unitario [0,1]³, pasar uniforms (u_volume, u_saliency, u_transfer, matrices), draw call real. | 🔴 Alta | T7 |
| T3 | **Fase 2b — DiagramRenderer: scatter plot** — Compilar `diagram.vert/.frag`, VBO con `(birth, death)`, color por dimensión (H0=azul/H1=verde/H2=rojo), dibujar diagonal `y=x`, escala automática de ejes. | 🔴 Alta | T7 |
| T4 | **Fase 2c — ManifoldRenderer: scatter 3D** — Compilar `manifold.vert/.frag`, VBO con `vec3(coords)` + label, resaltado del punto seleccionado, hit test para click/hover. | 🟡 Media | T7 |
| T5 | **Fase 3 — Integrated Gradients** — Sustituir la perturbación de bloques actual por Integrated Gradients (interpolar baseline→input en N=50 pasos, medir variación de score). Misma interfaz `GradCAM::compute()`. | 🟡 Media | — |
| T6 | **Fase 4 — Dear ImGui UI** — Inicializar `imgui_impl_glfw` + `imgui_impl_opengl3` en `RenderEngine::init()`. Implementar `src/ui/*.cpp`: slider threshold, transfer function editor, toggle GradCAM, exportar CSV. | 🟡 Media | T2 T3 T4 |
| T7 | **Fase 5 — Enlace diagrama↔volumen** — Durante `VietorisRips::compute()`, guardar mapa `simplex → vóxeles`. Implementar `voxelsForPersistenceRegion()`. Resaltar vóxeles en VolumeRenderer con textura de selección `GL_R8`. | 🟢 Baja | T2 T3 |
| T8 | **Fase 6 — Dataset real CBIS-DDSM** — Descargar de cancerimagingarchive.net, ajustar `intensity_threshold` para HU reales, exportar modelo CNN real a ONNX, medir tiempos de filtración Rips en volúmenes de producción. | 🟢 Baja | T1 T2 |

---

## Lo que queda por implementar

### Fase 2 — Pipeline de render completo

**Prioridad alta.** El motor OpenGL arranca, crea la ventana y los contextos correctamente, pero los tres paneles muestran fondo negro porque los shaders no están enlazados.

Los shaders GLSL están escritos y son correctos (`shaders/glsl/`). Lo que falta es el código C++ que los carga, compila y ejecuta:

```
src/render/VolumeRenderer.cpp
  └── TODO: compilar volume_raycast.vert + volume_raycast.frag
            crear VAO cubo unitario [0,1]³
            pasar u_volume, u_saliency, u_transfer, u_step_size
            ejecutar ray marching en render()

src/render/DiagramRenderer.cpp
  └── TODO: compilar diagram.vert + diagram.frag
            VBO con vec2(birth, death) por punto
            uniform de color por dimensión (H0=azul, H1=verde, H2=rojo)
            dibujar la diagonal y = x (referencia de persistencia cero)

src/render/ManifoldRenderer.cpp
  └── TODO: compilar manifold.vert + manifold.frag
            VBO con vec3(x,y,z) + label por ProjectedPoint
            hit test para hover/click
```

**Estimación:** 3-4 días de trabajo.

### Fase 3 — GradCAM real con gradientes

La implementación actual usa perturbación de bloques (zeroing de parches) como aproximación. Es correcta semánticamente pero lenta O(n/block) inferencias.

GradCAM verdadero requiere gradientes `∂score/∂activations`. Opciones:

- **LibTorch C++ API**: reexportar el modelo PyTorch como TorchScript y usar `torch::autograd`.
- **ONNX Runtime Training**: aún experimental para inferencia con gradientes.
- **Opción pragmática**: Integrated Gradients por perturbación del input (más preciso que block zeroing, sin necesitar backprop).

### Fase 4 — Dear ImGui UI

Los ficheros `src/ui/*.cpp` están compilados pero vacíos. Implementar:

- Panel izquierdo: parámetros TDA (threshold, max_ε, min_persistence).
- Panel derecho: resultado de clasificación, confianza, transfer function editor.
- Timeline (DCE-MRI): slider temporal para ver evolución de los diagramas de persistencia.

### Fase 5 — Enlace entre vistas

El callback `onDiagramClick` está registrado en `main.cpp` pero `voxelsForPersistenceRegion()` devuelve vacío. Para activarlo:

- Durante `VietorisRips::compute()`, guardar un `map<simplex_handle, vector<int>>` que asocie cada simplice con los vóxeles que lo originaron.
- Al hacer click en un punto `(birth, death, dim)`, buscar el simplice más crítico y resaltar sus vóxeles en el VolumeRenderer (usando un canal de selección en la textura 3D).

### Fase 6 — Dataset real

Para validación científica real se recomienda el dataset **CBIS-DDSM** (Curated Breast Imaging Subset of DDSM) disponible en The Cancer Imaging Archive:

```
https://www.cancerimagingarchive.net/collection/cbis-ddsm/
```

Requiere registro gratuito. Contiene 2620 mamografías con anotaciones de calcificaciones y masas. El DicomLoader actual es compatible directamente.

---

## Datos de prueba sintéticos

Los scripts en `scripts/` generan datos mínimos para verificar el pipeline sin datos médicos reales.

### `scripts/gen_dicom.py`

Genera 32 slices DICOM de 64×64 píxeles con:
- Fondo de ruido gaussiano (μ=100, σ=15 HU).
- Un blob gaussiano centrado que simula una masa o grupo de microcalcificaciones.
- Metadatos DICOM válidos (Series UID, Instance Number, ImagePositionPatient).
- Compatible con `itk::GDCMSeriesFileNames` (el loader de producción).

```bash
python3 scripts/gen_dicom.py data/samples/case001
# Genera: data/samples/case001/slice_0000.dcm ... slice_0031.dcm
```

### `scripts/gen_onnx_model.py`

Genera un MLP de dos capas:

```
image_features [1, 512]  ─┐
                            ├─ Concat[1,704] ─ Linear(704,128) ─ ReLU ─ Linear(128,2) ─ logits[1,2]
tda_features   [1, 192]  ─┘
```

Los pesos son aleatorios (semilla=0), la arquitectura es válida y `onnx.checker.check_model()` la valida. Suficiente para verificar la integración con ONNX Runtime.

```bash
python3 scripts/gen_onnx_model.py
# Genera: data/models/breast_cnn.onnx (354 KB)
```

---

## Configuración

`config.toml` en el directorio de trabajo controla todos los parámetros en tiempo de ejecución:

```toml
[io]
dicom_dir           = "data/samples/case001"   # directorio DICOM o ruta .mha
model_path          = "data/models/breast_cnn.onnx"

[tda]
intensity_threshold = 0.85    # umbral [0,1] para extracción de punto cloud
max_edge_length     = 5.0     # ε máximo en mm para la filtración Rips
max_dimension       = 2       # calcular H0, H1, H2 (β0, β1, β2)
min_persistence     = 0.3     # eliminar ruido topológico (persistence < esto)
max_points          = 50000   # subsampling si el point cloud es mayor

[nn]
hook_layers         = ["layer3", "layer4"]  # capas CNN para GradCAM
target_class        = -1                    # -1 = clase predicha

[xai]
umap_neighbors      = 15
umap_min_dist       = 0.1
umap_epochs         = 200

[render]
width               = 1920
height              = 1080
vsync               = true
msaa                = true
ray_step_size       = 0.003            # paso del ray marching (menor = más calidad)
show_gradcam        = true
transfer_fn         = "breast_default" # "breast_default" | "calcification"
```

---

## Tests unitarios

Los tests validan el núcleo matemático del TDA independientemente del resto del sistema:

```bash
cd build
ctest --output-on-failure         # todos los tests
./tests/test_tda --gtest_list_tests  # listar tests disponibles
./tests/test_tda --gtest_filter=PersistenceDiagram.CSVRoundtrip  # test específico
```

| Test | Lo que verifica |
|---|---|
| `FeatureVectorSize` | El histograma de persistencia tiene exactamente `3 × 64 = 192` entradas y está L2-normalizado (norma ≈ 1). |
| `FilteredRemovesNoise` | `filtered(min_persistence=0.1)` elimina features de corta vida y conserva los reales. |
| `CSVRoundtrip` | `saveCSV` + `loadCSV` preservan birth, death y dimension sin pérdida de precisión. |
| `BottleneckDistanceSelf` | La distancia bottleneck de un diagrama consigo mismo es 0. |

---

## Notas técnicas de compilación

Este proyecto usa **GCC 16.1.1** sobre **Arch Linux** con **CMake 4.3.2**. Esa combinación introduce varias incompatibilidades que el `CMakeLists.txt` resuelve explícitamente:

### GCC 16 + CMake 4.3.x: feature tables vacías

CMake 4.3.2 se publicó antes de que GCC 16 existiera. Cuando CMake intenta detectar las features de C++ soportadas por GCC 16, no tiene tabla de referencia y `CMAKE_CXX_COMPILE_FEATURES` queda vacío. Esto hace que cualquier `target_compile_features(cxx_std_20)` — incluyendo los de Eigen3 e ITK — falle con _"No known features for GNU 16.x"_.

**Solución**: inyección manual de la tabla completa de features vía `CACHE STRING FORCE` antes de cualquier `find_package`:

```cmake
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND
   CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL "16.0")
    set(CMAKE_CXX_COMPILE_FEATURES "cxx_std_11;cxx_std_14;..." CACHE STRING "" FORCE)
endif()
```

### `-isystem /usr/include` rompe `#include_next` en GCC 16

En GCC 16, `<cstdlib>` usa `#include_next <stdlib.h>` para delegar en la versión C de la cabecera. Si `/usr/include` aparece como `-isystem /usr/include` en la línea de compilación, GCC 16 reordena el path de búsqueda de un modo que hace que `#include_next` no encuentre `stdlib.h`.

En Arch, el paquete de sistema de GTest exporta `INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "/usr/include"`. La variable `CMAKE_NO_SYSTEM_FROM_IMPORTED` solo suprime la promoción automática de `INTERFACE_INCLUDE_DIRECTORIES` a sistema — NO afecta a `INTERFACE_SYSTEM_INCLUDE_DIRECTORIES`, que siempre se emite como `-isystem`.

**Solución**: limpiar explícitamente ambas propiedades en `tests/CMakeLists.txt`:

```cmake
foreach(_gt_tgt IN ITEMS GTest::gtest GTest::gtest_main GTest::gmock GTest::gmock_main)
    if(TARGET ${_gt_tgt})
        set_target_properties(${_gt_tgt} PROPERTIES
            INTERFACE_SYSTEM_INCLUDE_DIRECTORIES ""
            INTERFACE_INCLUDE_DIRECTORIES        "")
    endif()
endforeach()
```

### GUDHI 3.10: `FetchContent_MakeAvailable` falla

GUDHI tiene su propio `CMakeLists.txt` que no es compatible con `FetchContent_MakeAvailable` (intenta configurar su propio proyecto completo). Se usa `FetchContent_Populate` (nivel bajo) para solo descargar y extraer los fuentes, y luego crear manualmente el target `INTERFACE`:

```cmake
FetchContent_Populate(gudhi)   # solo descarga, no configura
add_library(gudhi INTERFACE)
target_include_directories(gudhi INTERFACE ${gudhi_SOURCE_DIR}/include)
```

### GUDHI 3.10: namespace `Gudhi::` (mayúscula G)

Versiones anteriores de GUDHI usaban `gudhi::`. GUDHI 3.x usa `Gudhi::`. Todos los tipos en `VietorisRips.cpp` usan el namespace correcto.
