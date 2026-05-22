# TODO — MammoViz-TDA

Hoja de ruta completa. Estimación: ~12 meses trabajando solo.

Objetivo: aprender C++ moderno y visión por computador en imágenes médicas reales.

**Principio de diseño:** los datos reales (CBIS-DDSM) son el camino principal desde la primera semana.
Los datos sintéticos existen solo como fallback para tests unitarios y para depurar el pipeline
cuando no hay conexión o los casos reales tardan en procesar. Nunca son el objetivo.

Convenciones de estado:
- `[ ]` pendiente
- `[~]` en progreso
- `[x]` completado
- `[!]` bloqueante (otro task depende de este)

---

## Fase 0 — Infraestructura + Dataset real (semanas 1-3)

Estas dos cosas van en paralelo. No esperes a tener el código para empezar a descargar datos.

### 0.1 Dataset CBIS-DDSM — acceso inmediato

CBIS-DDSM (Curated Breast Imaging Subset of DDSM) es el dataset estándar de referencia para
detección de cáncer de mama. Tiene 2620 mamografías con anotaciones de calcificaciones y masas,
divididas en benign/malignant. Es lo que vas a ver durante todo el año.

- [ ]  Registrarse en The Cancer Imaging Archive: `cancerimagingarchive.net` (gratuito)
- [ ]  Leer la descripción del dataset: qué contiene, cómo están organizadas las series, qué metadatos tienen
- [ ]  Descargar el cliente TCIA (`tcia-utils` en pip o el cliente Java oficial)
- [ ]  Descargar al menos **20 casos**: 10 calcificación benigna + 10 calcificación maligna
    - Priorizar casos de calcificación sobre masas al principio (estructuras más discretas, más fáciles de ver en TDA)
- [ ]  Explorar los DICOMs descargados con un visor externo (3D Slicer, Horos, o `pydicom` en Python) antes de escribir código
    - Anotar: rango de HU típico, resolución espacial (spacing en mm), tamaño en píxeles, número de slices
    - Entender la diferencia visual entre una calcificación benigna y una maligna
- [ ]  Crear `data/real/` con subdirectorios por caso, separados por clase: `data/real/calc_benign/`, `data/real/calc_malignant/`
- [ ]  Añadir `data/real/` a `.gitignore` (los DICOMs no se versiona, son datos de terceros)
- [ ]  Documentar en `data/real/README.md`: cómo descargamos los datos, IDs de casos seleccionados, fecha de descarga

**Por qué importa hacerlo ya:** si esperas a la Fase 8 para ver datos reales, habrás calibrado todo
el código contra blobs gaussianos perfectos. Un DICOM de mamografía real tiene artefactos de
compresión, valores HU en rangos distintos a los que asumes, slices desordenados, campos DICOM
nulos y resoluciones variables. Mejor que esas sorpresas aparezcan en la Fase 1 que en la Fase 8.

### 0.2 Entorno de compilación

- [ ]  Crear `CMakeLists.txt` raíz con políticas CMP0167/CMP0144 para CMake 4.x
- [ ]  Añadir workaround de tabla de features para GCC 16: inyectar `CMAKE_CXX_COMPILE_FEATURES`
manualmente antes de cualquier `find_package` (sin esto, Eigen3 e ITK fallan en configure)
- [ ]  Configurar preset Release (`O3 -march=native`) y Debug (`g -fsanitize=address,undefined`)
- [ ]  Integrar Ninja como generador por defecto
- [ ]  Verificar que `cmake -B build -G Ninja && ninja -C build` compila un `main.cpp` vacío sin warnings
- [ ]  Crear `.clang-format` (estilo: LLVM base, ColumnLimit=100, IndentWidth=4)
- [ ]  Crear `.clang-tidy` con checks: `modernize-*`, `readability-*`, `performance-*`
- [ ]  Crear `.gitignore`: `build/`, `.o`, `.a`, `data/real/`, `data/models/*.onnx`

### 0.3 Dependencias del sistema

- [ ]  Documentar en `DEPS.md` los paquetes Arch exactos: `itk`, `eigen`, `glfw`, `glew`, `gtest`
- [ ]  Escribir `install_deps.sh`: descarga ONNX Runtime 1.17 precompilado, clona Dear ImGui en `third_party/`
- [ ]  Integrar GUDHI 3.10 via `FetchContent_Populate` (no `MakeAvailable`) — solo headers
- [ ]  Crear target `INTERFACE` manual para GUDHI apuntando a `${gudhi_SOURCE_DIR}/include`
- [ ]  Verificar que un `#include <gudhi/Simplex_tree.h>` compila correctamente

### 0.4 Estructura de directorios

- [ ]  Crear árbol completo: `include/`, `src/`, `shaders/glsl/`, `tests/unit/`, `scripts/`, `data/`
- [ ]  Crear `CMakeLists.txt` por subdirectorio: `src/`, `tests/`
- [ ]  Crear headers vacíos con guard en: `io/`, `tda/`, `nn/`, `xai/`, `render/`, `utils/`
- [ ]  Añadir `config.toml` inicial con todos los parámetros

### 0.5 Utilidades base

- [ ]  `utils/Logger.h/.cpp`: niveles INFO/WARN/ERROR, timestamps, salida a `stderr`
- [ ]  `utils/Timer.h`: wrapper de `std::chrono::high_resolution_clock`, método `elapsed_ms()`
- [ ]  `utils/Config.h/.cpp`: parser TOML mínimo — secciones `[x]`, claves `k=v`, tipos string/float/int/bool
- [ ]  Test manual: leer `config.toml`, imprimir todos los valores, verificar tipos

**Criterio de éxito de Fase 0:** tienes DICOMs reales en disco y sabes qué contienen. `ninja -C build` compila sin warnings.

---

## Fase 1 — Carga de datos médicos (semanas 4-7)

Aprenderás: formato DICOM, coordenadas world-space en mm, orientación de imagen, unidades Hounsfield.
El criterio de éxito de esta fase es cargar un caso real de CBIS-DDSM, no datos sintéticos.

### 1.1 VolumeData — tensor 4D

- [ ]  `VolumeData.h`: struct con `data` (`std::vector<float>`), `dims[4]` (X,Y,Z,T), `spacing[3]` (mm/voxel), `origin[3]`, `patient_id` (string), `series_uid` (string)
- [ ]  Método `at(x,y,z,t)` — bounds checking en Debug, sin overhead en Release (`[[likely]]`)
- [ ]  Método `slice(z)` — puntero al plano Z como `const float*`
- [ ]  Método `normalize()` — escala a [0,1] in-place, guarda min/max original para revertir
- [ ]  Método `histogramHU(bins=256)` — distribución de intensidades, útil para calibrar threshold

### 1.2 DicomLoader — carga con ITK/GDCM

- [ ]  `DicomLoader.h/.cpp`: función `load(path)` que detecta directorio DICOM o archivo `.mha`
- [ ]  Caso directorio: `itk::GDCMSeriesFileNames` para ordenar slices por `ImagePositionPatient`
    - **Problema real a resolver:** CBIS-DDSM a veces tiene slices fuera de orden o con `ImagePositionPatient` inconsistente. Añadir fallback por `InstanceNumber`.
- [ ]  Leer `itk::Image<short,3>`, aplicar rescale slope/intercept, convertir a float
- [ ]  Normalizar HU a [0,1]. Para mamografía digital (FFDM) en CBIS-DDSM los valores son distintos
a tomografía — explorar el histograma de los casos descargados antes de hardcodear el rango
- [ ]  Caso `.mha`: `itk::ImageFileReader` con `MetaImageIO`, misma normalización
- [ ]  Extraer `spacing` en mm, `patient_id`, `series_uid` desde metadatos DICOM
- [ ]  Loggear siempre: ruta cargada, dimensiones, spacing, rango de HU encontrado

### 1.3 Validación con datos reales (no sintéticos)

- [ ]  Cargar los 20 casos de CBIS-DDSM con `DicomLoader`
- [ ]  Imprimir histograma de HU de cada caso — identificar el rango real de intensidades
- [ ]  Ajustar el rango de normalización en `config.toml` para que el tejido glandular quede ~0.4-0.6 y las calcificaciones queden > 0.8
- [ ]  Verificar que ningún caso falla en carga (pueden haber DICOMs corruptos o series incompletas)
- [ ]  Documentar casos problemáticos en `data/real/README.md`

### 1.4 Script sintético como fallback de tests

Los datos sintéticos no son el objetivo, pero los necesitas para tests unitarios que no requieran
tener los DICOMs reales en disco (CI, tests offline).

- [ ]  `scripts/gen_dicom.py`: genera 32 slices 64×64 con blob gaussiano + metadatos DICOM mínimos válidos
- [ ]  El script debe parametrizarse: `-size`, `-snr`, `-blob-intensity` para poder simular distintos escenarios
- [ ]  Usar solo en `tests/unit/` — nunca como sustituto del pipeline real

**Lectura recomendada:** ITK Software Guide Vol.1 Cap. 7, DICOM standard PS 3.3 Image Module.

**Recurso de dominio:** “Digital Mammography: A Practical Approach” (cualquier edición reciente) — entender qué ves en las imágenes antes de procesarlas.

**Criterio de éxito de Fase 1:** `./mammoviz data/real/calc_benign/case001` carga un caso real e imprime sus dimensiones y spacing correctos. Los 20 casos se cargan sin crash.

---

## Fase 2 — Análisis Topológico de Datos (semanas 8-14)

La parte más matemáticamente densa. El objetivo al final de esta fase es tener diagramas de
persistencia de casos reales y ver si hay diferencia visual entre casos benignos y malignos.

### 2.1 Teoría (antes de código)

- [ ]  Leer: “A User’s Guide to Topological Data Analysis” — Lum et al. (2013), primeras 15 páginas
- [ ]  Leer: “An Introduction to Topological Data Analysis” — Chazal & Michel (2021), primeras 20 páginas
- [ ]  Entender qué es una filtración de Vietoris-Rips y por qué ε es el parámetro crítico
- [ ]  Entender qué significan β₀, β₁, β₂ en el contexto de microcalcificaciones de mamografía
    - β₀: clusters separados de calcificaciones
    - β₁: distribuciones en anillo o ductos
    - β₂: cavidades (menos relevante en 2D pero presente en tomosíntesis)
- [ ]  Leer documentación GUDHI 3.10: `Rips_complex`, `Simplex_tree`, `persistent_cohomology`
- [ ]  Dibujar a mano un diagrama de persistencia de 5-6 puntos en R² (calcular H0 manualmente)
- [ ]  **Exploración cualitativa:** abrir un caso benigno y uno maligno en Python (`pydicom` + `matplotlib`), dibujar los píxeles de alta intensidad (candidatos a calcificaciones). ¿Tienen formas distintas a simple vista?

### 2.2 PointCloud — extracción de nube de puntos

- [ ]  `PointCloud.h`: `struct Point3D {float x,y,z; float intensity;}`, clase con `std::vector<Point3D>`
- [ ]  Método `fromVolume(VolumeData&, float threshold)`: vóxeles que superen el umbral con coordenadas en mm
- [ ]  Método `subsample(size_t max_points)`: muestreo aleatorio uniforme con `std::mt19937` + semilla fija
- [ ]  Método `exportOFF(path)`: formato OFF estándar para GUDHI
- [ ]  **Calibración con datos reales:** ejecutar con threshold=0.7, 0.8, 0.85 en 5 casos reales.
Anotar cuántos puntos produce cada threshold. El objetivo es ~5000-20000 puntos
(suficiente para capturar topología, manejable para Rips). Documentar el threshold elegido.
- [ ]  Añadir método `stats()` que imprime: número de puntos, bbox en mm, densidad media

### 2.3 VietorisRips — homología persistente

- [ ]  `VietorisRips.h`: clase con `Config {float max_edge_length; int max_dimension; bool sparse;}`
- [ ]  Método `compute(PointCloud&)` — construye `Gudhi::Simplex_tree`
- [ ]  Dense si n < 500 (`Gudhi::Rips_complex`), Sparse si n ≥ 500 (`Gudhi::Sparse_rips_complex`, aprox. 0.5)
- [ ]  Homología persistente con `Standard_persistence_algorithm`, coeficientes Z/2Z
- [ ]  Devolver `std::vector<PersistencePair>` con `{birth, death, dimension}`
- [ ]  Loggear: nº simplices, tiempo de cómputo, nº pares por dimensión
- [ ]  **Calibración con datos reales:** ejecutar en 5 casos reales. Medir tiempo. Si > 60 segundos,
ajustar `max_points` o `max_edge_length`. El objetivo es < 30 segundos por caso.
- [ ]  Soporte DCE-MRI temporal: `computeTemporal(VolumeData& 4D)` itera sobre T

### 2.4 PersistenceDiagram — análisis y exportación

- [ ]  Clase que envuelve `std::vector<PersistencePair>`
- [ ]  `filtered(float min_persistence)`: elimina pares con vida corta (ruido topológico)
- [ ]  `featureVector(int bins=64)` → 192 floats (3 dimensiones × 64 bins, L2-normalizado)
    - Histograma de `(death-birth)` por dimensión, concatenado y normalizado
- [ ]  `bottleneckDistance(PersistenceDiagram& other)`: aproximación O(n²)
- [ ]  `saveCSV(path)` / `loadCSV(path)`: round-trip sin pérdida
- [ ]  `wassersteinDistance(PersistenceDiagram& other)`: asignación greedy p=2

### 2.5 Exploración topológica de datos reales

Esta exploración no es código de producción — es ciencia. Hazla en Python para iterar rápido,
luego el conocimiento informa los parámetros del C++.

- [ ]  Script Python `scripts/explore_tda.py`:
    - Carga 5 casos benignos + 5 malignos (usando `pydicom`)
    - Extrae nube de puntos con el threshold calibrado
    - Computa Vietoris-Rips con `gudhi` (Python bindings)
    - Dibuja los diagramas de persistencia de todos los casos juntos
    - ¿Los benignos y malignos se separan visualmente en el diagrama?
- [ ]  Anotar observaciones: qué dimensión (H0/H1/H2) diferencia mejor las clases
- [ ]  Usar esos hallazgos para decidir `max_dimension` y `min_persistence` en `config.toml`

### 2.6 Tests unitarios TDA (con datos sintéticos como fallback)

Los tests unitarios usan datos sintéticos porque son deterministas y no requieren los DICOMs reales.

- [ ]  Workaround GCC 16 en `tests/CMakeLists.txt`: limpiar `INTERFACE_SYSTEM_INCLUDE_DIRECTORIES`
y `INTERFACE_INCLUDE_DIRECTORIES` de todos los targets de GTest
- [ ]  `FeatureVectorSize`: 192 valores, norma ≈ 1.0 (±1e-5)
- [ ]  `FilteredRemovesNoise`: pares cortos eliminados, largos conservados
- [ ]  `CSVRoundtrip`: save+load preserva todos los campos (±1e-6)
- [ ]  `BottleneckDistanceSelf`: 0.0 (±1e-10)
- [ ]  `WassersteinDistanceSelf`: 0.0
- [ ]  `EmptyDiagram`: featureVector de vacío → ceros sin crash

**Criterio de éxito de Fase 2:** diagramas de persistencia de 20 casos reales computados y guardados en CSV. Hay diferencia visual observable entre benignos y malignos en al menos una dimensión. `ctest` pasa los 6 tests.

---

## Fase 3 — Red Neuronal e Inferencia ONNX (semanas 15-20)

El objetivo de esta fase es tener un modelo real, no un dummy. El modelo dummy existe solo para
que el pipeline C++ compile y corra en CI sin necesitar el modelo real.

### 3.1 Teoría CNNs en imagen médica

- [ ]  Leer: “Deep Learning” Goodfellow cap. 9 (CNNs), primeras 30 páginas
- [ ]  Entender qué capas exporta una CNN estándar en ONNX y cómo se identifican
- [ ]  Leer 2-3 papers de clasificación en CBIS-DDSM:
    - “Deep Learning for Mammography” — Shen et al. (2019) es un buen punto de partida
    - Anotar: arquitectura usada, AUC reportado, cómo preprocesaron las imágenes
- [ ]  Buscar en Papers with Code “CBIS-DDSM” — ver el state of the art actual

### 3.2 Modelo pre-entrenado real

En lugar de entrenar desde cero (requiere semanas de compute y datos etiquetados completos),
usamos un modelo pre-entrenado disponible públicamente y lo exportamos a ONNX.

- [ ]  Buscar en Hugging Face Hub modelos entrenados en CBIS-DDSM o INbreast
    - Buscar: “mammography classification CBIS-DDSM”
    - Alternativa: usar ResNet-50 pre-entrenado en ImageNet y fine-tuned en los 20 casos descargados
- [ ]  Si se elige fine-tuning propio: `scripts/train_model.py` con PyTorch, 10 épocas, batch 8
    - Input: parche 224×224 de la región de interés (ROI anotada en CBIS-DDSM)
    - Output: 2 clases (benign/malignant)
    - Guardar el mejor checkpoint por AUC en validación
- [ ]  Exportar a ONNX con `torch.onnx.export()`, opset 17
- [ ]  Verificar con `netron.app`: ver la arquitectura, identificar los nombres de capas intermedias
- [ ]  Validar con `onnx.checker.check_model()`
- [ ]  Guardar como `data/models/breast_cnn.onnx`

### 3.3 Script modelo dummy (solo para CI/tests)

- [ ]  `scripts/gen_onnx_model.py`: MLP mínimo con entradas `image_features [1,512]` + `tda_features [1,192]`
- [ ]  Solo se usa en CI y en tests unitarios — no representa el modelo real
- [ ]  Guardar como `data/models/dummy_model.onnx` (nombre distinto para no confundirlo)

### 3.4 ModelInference — inferencia ONNX Runtime

- [ ]  Clase que wrappea `Ort::Session`
- [ ]  Constructor: carga `.onnx`, `Ort::Env` con WARNING, `SessionOptions` con 4 threads
- [ ]  **CRÍTICO:** leer nombres de inputs con `session.GetInputNameAllocated()` — no hardcodear.
El modelo real tendrá nombres distintos al dummy.
- [ ]  Método `run(image_features, tda_features)` → `Prediction {int class_id; float confidence; float probs[2];}`
- [ ]  Post-proceso: softmax sobre logits si el modelo devuelve logits (verificar con `netron`)
- [ ]  Test de humo: ejecutar con tensores de ceros, verificar predicción válida

### 3.5 FeatureExtractor — activaciones de capas intermedias

- [ ]  `scripts/add_intermediate_outputs.py`: reexporta el modelo con capas intermedias como outputs adicionales
    - Para ResNet-50: añadir `layer3` y `layer4` como outputs adicionales
    - Guardar como `breast_cnn_with_features.onnx`
- [ ]  `FeatureExtractor.h/.cpp`: carga el modelo extendido, extrae activaciones de la capa solicitada
- [ ]  Devolver `std::vector<float>` con las activaciones aplanadas

**Criterio de éxito de Fase 3:** `ModelInference::run()` produce predicciones con el modelo real en los 20 casos. El AUC en esos 20 casos es > 0.5 (mejor que azar, aunque sea por poco con solo 20 muestras).

---

## Fase 4 — XAI: Explicabilidad (semanas 21-27)

La parte más investigadora. El objetivo es poder abrir un caso maligno y ver visualmente qué
región activó el modelo.

### 4.1 Teoría XAI

- [ ]  Leer: “Grad-CAM” — Selvaraju et al. (2017)
- [ ]  Leer: “Axiomatic Attribution for Deep Networks” (Integrated Gradients) — Sundararajan et al. (2017)
- [ ]  Entender diferencia entre saliencia por perturbación, por gradiente, y por IG
- [ ]  Leer: “UMAP” — McInnes et al. (2018), sección 1 y 3

### 4.2 GradCAM por perturbación

- [ ]  `GradCAM.h/.cpp`: clase con referencia a `ModelInference`
- [ ]  Método `compute(VolumeData&, tda_features, block_size=8)`:
    1. Baseline: `confidence_0 = model.run(image_features, tda).confidence`
    2. Para cada bloque 3D `block_size³`: zeroing de features, reejecutar, medir `Δconf = conf_0 - conf_new`
    3. Asignar `Δconf` a los vóxeles del bloque
    4. Normalizar a [0,1]
- [ ]  Devolver `SaliencyMap` — mismas dimensiones que VolumeData
- [ ]  **Validación cualitativa:** abrir un caso maligno real, aplicar GradCAM, exportar slice
central como PNG (`stb_image_write` o script Python). ¿El highlight cae sobre las calcificaciones
visibles en la imagen? Si no, el threshold o el modelo tienen un problema.

### 4.3 Integrated Gradients

- [ ]  Método alternativo `computeIG(steps=50)`:
    1. Baseline = volumen de ceros (o ruido uniforme bajo)
    2. N interpolaciones lineales baseline→input
    3. Para cada interpolación: evaluar modelo, medir variación de score
    4. Integración trapezoidal
- [ ]  Comparar visualmente IG vs perturbación en el mismo caso real
- [ ]  Documentar cuál produce mapas más coherentes con la anatomía visible

### 4.4 ManifoldProjector — espacio latente

- [ ]  `ManifoldProjector.h/.cpp`: proyecta feature vectors a 3D
- [ ]  `fitPCA(features_matrix)`: SVD Jacobi con Eigen3
- [ ]  `projectPCA(features_matrix)` → `std::vector<ProjectedPoint>`
- [ ]  **Exploración con datos reales:** proyectar los feature vectors de los 20 casos.
¿Se separan los benignos de los malignos en el espacio latente? ¿Qué componente principal
tiene más poder discriminativo?
- [ ]  `fitUMAP` / `projectUMAP`: llamar a script Python via `subprocess`, leer CSV resultado
- [ ]  Comparar PCA vs UMAP en los 20 casos reales

**Criterio de éxito de Fase 4:** el mapa GradCAM de al menos un caso maligno real destaca visualmente la región de calcificaciones. El PCA de los 20 casos muestra alguna separación entre clases.

---

## Fase 5 — Render OpenGL 4.6 (semanas 28-39)

La fase más larga. Todo lo que construyas aquí se prueba con datos reales desde el primer día.

### 5.1 Teoría OpenGL

- [ ]  Completar learnopengl.com caps. 1-5 — escribir los ejemplos a mano
- [ ]  Entender: VAO/VBO, pipeline de rasterización, uniform buffers
- [ ]  Entender: texturas 3D (`GL_TEXTURE_3D`), formatos internos (`GL_R32F`, `GL_RGBA8`)
- [ ]  Leer “Real-Time Volume Graphics” cap. 1-2 (ray casting DVR)

### 5.2 RenderEngine — motor principal

- [ ]  Clase que maneja GLFW + GLEW
- [ ]  `init()`: ventana 1920×1080, OpenGL 4.6 Core Profile, MSAA 4x
- [ ]  Bucle: `pollEvents → updateState → render → swapBuffers`
- [ ]  Layout 3 paneles con `glViewport()`:
    - Panel izquierdo (50%): volumen 3D
    - Panel central (25%): diagrama de persistencia
    - Panel derecho (25%): espacio latente
- [ ]  Callbacks: Escape cierra; W/A/S/D orbitan cámara; click en diagrama → selección
- [ ]  Sistema de cámara: azimutal + elevación + distancia → matriz view 4×4

### 5.3 VolumeRenderer — ray casting

La tarea técnica más difícil.

- [ ]  GPU upload: `GL_TEXTURE_3D` con `GL_R32F`, subir VolumeData real
- [ ]  Textura 3D adicional para SaliencyMap (GradCAM)
- [ ]  Textura 1D para transfer function: intensidad [0,1] → RGBA
    - Calibrar los colores con los HU reales de CBIS-DDSM:
        - Tejido graso (baja densidad): transparente o amarillo tenue
        - Tejido glandular (densidad media): blanco semi-transparente
        - Calcificaciones (alta densidad): cian brillante opaco
- [ ]  VAO/VBO cubo unitario [0,1]³, 12 triángulos
- [ ]  Shader `volume_raycast.vert`: transformar vértices con MVP
- [ ]  Shader `volume_raycast.frag`:
    - Ray direction desde cámara al fragmento
    - Ray marching con paso `u_step_size`
    - Sample volumen + saliencia en cada paso, lookup transfer function
    - Compositing front-to-back: `C_out = C_in + (1-α_in) × C_sample × α_sample`
    - Early termination cuando α > 0.95
    - Overlay GradCAM: mezclar saliencia con color del volumen
- [ ]  Uniforms: `u_mvp`, `u_model_inv`, `u_cam_pos`, `u_volume`, `u_saliency`, `u_transfer`, `u_step_size`, `u_show_gradcam`
- [ ]  **Validación:** abrir un caso real de calcificación. Las calcificaciones deben verse como puntos
brillantes cian sobre el tejido. Si el volumen aparece todo blanco o todo negro, el rango de
normalización está mal calibrado.

### 5.4 DiagramRenderer — scatter plot de persistencia

- [ ]  Shader `diagram.vert/.frag`: puntos 2D en [0,1]², coloreados por dimensión
- [ ]  VBO con `vec4(birth, death, dimension, persistence)` por punto
- [ ]  Color por dimensión: H0=azul #4477CC, H1=verde #44BB88, H2=rojo #CC4444
- [ ]  `gl_PointSize` proporcional a persistencia
- [ ]  Línea diagonal `y=x` con `GL_LINES`
- [ ]  Escala automática de ejes (bbox + 10% margen)
- [ ]  Highlight del punto seleccionado en amarillo

### 5.5 ManifoldRenderer — scatter 3D latente

- [ ]  Shader `manifold.vert/.frag`: puntos 3D orbitables
- [ ]  VBO con `vec3(x,y,z)` + `float label` + `float confidence`
- [ ]  Color: benigno→azul, maligno→rojo, interpolado por confianza
- [ ]  Cámara propia: botón derecho + drag para orbitar
- [ ]  Hit test para hover: punto más cercano al cursor en 2D proyectado
- [ ]  Tooltip con label y confianza al hover

**Criterio de éxito de Fase 5:** ventana con 3 paneles. Un caso real de CBIS-DDSM se renderiza y las calcificaciones son visibles en el panel del volumen. El diagrama de persistencia muestra los puntos del caso. El manifold muestra los 20 casos proyectados con colores por clase.

---

## Fase 6 — Dear ImGui UI (semanas 40-45)

### 6.1 Setup ImGui

- [ ]  Integrar `imgui_impl_glfw.cpp` + `imgui_impl_opengl3.cpp`
- [ ]  Inicializar en `RenderEngine::init()`: `ImGui::CreateContext()`, `ImGui_ImplGlfw_InitForOpenGL()`, `ImGui_ImplOpenGL3_Init("#version 460")`
- [ ]  Bucle: `NewFrame()` → lógica UI → `Render()` → `RenderDrawData()`
- [ ]  Verificar demo window funciona

### 6.2 Panel de control del volumen

- [ ]  Slider “Intensity Threshold” [0,1] → regenera PointCloud al soltar
- [ ]  Slider “Ray Step Size” [0.001, 0.01] → calidad en tiempo real
- [ ]  Dropdown “Transfer Function”: breast_default / calcification / custom
- [ ]  Toggle “Show GradCAM” → activa/desactiva overlay de saliencia
- [ ]  Botón “Recompute TDA” → relanza VietorisRips (en thread separado, con barra de progreso)
- [ ]  Mostrar patient_id y series_uid del caso cargado

### 6.3 Panel del diagrama

- [ ]  Estadísticas: nº pares H0/H1/H2, rango de persistencia
- [ ]  Slider “Min Persistence” → filtra en tiempo real
- [ ]  Info del punto seleccionado: birth, death, dimensión, persistencia

### 6.4 Panel del manifold

- [ ]  Radio buttons: PCA / UMAP (recalcula al cambiar)
- [ ]  Resultado de clasificación: clase, confianza, barra de probabilidad
- [ ]  Botón “Export CSV”: guarda feature vector + proyección a `output/`
- [ ]  Mostrar todos los casos cargados con colores por label real (ground truth de CBIS-DDSM)

### 6.5 Ventana principal

- [ ]  Menu bar: File > Open DICOM / Open MHA / Open Case Folder / Exit
- [ ]  “Open Case Folder”: carga todos los casos en un directorio, los añade al manifold
- [ ]  Status bar: FPS, tiempo último cómputo TDA, nº casos cargados

**Criterio de éxito de Fase 6:** se pueden cargar los 20 casos desde la UI sin tocar el terminal. Los parámetros se ajustan sin recompilar.

---

## Fase 7 — Enlace entre vistas (semanas 46-49)

### 7.1 Simplex → vóxel mapping

- [ ]  Durante `VietorisRips::compute()`: guardar `point_idx → voxel_idx` para cada punto
- [ ]  Identificar simplice crítico de cada par `(birth, death, dim)`
- [ ]  `map<PersistencePair, vector<int> voxel_indices>` — mapeo completo
- [ ]  `voxelsForPair(birth, death, dim)` → `vector<int>`

### 7.2 Selección diagrama → volumen

- [ ]  Click en panel central → par `(birth,death)` más cercano al cursor
- [ ]  `voxelsForPair()` → índices de vóxeles
- [ ]  Actualizar textura de selección en VolumeRenderer

### 7.3 Highlight en VolumeRenderer

- [ ]  Textura `GL_R8` adicional para máscara de selección
- [ ]  Shader: si `texture(u_selection).r > 0.5` → mezclar con amarillo brillante
- [ ]  Animación: pulso sinusoidal para hacer el highlight visible

### 7.4 Hover volumen → diagrama (bidireccional)

- [ ]  Vóxel bajo el rayo del cursor → buscar en mapa inverso los pares que lo contienen
- [ ]  Resaltar esos pares en DiagramRenderer

**Criterio de éxito de Fase 7:** click en un punto del diagrama de un caso real ilumina la región del tejido correspondiente en el volumen.

---

## Fase 8 — Análisis comparativo y validación (semanas 50-52)

Con el pipeline completo funcionando sobre datos reales, esta fase hace análisis científico.

### 8.1 Ampliar el dataset

- [ ]  Descargar 50 casos adicionales de CBIS-DDSM (total ~70 casos)
- [ ]  Incluir también casos de masa (no solo calcificaciones) para comparar topología
- [ ]  Organizar por subtipos: `calc_benign`, `calc_malignant`, `mass_benign`, `mass_malignant`

### 8.2 Análisis estadístico de los diagramas de persistencia

- [ ]  Computar feature vectors de los 70 casos, guardar en `output/features.csv`
- [ ]  Script Python `scripts/analyze_features.py`:
    - t-SNE / PCA sobre los 70 feature vectors — ¿se separan las clases?
    - Test estadístico (Mann-Whitney U) entre benignos y malignos para cada bin del histograma
    - Identificar qué features topológicos (qué rango de persistencia, qué dimensión) tienen mayor poder discriminativo
- [ ]  Documentar hallazgos en `ANALYSIS.md`

### 8.3 Benchmarks de rendimiento

- [ ]  Medir tiempo de cada etapa del pipeline en los 70 casos: carga, PointCloud, Rips, ONNX, GradCAM
- [ ]  Tabular en `BENCHMARKS.md`: hardware, tamaño de volumen, tiempos por etapa
- [ ]  Identificar el cuello de botella principal y proponer optimización

### 8.4 Validación cualitativa del GradCAM

- [ ]  Para 10 casos malignos: ¿el GradCAM destaca las regiones anotadas en CBIS-DDSM (ROI)?
    - CBIS-DDSM incluye segmentaciones de las lesiones — comparar con el mapa de saliencia
- [ ]  Calcular IoU entre la máscara del GradCAM (threshold > 0.5) y la ROI anotada
- [ ]  Si el IoU < 0.2, revisar el modelo o el método de GradCAM

**Criterio de éxito de Fase 8:** análisis estadístico documentado. GradCAM con IoU > 0.2 en al menos 7/10 casos malignos.

---

## Mejoras opcionales (sin fecha fija)

### Rendimiento

- [ ]  Perfilar con `perf` / `gprof` — identificar hotspots
- [ ]  Paralelizar GradCAM con `std::async` / thread pool
- [ ]  Cachear feature vectors TDA en binario entre ejecuciones
- [ ]  Investigar GUDHI GPU si hay soporte experimental

### Algoritmos TDA adicionales

- [ ]  TDA Mapper para visualizar estructura global del dataset de 70 casos
- [ ]  Persistent Entropy como feature adicional en el featureVector
- [ ]  Alpha Complexes como alternativa a Vietoris-Rips (más eficiente en puntos uniformes)
- [ ]  Wasserstein distance completa via librería Hera

### Rendering avanzado

- [ ]  SSAO (ambient occlusion) en el ray caster del volumen
- [ ]  Segmentación interactiva: brush tool para pintar vóxeles en OpenGL
- [ ]  Exportar render a PNG/EXR con `glReadPixels` para figuras de publicación
- [ ]  Superposición de la ROI anotada de CBIS-DDSM sobre el volumen renderizado

### Arquitectura

- [ ]  GitHub Actions: compilar en Ubuntu + Arch, correr `ctest`
- [ ]  Cobertura de código con `gcov` / `lcov`
- [ ]  Doxygen para API pública
- [ ]  Serialización completa de sesión (JSON): parámetros, diagrama, proyección, caso cargado

---

## Notas de aprendizaje

### C++20

- [ ]  Ranges y views: `std::views::filter`, `std::views::transform` — usar en el pipeline de PointCloud
- [ ]  Concepts: definir `concept Volumetric` que requiera `at()` y `dims()`
- [ ]  Coroutines: explorar para el pipeline asíncrono de GradCAM
- [ ]  Regla de los 5 en clases con recursos GPU: move constructor + move assign
- [ ]  RAII consistente: todos los recursos OpenGL liberados en destructores

### Imagen médica

- [ ]  Diferencias entre mamografía FFDM, tomosíntesis DBT y DCE-MRI 4D
- [ ]  Por qué las mamografías son distintas de imágenes naturales: ruido cuántico, bajo contraste
- [ ]  Leer al menos 2 papers de clasificación en CBIS-DDSM — qué AUC reportan, qué arquitectura usan
- [ ]  Entender las anotaciones de CBIS-DDSM: ROI, pathology, assessment categories (BI-RADS)

---

## Registro de decisiones técnicas

| Fecha | Decisión | Alternativa considerada | Razón |
| --- | --- | --- | --- |
| — | CBIS-DDSM como dataset principal desde Fase 0 | Datos sintéticos como base | Los sintéticos ocultan problemas reales de HU, artefactos, slices desordenados |
| — | ONNX Runtime en lugar de LibTorch | LibTorch C++ | Más ligero, no requiere CUDA para CPU inference |
| — | GradCAM por perturbación inicialmente | GradCAM real con backprop | ONNX RT no expone gradientes; perturbación es semánticamente correcta |
| — | PCA por defecto para manifold | UMAP | PCA es determinista, sin hiperparámetros, resultado inmediato |
| — | GUDHI via FetchContent_Populate | Submodule | FetchContent gestiona descarga automáticamente; submodule más frágil |
| — | Dense/Sparse split en n=500 | Umbral fijo mayor | GUDHI recomienda Sparse >1000 pts; 500 da margen cómodo |
| — | Z/2Z en homología | Coeficientes enteros Z | Estándar en TDA aplicado, más rápido en GUDHI |

---

## Métricas de progreso

| Fase | Hito medible |
| --- | --- |
| 0 | 20 casos CBIS-DDSM descargados y explorados. `ninja -C build` sin warnings. |
| 1 | Los 20 casos reales se cargan sin crash. Histograma de HU documentado. |
| 2 | Diagramas de persistencia de 20 casos reales computados. Diferencia visual entre clases. |
| 3 | Modelo real en ONNX. Predicciones con AUC > 0.5 en los 20 casos. |
| 4 | GradCAM destaca visualmente calcificaciones en al menos un caso maligno real. |
| 5 | Ventana con 3 paneles. Caso real renderizado con calcificaciones visibles. |
| 6 | 20 casos cargados y navegables desde la UI. |
| 7 | Click en diagrama ilumina vóxeles en el volumen. |
| 8 | Análisis estadístico de 70 casos. GradCAM IoU > 0.2 en 7/10 malignos. |
