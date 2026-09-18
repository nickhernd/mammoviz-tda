# mammoviz-tda
## Enunciado TFG
El cáncer de mama es uno de los problemas de salud más relevantes a nivel mundial, y la mamografía sigue siendo una de las herramientas principales para su detección temprana. Sin embargo, interpretar una mamografía no es sencillo: las lesiones pueden ser pequeñas, sutiles o difíciles de distinguir del tejido mamario normal. En este TFG se propone desarrollar un sistema de inteligencia artificial capaz de analizar mamografías y detectar automáticamente posibles regiones sospechosas.
El trabajo permitirá al estudiante introducirse en uno de los campos más activos de la IA médica: el diagnóstico asistido por ordenador. Se utilizarán bases de datos públicas de mamografía, como CBIS-DDSM o VinDr-Mammo, que contienen imágenes anotadas por especialistas. A partir de estas imágenes, se entrenarán modelos de aprendizaje profundo para identificar masas, microcalcificaciones u otras anomalías asociadas al cáncer de mama.
El proyecto incluirá tareas de preprocesamiento de imagen médica, mejora de contraste, extracción de regiones de interés y entrenamiento de redes neuronales convolucionales o modelos actuales de visión artificial. Una parte especialmente interesante será la explicabilidad: no bastará con que el modelo prediga si una imagen es sospechosa, sino que deberá mostrar qué zonas de la mamografía han influido en su decisión mediante técnicas como Grad-CAM o mapas de activación.
El resultado final será un prototipo capaz de cargar una mamografía, estimar la probabilidad de lesión sospechosa y visualizar las regiones más relevantes para el modelo.




**Análisis Topológico de Datos (TDA) aplicado a la clasificación y visualización de mamografías digitales.**

Trabajo de Fin de Grado. Este repositorio contiene la memoria (LaTeX) y el
código del pipeline que combina homología persistente con aprendizaje profundo
sobre el conjunto de datos [CBIS-DDSM](https://www.cancerimagingarchive.net/collection/cbis-ddsm/).

## Objetivo

Estudiar si los descriptores topológicos (diagramas de persistencia, imágenes de
persistencia) permiten discriminar lesiones **benignas** de **malignas** en
mamografía, de forma más interpretable y robusta, y compararlos y combinarlos con
una CNN de referencia.

## Estructura del repositorio

```
mammoviz-tda/
├── data/                # Datos (NO versionados; ver "Datos" más abajo)
│   ├── csv/             #   CSV de descripción de casos de CBIS-DDSM
│   └── images/          #   Imágenes DICOM/PNG
├── src/                 # Código fuente del pipeline
│   ├── data.py          #   Carga de metadatos y partición POR PACIENTE
│   ├── preprocessing.py #   Preprocesado de imágenes (ROI, CLAHE, resize)
│   ├── topology.py      #   Homología persistente y vectorización (giotto-tda)
│   ├── models.py        #   Clasificadores (SVM, RF) y CNN de referencia
│   └── evaluate.py      #   Métricas clínicas y visualización
├── notebooks/           # Experimentos reproducibles
├── bib/                 # Bibliografía en PDF
├── tfg/                 # Memoria del TFG en LaTeX
│   ├── main.tex
│   ├── portada.tex
│   ├── bibliografia.bib
│   └── capitulos/
└── requirements.txt
```

## Instalación

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

pip install -r requirements.txt
```

## Datos

El fichero `CBIS-DDSM-All-...-nbia-digest.xlsx` incluido es el **manifiesto de
TCIA**: describe las series pero **no contiene la etiqueta benigno/maligno**.
Para reproducir los experimentos necesitas descargar de CBIS-DDSM:

1. Los **CSV de descripción de casos** (`mass_case_description_train_set.csv`,
   `calc_case_description_*.csv`, …) → descárgalos con:
   ```bash
   curl -O https://www.cancerimagingarchive.net/wp-content/uploads/mass_case_description_train_set.csv
   # (y lo mismo para mass_case_description_test_set.csv, calc_case_description_train_set.csv,
   #  calc_case_description_test_set.csv) -> colócalos todos en data/csv/
   ```
2. Las **imágenes** (mamografías completas y máscaras de ROI): el dataset completo son
   163.5 GB (6775 series DICOM), demasiado para descargar de golpe. Usa el script
   incremental, que descarga un piloto balanceado benigno/maligno y se puede
   relanzar cuando quieras para ampliarlo:
   ```bash
   python scripts/download_cbis_ddsm.py --add 15 15 --category mass
   ```
   Va directamente contra la API REST de TCIA (no hace falta el NBIA Data
   Retriever), cachea el índice de series y no repite descargas ya hechas.

> ⚠️ **Nota metodológica importante.** El notebook original
> (`Demo_Notebook.ipynb`) alcanzaba una exactitud del 100 % desde la primera
> época, lo cual es un síntoma claro de **fuga de datos (data leakage)**. Este
> proyecto lo corrige con una **partición estricta por paciente**
> (`src/data.py:split_by_patient`), de modo que ninguna imagen de un paciente
> aparece en dos particiones distintas.

## Uso rápido

Para ejecutar el pipeline completo de un tirón (carga real → preprocesado →
topología → clasificador → métricas clínicas) sobre los casos ya descargados:

```bash
python scripts/run_pipeline.py
```

Guarda en `results/` la matriz de confusión, un diagrama de persistencia de
ejemplo y `metrics.txt`. Si prefieres ir paso a paso o integrarlo en tu propio
notebook, las mismas piezas están disponibles por separado:

```python
from src.data import load_case_descriptions, split_by_patient
from src.preprocessing import preprocess
from src.topology import persistence_diagram, persistence_image_vector
from src.models import build_svm
from src.evaluate import clinical_metrics

# 1. Cargar metadatos reales y particionar por paciente
df = load_case_descriptions("data/csv")
partitions = split_by_patient(df)

# 2. Preprocesar imagen -> 3. extraer descriptor topológico
# img_pp = preprocess(imagen, mascara)
# diagrama = persistence_diagram(img_pp)
# feat = persistence_image_vector(diagrama)

# 4. Entrenar clasificador y 5. evaluar con métricas clínicas
# clf = build_svm(); clf.fit(X_train, y_train)
# metrics = clinical_metrics(y_test, clf.predict(X_test))
```

## Memoria (LaTeX)

```bash
cd tfg
latexmk -pdf main.tex     # requiere una distribución LaTeX + biber
```

## Notebooks

- `notebooks/01_pipeline_demo.ipynb` — **demostración de extremo a extremo** del
  pipeline (datos → preprocesado → topología → modelos → evaluación). Usa datos
  **sintéticos** para poder ejecutarse sin CBIS-DDSM descargado. El backend
  topológico se elige automáticamente (`src.topology.backend()`): imágenes de
  persistencia reales (H0+H1) con `gudhi` o `giotto-tda` si alguno está
  instalado, o si no un respaldo basado en la **curva de Betti-0** (`scipy`).
  Ejecutado con `gudhi`, da accuracy ≈ 0.67–0.73, sensibilidad ≈ 0.60–0.80,
  AUC ≈ 0.83–0.84.

## Estado del proyecto

- [x] Estructura de memoria (capítulos 1–8 + apéndice), compila con MiKTeX
- [x] Bibliografía inicial
- [x] Pipeline (`src/`) documentado a nivel profesional
- [x] Notebook de demostración ejecutable de extremo a extremo
- [x] Tests unitarios de `src/` (`pytest`, datos sintéticos)
- [x] Capítulos 7–8 con validación preliminar del *pipeline* (datos sintéticos)
- [ ] Descarga completa de datos e imágenes de CBIS-DDSM
- [ ] Ejecución de experimentos con datos reales y resultados
- [ ] Entrenamiento de la CNN de referencia y experimento de fusión
- [ ] Visualizaciones finales para la memoria

## Tests

```bash
pip install -r requirements.txt
pytest
```

Los tests cubren `src/data.py`, `preprocessing.py`, `models.py` y `evaluate.py`
con datos sintéticos (no requieren CBIS-DDSM descargado). Los de `topology.py`
que necesitan `giotto-tda` se saltan automáticamente si no está instalado.
