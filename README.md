# mammoviz-tda

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
   `calc_case_description_*.csv`, …) → colócalos en `data/csv/`.
2. Las **imágenes** (mamografías completas y máscaras de ROI) → `data/images/`.

> ⚠️ **Nota metodológica importante.** El notebook original
> (`Demo_Notebook.ipynb`) alcanzaba una exactitud del 100 % desde la primera
> época, lo cual es un síntoma claro de **fuga de datos (data leakage)**. Este
> proyecto lo corrige con una **partición estricta por paciente**
> (`src/data.py:split_by_patient`), de modo que ninguna imagen de un paciente
> aparece en dos particiones distintas.

## Uso rápido

```python
from src.data import load_manifest, split_by_patient
from src.preprocessing import preprocess
from src.topology import extract_features
from src.models import build_svm
from src.evaluate import clinical_metrics

# 1. Cargar metadatos y particionar por paciente
manifest = load_manifest("CBIS-DDSM-All-doiJNLP-zzWs5zfZ-nbia-digest.xlsx")

# 2. Preprocesar imagen -> 3. extraer descriptor topológico
# img_pp = preprocess(imagen, mascara)
# feat   = extract_features(img_pp, method="image")

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
  **sintéticos** para poder ejecutarse sin CBIS-DDSM descargado, e incluye un
  respaldo topológico basado en la **curva de Betti-0** (`scipy`) si `giotto-tda`
  no está instalado. Ejecutado, da métricas realistas (accuracy ≈ 0.70, AUC ≈ 0.78).

## Estado del proyecto

- [x] Estructura de memoria (capítulos 1–8 + apéndice), compila con MiKTeX
- [x] Bibliografía inicial
- [x] Pipeline (`src/`) documentado a nivel profesional
- [x] Notebook de demostración ejecutable de extremo a extremo
- [ ] Descarga completa de datos e imágenes de CBIS-DDSM
- [ ] Ejecución de experimentos con datos reales y resultados
- [ ] Entrenamiento de la CNN de referencia y experimento de fusión
- [ ] Visualizaciones finales para la memoria
```
