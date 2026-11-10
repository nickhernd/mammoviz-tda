# mammoviz-tda

**Análisis Topológico de Datos (TDA) aplicado a la detección anticipada de
cáncer de mama mediante comparación longitudinal de mamografías.**

Trabajo de Fin de Grado. Este repositorio contiene la memoria (LaTeX) y el
código del *pipeline* que combina homología persistente con aprendizaje
profundo sobre el conjunto de datos
[CBIS-DDSM](https://www.cancerimagingarchive.net/collection/cbis-ddsm/).

## Hipótesis principal

¿Existe ya una señal topológica medible en la mamografía *previa* (leída
clínicamente como normal) de una paciente que más adelante es diagnosticada
de cáncer? El proyecto compara, para la misma paciente, el diagrama de
persistencia de la exploración en $t-1$ con el de la exploración en $t$
(donde se diagnostica la lesión), usando dos señales:

- **Anticipación**: descriptores topológicos de la imagen en $t-1$ sola.
- **Deriva**: distancia *bottleneck*/Wasserstein entre los diagramas de
  $t-1$ y $t$.

Esta hipótesis **sustituye** al planteamiento inicial del enunciado
(interpretabilidad TDA vs. Grad-CAM/SHAP/LIME sobre una sola imagen), que se
mantiene como base necesaria (clasificación benigno/maligno aislada) pero ya
no es el objetivo central. Ver `tfg/capitulos/02-objetivos.tex` (objetivos
O7/O8) y `tfg/capitulos/05-metodologia.tex`.

**Bloqueo real y actual**: no existe todavía un dataset longitudinal público
de descarga directa. Los dos candidatos identificados, CSAW-CC y EMBED,
requieren solicitud formal de acceso (ver issue
[#92](../../issues/92) y `tfg/bibliografia.bib`:
`dataset_csawcc`, `jeong2021embed`). Mientras se tramita ese acceso, la
metodología se valida con **datos sintéticos longitudinales**
(`scripts/synthetic_longitudinal_validation.py`).

## Objetivo (base necesaria)

Estudiar si los descriptores topológicos (diagramas de persistencia,
imágenes de persistencia) permiten discriminar lesiones **benignas** de
**malignas** en una mamografía aislada, compararlos y combinarlos con una CNN
de referencia, y usar esa base como paso previo a la comparación
longitudinal.

## Estructura del repositorio

```
mammoviz-tda/
├── data/                        # Datos (NO versionados; ver "Datos" más abajo)
│   ├── csv/                     #   CSV de descripción de casos de CBIS-DDSM
│   └── images/                  #   Imágenes DICOM reales (completas + ROI)
├── src/                         # Código fuente del pipeline
│   ├── data.py                  #   Carga de metadatos y partición POR PACIENTE
│   ├── preprocessing.py         #   Preprocesado de imágenes (ROI, CLAHE, resize)
│   ├── topology.py              #   Homología persistente y vectorización (gudhi/giotto-tda)
│   ├── models.py                #   Clasificadores (SVM, RF) y CNN de referencia (PyTorch)
│   └── evaluate.py               #   Métricas clínicas y visualización
├── scripts/
│   ├── download_cbis_ddsm.py    #   Descarga incremental de imágenes reales (API TCIA)
│   ├── run_pipeline.py          #   Pipeline completo de un tirón sobre datos reales
│   ├── train_cnn.py             #   Entrenamiento de la CNN de referencia (PyTorch)
│   ├── fuse_and_evaluate.py     #   Comparación TDA vs. CNN vs. fusión (+ --tune por vía)
│   ├── explainability_benchmark.py  # Grad-CAM vs. región TDA vs. SHAP vs. LIME
│   ├── synthetic_longitudinal_validation.py  # Validación de la hipótesis con datos sintéticos
│   ├── download_embed_longitudinal.py  # Descarga del piloto longitudinal REAL (EMBED)
│   └── embed_longitudinal_evaluation.py  # Evalúa O7/O8 sobre el piloto real de EMBED
├── notebooks/                   # Entorno de trabajo interactivo (ver docs/guia_uso.pdf)
├── bib/                         # Bibliografía en PDF
├── docs/
│   └── guia_uso.pdf             #   Guía de uso: instalación, comandos y flujo con notebooks
├── tfg/                         # Memoria del TFG en LaTeX
│   ├── main.tex
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

> ⚠️ `data/` está en `.gitignore` (163.5 GB el dataset completo de
> CBIS-DDSM): desaparece al clonar o cambiar de máquina/entorno. Hay que
> volver a descargarlo con los pasos de abajo; no es un error.

1. Los **CSV de descripción de casos** (`mass_case_description_train_set.csv`,
   `calc_case_description_*.csv`, …) → colócalos en `data/csv/`.
2. Las **imágenes reales** (mamografías completas + máscaras de ROI): el
   dataset completo son 163.5 GB (6775 series DICOM), demasiado para
   descargar de golpe. Usa el script incremental, que descarga un piloto
   balanceado benigno/maligno directamente contra la API REST de TCIA (no
   hace falta el NBIA Data Retriever) y se puede relanzar cuando quieras para
   ampliarlo sin repetir descargas:
   ```bash
   python scripts/download_cbis_ddsm.py --add 15 15 --category mass
   ```

> ⚠️ **Nota metodológica importante.** El notebook original
> (`Demo_Notebook.ipynb`) alcanzaba una exactitud del 100 % desde la primera
> época, lo cual es un síntoma claro de **fuga de datos (data leakage)**. Este
> proyecto lo corrige con una **partición estricta por paciente**
> (`src/data.py:split_by_patient`), de modo que ninguna imagen de un paciente
> aparece en dos particiones distintas.

## Datos longitudinales (EMBED)

Para la hipótesis principal (O7/O8) se usa [EMBED](https://github.com/Emory-HITI/EMBED_Open_Data),
de acceso controlado (ver `docs/acceso_datasets_longitudinales.pdf` para
cómo solicitarlo). Con el acceso concedido:

1. Configura `~/.aws/credentials` con el perfil que te den (formato
   `[emory-embed]` + `aws_access_key_id`/`aws_secret_access_key`). **Nunca
   subas este fichero ni lo pegues en ningún sitio versionado.**
2. Descarga las tablas de metadatos (una vez, ~210 MB) y el piloto
   longitudinal:
   ```bash
   mkdir -p data/embed
   python -c "import boto3; s = boto3.Session(profile_name='emory-embed', region_name='us-west-2'); c = s.client('s3'); [c.download_file('embed-dataset-open', f'tables/{f}', f'data/embed/{f.replace(\"EMBED_OpenData_\", \"\")}') for f in ['EMBED_OpenData_clinical.csv', 'EMBED_OpenData_metadata_reduced.csv']]"
   python scripts/download_embed_longitudinal.py --n-incident 20 --n-control 20
   python scripts/embed_longitudinal_evaluation.py
   ```
`data/embed/` está en `.gitignore` (igual que `data/csv/`/`data/images/`):
las imágenes y los metadatos de EMBED no se versionan.

## Uso rápido

Para ejecutar el pipeline completo de un tirón (carga real → preprocesado →
topología → clasificador → métricas clínicas) sobre los casos ya descargados:

```bash
python scripts/run_pipeline.py            # Random Forest, n_bins=4 (ver issue #77)
python scripts/run_pipeline.py --tune     # + GridSearchCV sobre el train actual
python scripts/train_cnn.py               # CNN de referencia (PyTorch, DenseNet121)
python scripts/fuse_and_evaluate.py       # TDA vs. CNN vs. fusión, con CV-AUC
python scripts/fuse_and_evaluate.py --tune --classifier svm   # + ajuste de hiperparametros por via
python scripts/explainability_benchmark.py   # Grad-CAM vs. TDA vs. SHAP vs. LIME (linea secundaria)
python scripts/synthetic_longitudinal_validation.py --sweep   # hipótesis principal, datos sintéticos
```

Guarda en `results/` la matriz de confusión, un diagrama de persistencia de
ejemplo y `metrics.txt` (incluye el AUC por validación cruzada, más fiable
que el AUC de un único split de test pequeño — ver "Problemas conocidos" en
`docs/guia_uso.pdf`).

**Guía completa de todos los comandos y del entorno de trabajo**: ver
[`docs/guia_uso.pdf`](docs/guia_uso.pdf).

## Entorno de trabajo interactivo (notebook)

`notebooks/02_real_data_pipeline.ipynb` reutiliza las mismas funciones que
los scripts anteriores (no duplica código) para explorar el pipeline celda a
celda sobre los casos reales ya descargados: carga de metadatos, inspección
de un caso real (imagen + máscara de ROI), diagrama de persistencia,
entrenamiento y evaluación del clasificador topológico, CV-AUC, CNN de
referencia y la validación sintética de la hipótesis longitudinal. Validado
de extremo a extremo con `jupyter nbconvert --execute`.

## Memoria (LaTeX)

```bash
cd tfg
latexmk -pdf main.tex     # requiere una distribución LaTeX + biber
```

## Estado del proyecto

- [x] Pivote de hipótesis: de interpretabilidad aislada a detección anticipada longitudinal
- [x] Pipeline (`src/`) documentado a nivel profesional, migrado a PyTorch
- [x] Datos reales de CBIS-DDSM descargados (piloto: 70 pacientes, benigno/maligno balanceado)
- [x] Bug de `topology.py` corregido (dimensiones H0/H1 fijas) + test de regresión
- [x] Bug de `breast_density` corregido (columnas duplicadas mass/calc) + test de regresión
- [x] Overfitting diagnosticado y corregido (n_bins, Random Forest, `--tune`, CV-AUC)
- [x] CNN de referencia (PyTorch/DenseNet121) entrenada de forma reproducible
- [x] Comparación y fusión TDA + CNN, con ajuste de hiperparámetros por vía
      (CV-AUC sobre 70 pacientes: TDA 0.66, CNN 0.83, fusión 0.82 — con el
      piloto ampliado, CNN y fusión quedan prácticamente empatadas; con 30
      pacientes el número de fusión parecía mejor pero era ruido de muestra)
- [x] Interpretabilidad: Grad-CAM, región topológica dominante, SHAP y LIME
      comparados sobre el piloto real, con protocolo estadístico (bootstrap
      CI, test pareado) y desglose por `mass_margins`/`breast_density`;
      acuerdo bajo entre los 4 métodos, TDA tan rápida como Grad-CAM y
      ~300x más rápida que SHAP (`docs/` y `tfg/capitulos/05-metodologia.tex`)
- [x] Metodología longitudinal diseñada y **validada con datos sintéticos**
      (AUC anticipación 0.871, deriva 0.937 — ver `tfg/capitulos/05-metodologia.tex`)
- [x] Notebook interactivo + guía de uso en PDF (`docs/guia_uso.pdf`)
- [x] Acceso a EMBED concedido (cohorte real: 23253 pacientes, 14789 con
      2+ exploraciones; 644 incidentes + 7256 control identificados)
- [x] Primera evaluación REAL (no sintética) de la hipótesis de detección
      anticipada: piloto de 40 pacientes, AUC deriva 0.688, anticipación
      0.525 — consistente con el hallazgo sintético, todavía preliminar
      (`scripts/download_embed_longitudinal.py`, `scripts/embed_longitudinal_evaluation.py`)
- [ ] Ampliar el piloto de EMBED más allá de 40 pacientes (hay 900+ candidatos)
- [ ] Solicitar también acceso a CSAW-CC como fuente cruzada, issue [#92](../../issues/92)
- [ ] Ampliación del piloto de CBIS-DDSM más allá de 70 pacientes
- [ ] Visualizaciones finales para la memoria

## Tests

```bash
pip install -r requirements.txt
pytest
```

Los tests cubren `src/data.py`, `preprocessing.py`, `models.py` y
`evaluate.py` con datos sintéticos (no requieren CBIS-DDSM descargado). Los
de `topology.py` que necesitan `giotto-tda` se saltan automáticamente si no
está instalado (en Python ≥ 3.13 se usa `gudhi`+`persim`).
