# mammoviz-tda

**Homología persistente para la detección anticipada del cáncer de mama en
mamografías de cribado.** Trabajo de Fin de Grado: código, notebooks y
memoria en LaTeX.

## La pregunta

¿Había ya una señal topológica medible en la mamografía de cribado anterior,
leída como normal, de las mujeres a las que después se diagnosticó un cáncer?

## Resultado principal

La topología (imagen de persistencia H0+H1) de una ventana de ~3.5 cm en el
**sitio donde aparecerá el cáncer**, medida **un cribado antes**, separa casos
de controles emparejados:

| | AUC | p |
|---|---|---|
| EMBED (desarrollo, controles emparejados por edad) | 0.571 | 0.02 |
| **CSAW-CC (validación externa preregistrada)** | **0.614** | **0.001** |
| CSAW-CC, donde el radiólogo no veía nada | 0.596 | 0.01 |
| CSAW-CC, ventana desplazada media mama (control) | 0.519 | 0.27 |

- Es **local**: desaparece media mama más allá y en la mama contralateral.
- No la explican la edad, la densidad ni los hallazgos benignos previos.
- Su patrón topológico (una densificación local) es el mismo en las dos
  cohortes (r = 0.88), y la topología aporta sobre la densidad sola.
- Una red neuronal independiente (DenseNet121) sobre la misma ventana
  confirma la señal.
- El plan de la validación quedó en `git` antes de descargar ningún dato de
  CSAW-CC: [`docs/plan_validacion_csaw.md`](docs/plan_validacion_csaw.md).

Por el camino se documentan **tres confusores de diseño** (intervalo entre
exploraciones, época de adquisición y tipo de exploración en el diagnóstico)
que producían señales falsas de hasta AUC 0.89. Como base, el trabajo incluye
la clasificación benigno/maligno sobre CBIS-DDSM y la interpretabilidad del
TDA frente a Grad-CAM, SHAP y LIME.

La historia completa, con los errores y los resultados negativos, está en el
diario: [`docs/diario.md`](docs/diario.md).

## Datos

| Conjunto | Papel | Acceso |
|---|---|---|
| [CBIS-DDSM](https://www.cancerimagingarchive.net/collection/cbis-ddsm/) | clasificación aislada | público (TCIA) |
| [EMBED](https://github.com/Emory-HITI/EMBED_Open_Data) | desarrollo longitudinal | controlado (Emory) |
| CSAW-CC | validación longitudinal | controlado (Karolinska, vía SND) |

Ninguna imagen ni credencial se versiona (`data/` está en `.gitignore`). Las
credenciales de EMBED y CSAW-CC van en `~/.aws/credentials`, perfiles
`[emory-embed]` y `[csaw-cc]`. Cómo solicitar el acceso:
`docs/acceso_datasets_longitudinales.pdf`.

Condiciones de uso:
- EMBED: citar *EMory BrEast Imaging Dataset (EMBED). HITI Lab, Emory
  University*.
- CSAW-CC: en publicaciones, invitar como coautores a los curadores del
  dataset (criterios ICMJE).

## Estructura

```
src/            biblioteca: datos, preprocesado, topología, ventanas, registro, modelos
scripts/        un script por análisis; cada uno escribe en results/
notebooks/      02 CBIS-DDSM · 03 EMBED (confusores, sitio) · 04 CSAW-CC (validación)
tests/          93 tests (pytest), también en GitHub Actions
docs/           plan de validación, diario, guías
tfg/            memoria en LaTeX
results/        resultados en texto de cada análisis
```

La memoria (apéndice) relaciona cada resultado con el script que lo produce.

## Instalación

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (Linux/Mac: source .venv/bin/activate)
pip install -r requirements.txt
```

## Reproducir los resultados principales

```bash
# Cohortes (requieren acceso y credenciales)
python scripts/download_embed_longitudinal.py --lesion --n-controls 3 --age-caliper 3
python scripts/download_csaw.py

# Análisis
python scripts/embed_lesion_site.py --manifest longitudinal_manifest_lesion_edad.csv
python scripts/csaw_lesion_site.py          # validación preregistrada
python scripts/interpret_anticipation.py    # patrón topológico EMBED vs CSAW
python scripts/density_baseline.py --dataset csaw
python scripts/cnn_vs_topology.py --dataset csaw

# Figuras de la memoria y memoria
python scripts/figuras_memoria.py
cd tfg && latexmk -pdf main.tex
```

Los notebooks `03_embed_longitudinal.ipynb` y `04_validacion_csaw.ipynb`
recorren los mismos análisis celda a celda y se ejecutan de principio a fin.

## Tests

```bash
pytest
```

Los tests usan arrays construidos a mano y no necesitan datos descargados.
La equivalencia entre los backends `giotto-tda` y `gudhi` se comprueba en un
trabajo aparte de CI con Python 3.12 (`giotto-tda` no tiene binarios para
versiones posteriores).
