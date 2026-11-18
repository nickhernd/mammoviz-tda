# Diario de trabajo del TFG

Cuaderno de laboratorio: qué se hizo cada día, qué se descubrió y qué se
decidió. **No es la memoria**: aquí van también los errores, los resultados
que luego resultaron ser artefactos y los porqués de cada cambio de rumbo,
para poder reconstruir el razonamiento al redactar.

Las entradas más recientes van arriba. Los commits concretos se pueden ver
con `git log --since=<fecha> --until=<fecha>`.

---

## Hallazgos clave hasta ahora (resumen vivo)

1. **La fuga de datos del notebook original** (100 % de exactitud desde la
   primera época) se corrigió con partición por paciente.
2. **Tamaño de muestra**: todos los resultados "buenos" de pilotos pequeños
   se han desinflado al crecer la muestra (CV-AUC de TDA 0.85 → 0.66;
   fusión 0.97 → 0.82). Reportar siempre con intervalo de confianza.
3. **Interpretabilidad**: Grad-CAM, SHAP, LIME y la región topológica apenas
   coinciden entre sí (IoU 0.07–0.14, n=122). La ventaja de TDA es el coste
   (300× más rápida que SHAP), no la concordancia.
4. **Hipótesis longitudinal (EMBED)**: en el camino se han encontrado
   **tres confusores de diseño**, cada uno de los cuales producía una señal
   falsa:
   - intervalo t-1 → t distinto entre grupos ("intervalo solo" daba AUC 0.745);
   - época de adquisición de t-1 distinta (inflaba la anticipación a 0.63);
   - la exploración t de los casos era diagnóstica con vistas ampliadas
     (spot/magnificación), la de los controles de cribado (inflaba la deriva).
   El diseño definitivo es una **cohorte de cribado** con cánceres
   confirmados por patología y controles emparejados (ver 2026-11-17).
5. **Resultado principal**: el descriptor global no ve las lesiones de
   cribado (control positivo 0.53). En el sitio de la lesión sí las ve
   (0.67, p = 0.02), pero en ese mismo sitio un cribado antes no hay señal
   (0.50). Nulo informativo.

---

## 2026-11-18

**Hecho**
- **Acceso a CSAW-CC concedido** (Karolinska, vía SND). Claves en
  `~/.aws/credentials`, perfil `csaw-cc`; datos en `data/csaw/`, añadido a
  `.gitignore` antes de descargar nada.
- Inventario: 8 723 pacientes (873 cánceres), 98 788 imágenes Hologic
  (2.3 TB), 1 006 máscaras. Fechas anonimizadas (solo el orden de las
  exploraciones) y edad en dos bandas.
- **Plan de validación fijado antes de ver los datos**
  (`docs/plan_validacion_csaw.md`, commit 17f3e70): hipótesis, cohorte,
  localización, emparejamiento, medida idéntica a EMBED, análisis
  principal con 1 000 permutaciones, criterio de éxito y secundarios. El
  script del análisis también se guardó antes de tener las imágenes.
- Cohorte (`scripts/download_csaw.py`): 109 cánceres detectados en cribado
  con cribado anterior normal y máscara, 204 vistas (130 con el sitio
  marcado directamente en t-1), 612 controles por banda de edad, vista y
  orden de exploración.

**Descubierto**
- El README de CSAW-CC dice que el radiólogo puso **un punto en el cribado
  anterior donde después apareció el cáncer**, aunque no hubiera nada
  visible. Es mejor que EMBED: no hace falta trasladar el sitio desde t.
- Condición de uso: en publicaciones hay que invitar como coautores a los
  curadores (criterios ICMJE). Para el TFG basta con citar.

**Resultado de la validación (`results/csaw_validacion.txt`)**
- **Análisis principal: AUC 0.614, p = 0.001 (1 000 permutaciones);
  regresión condicional OR 1.59 por DE (p = 3·10⁻⁷). Criterio del plan
  CUMPLIDO: la hipótesis se replica en una cohorte independiente.**
- Control positivo (t): 0.725. Sitio alejado: 0.519 (anticipación) y
  0.516 (control positivo), es decir, señal local. MLO 0.627, CC 0.579.
  H0 0.614, H1 0.588.
- Por regla de localización: marcado en t-1, 0.660; trasladado desde t,
  0.491.
- Exploratorio (`results/csaw_exploratorio.txt`): la marca en t-1 es un
  **punto** (nada visible, 88 vistas) en dos tercios de los casos, y un
  contorno (signos visibles no detectados, 42) en el resto.
  - **Con solo los puntos: 0.596 (p = 0.01).** Hay señal donde el
    radiólogo no veía nada, no solo cánceres visibles omitidos.
  - Con contorno: 0.669.
  - Trasladado desde t: 0.491. La alineación por la caja de la mama es
    demasiado imprecisa, lo que explica por qué en EMBED (todo trasladado)
    el efecto era más débil.
- No se pudo emparejar por intervalo ni fecha (fechas anonimizadas); todo
  Hologic del mismo programa.
- **Por qué eso no invalida el resultado**: los confusores de este tipo
  (intervalo, época, equipo, tipo de exploración: los de EMBED) actúan
  sobre toda la mamografía. Si explicaran la señal, también aparecería en
  la ventana desplazada media mama, y ahí da 0.519. El control negativo de
  localización descarta los confusores globales, incluidos los que no se
  pudieron emparejar.
- Notebook 04 (`04_validacion_csaw.ipynb`) con todo el análisis.

---

## 2026-11-17

**Hecho**
- Notebooks 02 y 03 ejecutados y validados de principio a fin (0 errores).
- H2 (edad): no es confusor; deriva H0 significativa ajustando por edad
  (z = 8.4). H3: la deriva funciona mejor en mamas poco densas (0.82) que
  densas (0.61). H5: la señal va por H0; H1 sola al azar.
- **Anticipación local** (`scripts/embed_local_anticipation.py`,
  `src/local_topology.py`): mama recortada, rejilla 4×4, homología por
  parche; "sitio del cambio" = parche con mayor deriva.
- Corregido: 13 imágenes Fujifilm venían en `MONOCHROME1` (invertidas, fondo
  blanco). Nuevo lector común `read_dicom_gray`.

**Descubierto**
- **El sitio de mayor cambio no es la lesión**: validado contra las ROI
  anotadas de EMBED, cae en la vecindad de la lesión en 1/12 casos (azar:
  39 %). Las variantes de anticipación local que parecían dar señal (0.67)
  eran un efecto de selección del parche, no un precursor.
- Investigando por qué, **tercer confusor**: la t de los "incidentes" era un
  estudio diagnóstico en 193/194 casos, y en 136/194 una vista de
  compresión focal/magnificación; la de los controles, cribado normal. La
  deriva de 0.751 medía en parte el tipo de vista. Además, los "incidentes"
  se definían por BI-RADS 4/5, y la mayoría de esas biopsias son benignas.

**Decidido**
- Rehacer el diseño como **cohorte de cribado**: 162 cánceres confirmados por
  patología (`path_severity` 0/1) detectados en cribado; t = ese cribado,
  t-1 = cribado negativo previo; controles con dos cribados negativos
  emparejados por intervalo, época (±1 año) y vista; solo imágenes de campo
  completo de la mama del lado del cáncer. Intervalos (385 vs 390 días) y año
  de t-1 equilibrados.

**Resultados de la cohorte de cribado (324 pacientes)**
- Global (mamografía entera a 224×224): anticipación 0.459, deriva 0.535,
  "intervalo solo" 0.503. **Todo al azar.** El 0.751 de deriva del diseño
  anterior venía entero del tipo de exploración.
- **Control positivo** (nuevo): la imagen t sola, con el cáncer ya presente,
  tampoco separa (0.530). Conclusión: el descriptor global no "ve" una
  lesión de cribado, que a esa escala son pocos píxeles. Un nulo de
  anticipación con un descriptor ciego no dice nada.
- Topología local en rejilla 4×4: todo al azar; el parche de máximo cambio
  cae en la lesión 3/82 (azar 6 %).
- **Análisis en el sitio de la lesión** (`scripts/embed_lesion_site.py`):
  ventana de 512 px (~3.3 cm) a resolución completa centrada en la ROI de t;
  la misma posición relativa se aplica a t-1 y al control emparejado
  (sin sesgo de selección del sitio). 82 pares, CV agrupada por par.
  - control positivo (t en la lesión): **0.668**, permutación p = 0.02.
    La topología sí detecta el cáncer cuando se mira donde está (coherente
    con el 0.66 de CBIS-DDSM, que también usa la lesión recortada).
  - anticipación (t-1 en el mismo sitio): **0.498**, p = 0.5.
  - deriva en el sitio: 0.501.
  - Sensibilidad con ventana de 1024 px (por si la alineación sin registro
    falla): control positivo 0.629 (p = 0.05), anticipación 0.589
    (p = 0.089, no significativa; a vigilar con más pares).
  - La densidad no está desequilibrada (p = 0.50), no confunde.

- **Prueba definitiva del tercer confusor**: recalculados los diseños
  antiguos con el lector corregido (MONOCHROME1), sus números se mantienen
  (emparejado: deriva 0.748, anticipación 0.496). Y la exploración t sola
  da **0.888 en el diseño emparejado frente a 0.530 en la cohorte de
  cribado**: el descriptor global reconocía el tipo de vista (ampliada
  diagnóstica frente a cribado), no el cáncer.
- Notebook 03 rehecho sobre la cohorte de cribado: balance de covariables,
  señales globales con control positivo, sitio de la lesión con
  permutación, H2-H4 y la comparación de los tres diseños. Ejecutado de
  principio a fin (14 celdas, 0 errores, ~28 min por las permutaciones).
  - En el sitio de la lesión, H1 (ciclos) ve más la lesión que H0
    (0.685 frente a 0.615), como en CBIS-DDSM con las espiculadas.
  - Anticipación por cercanía al diagnóstico: cerca 0.45, lejos 0.56;
    ninguna señal (n pequeño).
  - Control positivo por densidad: poco densa 0.66, densa 0.60.

- **¿Resolución o dilución?** (`scripts/embed_global_resolution.py`):
  control positivo global con la mama recortada a 224 / 512 / 1024 px:
  0.549 / 0.587 / 0.600 (±0.07–0.11). Mejora un poco con la resolución,
  pero no llega al 0.67 de la ventana en la lesión: pesa más la dilución
  (un rasgo entre miles) que la resolución. Hace falta un descriptor local.
- **Cohorte de lesión** (`--lesion`): con la versión abierta de EMBED no hay
  más cánceres elegibles (658 confirmados → 313 con cribado en el
  diagnóstico → 162 con cribado negativo previo; bajar el margen a 90 d no
  añade ninguno). Para ganar datos: todas las vistas del lado del cáncer
  con ROI (MLO y CC: 174 vistas de 97 cánceres, antes 82) y 3 controles
  por vista (522). CV agrupada por paciente del caso. Descargando.
- CI: el job giotto-tda vs gudhi (#88) fallaba porque giotto-tda 0.6.2 fija
  scikit-learn 1.3.2 (numpy 1.x) y pip retrocedía a una versión sin wheel.
  Versiones fijadas tras comprobar la resolución en seco solo con wheels.

- **Resultados de la cohorte de lesión** (174 vistas de 97 cánceres, 522
  controles; CV agrupada por paciente; permutación dentro de cada caso y
  sus controles):

  | Ventana | Control positivo | Anticipación | Deriva |
  |---|---|---|---|
  | 512 px | 0.693 (p = 0.01) | **0.569 (p = 0.04)** | 0.577 (p = 0.03) |
  | 1024 px | 0.636 (p = 0.01) | 0.534 (p = 0.21) | 0.486 (p = 0.61) |

- **Comprobación de confusores de esa anticipación**:
  - Edad desequilibrada (casos 64.2 frente a 60.9 años, p = 2·10⁻⁴; las
    covariables solas dan 0.56). Pero el score no codifica la edad (en
    controles R² ≈ 0, ρ = −0.01) y en regresión logística condicional
    ajustada por edad y densidad mantiene OR 1.29 por DE (p = 0.008).
  - **Hallazgo clínico**: la t-1 de los casos tiene BI-RADS B (benigno
    anotado) en el 17 % frente al 7 % de los controles (p = 0.001). Un año
    antes el radiólogo ya veía algo con más frecuencia. Ajustando también
    por B: OR 1.30 (p = 0.007); solo con t-1 = N: 0.569 (p = 0.06).
  - Igual en MLO (0.566) y CC (0.571).
  - **Pero frágil**: con un control por caso 0.522 (p = 0.27); con ventana
    de 1024 px 0.534; el análisis 1:1 original daba 0.498.
  - La deriva no se replica por vista (0.51 en MLO y en CC frente a 0.575
    conjunta): probable artefacto, no se interpreta.
  - **Control negativo de localización** (`--shift 0.5`: ventana desplazada
    media mama en vertical, misma profundidad, igual en casos y controles):
    control positivo 0.524 (p = 0.26), anticipación 0.472 (p = 0.86),
    deriva 0.503. Fuera del sitio no hay nada: el indicio es **local**, no
    una propiedad de toda la mama (tipo marcador de riesgo).
  - Por dimensión: anticipación solo H1 0.568, solo H0 0.544 (como el
    control positivo, H1 ve algo más).
  - Ninguna imagen t-1 (casos ni controles) tiene ROI anotada: no se puede
    comprobar si el radiólogo marcó algo en el sitio un año antes.
  - **Replicación con controles emparejados por edad** (`--age-caliper 3`,
    `longitudinal_manifest_lesion_edad.csv`; 451 de 522 controles nuevos):
    edad 64.2 frente a 64.2 (p = 0.90); control positivo 0.679 (p = 0.01);
    **anticipación 0.571 (p = 0.02)**; 1:1 0.616 (p = 0.01); regresión
    condicional ajustada por densidad y B: OR 1.24 (p = 0.02); MLO 0.588
    (p = 0.06), CC 0.579 (p = 0.08). La "fragilidad" del 1:1 anterior era
    ruido del muestreo de controles.
  - **Lectura**: indicio débil, no concluyente, tras muchas variantes
    probadas. Hipótesis para validar fuera (CSAW-CC) con el análisis fijado
    de antemano; no un resultado.

**Lectura actualizada (tras la replicación por edad)**: la anticipación en
el sitio de la lesión es un **efecto modesto pero consistente** (AUC
≈0.57–0.62): local (desaparece media mama más allá), presente en MLO y CC,
no explicado por edad, densidad ni BI-RADS B, y replicado con un conjunto
de controles casi nuevo y emparejado por edad. Sigue siendo un único
dataset, los mismos 97 cánceres y un análisis elegido tras varias variantes:
hace falta validación externa con el análisis fijado de antemano.

**¿Estaba ya formulada la hipótesis?** (revisión, detalle en la issue #93)
La idea general sí: entre un tercio y la mitad de los cánceres detectados
en cribado tienen signos mínimos u omitidos en la previa (revisiones
radiológicas: Noruega 2020, Flandes 2022), y modelos de IA marcan el sitio
del cáncer futuro 2–4 años antes en el 57–81 % de casos seleccionados
(Noruega 2026). Lo que no se ha encontrado es topología o TDA sobre
mamografías previas, ni un diseño emparejado con control positivo y de
localización, ni la descripción de los confusores. La aportación es el
descriptor interpretable, el diseño controlado y los confusores, no el
descubrimiento de la señal.

**Conclusión del día**: con un diseño sin confusores y un descriptor que sí
detecta la lesión cuando está, **no hay señal topológica detectable en el
mismo sitio un cribado antes** (~1 año). Es un nulo informativo, no un nulo
por ceguera del método. Pendiente: más pares con ROI (solo 82 de 162
incidentes tienen ROI en la vista usada) para cerrar la ventana de 1024.

---

## 2026-11-16

**Hecho**
- Cohorte EMBED ampliada a 394 pacientes. Notebook 03 extendido con
  hipótesis secundarias (edad, densidad, intervalo, H0 vs H1, casos extremos).
- Eliminado todo lo sintético: script de validación sintética, notebook 01
  y figuras. `topological_drift` movido a `src/topology.py`.
- Test de equivalencia giotto-tda vs gudhi en CI (Python 3.12; en local no
  hay wheels de giotto-tda para 3.14).
- 11 papers de la bibliografía descargados en `bib/`.

**Descubierto**
- Con 394 pacientes la anticipación parecía significativa (0.627), pero
  **el intervalo t-1 → t por sí solo daba AUC 0.745**: los controles usaban
  sus dos primeras exploraciones (~1 año) y los casos años de diferencia.
- Diseño emparejado 1:1 por intervalo (194 + 194): "intervalo solo" cae a
  0.527, la deriva se mantiene (0.751) y la **anticipación cae al azar
  (0.488)**: el 0.627 venía de que el año de t-1 difería (2014 vs 2013,
  p = 3.5·10⁻⁴), un artefacto de época de adquisición.

---

## 2026-11-10

**Hecho**
- Primer resultado longitudinal real (EMBED): 40 → 99 pacientes.
- Protocolo de hold-out congelado documentado (issue #89).
- Filtración alternativa (transformada de distancia): mismo CV-AUC que la
  intensidad (0.658), el resultado no es un artefacto de la filtración.

**Descubierto**
- `persim.bottleneck` tardaba ~121 s por par con diagramas reales (miles de
  puntos frente a decenas en los sintéticos). Recortando a los 100 puntos
  más persistentes baja a ~0.2 s.

---

## 2026-11-09

**Hecho**
- Estudio de interpretabilidad completo (Grad-CAM, región topológica
  dominante, SHAP, LIME) con bootstrap y test de Wilcoxon pareado.
- Piloto CBIS-DDSM ampliado de 30 a 70 pacientes.
- Guía de acceso a datasets longitudinales (`docs/acceso_datasets_longitudinales.pdf`).
- **Acceso a EMBED concedido.** Confirmada su estructura longitudinal:
  23 253 pacientes, 14 789 con 2+ exploraciones.
- Historial de git reescrito para quitar la atribución de Claude (decisión
  del autor: no debe aparecer en GitHub).

**Descubierto**
- Al ampliar el piloto, el CV-AUC de TDA bajó de 0.850 a 0.658: el número
  del piloto pequeño era optimista.
- Bug real en `load_case_descriptions`: el CSV de masas usa
  `breast_density` y el de calcificaciones `breast density`; quedaban dos
  columnas duplicadas.
- Los cuatro métodos de interpretabilidad apenas coinciden entre sí.

---

## 2026-11-05 / 2026-11-06

**Hecho**
- Resuelto un conflicto de Git sin resolver que el backup automático había
  dejado dentro de `run_pipeline.py`.
- Notebook 02 con datos reales; guía de uso en PDF; CI con pytest;
  dependencias fijadas; sanity check visual espiculada benigna vs maligna.
- Veredicto de novedad documentado: nadie combina TDA con análisis
  longitudinal de mamografía.

**Descubierto**
- El backup automático de la máquina comitea y pushea solo, y en ocasiones
  revierte cambios recientes. Revisar `git log` si algo "desaparece".

---

## Octubre 2026 (antes de este diario)

- Pivote de la hipótesis: de "interpretabilidad TDA vs Grad-CAM" a
  **detección anticipada longitudinal** (issue #91).
- Descarga real de CBIS-DDSM vía API de TCIA, migración de la CNN a PyTorch,
  diagnóstico del sobreajuste (demasiadas features para tan pocos casos).
- Búsqueda de datasets longitudinales: CSAW-CC y EMBED, ambos de acceso
  controlado (2026-10-07).

## Mayo – septiembre 2026

- Arranque del repositorio, primeras descargas, bibliografía inicial y
  planificación en issues.
