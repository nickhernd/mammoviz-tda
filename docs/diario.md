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
