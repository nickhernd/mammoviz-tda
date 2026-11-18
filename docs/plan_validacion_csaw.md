# Plan de validación externa en CSAW-CC (fijado antes de ver los datos)

**Fecha de fijación:** 2026-11-18, antes de descargar ninguna imagen de
CSAW-CC y antes de calcular ningún resultado sobre ese dataset. El commit
que introduce este fichero es la marca temporal. Cualquier análisis que no
esté descrito aquí se presentará como **exploratorio**.

## Hipótesis a validar

En EMBED (cohorte de lesión, 97 cánceres, controles emparejados por edad),
la topología de una ventana de 512 px centrada donde aparecerá el cáncer,
medida en el **cribado anterior leído como normal**, separa casos de
controles con AUC ≈ 0.57 (permutación p ≈ 0.02). La señal es local: con la
ventana desplazada media mama desaparece.

**H1 (principal):** en CSAW-CC, con el mismo análisis, el AUC de anticipación
es mayor que 0.5.

## Datos

CSAW-CC, parte no reservada (8 723 pacientes, mamografías Hologic de
Karolinska, 2008–2016). Fechas anonimizadas: solo se conoce el orden de las
exploraciones (sufijo `_N` del nombre de fichero) y la variable `rad_timing`.

## Cohorte

**Casos:**
- `x_case = 1`.
- t = la última exploración con `rad_timing = 1` (cáncer detectado en
  cribado, menos de 60 días del cribado al diagnóstico).
- t-1 = la exploración inmediatamente anterior (`N - 1`) con
  `rad_recall = 0` (leída como normal).
- Vistas: MLO y CC de la mama del lado del cáncer (`x_cancer_laterality`).
  Cada vista que cumpla la regla de localización de abajo es una fila.

**Localización del sitio en t-1:**
1. Si la imagen t-1 tiene máscara, se usa el centroide de los píxeles
   marcados (el radiólogo puso un punto donde después apareció el cáncer, o
   contorneó un signo visible).
2. Si no la tiene pero la imagen t sí, se usa el centroide de la máscara de
   t trasladado a t-1 con la misma posición relativa dentro de la caja de la
   mama (`src.local_topology.map_point`), como en EMBED.
3. Si no hay máscara en ninguna de las dos, la vista se excluye.

**Controles:** 3 por cada fila de caso.
- `x_case = 0`.
- Dos exploraciones consecutivas (`N - 1`, `N`), ambas con `rad_recall = 0`.
- Misma banda de edad (`x_age`), misma lateralidad y vista que el caso.
- Mismo número de orden de t-1 si es posible; si no, el más cercano.
- Cada control se usa una sola vez. Selección aleatoria con semilla 42.
- El sitio del control es la posición relativa del sitio del caso en la
  caja de la mama del caso, trasladada a la del control (igual que en
  EMBED).

No se puede emparejar por intervalo ni por fecha, porque las fechas están
anonimizadas. Todas las imágenes son Hologic del mismo programa y periodo.

## Medida

Sin cambios respecto a EMBED (`scripts/embed_lesion_site.py`):
- ventana de 512×512 px a resolución completa (≈ 36 mm con píxel de 0.07 mm);
- reducción a 256×256, CLAHE, homología cúbica (gudhi);
- imagen de persistencia de 6×6 por dimensión (H0 y H1, 72 valores).

## Análisis principal

- Random Forest (200 árboles, `class_weight="balanced"`).
- AUC por validación cruzada de 5 particiones agrupada por paciente del caso
  (`GroupKFold`).
- p-valor por **1 000 permutaciones** dentro de cada conjunto caso + controles
  (`embed_lesion_site.permutation_auc`).
- Apoyo: regresión logística condicional (estratos = conjunto) del score
  fuera de fold; se informa el OR por desviación típica.

**Criterio de éxito de H1:** AUC > 0.5 con p < 0.05 en la permutación, y el
OR de la regresión condicional > 1.

## Análisis secundarios (también fijados)

1. **Control positivo:** la misma ventana en t, sobre la máscara de t.
   Esperado: AUC claramente > 0.5. Si no lo es, el análisis principal no se
   interpreta.
2. **Control negativo de localización:** ventana desplazada media mama en
   vertical (`--shift 0.5`). Esperado: AUC ≈ 0.5.
3. Por vista: MLO y CC por separado.
4. Por dimensión: solo H0 y solo H1.
5. Subgrupo con el sitio marcado directamente en t-1 (regla 1) frente a
   trasladado desde t (regla 2).

## Lo que no se hará

- No se cambiará el tamaño de ventana, el número de bins, el clasificador ni
  el preprocesado tras ver resultados.
- No se descartarán casos por su resultado.
- Si el análisis principal no alcanza el criterio, se informará como
  **no replicado**, sea cual sea el resultado de los análisis secundarios.

## Condiciones de uso del dataset

Según el README de CSAW-CC, una publicación debe seguir los criterios ICMJE
e **invitar como coautores a los curadores** del dataset (Fredrik Strand y
colaboradores, Karolinska Institutet). Para el TFG basta con citarlo; para un
artículo hay que contactarles.

---

# Adenda 1 (2026-11-18): registro t -> t-1

**Fijada antes de calcular nada con registro sobre CSAW-CC.** El commit que
introduce esta adenda es la marca temporal.

## Motivo

En el análisis principal, las 74 vistas de caso sin marca en t-1 (el sitio
se trasladaba desde t con la caja de la mama) daban AUC 0.491. En CSAW-CC,
el error geométrico de ese traslado es de 12.2 mm de mediana, frente a
6.1 mm con un registro elástico (`results/registro_csaw.txt`; se mide solo
con posiciones, sin etiquetas ni AUC). En EMBED, que es el conjunto de
desarrollo, el registro elástico sube la anticipación de 0.573 a 0.632
(`results/registro_sitio_embed_*`).

## Hipótesis H2

En CSAW-CC, en las vistas sin marca en t-1, la anticipación con el sitio
trasladado mediante **registro elástico** (`src/registration.py`, método
`elastico`) tiene AUC > 0.5.

## Análisis

- `python scripts/registered_lesion_site.py --dataset csaw --metodo elastico --subconjunto trasladado --permutations 1000`.
- Misma medida, clasificador, validación cruzada y permutaciones dentro de
  conjunto que el análisis principal.
- **Criterio:** AUC > 0.5, p < 0.05 (1 000 permutaciones) y OR condicional > 1.
- Secundario: la cohorte completa con registro (`--subconjunto todo`). Las
  vistas con marca en t-1 no cambian.

## Limitación declarada

Estas 74 vistas ya se vieron en el análisis principal (con la caja de la
mama). El método de registro se eligió por geometría y se probó en EMBED,
no en ellas, pero no son datos "vírgenes". La prueba limpia sería el 30 %
reservado de CSAW-CC, en colaboración con Karolinska.
