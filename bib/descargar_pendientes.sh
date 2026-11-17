#!/usr/bin/env bash
# Descarga los papers de acceso abierto que faltan en bib/.
# Uso (desde la raiz del repo):  bash bib/descargar_pendientes.sh
# Cada fichero se descarga a bib/ y se comprueba que empieza por %PDF; si
# un servidor devuelve una pagina HTML en vez del PDF, se borra y se avisa
# para bajarlo a mano desde el enlace.
set -u
cd "$(dirname "$0")"
UA="Mozilla/5.0 (Windows NT 10.0; Win64; x64)"

dl() {
  local out="$1" url="$2"
  if [ -s "$out" ]; then echo "ya existe: $out"; return; fi
  curl -sL -A "$UA" --max-time 120 -o "$out" "$url"
  if [ "$(head -c 4 "$out" 2>/dev/null)" = "%PDF" ]; then
    echo "OK      : $out"
  else
    rm -f "$out"; echo "A MANO  : $out  <-  $url"
  fi
}

# Citados ya en tfg/bibliografia.bib
dl jetomo2025_peerj_cs3374.pdf               "https://peerj.com/articles/cs-3374.pdf"

# Estado del arte de la hipotesis de anticipacion (issue #93)
dl hovda2020_prior_review_screen_detected.pdf "https://europepmc.org/articles/PMC7979605?pdf=render"
dl flanders2022_negative_mammograms_review.pdf "https://europepmc.org/articles/PMC9673605?pdf=render"
dl norway2026_ai_markers_prior.pdf            "https://europepmc.org/articles/PMC13661703?pdf=render"
dl diagnostics2022_missed_cancer_aicad.pdf    "https://europepmc.org/articles/PMC8871484?pdf=render"

# Homologia persistente en mamografia (una sola exploracion)
dl ph_mass_classification_2201.02295.pdf      "https://arxiv.org/pdf/2201.02295"
dl ph_microcalcifications_pmc10177619.pdf     "https://europepmc.org/articles/PMC10177619?pdf=render"

# Herramientas y teoria que usa el pipeline
dl gudhi_maria2014.pdf                        "https://hal.inria.fr/hal-01005601/document"
