"""Figuras de la memoria (tfg/figuras/) a partir de los ficheros de results/.

Uso:
    python scripts/figuras_memoria.py

Ningun numero esta escrito a mano: todos se leen de los ficheros de
resultados que generan los scripts de analisis, asi que las figuras siempre
coinciden con el ultimo resultado guardado.
"""
from __future__ import annotations

import re
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
OUT = ROOT / "tfg" / "figuras"
GRIS, AZUL, NARANJA, OSCURO = "#9AA3AE", "#1F4E8C", "#B4531A", "#1A1F2B"


def auc_global(fichero: str, clave: str) -> float:
    txt = (RES / fichero).read_text(encoding="utf-8")
    return float(re.search(rf"auc_{re.escape(clave)}: ([0-9.]+)", txt).group(1))


def auc_linea(fichero: str, patron: str) -> float:
    txt = (RES / fichero).read_text(encoding="utf-8")
    return float(re.search(patron + r".*?AUC[^=]*= ([0-9.]+)", txt).group(1))


def tres_disenos() -> None:
    disenos = {"original": "embed_longitudinal_pilot.txt",
               "emparejado por intervalo": "embed_longitudinal_pilot_matched.txt",
               "cohorte de cribado": "embed_longitudinal_cribado.txt"}
    senales = {"solo intervalo\n(sin imagen)": "intervalo_solo", "anticipación\n(t-1)": "anticipacion",
               "deriva\n(t-1 frente a t)": "deriva", "solo t\n(control positivo)": "diagnostico_(solo_t)"}
    colores = [GRIS, NARANJA, AZUL]
    fig, ax = plt.subplots(figsize=(8, 3.6))
    x = np.arange(len(senales)); w = 0.26
    for k, (nombre, f) in enumerate(disenos.items()):
        vals = [auc_global(f, c) for c in senales.values()]
        ax.bar(x + (k - 1) * w, vals, w, color=colores[k], label=nombre)
    ax.axhline(0.5, color=OSCURO, ls="--", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(list(senales)); ax.set_ylim(0.3, 1.0)
    ax.set_ylabel("AUC (validación cruzada)")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(OUT / "tres_disenos.png", dpi=200); plt.close(fig)


def embed_vs_csaw() -> None:
    etiquetas = ["control positivo\n(t, en el sitio)", "anticipación\n(t-1, en el sitio)", "sitio alejado\n(t-1, media mama)"]
    embed = [auc_linea("embed_lesion_site_lesion_edad.txt", "control positivo"),
             auc_linea("embed_lesion_site_lesion_edad.txt", "anticipacion: t-1"),
             auc_linea("embed_lesion_site_lesion_alejado0.5.txt", "anticipacion: t-1")]
    csaw_txt = (RES / "csaw_validacion.txt").read_text(encoding="utf-8")
    csaw = [float(re.search(r"control positivo \(t en el sitio\)\s*: AUC = ([0-9.]+)", csaw_txt).group(1)),
            float(re.search(r"ANALISIS PRINCIPAL.*?AUC = ([0-9.]+)", csaw_txt, re.S).group(1)),
            float(re.search(r"sitio alejado.*?anticipacion AUC = ([0-9.]+)", csaw_txt).group(1))]
    fig, ax = plt.subplots(figsize=(7, 3.4))
    x = np.arange(3); w = 0.36
    ax.bar(x - w / 2, embed, w, color=GRIS, label="EMBED (descubrimiento)")
    ax.bar(x + w / 2, csaw, w, color=AZUL, label="CSAW-CC (validación prefijada)")
    for i, (a, b) in enumerate(zip(embed, csaw)):
        ax.text(i - w / 2, a + 0.008, f"{a:.2f}", ha="center", fontsize=8)
        ax.text(i + w / 2, b + 0.008, f"{b:.2f}", ha="center", fontsize=8)
    ax.axhline(0.5, color=OSCURO, ls="--", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(etiquetas); ax.set_ylim(0.4, 0.8)
    ax.set_ylabel("AUC (validación cruzada)")
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(OUT / "embed_vs_csaw.png", dpi=200); plt.close(fig)


def tipo_marca() -> None:
    txt = (RES / "csaw_exploratorio.txt").read_text(encoding="utf-8")
    filas = [("punto\n(nada visible)", r"sitio por punto"), ("contorno\n(signos visibles)", r"sitio por contorno"),
             ("trasladado\ndesde t", r"sitio trasladado")]
    vals = [float(re.search(p + r".*?AUC = ([0-9.]+)", txt).group(1)) for _, p in filas]
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    ax.bar(range(3), vals, color=[AZUL, NARANJA, GRIS], width=0.55)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.008, f"{v:.2f}", ha="center", fontsize=9)
    ax.axhline(0.5, color=OSCURO, ls="--", lw=0.8)
    ax.set_xticks(range(3)); ax.set_xticklabels([f for f, _ in filas]); ax.set_ylim(0.4, 0.75)
    ax.set_ylabel("AUC anticipación (t-1)")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(OUT / "csaw_tipo_marca.png", dpi=200); plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    tres_disenos()
    embed_vs_csaw()
    tipo_marca()
    src = RES / "figuras" / "interpretacion_anticipacion.png"
    if src.exists():
        shutil.copy(src, OUT / "interpretacion_anticipacion.png")
    print("Figuras en", OUT, sorted(p.name for p in OUT.glob("*.png")))


if __name__ == "__main__":
    main()
