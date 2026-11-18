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


def localizacion() -> None:
    """Percentil del sitio en su mama: caso frente a la misma posicion en su control."""
    import pandas as pd

    f0 = ROOT / "data" / "csaw" / "localizacion_t0.csv"
    f1 = ROOT / "data" / "csaw" / "localizacion_t1.csv"
    if not (f0.exists() and f1.exists()):
        print("  (sin datos de localizacion: se omite)")
        return
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
    bins = np.linspace(0, 1, 11)
    for ax, f, titulo in ((axes[0], f1, "t: diagnóstico (control positivo)"),
                          (axes[1], f0, "t-1: cribado anterior")):
        d = pd.read_csv(f).dropna(subset=["percentil_control"])
        ax.hist(d["percentil_control"], bins=bins, color=GRIS, alpha=0.85, label="misma posición en su control sano")
        ax.hist(d["percentil"], bins=bins, color=AZUL, alpha=0.6, label="sitio del cáncer")
        ax.axvline(d["percentil"].median(), color=AZUL, lw=1.6)
        ax.axvline(d["percentil_control"].median(), color=OSCURO, lw=1.2, ls="--")
        ax.set_title(f"{titulo} ({len(d)} mamas)", fontsize=10)
        ax.set_xlabel("percentil de la puntuación dentro de su mama")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("mamas")
    axes[1].legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout(); fig.savefig(OUT / "localizacion.png", dpi=200); plt.close(fig)


def registro() -> None:
    """Error del traslado t -> t-1 con cada metodo de registro."""
    import pandas as pd

    f = ROOT / "data" / "csaw" / "registro_errores.csv"
    if not f.exists():
        print("  (sin datos de registro: se omite)")
        return
    d = pd.read_csv(f)
    metodos = [("caja", "caja de la mama"), ("afin", "afín (ECC)"), ("elastico", "elástico (TV-L1)")]
    fig, ax = plt.subplots(figsize=(6.4, 3.3))
    datos = [d[m].dropna().values for m, _ in metodos]
    bp = ax.boxplot(datos, tick_labels=[e for _, e in metodos], showfliers=False, patch_artist=True, widths=0.5)
    for patch, c in zip(bp["boxes"], [GRIS, NARANJA, AZUL]):
        patch.set_facecolor(c); patch.set_alpha(0.7)
    ax.axhline(18, color=OSCURO, ls="--", lw=0.8)
    ax.text(3.45, 18.6, "media ventana (18 mm)", fontsize=8, ha="right")
    for i, v in enumerate(datos, 1):
        ax.text(i, np.median(v) + 0.6, f"{np.median(v):.1f} mm", ha="center", fontsize=8)
    ax.set_ylabel("error de localización (mm)")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(OUT / "registro.png", dpi=200); plt.close(fig)


def filtracion() -> None:
    """Filtracion por subniveles sobre una ventana real de mamografia y su diagrama."""
    import sys
    sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
    import pickle

    import pandas as pd

    m_path = ROOT / "data" / "csaw" / "manifest_csaw.csv"
    if not m_path.exists():
        print("  (sin datos de CSAW: se omite la figura de filtracion)")
        return
    import case_studies as cst
    import csaw_lesion_site as cs
    from src.topology import persistence_diagram

    m = pd.read_csv(m_path)
    r = m[m["kind"] == "incidente"].iloc[0]
    rel0, _ = pickle.loads(cs.CACHE_PATH.read_bytes())[("rel", r.path_t0, r.path_t1)]
    w = cst.window(ROOT / r.path_t0, rel0)
    dgm = persistence_diagram(w)[0]

    fig, axes = plt.subplots(1, 4, figsize=(12, 3.2))
    axes[0].imshow(w, cmap="gray"); axes[0].set_title("ventana (intensidad)", fontsize=9)
    for ax, thr in zip(axes[1:3], (0.35, 0.65)):
        ax.imshow(w <= thr, cmap="Blues", vmin=0, vmax=1.4)
        ax.set_title(f"subnivel: píxeles con intensidad ≤ {thr}", fontsize=9)
    for ax in axes[:3]:
        ax.axis("off")
    ax = axes[3]
    for dim, color, lab in ((0, AZUL, "$H_0$ componentes"), (1, NARANJA, "$H_1$ ciclos")):
        p = dgm[dgm[:, 2] == dim]
        ax.scatter(p[:, 0], p[:, 1], s=4, color=color, alpha=0.6, label=lab, linewidths=0)
    ax.plot([0, 1], [0, 1], color=OSCURO, lw=0.8)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02); ax.set_aspect("equal")
    ax.set_xlabel("nacimiento"); ax.set_ylabel("muerte")
    ax.set_title("diagrama de persistencia", fontsize=9)
    ax.legend(frameon=False, fontsize=7, loc="lower right", markerscale=3)
    fig.tight_layout(); fig.savefig(OUT / "filtracion.png", dpi=200); plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    tres_disenos()
    embed_vs_csaw()
    tipo_marca()
    localizacion()
    registro()
    filtracion()
    src = RES / "figuras" / "interpretacion_anticipacion.png"
    if src.exists():
        shutil.copy(src, OUT / "interpretacion_anticipacion.png")
    print("Figuras en", OUT, sorted(p.name for p in OUT.glob("*.png")))


if __name__ == "__main__":
    main()
