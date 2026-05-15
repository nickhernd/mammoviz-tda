"""
Genera un fantoma mamográfico 3D sintético con:
  - Tejido graso (background), glandular (heterogéneo) y piel
  - Microcalcificaciones en clústeres (patrón maligno lineal / benigno redondo)
  - Masa espiculada o redonda configurable
  - Metadatos DICOM válidos (Modality=MG, DICOM-CT-like UIDs)

Uso:
    python3 scripts/gen_realistic_phantom.py data/samples/malignant   --pattern malignant
    python3 scripts/gen_realistic_phantom.py data/samples/benign      --pattern benign
    python3 scripts/gen_realistic_phantom.py data/samples/case001     --pattern malignant  # sobreescribe el sintético básico

Requiere:  pip install pydicom numpy
"""

import argparse
import os
import sys
import numpy as np
import pydicom
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.sequence import Sequence
from pydicom.uid import generate_uid, ExplicitVRLittleEndian

# ── Dimensiones del volumen ───────────────────────────────────────────────────
NX, NY, NZ = 128, 128, 48   # x, y (in-plane), z (slices)
SPACING_XY  = 0.1           # mm por pixel (0.1 mm ≈ resolución mamografía digital)
SPACING_Z   = 1.0           # mm entre slices

RNG = np.random.default_rng(42)


def sigmoid(x, k=10):
    return 1.0 / (1.0 + np.exp(-k * x))


# ── Construcción del volumen ──────────────────────────────────────────────────

def make_volume(pattern: str) -> np.ndarray:
    """Devuelve float32 [NX, NY, NZ] normalizado a [0, 1]."""
    vol = np.zeros((NX, NY, NZ), dtype=np.float32)

    cx, cy = NX // 2, NY // 2
    xx, yy, zz = np.meshgrid(
        np.linspace(-1, 1, NX),
        np.linspace(-1, 1, NY),
        np.linspace(-1, 1, NZ),
        indexing='ij'
    )

    # 1. Forma elíptica de la mama (máscara de tejido)
    r_breast = np.sqrt((xx / 0.90)**2 + (yy / 0.85)**2 + (zz / 0.50)**2)
    breast_mask = r_breast < 1.0

    # 2. Tejido graso (fondo bajo, ruido suave)
    fat = 0.08 + 0.06 * RNG.standard_normal((NX, NY, NZ)).astype(np.float32)
    fat = np.clip(fat, 0, 1)

    # 3. Tejido glandular heterogéneo — perlin-like con múltiples escalas
    gland = np.zeros((NX, NY, NZ), dtype=np.float32)
    for scale, amp in [(0.15, 0.35), (0.08, 0.20), (0.04, 0.10)]:
        noise = RNG.standard_normal((NX, NY, NZ)).astype(np.float32)
        # Suavizado simple con convolución gaussiana
        from scipy.ndimage import gaussian_filter
        gland += amp * gaussian_filter(noise, sigma=scale * NX)

    gland = (gland - gland.min()) / (gland.max() - gland.min() + 1e-8)
    gland_mask = gland > 0.55   # regiones densas

    # Componer: grasa + glandular
    vol = fat.copy()
    vol[gland_mask] = 0.35 + 0.20 * gland[gland_mask]

    # 4. Capa de piel (borde fino de alta densidad)
    skin = (r_breast > 0.90) & (r_breast < 1.0)
    vol[skin] = 0.65 + 0.10 * RNG.random((NX, NY, NZ)).astype(np.float32)[skin]

    # Enmascarar fuera de la mama
    vol[~breast_mask] = 0.0

    # 5. Microcalcificaciones en clústeres
    _add_calcifications(vol, breast_mask, pattern)

    # 6. Masa principal
    _add_mass(vol, breast_mask, pattern)

    return np.clip(vol, 0, 1).astype(np.float32)


def _add_calcifications(vol: np.ndarray, breast_mask: np.ndarray, pattern: str):
    """
    Maligno:  3-5 clústeres, distribución lineal/ramificada (ductal).
    Benigno:  2-3 clústeres, distribución redonda y regular.
    """
    n_clusters = 4 if pattern == 'malignant' else 2

    for _ in range(n_clusters):
        # Centro del clúster aleatorio dentro de la mama
        candidates = np.argwhere(breast_mask)
        center = candidates[RNG.integers(len(candidates))]
        cx, cy, cz = center

        n_calcs = RNG.integers(8, 20)   # calcificaciones por clúster

        if pattern == 'malignant':
            # Distribución lineal (a lo largo de un ducto)
            direction = RNG.standard_normal(3)
            direction /= np.linalg.norm(direction) + 1e-8
            spread = 12.0
            offsets = RNG.standard_normal((n_calcs, 3)) * 1.5
            offsets += np.outer(np.linspace(-spread/2, spread/2, n_calcs), direction)
        else:
            # Distribución redonda y regular
            offsets = RNG.standard_normal((n_calcs, 3)) * 6.0

        for off in offsets:
            px = int(np.clip(cx + off[0], 1, NX - 2))
            py = int(np.clip(cy + off[1], 1, NY - 2))
            pz = int(np.clip(cz + off[2], 1, NZ - 2))
            if breast_mask[px, py, pz]:
                # Microcalcificación: 1-2 voxels de alta intensidad
                vol[px, py, pz] = 0.92 + 0.07 * RNG.random()
                if RNG.random() > 0.5:
                    vol[px+1, py, pz] = 0.88 + 0.07 * RNG.random()


def _add_mass(vol: np.ndarray, breast_mask: np.ndarray, pattern: str):
    """
    Maligno:  masa espiculada (bordes irregulares, alta densidad central).
    Benigno:  masa redonda/oval de bordes lisos.
    """
    candidates = np.argwhere(breast_mask)
    center = candidates[RNG.integers(len(candidates))]
    cx, cy, cz = center

    xx, yy, zz = np.meshgrid(
        np.arange(NX), np.arange(NY), np.arange(NZ), indexing='ij'
    )

    if pattern == 'malignant':
        # Elipsoide con perturbación espiculada
        from scipy.ndimage import gaussian_filter
        noise = gaussian_filter(
            RNG.standard_normal((NX, NY, NZ)).astype(np.float32), sigma=3
        )
        rx, ry, rz = 8, 7, 5
        dist = np.sqrt(
            ((xx - cx) / rx)**2 +
            ((yy - cy) / ry)**2 +
            ((zz - cz) / rz)**2
        )
        spicules = 1.0 + 0.35 * noise   # modulación del radio
        mass_mask = (dist * spicules) < 1.0
        intensity_profile = np.clip(1.0 - dist * spicules, 0, 1) * 0.75
    else:
        # Masa redonda lisa
        rx, ry, rz = 7, 7, 5
        dist = np.sqrt(
            ((xx - cx) / rx)**2 +
            ((yy - cy) / ry)**2 +
            ((zz - cz) / rz)**2
        )
        mass_mask = dist < 1.0
        intensity_profile = np.clip(1.0 - dist, 0, 1) * 0.65

    mass_mask &= breast_mask
    vol[mass_mask] = np.maximum(vol[mass_mask], intensity_profile[mass_mask].astype(np.float32))


# ── Conversión a DICOM ────────────────────────────────────────────────────────

def save_dicom_series(vol: np.ndarray, out_dir: str, patient_id: str, pattern: str):
    """Guarda cada slice Z como un archivo .dcm válido."""
    os.makedirs(out_dir, exist_ok=True)

    series_uid   = generate_uid()
    study_uid    = generate_uid()
    frame_of_ref = generate_uid()

    # Escalar a uint16 (0–4095, rango mamografía digital típico)
    hu = (vol * 4095).astype(np.uint16)

    for z in range(NZ):
        fmeta = FileMetaDataset()
        fmeta.MediaStorageSOPClassUID    = '1.2.840.10008.5.1.4.1.1.2'  # CT Storage (compatible)
        fmeta.MediaStorageSOPInstanceUID = generate_uid()
        fmeta.TransferSyntaxUID          = ExplicitVRLittleEndian

        ds = Dataset()
        ds.file_meta = fmeta
        ds.is_implicit_VR  = False
        ds.is_little_endian = True

        # Patient / Study / Series
        ds.PatientID          = patient_id
        ds.PatientName        = f"PHANTOM^{pattern.upper()}"
        ds.StudyInstanceUID   = study_uid
        ds.SeriesInstanceUID  = series_uid
        ds.SOPInstanceUID     = fmeta.MediaStorageSOPInstanceUID
        ds.SOPClassUID        = fmeta.MediaStorageSOPClassUID
        ds.FrameOfReferenceUID = frame_of_ref

        ds.Modality           = "MG"
        ds.BodyPartExamined   = "BREAST"
        ds.StudyDescription   = f"Synthetic phantom - {pattern}"
        ds.SeriesDescription  = f"DBT-like 3D volume - {pattern}"

        # Image geometry
        ds.Rows               = NY
        ds.Columns            = NX
        ds.InstanceNumber     = str(z + 1)
        ds.SliceLocation      = str(round(z * SPACING_Z, 4))
        ds.ImagePositionPatient = [0.0, 0.0, round(z * SPACING_Z, 4)]
        ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
        ds.PixelSpacing       = [SPACING_XY, SPACING_XY]
        ds.SliceThickness     = SPACING_Z
        ds.SpacingBetweenSlices = SPACING_Z

        # Pixel data
        ds.SamplesPerPixel    = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.BitsAllocated      = 16
        ds.BitsStored         = 12
        ds.HighBit            = 11
        ds.PixelRepresentation = 0
        ds.PixelData          = hu[:, :, z].T.tobytes()   # row-major: (NY, NX)

        # Rescale para que DicomLoader convierta bien a [0,1]
        ds.RescaleIntercept   = "0"
        ds.RescaleSlope       = "1"

        fname = os.path.join(out_dir, f"slice_{z:04d}.dcm")
        pydicom.dcmwrite(fname, ds, write_like_original=False)

    print(f"Saved {NZ} DICOM slices ({NX}×{NY}) → {out_dir}")
    print(f"  Pattern  : {pattern}")
    print(f"  Spacing  : {SPACING_XY}×{SPACING_XY}×{SPACING_Z} mm")
    print(f"  Pixel range: 0–4095 (uint16 → rescaled to [0,1] by DicomLoader)")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Check scipy available
    try:
        import scipy
    except ImportError:
        print("ERROR: scipy required.  pip install scipy", file=sys.stderr)
        sys.exit(1)

    parser = argparse.ArgumentParser(description="Generate realistic mammography phantom")
    parser.add_argument("output_dir", nargs="?", default="data/samples/case001",
                        help="Output directory for DICOM series")
    parser.add_argument("--pattern", choices=["malignant", "benign"],
                        default="malignant",
                        help="malignant: spiculated mass + linear calcs; "
                             "benign: round mass + round calcs")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    RNG = np.random.default_rng(args.seed)

    patient_id = f"PHANTOM_{args.pattern.upper()}_{args.seed:03d}"

    print(f"Generating {args.pattern} phantom ({NX}×{NY}×{NZ} voxels)...")
    volume = make_volume(args.pattern)

    save_dicom_series(volume, args.output_dir, patient_id, args.pattern)
    print("Done.")
