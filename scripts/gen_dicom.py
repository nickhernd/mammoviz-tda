"""
Generate a synthetic 3D DICOM series that mimics a mammography volume.
Produces N slices with a gaussian blob (simulated tumor/mass) + noise.
"""
import numpy as np
import pydicom
from pydicom.dataset import Dataset, FileDataset
from pydicom.sequence import Sequence
from pydicom.uid import generate_uid, ExplicitVRLittleEndian
import datetime
import os
import sys

OUT_DIR = sys.argv[1] if len(sys.argv) > 1 else "data/samples/case001"
NX, NY, NZ = 64, 64, 32   # small volume so TDA is fast

os.makedirs(OUT_DIR, exist_ok=True)

rng = np.random.default_rng(42)

# Create volume: background noise + a bright blob (simulated microcalcification cluster)
volume = rng.normal(100, 15, (NZ, NY, NX)).astype(np.int16)
# Gaussian blob at centre
cx, cy, cz = NX // 2, NY // 2, NZ // 2
for z in range(NZ):
    for y in range(NY):
        for x in range(NX):
            d2 = ((x-cx)**2 + (y-cy)**2)/64.0 + ((z-cz)**2)/16.0
            volume[z, y, x] += int(800 * np.exp(-d2 / 20.0))
volume = np.clip(volume, -1000, 3000).astype(np.int16)

series_uid = generate_uid()
study_uid  = generate_uid()
frame_uid  = generate_uid()

for z_idx in range(NZ):
    ds = Dataset()
    ds.file_meta = Dataset()
    ds.file_meta.MediaStorageSOPClassUID    = "1.2.840.10008.5.1.4.1.1.2"
    ds.file_meta.MediaStorageSOPInstanceUID = generate_uid()
    ds.file_meta.TransferSyntaxUID          = ExplicitVRLittleEndian

    ds.is_implicit_VR = False
    ds.is_little_endian = True

    ds.SOPClassUID              = "1.2.840.10008.5.1.4.1.1.2"
    ds.SOPInstanceUID           = ds.file_meta.MediaStorageSOPInstanceUID
    ds.StudyDate                = datetime.date.today().strftime("%Y%m%d")
    ds.SeriesDate               = ds.StudyDate
    ds.Modality                 = "MG"
    ds.Manufacturer             = "SyntheticGen"
    ds.StudyInstanceUID         = study_uid
    ds.SeriesInstanceUID        = series_uid
    ds.FrameOfReferenceUID      = frame_uid
    ds.InstanceNumber           = z_idx + 1
    ds.ImagePositionPatient     = [0.0, 0.0, float(z_idx)]
    ds.ImageOrientationPatient  = [1, 0, 0, 0, 1, 0]
    ds.SliceThickness           = "1.0"
    ds.PixelSpacing             = [1.0, 1.0]
    ds.Rows                     = NY
    ds.Columns                  = NX
    ds.BitsAllocated            = 16
    ds.BitsStored               = 16
    ds.HighBit                  = 15
    ds.PixelRepresentation      = 1  # signed
    ds.RescaleIntercept         = "0"
    ds.RescaleSlope             = "1"
    ds.SamplesPerPixel          = 1
    ds.PhotometricInterpretation= "MONOCHROME2"
    ds.PixelData                = volume[z_idx].tobytes()

    fname = os.path.join(OUT_DIR, f"slice_{z_idx:04d}.dcm")
    pydicom.dcmwrite(fname, ds, write_like_original=False)

print(f"Generated {NZ} DICOM slices ({NX}x{NY}x{NZ}) in '{OUT_DIR}'")
