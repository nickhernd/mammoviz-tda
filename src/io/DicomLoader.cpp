#include "io/DicomLoader.h"
#include "utils/Logger.h"

// ITK headers
#include <itkImage.h>
#include <itkImageSeriesReader.h>
#include <itkGDCMImageIO.h>
#include <itkGDCMSeriesFileNames.h>
#include <itkMetaImageIO.h>
#include <itkImageFileReader.h>
#include <itkRescaleIntensityImageFilter.h>
#include <itkCastImageFilter.h>

#include <algorithm>
#include <filesystem>
#include <cmath>

namespace mmviz::io {

using PixelType    = float;
using ImageType3D  = itk::Image<PixelType, 3>;
using ImageType4D  = itk::Image<PixelType, 4>;

DicomLoader::DicomLoader(const DicomLoadOptions& opts) : m_opts(opts) {}

static std::unique_ptr<VolumeData> itkImageToVolume(
    typename ImageType3D::Pointer img,
    const DicomLoadOptions& opts)
{
    auto region  = img->GetLargestPossibleRegion();
    auto size    = region.GetSize();
    auto spacing = img->GetSpacing();

    auto vol      = std::make_unique<VolumeData>();
    vol->shape    = { (int)size[0], (int)size[1], (int)size[2], 1 };
    vol->spacing  = { (float)spacing[0], (float)spacing[1], (float)spacing[2] };
    vol->data.resize(size[0] * size[1] * size[2]);

    float* dst = vol->data.data();
    itk::ImageRegionConstIterator<ImageType3D> it(img, region);

    if (opts.normalize_intensity) {
        // First pass: find min/max
        float vmin =  1e9f, vmax = -1e9f;
        for (it.GoToBegin(); !it.IsAtEnd(); ++it) {
            float v = it.Get();
            vmin = std::min(vmin, v);
            vmax = std::max(vmax, v);
        }
        float range = (vmax - vmin) > 0 ? (vmax - vmin) : 1.0f;
        for (it.GoToBegin(); !it.IsAtEnd(); ++it)
            *dst++ = (it.Get() - vmin) / range;
    } else {
        for (it.GoToBegin(); !it.IsAtEnd(); ++it)
            *dst++ = it.Get();
    }

    return vol;
}

std::unique_ptr<VolumeData> DicomLoader::load(const std::string& dicom_dir) const {
    LOG_INFO("Reading DICOM series from: {}", dicom_dir);

    auto nameGenerator = itk::GDCMSeriesFileNames::New();
    nameGenerator->SetUseSeriesDetails(true);
    nameGenerator->SetDirectory(dicom_dir);

    auto seriesUID = nameGenerator->GetSeriesUIDs();
    if (seriesUID.empty()) {
        LOG_ERROR("No DICOM series found in: {}", dicom_dir);
        return nullptr;
    }

    // Load the first series (or largest if multiple)
    const std::string& uid = seriesUID[0];
    LOG_INFO("Loading series UID: {}", uid);

    auto fileNames = nameGenerator->GetFileNames(uid);
    LOG_INFO("Found {} DICOM slices", fileNames.size());

    auto dicomIO = itk::GDCMImageIO::New();
    auto reader  = itk::ImageSeriesReader<ImageType3D>::New();
    reader->SetImageIO(dicomIO);
    reader->SetFileNames(fileNames);

    try {
        reader->Update();
    } catch (const itk::ExceptionObject& e) {
        LOG_ERROR("ITK DICOM read failed: {}", e.what());
        return nullptr;
    }

    auto vol = itkImageToVolume(reader->GetOutput(), m_opts);
    if (vol) downsample(*vol, m_opts.max_dimension);
    return vol;
}

std::unique_ptr<VolumeData> DicomLoader::loadMHA(const std::string& mha_path) const {
    LOG_INFO("Reading MHA/MHD file: {}", mha_path);

    auto reader = itk::ImageFileReader<ImageType3D>::New();
    reader->SetFileName(mha_path);

    try {
        reader->Update();
    } catch (const itk::ExceptionObject& e) {
        LOG_ERROR("ITK MHA read failed: {}", e.what());
        return nullptr;
    }

    auto vol = itkImageToVolume(reader->GetOutput(), m_opts);
    if (vol) downsample(*vol, m_opts.max_dimension);
    return vol;
}

// ── Trilinear downsampling ────────────────────────────────────────────────────
void DicomLoader::downsample(VolumeData& vol, int max_dim) {
    if (max_dim <= 0) return;
    int ox = vol.shape[0], oy = vol.shape[1], oz = vol.shape[2];
    if (ox <= max_dim && oy <= max_dim && oz <= max_dim) return;

    float scale = (float)max_dim / (float)std::max({ox, oy, oz});
    int nx = std::max(1, (int)std::round(ox * scale));
    int ny = std::max(1, (int)std::round(oy * scale));
    int nz = std::max(1, (int)std::round(oz * scale));

    LOG_INFO("Resampling volume {}x{}x{} → {}x{}x{}", ox, oy, oz, nx, ny, nz);

    std::vector<float> resampled(nx * ny * nz);
    for (int z = 0; z < nz; ++z)
    for (int y = 0; y < ny; ++y)
    for (int x = 0; x < nx; ++x) {
        float sx = (x + 0.5f) * ox / nx - 0.5f;
        float sy = (y + 0.5f) * oy / ny - 0.5f;
        float sz = (z + 0.5f) * oz / nz - 0.5f;

        int x0 = std::clamp((int)sx, 0, ox-1), x1 = std::min(x0+1, ox-1);
        int y0 = std::clamp((int)sy, 0, oy-1), y1 = std::min(y0+1, oy-1);
        int z0 = std::clamp((int)sz, 0, oz-1), z1 = std::min(z0+1, oz-1);
        float fx = sx - x0, fy = sy - y0, fz = sz - z0;
        fx = std::clamp(fx, 0.f, 1.f);
        fy = std::clamp(fy, 0.f, 1.f);
        fz = std::clamp(fz, 0.f, 1.f);

        auto v = [&](int xi, int yi, int zi) {
            return vol.data[zi * ox * oy + yi * ox + xi];
        };
        float val =
            (1-fz)*((1-fy)*((1-fx)*v(x0,y0,z0) + fx*v(x1,y0,z0)) +
                       fy *((1-fx)*v(x0,y1,z0) + fx*v(x1,y1,z0))) +
               fz *((1-fy)*((1-fx)*v(x0,y0,z1) + fx*v(x1,y0,z1)) +
                       fy *((1-fx)*v(x0,y1,z1) + fx*v(x1,y1,z1)));

        resampled[z * nx * ny + y * nx + x] = val;
    }

    vol.data     = std::move(resampled);
    vol.shape    = {nx, ny, nz, 1};
    vol.spacing  = { vol.spacing[0] * ox / nx,
                     vol.spacing[1] * oy / ny,
                     vol.spacing[2] * oz / nz };
}

} // namespace mmviz::io
