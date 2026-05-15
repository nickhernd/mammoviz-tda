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

namespace mmviz::io {

using PixelType    = float;
using ImageType3D  = itk::Image<PixelType, 3>;
using ImageType4D  = itk::Image<PixelType, 4>;

DicomLoader::DicomLoader(const LoadOptions& opts) : m_opts(opts) {}

static std::unique_ptr<VolumeData> itkImageToVolume(
    typename ImageType3D::Pointer img,
    const DicomLoader::LoadOptions& opts)
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

    return itkImageToVolume(reader->GetOutput(), m_opts);
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

    return itkImageToVolume(reader->GetOutput(), m_opts);
}

} // namespace mmviz::io
