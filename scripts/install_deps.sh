#!/usr/bin/env bash
# Installs system-level dependencies for mammoviz-tda on Arch Linux
# Run once before building: bash scripts/install_deps.sh

set -euo pipefail

echo "==> Updating system packages..."
sudo pacman -Syu --noconfirm

echo "==> Installing core build tools..."
sudo pacman -S --noconfirm \
    cmake ninja gcc \
    git pkg-config

echo "==> Installing math/linear algebra libraries..."
sudo pacman -S --noconfirm \
    eigen \
    cgal

echo "==> Installing medical imaging (ITK)..."
if ! command -v yay &>/dev/null; then
    echo "  WARNING: yay not found. Install ITK manually from https://itk.org"
else
    yay -S --noconfirm itk
fi

echo "==> Installing GUDHI (TDA library)..."
pip install gudhi --user 2>/dev/null || \
    echo "  NOTE: Install GUDHI headers from https://gudhi.inria.fr"

echo "==> Installing rendering dependencies..."
sudo pacman -S --noconfirm \
    glfw-x11 \
    glew \
    vulkan-devel \
    shaderc

echo "==> Installing ONNX Runtime..."
ONNX_VER="1.17.0"
ONNX_URL="https://github.com/microsoft/onnxruntime/releases/download/v${ONNX_VER}/onnxruntime-linux-x64-${ONNX_VER}.tgz"
mkdir -p third_party/onnxruntime
curl -L "${ONNX_URL}" | tar xz -C third_party/onnxruntime --strip-components=1
echo "  ONNX Runtime installed to third_party/onnxruntime"

echo "==> Cloning Dear ImGui..."
if [ ! -d third_party/imgui ]; then
    git clone --depth=1 https://github.com/ocornut/imgui.git third_party/imgui
fi

echo ""
echo "==> Done. Now build with:"
echo "    cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release"
echo "    cmake --build build -j\$(nproc)"
