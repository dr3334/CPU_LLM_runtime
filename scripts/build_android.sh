#!/usr/bin/env bash
# Android cross-compile via the NDK's own CMake toolchain file.
#
#   ./scripts/build_android.sh
#   ABI=arm64-v8a API=24 ./scripts/build_android.sh
#
# The resulting binary is a PIE executable that can be pushed with adb:
#   adb push build-android-arm64-v8a/llmrt /data/local/tmp/
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

NDK="${ANDROID_NDK_HOME:-/home/dr/android-sdk/ndk/26.3.11579264}"
ABI="${ABI:-arm64-v8a}"
API="${API:-24}"
BUILD_DIR="${BUILD_DIR:-${ROOT}/build-android-${ABI}}"

if [[ ! -f "${NDK}/build/cmake/android.toolchain.cmake" ]]; then
  echo "error: NDK toolchain not found under ${NDK}" >&2
  echo "       set ANDROID_NDK_HOME or pass NDK=<path>" >&2
  exit 1
fi

cmake -S "${ROOT}" -B "${BUILD_DIR}" -G Ninja \
  -DCMAKE_TOOLCHAIN_FILE="${NDK}/build/cmake/android.toolchain.cmake" \
  -DANDROID_ABI="${ABI}" \
  -DANDROID_PLATFORM="android-${API}" \
  -DANDROID_STL=c++_static \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLMRT_BUILD_TESTS=ON

cmake --build "${BUILD_DIR}" -j "$(nproc)"

echo
echo "built: ${BUILD_DIR}/llmrt  (${ABI}, android-${API})"
