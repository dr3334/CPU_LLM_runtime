#!/usr/bin/env bash
# Desktop (x86_64 Linux / WSL) build.
#
#   ./scripts/build_desktop.sh                 # RelWithDebInfo
#   BUILD_TYPE=Debug ./scripts/build_desktop.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-${ROOT}/build}"
BUILD_TYPE="${BUILD_TYPE:-RelWithDebInfo}"

cmake -S "${ROOT}" -B "${BUILD_DIR}" -G Ninja \
  -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
  -DLLMRT_BUILD_TESTS=ON

cmake --build "${BUILD_DIR}" -j "$(nproc)"

echo
echo "built: ${BUILD_DIR}/llmrt"
echo "tests: ctest --test-dir ${BUILD_DIR} --output-on-failure"
