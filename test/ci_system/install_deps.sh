#!/bin/bash
set -e

# ============================================================
# Platform dispatcher
#
# AMD/ROCm runners (e.g. linux-mi355-*) share the same install entry
# point in CI yaml configs, but need a different toolchain. Hand off
# to the ROCm-specific script when running on an AMD runner.
# ============================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AMD_RUNNER_LABEL_PATTERNS=(linux-mi350* linux-mi355*)

for pat in "${AMD_RUNNER_LABEL_PATTERNS[@]}"; do
    if [[ "${CI_RUNNER_LABEL:-}" == ${pat} ]]; then
        echo "Detected AMD runner '${CI_RUNNER_LABEL}', delegating to install_deps_rocm.sh"
        exec bash "${SCRIPT_DIR}/install_deps_rocm.sh" "$@"
    fi
done

# ============================================================
# Configuration
# ============================================================
CUDA_VERSION=${CUDA_VERSION:-13.0.1}
SM=${SM:-sm100}
BUILD_AND_DOWNLOAD_PARALLEL=${BUILD_AND_DOWNLOAD_PARALLEL:-16}

export MAX_JOBS=${BUILD_AND_DOWNLOAD_PARALLEL}
export CPLUS_INCLUDE_PATH="/usr/local/cuda/include/cccl"
export C_INCLUDE_PATH="/usr/local/cuda/include/cccl"

WORKSPACE=${WORKSPACE:-$(pwd)}

# Wrap pip install in a retry loop. PyPI's CDN occasionally returns a
# bad Content-Type for /simple/<pkg>/ pages (most recently observed for
# starlette on 2026-04-30); pip silently skips those pages, fails to
# find any version, and the resolver gives up. pip's own --retries flag
# does not retry past that warning, so we wrap the whole invocation.
pip_install_with_retry() {
    local max_attempts=5
    local attempt=1
    local delay=10
    while [ "${attempt}" -le "${max_attempts}" ]; do
        if "$@"; then
            return 0
        fi
        if [ "${attempt}" -eq "${max_attempts}" ]; then
            echo "pip install failed after ${max_attempts} attempts: $*" >&2
            return 1
        fi
        echo "pip install attempt ${attempt}/${max_attempts} failed; retrying in ${delay}s..." >&2
        sleep "${delay}"
        attempt=$((attempt + 1))
        delay=$((delay * 2))
    done
}

echo "=========================================="
echo "SM=${SM}, CUDA_VERSION=${CUDA_VERSION}"
echo "WORKSPACE=${WORKSPACE}"
echo "=========================================="

# ============================================================
# Step 1: Determine CUDA index and FlashInfer architecture
# ============================================================
echo "=== Step 1: Determine CUDA index and architecture ==="
case "${CUDA_VERSION}" in
    12.9.1) CUINDEX=129 ;;
    13.0.1) CUINDEX=130 ;;
    *)      CUINDEX=130 ;;
esac
echo "PyTorch CUDA index: cu${CUINDEX}"

case "${SM}" in
    sm103) FI_ARCH="10.3a" ;;
    sm100) FI_ARCH="10.0a" ;;
    sm90)  FI_ARCH="9.0a" ;;
    *)     echo "Unknown SM: ${SM}" && exit 1 ;;
esac
echo "FlashInfer architecture: ${FI_ARCH}"

# ============================================================
# Step 2: Upgrade base tools
# ============================================================
sudo apt install -y openmpi-bin libopenmpi-dev libssl-dev pkg-config -y
echo "=== Step 2: Upgrade pip/setuptools/wheel ==="
python3 -m pip install --upgrade pip setuptools wheel

# ============================================================
# Step 3: Build and install tokenspeed-kernel split wheels
# ============================================================
echo "=== Step 3: Build and install tokenspeed-kernel split wheels ==="
cd "${WORKSPACE}"
export PIP_EXTRA_INDEX_URL="https://download.pytorch.org/whl/cu${CUINDEX}"
pip_install_with_retry pip3 install build

KERNEL_WHEEL_DIR="${WORKSPACE}/.ci-tokenspeed-kernel-wheels"
rm -rf "${KERNEL_WHEEL_DIR}"
mkdir -p "${KERNEL_WHEEL_DIR}"

build_tokenspeed_kernel_wheel() {
    local package="$1"
    shift

    echo "Building tokenspeed-kernel package: ${package}"
    rm -rf "${WORKSPACE}/tokenspeed-kernel/python/build" "${WORKSPACE}"/tokenspeed-kernel/python/*.egg-info
    env TOKENSPEED_KERNEL_PACKAGE="${package}" "$@" \
        python3 -m build --wheel --no-isolation \
        --outdir "${KERNEL_WHEEL_DIR}" \
        "${WORKSPACE}/tokenspeed-kernel/python"
}

install_tokenspeed_kernel_wheel() {
    local distribution="$1"
    local wheel_prefix="${distribution//-/_}"
    local wheels=()
    mapfile -t wheels < <(find "${KERNEL_WHEEL_DIR}" -maxdepth 1 -type f -name "${wheel_prefix}-*.whl" | sort)
    if [ "${#wheels[@]}" -ne 1 ]; then
        echo "Expected exactly one ${distribution} wheel in ${KERNEL_WHEEL_DIR}, found ${#wheels[@]}" >&2
        printf "%s\n" "${wheels[@]}" >&2
        return 1
    fi

    echo "Installing ${distribution} wheel: ${wheels[0]}"
    pip_install_with_retry pip3 install --force-reinstall "${wheels[0]}" -v
}

pip3 uninstall -y tokenspeed-kernel tokenspeed-kernel-nvidia tokenspeed-kernel-amd

build_tokenspeed_kernel_wheel core TOKENSPEED_KERNEL_SKIP_NATIVE_BUILD=1
install_tokenspeed_kernel_wheel tokenspeed-kernel

build_tokenspeed_kernel_wheel nvidia \
    TOKENSPEED_KERNEL_BACKEND=cuda \
    FLASHINFER_CUDA_ARCH_LIST="${FI_ARCH}"
install_tokenspeed_kernel_wheel tokenspeed-kernel-nvidia

# ============================================================
# Step 4: Install TokenSpeed Scheduler (C++)
# ============================================================
echo "=== Step 4: Install TokenSpeed Scheduler ==="
pip_install_with_retry pip3 install cmake ninja
pip_install_with_retry pip3 install tokenspeed-scheduler/

# ============================================================
# Step 5: Install TokenSpeed
# ============================================================
echo "=== Step 5: Install TokenSpeed ==="
# tokenspeed-smg / -grpc-servicer / -grpc-proto are pinned in
# python/pyproject.toml; pip resolves them from PyPI as part of the
# editable install below.
pip_install_with_retry pip3 install -e "./python" \
    --extra-index-url https://download.pytorch.org/whl/cu${CUINDEX}

# ============================================================
# Step 6: Optionally override tokenspeed-mla with in-tree source
# ============================================================
# Set by `.github/workflows/pr-test.yml` when the diff touches
# `tokenspeed-mla/`. Without this override CI exercises whichever
# `tokenspeed-mla` version is pinned in
# `tokenspeed-kernel/python/requirements/cuda-thirdparty.txt` and the
# in-tree change is silently ignored.
if [ "${INSTALL_TOKENSPEED_MLA_FROM_SOURCE:-0}" = "1" ]; then
    echo "=== Step 6: Reinstall tokenspeed-mla from in-tree source ==="
    pip_install_with_retry pip3 install --break-system-packages \
        --force-reinstall --no-deps "${WORKSPACE}/tokenspeed-mla"
fi

# ============================================================
# Step 7: Fix Triton ptxas (CUDA 13+ only)
# ============================================================
echo "=== Step 7: Fix Triton ptxas ==="
if [ "${CUDA_VERSION%%.*}" = "13" ]; then
    TRITON_BIN="/usr/local/lib/python3.12/dist-packages/triton/backends/nvidia/bin"
    if [ -d "${TRITON_BIN}" ]; then
        rm -f "${TRITON_BIN}/ptxas" 2>/dev/null || sudo rm -f "${TRITON_BIN}/ptxas" 2>/dev/null || true
        ln -sf /usr/local/cuda/bin/ptxas "${TRITON_BIN}/ptxas" 2>/dev/null || sudo ln -sf /usr/local/cuda/bin/ptxas "${TRITON_BIN}/ptxas" 2>/dev/null || true
    fi
fi

echo ""
echo "=========================================="
echo "Installed successfully! CUDA_VERSION=${CUDA_VERSION}, SM=${SM}"
echo "=========================================="
