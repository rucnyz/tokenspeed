#!/bin/bash
set -e

# ============================================================
# ROCm/AMD MI355 install script for TokenSpeed CI.
# ============================================================
GFX_ARCH=${GFX_ARCH:-gfx950}
ROCM_VERSION=${ROCM_VERSION:-7.2}
BUILD_AND_DOWNLOAD_PARALLEL=${BUILD_AND_DOWNLOAD_PARALLEL:-16}

ROCM_INDEX="https://download.pytorch.org/whl/rocm${ROCM_VERSION}"

export MAX_JOBS=${BUILD_AND_DOWNLOAD_PARALLEL}
WORKSPACE=${WORKSPACE:-$(pwd)}

# Wrap pip install in a retry loop for transient package-index failures.
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
echo "GFX_ARCH=${GFX_ARCH}"
echo "ROCM_VERSION=${ROCM_VERSION}"
echo "WORKSPACE=${WORKSPACE}"
echo "=========================================="

echo "=== Step 1: apt deps ==="
sudo apt-get install -y openmpi-bin libopenmpi-dev libssl-dev pkg-config

echo "=== Step 2: Upgrade pip/setuptools/wheel ==="
python3 -m pip install --upgrade pip setuptools wheel

# ============================================================
# Step 3: Build and install tokenspeed-kernel split wheels
# ============================================================
echo "=== Step 3: Build and install tokenspeed-kernel split wheels ==="
cd "${WORKSPACE}"
export PIP_EXTRA_INDEX_URL="${ROCM_INDEX}"
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

build_tokenspeed_kernel_wheel amd \
    TOKENSPEED_KERNEL_BACKEND=rocm \
    TOKENSPEED_KERNEL_SKIP_NATIVE_BUILD=1
install_tokenspeed_kernel_wheel tokenspeed-kernel-amd

echo "=== Step 4: Install TokenSpeed Scheduler ==="
pip3 install cmake ninja
pip3 install tokenspeed-scheduler/

echo "=== Step 5: Install TokenSpeed ==="
# tokenspeed-smg / -grpc-servicer / -grpc-proto are pinned in
# python/pyproject.toml; pip resolves them from PyPI as part of the
# editable install below.
pip3 install -e ./python --no-build-isolation \
    --extra-index-url "${ROCM_INDEX}"

echo ""
echo "=========================================="
echo "ROCm install completed (GFX_ARCH=${GFX_ARCH})"
echo "=========================================="
