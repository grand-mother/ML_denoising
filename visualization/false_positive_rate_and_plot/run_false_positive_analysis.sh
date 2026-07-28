#!/bin/bash
#
# run_false_positive_analysis.sh
#
# Shell script to run the false positive rate analysis pipeline.
# This script automates the 3-step workflow described in false_positive_rate_and_plot.py
#
# Usage:
#   ./run_false_positive_analysis.sh [step1|step3|all]
#
# Requirements:
#   - Python environment with numpy, scipy, matplotlib
#   - Input files: clean_waveforms.npy, noisy_waveforms.npy
#   - For step3: denoised_signal.npy, denoised_noise_only.npy
#

set -e  # Exit on error

# ============================================
# Configuration - Modify these paths as needed
# ============================================
DATA_DIR="/sps/grand/blevy/sims/sims_for_denoising_sept2025"
CLEAN_FILE="${DATA_DIR}/noiseless_traces.npy"
NOISY_FILE="${DATA_DIR}/noise_microV_lst0.npy"  # Change lst0 to lst1-lst23 as needed
NOISE_ONLY_OUT="noise_only.npy"
NOISE_META_OUT="noise_only_meta.npz"

# Denoiser output files (for step 3)
DENOISED_SIGNAL="denoised_signal.npy"
DENOISED_NOISE_ONLY="denoised_noise_only.npy"

# Output figure
SAVEFIG="fig_false_alarm_panel.pdf"

# Script location
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT="${SCRIPT_DIR}/false_positive_rate_and_plot_v3.py"

# ============================================
# Functions
# ============================================

step1() {
    echo "============================================"
    echo "Step 1: Generate noise_only.npy"
    echo "============================================"
    python "${SCRIPT}" \
        --clean "${CLEAN_FILE}" \
        --noisy "${NOISY_FILE}" \
        --noise-only-out "${NOISE_ONLY_OUT}" \
        --noise-meta-out "${NOISE_META_OUT}" \
        --make-noise-only-only
    
    echo ""
    echo "[INFO] Step 1 completed!"
    echo "[INFO] Generated: ${NOISE_ONLY_OUT}, ${NOISE_META_OUT}"
    echo ""
    echo "Next steps:"
    echo "  1. Run your denoiser inference on ${NOISY_FILE} -> ${DENOISED_SIGNAL}"
    echo "  2. Run your denoiser inference on ${NOISE_ONLY_OUT} -> ${DENOISED_NOISE_ONLY}"
    echo "  3. Then run: $0 step3"
}

step3() {
    echo "============================================"
    echo "Step 3: Compute FPR/ROC and generate figure"
    echo "============================================"
    
    # Check if required files exist
    if [[ ! -f "${DENOISED_SIGNAL}" ]]; then
        echo "[ERROR] Missing ${DENOISED_SIGNAL}. Run your denoiser first."
        exit 1
    fi
    if [[ ! -f "${DENOISED_NOISE_ONLY}" ]]; then
        echo "[ERROR] Missing ${DENOISED_NOISE_ONLY}. Run your denoiser on ${NOISE_ONLY_OUT} first."
        exit 1
    fi
    
    python "${SCRIPT}" \
        --clean "${CLEAN_FILE}" \
        --noisy "${NOISY_FILE}" \
        --sig-ml "${DENOISED_SIGNAL}" \
        --noise-ml "${DENOISED_NOISE_ONLY}" \
        --noise-only-out "${NOISE_ONLY_OUT}" \
        --noise-meta-out "${NOISE_META_OUT}" \
        --savefig "${SAVEFIG}"
    
    echo ""
    echo "[INFO] Step 3 completed!"
    echo "[INFO] Figure saved to: ${SAVEFIG}"
}

usage() {
    echo "Usage: $0 [step1|step3|all]"
    echo ""
    echo "Steps:"
    echo "  step1  - Generate noise_only.npy from clean/noisy waveforms"
    echo "  step3  - Compute FPR/ROC and generate the figure (requires denoised files)"
    echo "  all    - Run step1, then pause for manual denoising, then step3"
    echo ""
    echo "Typical workflow:"
    echo "  1. Run: $0 step1"
    echo "  2. Run your denoiser on noisy_waveforms.npy -> denoised_signal.npy"
    echo "  3. Run your denoiser on noise_only.npy -> denoised_noise_only.npy"
    echo "  4. Run: $0 step3"
}

# ============================================
# Main
# ============================================

case "${1:-}" in
    step1)
        step1
        ;;
    step3)
        step3
        ;;
    all)
        step1
        echo ""
        echo "============================================"
        echo "Step 2: Manual denoising required"
        echo "============================================"
        echo "Please run your denoiser inference to produce:"
        echo "  - ${DENOISED_SIGNAL} from ${NOISY_FILE}"
        echo "  - ${DENOISED_NOISE_ONLY} from ${NOISE_ONLY_OUT}"
        echo ""
        read -p "Press Enter when ready to continue with Step 3..."
        step3
        ;;
    -h|--help)
        usage
        ;;
    *)
        usage
        exit 1
        ;;
esac

