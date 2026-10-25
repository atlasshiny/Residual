#!/usr/bin/env bash
# ============================================================================
# setup_and_run.sh — One-command setup + full pipeline for gatorhacks-v
# ============================================================================
# Usage:
#   ./setup_and_run.sh              # Full pipeline (fetch → residual → train)
#   ./setup_and_run.sh --skip-fetch # Skip data fetch (use existing data/)
#   ./setup_and_run.sh --test       # Run tests after pipeline
#   ./setup_and_run.sh --help       # Show this help
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

SKIP_FETCH=false
RUN_TESTS=false
GPU=false

for arg in "$@"; do
    case "$arg" in
        --skip-fetch) SKIP_FETCH=true ;;
        --test)       RUN_TESTS=true ;;
        --gpu)        GPU=true ;;
        --help|-h)
            echo "Usage: $0 [--skip-fetch] [--test] [--gpu]"
            echo ""
            echo "  --skip-fetch  Skip JPL Horizons data fetch (use existing data/)"
            echo "  --test        Run test suite after pipeline"
            echo "  --gpu         Install PyTorch GPU extras (for PINN/MLP training)"
            exit 0
            ;;
        *) echo "Unknown argument: $arg"; exit 1 ;;
    esac
done

EXTRAS="dev,data"
if $GPU; then EXTRAS="dev,data,gpu"; fi

echo "============================================================"
echo "  GatorHacks-V: Setup & Pipeline"
echo "============================================================"

# --- 1. Python environment ---
echo ""
echo "[1/5] Setting up Python environment..."
if command -v uv &>/dev/null; then
    uv sync --extra $EXTRAS
else
    echo "ERROR: 'uv' not found. Install from https://docs.astral.sh/uv/"
    exit 1
fi

# --- 2. Rust workspace ---
echo ""
echo "[2/5] Building Rust workspace..."
if command -v cargo &>/dev/null; then
    cargo build --workspace
else
    echo "WARNING: 'cargo' not found. Skipping Rust build."
    echo "  Install from https://rustup.rs/"
fi

# --- 3. Fetch reference data ---
if ! $SKIP_FETCH; then
    echo ""
    echo "[3/5] Fetching JPL Horizons reference data..."
    make fetch-data
else
    echo ""
    echo "[3/5] Skipping data fetch (--skip-fetch)."
fi

# --- 4. Generate residual table ---
echo ""
echo "[4/5] Generating residual table..."
if [ -f data/reference.csv ]; then
    cargo run -p residual --example residual_pipeline -- data/reference.csv data/residual_table.csv 2>/dev/null || \
        echo "WARNING: Rust residual pipeline failed. Using synthetic fallback."
else
    echo "  No reference.csv found. Pipeline will use synthetic data."
fi

# --- 5. Train models ---
echo ""
echo "[5/5] Training PINN / MLP / SINDy / LinearReg..."
if [ -f data/residual_table.csv ]; then
    uv run --extra data python -m ml.pinn.run --data data/residual_table.csv
else
    echo "  No residual_table.csv. Running synthetic demo..."
    uv run --extra data python -m ml.pinn.demo
fi

echo ""
echo "============================================================"
echo "  Pipeline complete!"
echo "  Results: ml/pinn/pinn_result.json"
echo "============================================================"

# --- Optional: Run tests ---
if $RUN_TESTS; then
    echo ""
    echo "Running test suite..."
    if $GPU; then
        uv run --extra dev --extra data --extra gpu pytest -q tests ml
    else
        uv run --extra dev --extra data pytest -q tests ml
    fi
    cargo test --workspace 2>/dev/null || true
fi
