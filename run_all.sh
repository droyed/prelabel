#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

run() {
    local label="$1"
    shift
    echo ""
    echo "=========================================="
    echo " $label"
    echo "=========================================="
    python "$@"
    echo ""
    echo "--- $label: DONE ---"
}

run "Demo: basic_usage with YOLO"          demos/demo_det.py yolo
run "Demo: basic_usage with SAM3"          demos/demo_det.py sam3
run "Demo: demo_LabelStudioClient"         demos/demo_LabelStudioClient.py
run "Tests: test_LabelStudioClient"        tests/test_LabelStudioClient.py

echo ""
echo "=========================================="
echo " All scripts completed successfully."
echo "=========================================="
