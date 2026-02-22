#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

run() {
    local label="$1"
    local script="$2"
    echo ""
    echo "=========================================="
    echo " $label"
    echo "=========================================="
    python "$script"
    echo ""
    echo "--- $label: DONE ---"
}

run "Demo: basic_usage"           "demos/basic_usage.py"
run "Demo: demo_LabelStudioClient" "demos/demo_LabelStudioClient.py"
run "Tests: test_LabelStudioClient" "tests/test_LabelStudioClient.py"

echo ""
echo "=========================================="
echo " All scripts completed successfully."
echo "=========================================="
