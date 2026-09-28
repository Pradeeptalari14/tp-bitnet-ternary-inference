#!/usr/bin/env bash
set -eo pipefail

echo "=========================================================="
echo "⚡ Validating BitNet 1.58-bit Ternary Inference Engine"
echo "=========================================================="

python3 -m py_compile bitlinear_kernel.py
echo "✓ bitlinear_kernel.py syntax validated."

python3 -m py_compile manim_flow.py
echo "✓ manim_flow.py syntax validated."

python3 -c "
from bitlinear_kernel import BitLinearTernaryLayer
layer = BitLinearTernaryLayer(32, 16)
out = layer.forward([1] * 32)
assert len(out) == 16, 'Output dimension mismatch'
metrics = layer.get_finops_metrics()
assert metrics['required_gpus'] == 0, 'Should be 0 GPUs'
print('✓ Unit verification passed: BitLinear integer addition clean.')
"

echo "=========================================================="
echo "✅ All SRE validation checks passed successfully!"
echo "=========================================================="
