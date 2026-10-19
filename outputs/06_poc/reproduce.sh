#!/bin/bash
set -e
echo "=== Phase 0: Tests ==="
python -m pytest tests/ -v

echo "=== Phase 0: Reproduce anchors ==="
python scripts/reproduce_anchors.py

echo "=== Phase 1: Prior-correction controls ==="
python scripts/phase1_prior_controls.py

echo "=== Phase 2: Appearance alignment ==="
python scripts/phase2_appearance.py

echo "=== Phase 3: Prior-matched alignment ==="
python scripts/phase3_prior_matched.py

echo "=== Phase 4: Deliverables ==="
python scripts/phase4_deliverables.py

echo "=== Done ==="
echo "Results in outputs/06_poc/"
