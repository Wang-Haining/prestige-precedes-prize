#!/usr/bin/env bash
# Reproduce every result, figure and supporting table from the data in data/.
# B=1000 bootstrap replicates; the full run takes many CPU hours (same_scientist.py dominates).
set -euo pipefail
cd "$(dirname "$0")"
python -m analysis.between
python -m analysis.same_scientist
python -m analysis.first_paper
python -m analysis.checks
python -m analysis.linkage_audit
python -m analysis.figures
python -m analysis.tables
