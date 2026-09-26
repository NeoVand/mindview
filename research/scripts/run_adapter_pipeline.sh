#!/bin/zsh
# Wait for Qwen targets, then collect ternary-LLM stats, fit adapters, and render test images.
cd /Users/neo/repos/mindview/research
until grep -a -q -E "done [0-9]+ rows|Traceback" data/adapter_targets.log; do sleep 10; done
grep -a -q Traceback data/adapter_targets.log && { echo "targets failed"; exit 1; }
P1="a bonsai tree made of glowing circuitry in a dark museum, volumetric light"; P2="a portrait of an old fisherman, weathered skin, soft window light"; P3="an ocean wave frozen in glass at sunrise"; P4="a red fox sleeping in fresh snow"
for src in bonsai4b bonsai17; do
  .venv/bin/python -u scripts/adapter.py stats $src > data/adapter_stats_$src.log 2>&1 || { echo "stats $src failed"; tail -5 data/adapter_stats_$src.log; exit 1; }
  for lam in 0.001 0.01 0.1; do .venv/bin/python scripts/adapter.py fit $src $lam 2>&1 | grep -a lambda; done
done
