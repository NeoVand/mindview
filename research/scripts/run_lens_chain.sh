#!/bin/zsh
# tuned lens (16 prompts) -> comparison sheets -> re-export trace with tuned lens -> decoder stages -> copy to app
cd /Users/neo/repos/mindview/research
set -e
LENS_PROMPTS=16 .venv/bin/python -u scripts/tuned_lens.py fit > data/tuned_lens_fit.log 2>&1
.venv/bin/python -u scripts/tuned_lens.py render "a bonsai tree made of glowing circuitry in a dark museum, volumetric light" >> data/tuned_lens_fit.log 2>&1
echo "SHEET READY"
.venv/bin/python -u scripts/export_trace.py "a bonsai tree made of glowing circuitry in a dark museum, volumetric light" --encoder bonsai17t_adapt --name bonsai-museum-ternary > data/export_trace_2.log 2>&1
.venv/bin/python -u scripts/decoder_stages.py fit > data/decoder_stages.log 2>&1
.venv/bin/python -u scripts/decoder_stages.py render traces/bonsai-museum-ternary >> data/decoder_stages.log 2>&1
rm -rf ../static/traces/bonsai-museum-ternary && cp -R traces/bonsai-museum-ternary ../static/traces/ >/dev/null
echo "TRACE READY"
