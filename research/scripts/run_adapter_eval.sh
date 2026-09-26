#!/bin/zsh
# After run_adapter_pipeline.sh: refit each adapter at its best lambda, render the 4 test prompts, compose a grid.
cd /Users/neo/repos/mindview/research
PIPE_OUT=$1
until grep -a -q -E "bonsai17 lambda=0.1|failed" $PIPE_OUT 2>/dev/null; do sleep 15; done
grep -a -q failed $PIPE_OUT && { echo "pipeline failed"; cat $PIPE_OUT; exit 1; }
P1="a bonsai tree made of glowing circuitry in a dark museum, volumetric light"; P2="a portrait of an old fisherman, weathered skin, soft window light"; P3="an ocean wave frozen in glass at sunrise"; P4="a red fox sleeping in fresh snow"
for src in bonsai4b bonsai17; do
  best=$(grep -a "^$src lambda" $PIPE_OUT | sed -E 's/.*lambda=([0-9.e-]+).*cosine[^=]*= ([0-9.-]+).*/\2 \1/' | sort -rn | head -1 | awk '{print $2}')
  echo "$src best lambda $best"
  .venv/bin/python scripts/adapter.py fit $src $best 2>&1 | grep -a lambda
  .venv/bin/python scripts/adapter.py apply $src "$P1" "$P2" "$P3" "$P4" 2>&1 | grep -a wrote
  ENCODER=${src}_adapt .venv/bin/python scripts/image_speed.py ternary 512 "$P1" "$P2" "$P3" "$P4" 2>&1 | grep -a "steps:"
done
cd renders/image/compare && ../../../.venv/bin/python -c "
from PIL import Image
rows=['qwen','bonsai4b','bonsai4b_adapt','bonsai17_adapt']
ims=[[Image.open(f'ternary_{e}_512_s4_seed7_p{i}.png').resize((320,320)) for i in range(4)] for e in rows]
o=Image.new('RGB',(1280,320*len(rows))); [o.paste(im,(c*320,r*320)) for r,row in enumerate(ims) for c,im in enumerate(row)]; o.save('grid_adapters.png'); print('grid written')"
