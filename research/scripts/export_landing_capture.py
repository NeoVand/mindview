"""capture_fast.py's run, in the form the landing prototype reads (static/_dev/landing/<name>/, not deployed):
    meta.json      prompt, tokens, sizes, sigmas
    reader.json    the reader's attention per layer (1-9), mean over heads, between the prompt's tokens [9][n][n]
    maps_s0.bin    step 0 (16 x 16): per block, per token, the image rows' attention to it, f16 [25][n][256]
    maps_s1.bin    step 1 (32 x 32): the same, f16 [25][n][1024]
    state_s0.png   the noise it starts from; sketch.png the first step's guess; state_s1.png that guess upsampled and
                   noised back; final.png the picture

    python scripts/export_landing_capture.py [name]
"""
import json
import os
import shutil
import sys

import numpy as np

os.chdir(os.path.join(os.path.dirname(__file__), '..'))
NAME = sys.argv[1] if len(sys.argv) > 1 else 'bonsai'
SRC = f'renders/landing_capture/{NAME}'
DST = f'../static/_dev/landing/{NAME}'
os.makedirs(DST, exist_ok=True)
meta = json.load(open(f'{SRC}/meta.json'))
n = meta['n_real']
A = np.load(f'{SRC}/reader_attn.npy').astype(np.float32).mean(1)  # [9, n, n]
json.dump([[[round(float(x), 4) for x in row] for row in layer] for layer in A], open(f'{DST}/reader.json', 'w'))
for s in (0, 1):
    M = np.load(f'{SRC}/attn_s{s}.npy').astype(np.float32)  # [25, ni, n]
    np.ascontiguousarray(M.transpose(0, 2, 1)).astype('<f2').tofile(f'{DST}/maps_s{s}.bin')  # [25, n, ni]
for f in ('state_s0.png', 'sketch.png', 'state_s1.png', 'final.png'):
    shutil.copy(f'{SRC}/{f}', f'{DST}/{f}')
json.dump(meta, open(f'{DST}/meta.json', 'w'), indent=1)
print('wrote', DST)
