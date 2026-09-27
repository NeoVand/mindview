"""Does the painter need its 512 text rows? The prompt is padded to 512 tokens, and the DiT runs over all of them
(1,536 rows a step with the 1,024 image patches). This paints cached prompts with the text stream cut to fewer rows
and compares each picture with the full 512.

The stock encoder is causal, so its first k rows are the same whether the prompt is padded to 512 or to k: cutting
the cached embeddings is what encoding with max_sequence_length = k would give.

    python scripts/text_rows.py [encoder] [n_prompts]     # encoder as in data/prompt_cache (default qwen)
"""
import glob
import json
import os
import sys
import time

import numpy as np
import torch
from diffusers import Flux2KleinPipeline
from PIL import Image

os.chdir(os.path.join(os.path.dirname(__file__), '..'))
encoder = sys.argv[1] if len(sys.argv) > 1 else 'qwen'
n_prompts = int(sys.argv[2]) if len(sys.argv) > 2 else 8
SEED, SIZE, STEPS = 7, 512, 4
OUT = f'renders/text_rows/{encoder}'
os.makedirs(OUT, exist_ok=True)

cache = []
for f in sorted(glob.glob('data/prompt_cache/*.pt')):
    d = torch.load(f, map_location='cpu')
    if d['encoder'] == encoder:
        cache.append(d)
cache = cache[:n_prompts]

root = glob.glob('/Users/neo/.cache/huggingface/hub/models--prism-ml--bonsai-image-ternary-4B-unpacked/snapshots/*/')[0]
pipe = Flux2KleinPipeline.from_pretrained(root, text_encoder=None, torch_dtype=torch.bfloat16).to('mps')
pipe.set_progress_bar_config(disable=True)


def paint(emb):
    t0 = time.time()
    with torch.no_grad():
        img = pipe(prompt_embeds=emb.to('mps'), height=SIZE, width=SIZE, num_inference_steps=STEPS,
                   generator=torch.Generator('cpu').manual_seed(SEED)).images[0]
    torch.mps.synchronize()
    return np.asarray(img).astype(np.float64), time.time() - t0


def psnr(a, b):
    mse = ((a - b) ** 2).mean()
    return float('inf') if mse == 0 else 10 * np.log10(255 ** 2 / mse)


paint(cache[0]['embeds'][:, :64])  # warm up
results = []
for i, d in enumerate(cache):
    emb, n = d['embeds'], d['n_real']
    lengths = {'512': 512, '256': 256, '128': 128, '64': 64, 'real': n, 'real+8': n + 8}
    row = dict(prompt=d['prompt'], n_real=n)
    ref = None
    for label, k in lengths.items():
        img, dt = paint(emb[:, :k])
        Image.fromarray(img.astype(np.uint8)).save(f'{OUT}/p{i}_{label}.png')
        if label == '512':
            ref = img
        row[label] = dict(rows=k, seconds=round(dt, 2), psnr_vs_512=round(psnr(img, ref), 2))
        print(i, label, k, f'{dt:.1f}s', f'{psnr(img, ref):.1f} dB', flush=True)
    results.append(row)
json.dump(results, open(f'{OUT}/results.json', 'w'), indent=1)

# one sheet: a row per prompt, a column per length
labels = list(lengths)
sheet = Image.new('RGB', (len(labels) * 256, len(cache) * 256))
for i in range(len(cache)):
    for j, label in enumerate(labels):
        sheet.paste(Image.open(f'{OUT}/p{i}_{label}.png').resize((256, 256)), (j * 256, i * 256))
sheet.save(f'{OUT}/sheet.jpg', quality=88)
print('columns:', labels)
for label in labels:
    print(label, 'mean PSNR vs 512:', round(np.mean([r[label]['psnr_vs_512'] for r in results if label != '512']), 2)
          if label != '512' else '', 'mean seconds:', round(np.mean([r[label]['seconds'] for r in results]), 2))
