"""Can the ternary painter paint in 1–2 steps with a few-step LoRA trained on the original FLUX.2 klein 4B?

radames/FLUX.2-klein-Sana-Sprint is a rank-256 LoRA (alpha 256, so scale 1) distilled from klein 4B for 1–2 steps.
Bonsai Image 4B is klein with ternary weights, so the LoRA may or may not carry over. Here it is merged into the
ternary DiT's weights (W + B A, computed in f32); in the browser it would run as a side branch on the ternary GEMM.

Paints cached prompts (stock encoder) the same seed several ways and puts them on one sheet:
    base 4 steps (today) | base 2 steps (control) | LoRA 2 steps | LoRA 1 step | LoRA 2 steps, 64 text rows

    python scripts/lora_steps.py [encoder] [n_prompts]
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
from safetensors import safe_open

os.chdir(os.path.join(os.path.dirname(__file__), '..'))
encoder = sys.argv[1] if len(sys.argv) > 1 else 'qwen'
n_prompts = int(sys.argv[2]) if len(sys.argv) > 2 else 8
SEED, SIZE = 7, 512
OUT = f'renders/lora_steps/{encoder}'
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
LORA = glob.glob('/Users/neo/.cache/huggingface/hub/models--radames--FLUX.2-klein-Sana-Sprint/snapshots/*/'
                 'pytorch_lora_weights.safetensors')[0]


def paint(emb, steps):
    t0 = time.time()
    with torch.no_grad():
        img = pipe(prompt_embeds=emb.to('mps'), height=SIZE, width=SIZE, num_inference_steps=steps,
                   generator=torch.Generator('cpu').manual_seed(SEED)).images[0]
    torch.mps.synchronize()
    return img, time.time() - t0


def merge_lora():
    with safe_open(LORA, 'pt') as s:
        mods = sorted({k.rsplit('.lora_', 1)[0] for k in s.keys()})
        for m in mods:
            lin = pipe.transformer.get_submodule(m.removeprefix('transformer.'))
            A = s.get_tensor(f'{m}.lora_A.weight').float()
            B = s.get_tensor(f'{m}.lora_B.weight').float()
            W = lin.weight.data
            lin.weight.data = (W.float().cpu() + B @ A).to(W.dtype).to(W.device)
    return len(mods)


CONFIGS = [('base-4', False, 4, 512), ('base-2', False, 2, 512), ('lora-2', True, 2, 512), ('lora-1', True, 1, 512),
           ('lora-2-text64', True, 2, 64)]
times = {c[0]: [] for c in CONFIGS}
paint(cache[0]['embeds'], 1)  # warm up
merged = False
for label, lora, steps, rows in CONFIGS:
    if lora and not merged:
        print('merged', merge_lora(), 'LoRA modules', flush=True)
        merged = True
        paint(cache[0]['embeds'], 1)  # warm up again
    for i, d in enumerate(cache):
        img, dt = paint(d['embeds'][:, :rows], steps)
        img.save(f'{OUT}/p{i}_{label}.png')
        times[label].append(dt)
        print(label, i, f'{dt:.1f}s', flush=True)

labels = [c[0] for c in CONFIGS]
sheet = Image.new('RGB', (len(labels) * 256, len(cache) * 256))
for i in range(len(cache)):
    for j, label in enumerate(labels):
        sheet.paste(Image.open(f'{OUT}/p{i}_{label}.png').resize((256, 256)), (j * 256, i * 256))
sheet.save(f'{OUT}/sheet.jpg', quality=88)
summary = {k: round(float(np.mean(v)), 2) for k, v in times.items()}
json.dump(dict(columns=labels, prompts=[d['prompt'] for d in cache], mean_seconds=summary), open(f'{OUT}/results.json', 'w'),
          indent=1)
print('columns:', labels)
print('mean seconds:', summary)
