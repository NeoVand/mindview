"""How small can the few-step LoRA get? (It is 735 MB in bf16: rank 256 on all 100 ternary linears of the DiT.)

Paints cached prompts at 2 steps with the LoRA
    full    as released (rank 256)
    r<n>    cut to rank n by SVD (per module: the best rank-n approximation of B A), e.g. r128, r64, r32
    (CONFIGS=full,r32,r16 SUFFIX=-low picks others)
    tern    merged into the weights and made ternary again (trits re-chosen with the original group scales, then the
            scales re-fitted by least squares): if this held up, 2 steps would cost nothing at all
and prints, for 'tern', how many trits the merge changes. Each configuration reloads the ternary DiT.

    python scripts/lora_compress.py [encoder] [n_prompts]
"""
import glob
import json
import os
import sys
import time

import numpy as np
import torch
from diffusers import Flux2KleinPipeline, Flux2Transformer2DModel
from PIL import Image
from safetensors import safe_open

os.chdir(os.path.join(os.path.dirname(__file__), '..'))
encoder = sys.argv[1] if len(sys.argv) > 1 else 'qwen'
n_prompts = int(sys.argv[2]) if len(sys.argv) > 2 else 8
SEED, SIZE, STEPS = 7, 512, 2
OUT = f'renders/lora_compress/{encoder}' + os.environ.get('SUFFIX', '')
os.makedirs(OUT, exist_ok=True)

cache = []
for f in sorted(glob.glob('data/prompt_cache/*.pt')):
    d = torch.load(f, map_location='cpu')
    if d['encoder'] == encoder:
        cache.append(d)
cache = cache[:n_prompts]
root = glob.glob('/Users/neo/.cache/huggingface/hub/models--prism-ml--bonsai-image-ternary-4B-unpacked/snapshots/*/')[0]
LORA = glob.glob('/Users/neo/.cache/huggingface/hub/models--radames--FLUX.2-klein-Sana-Sprint/snapshots/*/'
                 'pytorch_lora_weights.safetensors')[0]
pipe = Flux2KleinPipeline.from_pretrained(root, text_encoder=None, torch_dtype=torch.bfloat16)
pipe.set_progress_bar_config(disable=True)


def lora_pairs():
    with safe_open(LORA, 'pt') as s:
        for m in sorted({k.rsplit('.lora_', 1)[0] for k in s.keys()}):
            yield m.removeprefix('transformer.'), s.get_tensor(f'{m}.lora_A.weight').float(), \
                s.get_tensor(f'{m}.lora_B.weight').float()


def truncate(A, B, r):
    """The best rank-r approximation of B A (A [R, in], B [out, R]) as (A_r, B_r)."""
    Qb, Rb = torch.linalg.qr(B)  # B = Qb Rb
    Qa, Ra = torch.linalg.qr(A.T)  # A^T = Qa Ra
    U, S, Vh = torch.linalg.svd(Rb @ Ra.T)
    return (S[:r, None] * (Vh[:r] @ Qa.T)), Qb @ U[:, :r]


def ternarise(W, W0):
    """W (merged) to trits x group scales, starting from the original ternary W0's group scales."""
    out, inn = W.shape
    Wg, W0g = W.reshape(out, inn // 128, 128), W0.reshape(out, inn // 128, 128)
    s0 = W0g.abs().amax(-1, keepdim=True).clamp_min(1e-12)  # a ternary group's scale is its largest magnitude
    t0 = torch.round(W0g / s0)
    t = torch.round(Wg / s0).clamp(-1, 1)
    flips = (t != t0).float().mean().item()
    s = (Wg * t).sum(-1, keepdim=True) / (t * t).sum(-1, keepdim=True).clamp_min(1)  # least squares per group
    s = torch.where((t * t).sum(-1, keepdim=True) > 0, s, s0)
    return (t * s).reshape(out, inn), flips


def load(config):
    global pipe
    pipe.transformer = Flux2Transformer2DModel.from_pretrained(root, subfolder='transformer', torch_dtype=torch.bfloat16)
    tr = pipe.transformer
    flips = []
    for name, A, B in lora_pairs():
        lin = tr.get_submodule(name)
        W0 = lin.weight.data.float()
        if config == 'full':
            W = W0 + B @ A
        elif config.startswith('r'):
            Ar, Br = truncate(A, B, int(config[1:]))
            W = W0 + Br @ Ar
        else:
            W, f = ternarise(W0 + B @ A, W0)
            flips.append(f)
        lin.weight.data = W.to(torch.bfloat16)
    pipe.to('mps')
    return float(np.mean(flips)) if flips else None


def paint(emb):
    t0 = time.time()
    with torch.no_grad():
        img = pipe(prompt_embeds=emb.to('mps'), height=SIZE, width=SIZE, num_inference_steps=STEPS,
                   generator=torch.Generator('cpu').manual_seed(SEED)).images[0]
    torch.mps.synchronize()
    return img, time.time() - t0


CONFIGS = os.environ.get('CONFIGS', 'full,r128,r64,tern').split(',')
notes = {}
for c in CONFIGS:
    f = load(c)
    if f is not None:
        notes['tern_trits_changed'] = f
        print(f'tern: {100 * f:.3f}% of trits change', flush=True)
    for i, d in enumerate(cache):
        img, dt = paint(d['embeds'])
        img.save(f'{OUT}/p{i}_{c}.png')
        print(c, i, f'{dt:.1f}s', flush=True)
    pipe.to('cpu')

sheet = Image.new('RGB', (len(CONFIGS) * 256, len(cache) * 256))
for i in range(len(cache)):
    for j, c in enumerate(CONFIGS):
        sheet.paste(Image.open(f'{OUT}/p{i}_{c}.png').resize((256, 256)), (j * 256, i * 256))
sheet.save(f'{OUT}/sheet.jpg', quality=88)
json.dump(dict(columns=CONFIGS, prompts=[d['prompt'] for d in cache], **notes), open(f'{OUT}/results.json', 'w'), indent=1)
print('columns:', CONFIGS, notes)
