"""The low-resolution first step with fewer text rows: does it need the pads?

At 2 steps the painter keeps 256 text rows (prompt + pads). With the first step at 256 x 256 (see progressive.py) those
256 text rows are half of that step's rows. The reader is causal, so the first n rows of a 256-row encoding are the
n-row encoding: the first step can simply take a prefix. Columns: the first step with 256, 128 and 64 text rows (the
second always 256), all with the first step at 256 x 256, the LoRA merged into the ternary DiT, stock-encoder prompts.

    python scripts/progressive_rows.py [n_prompts]
"""
import glob
import os
import sys

import torch
import torch.nn.functional as F
from diffusers import Flux2KleinPipeline
from PIL import Image
from safetensors import safe_open

os.chdir(os.path.join(os.path.dirname(__file__), '..'))
n_prompts = int(sys.argv[1]) if len(sys.argv) > 1 else 8
S1, SEED = 0.8854396343231201, 7
OUT = 'renders/progressive_rows'
os.makedirs(OUT, exist_ok=True)

cache = [d for d in (torch.load(f, map_location='cpu') for f in sorted(glob.glob('data/prompt_cache/*.pt')))
         if d['encoder'] == 'qwen'][:n_prompts]
root = glob.glob('/Users/neo/.cache/huggingface/hub/models--prism-ml--bonsai-image-ternary-4B-unpacked/snapshots/*/')[0]
pipe = Flux2KleinPipeline.from_pretrained(root, text_encoder=None, torch_dtype=torch.bfloat16)
LORA = glob.glob('/Users/neo/.cache/huggingface/hub/models--radames--FLUX.2-klein-Sana-Sprint/snapshots/*/'
                 'pytorch_lora_weights.safetensors')[0]
with safe_open(LORA, 'pt') as s:
    for m in sorted({k.rsplit('.lora_', 1)[0] for k in s.keys()}):
        lin = pipe.transformer.get_submodule(m.removeprefix('transformer.'))
        W = lin.weight.data
        lin.weight.data = (W.float() + s.get_tensor(f'{m}.lora_B.weight').float() @ s.get_tensor(f'{m}.lora_A.weight').float()).to(W.dtype)
pipe.to('mps')
tr = pipe.transformer
dev, dt = 'mps', torch.bfloat16


def velocity(x, ids, sigma, emb):
    with torch.no_grad():
        return tr(hidden_states=x.to(dt), timestep=torch.tensor([sigma], device=dev, dtype=dt), guidance=None,
                  encoder_hidden_states=emb, txt_ids=pipe._prepare_text_ids(emb).to(dev), img_ids=ids,
                  return_dict=False)[0].float()


def packed_noise(side, gen):
    z = torch.randn((1, 128, side, side), generator=gen, dtype=torch.float32).to(dev)
    return pipe._pack_latents(z), pipe._prepare_latent_ids(z).to(dev)


def to_grid(x, side):
    return x.permute(0, 2, 1).reshape(1, 128, side, side)


def decode(x0):
    lat = to_grid(x0, 32)
    mean = pipe.vae.bn.running_mean.view(1, -1, 1, 1).to(dev, torch.float32)
    std = torch.sqrt(pipe.vae.bn.running_var.view(1, -1, 1, 1) + pipe.vae.config.batch_norm_eps).to(dev, torch.float32)
    lat = pipe._unpatchify_latents(lat * std + mean)
    with torch.no_grad():
        img = pipe.vae.decode(lat.to(pipe.vae.dtype), return_dict=False)[0]
    return pipe.image_processor.postprocess(img, output_type='pil')[0]


def paint(emb, rows1):
    gen = torch.Generator('cpu').manual_seed(SEED)
    x, ids = packed_noise(16, gen)
    x0 = x - velocity(x, ids, 1.0, emb[:, :rows1])
    lat = F.interpolate(pipe._unpatchify_latents(to_grid(x0, 16)), scale_factor=2, mode='bilinear', align_corners=False)
    x0 = pipe._pack_latents(pipe._patchify_latents(lat))
    eps, ids = packed_noise(32, gen)
    x = (1 - S1) * x0 + S1 * eps
    return decode(x - S1 * velocity(x, ids, S1, emb))


COLS = [256, 128, 64]
for i, d in enumerate(cache):
    emb = d['embeds'][:, :256].to(dev, dt)
    for rows1 in COLS:
        paint(emb, rows1).save(f'{OUT}/p{i}_t{rows1}.png')
        print(rows1, i, flush=True)
sheet = Image.new('RGB', (len(COLS) * 256, len(cache) * 256))
for i in range(len(cache)):
    for j, rows1 in enumerate(COLS):
        sheet.paste(Image.open(f'{OUT}/p{i}_t{rows1}.png').resize((256, 256)), (j * 256, i * 256))
sheet.save(f'{OUT}/sheet.jpg', quality=88)
print('columns: first step (at 256 x 256) with 256, 128, 64 text rows; the second with 256')
