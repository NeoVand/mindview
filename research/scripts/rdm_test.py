"""Does EPFL's 1-step fine-tune of klein carry over to the ternary painter, and at what rank?

Paints cached prompts (stock encoder, 512 x 512, one seed) with
    rdm        the 1-step model itself: klein with epfl-vita/flux2-klein-1step-rdm's weights, 1 step (the ceiling)
    tern+full  the ternary DiT plus RDM's whole change from klein (every tensor), 1 step
    tern+r256  the ternary DiT plus that change at rank 256 for the 100 big matrices (exact for the rest), 1 step
    sana-1     the ternary DiT with the SANA-Sprint LoRA, 1 step (today's 1 step)
    sana-2     the same, 2 steps (today's Fast)
Needs data/rdm/rdm_lora_r256.safetensors from rdm_delta.py.

    python scripts/rdm_test.py [n_prompts]
"""
import glob
import os
import sys
import time

import torch
from diffusers import Flux2KleinPipeline, Flux2Transformer2DModel
from PIL import Image
from safetensors import safe_open

os.chdir(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, 'scripts')
from rdm_delta_map import pieces  # noqa: E402

n_prompts = int(sys.argv[1]) if len(sys.argv) > 1 else 8
S1, SEED = 0.8854396343231201, 7
OUT = 'renders/rdm_test'
os.makedirs(OUT, exist_ok=True)
HUB = '/Users/neo/.cache/huggingface/hub/'
TERN = glob.glob(HUB + 'models--prism-ml--bonsai-image-ternary-4B-unpacked/snapshots/*/')[0]
KLEIN = glob.glob(HUB + 'models--black-forest-labs--FLUX.2-klein-4B/snapshots/*/')[0]
RDM = glob.glob(HUB + 'models--epfl-vita--flux2-klein-1step-rdm/snapshots/*/model.safetensors')[0]
SANA = glob.glob(HUB + 'models--radames--FLUX.2-klein-Sana-Sprint/snapshots/*/pytorch_lora_weights.safetensors')[0]
LOWRANK = 'data/rdm/rdm_lora_r256.safetensors'

cache = [d for d in (torch.load(f, map_location='cpu') for f in sorted(glob.glob('data/prompt_cache/*.pt')))
         if d['encoder'] == 'qwen'][:n_prompts]
pipe = Flux2KleinPipeline.from_pretrained(TERN, text_encoder=None, torch_dtype=torch.bfloat16)
dev, dt = 'mps', torch.bfloat16
pipe.vae.to(dev)


def rdm_delta():
    """{diffusers key: RDM - klein} for every tensor."""
    out = {}
    with safe_open(RDM, 'pt') as fr, safe_open(KLEIN + 'transformer/diffusion_pytorch_model.safetensors', 'pt') as fk:
        for k in fr.keys():
            full = fr.get_tensor(k).float()
            for dk, sl in pieces(k):
                if sl == 'swap':
                    h = full.shape[0] // 2
                    w1 = torch.cat([full[h:], full[:h]])
                elif sl is not None:
                    w1 = full[sl[0]:sl[1]]
                else:
                    w1 = full
                out[dk] = w1 - fk.get_tensor(dk).float()
    return out


def load(config, delta):
    base = KLEIN if config == 'rdm' else TERN
    tr = Flux2Transformer2DModel.from_pretrained(base, subfolder='transformer', torch_dtype=dt)
    params = dict(tr.named_parameters())
    if config in ('rdm', 'tern+full', 'tern+r256'):
        low = {}
        if config == 'tern+r256':
            with safe_open(LOWRANK, 'pt') as s:
                for m in {k.rsplit('.lora_', 1)[0] for k in s.keys()}:
                    low[m.removeprefix('transformer.') + '.weight'] = (
                        s.get_tensor(f'{m}.lora_B.weight').float() @ s.get_tensor(f'{m}.lora_A.weight').float())
        for k, d in delta.items():
            p = params[k]
            p.data = (p.data.float() + low.get(k, d)).to(dt)
    if config.startswith('sana'):
        with safe_open(SANA, 'pt') as s:
            for m in {k.rsplit('.lora_', 1)[0] for k in s.keys()}:
                lin = tr.get_submodule(m.removeprefix('transformer.'))
                lin.weight.data = (lin.weight.data.float() + s.get_tensor(f'{m}.lora_B.weight').float()
                                   @ s.get_tensor(f'{m}.lora_A.weight').float()).to(dt)
    return tr.to(dev)


def decode(x0):
    lat = x0.permute(0, 2, 1).reshape(1, 128, 32, 32)
    mean = pipe.vae.bn.running_mean.view(1, -1, 1, 1).to(dev, torch.float32)
    std = torch.sqrt(pipe.vae.bn.running_var.view(1, -1, 1, 1) + pipe.vae.config.batch_norm_eps).to(dev, torch.float32)
    lat = pipe._unpatchify_latents(lat * std + mean)
    with torch.no_grad():
        img = pipe.vae.decode(lat.to(pipe.vae.dtype), return_dict=False)[0]
    return pipe.image_processor.postprocess(img, output_type='pil')[0]


CONFIGS = [('rdm', 1), ('tern+full', 1), ('tern+r256', 1), ('sana-1', 1), ('sana-2', 2)]
delta = rdm_delta()
print('delta tensors', len(delta), flush=True)
for config, steps in CONFIGS:
    tr = load(config, delta)
    for i, d in enumerate(cache):
        emb = d['embeds'].to(dev, dt)
        txt_ids = pipe._prepare_text_ids(emb).to(dev)
        z = torch.randn((1, 128, 32, 32), generator=torch.Generator('cpu').manual_seed(SEED)).to(dev)
        eps, ids = pipe._pack_latents(z), pipe._prepare_latent_ids(z).to(dev)
        t0 = time.time()

        def v(x, sigma):
            with torch.no_grad():
                return tr(hidden_states=x.to(dt), timestep=torch.tensor([sigma], device=dev, dtype=dt), guidance=None,
                          encoder_hidden_states=emb, txt_ids=txt_ids, img_ids=ids, return_dict=False)[0].float()

        x0 = eps - v(eps, 1.0)
        if steps == 2:
            x = (1 - S1) * x0 + S1 * eps
            x0 = x - S1 * v(x, S1)
        decode(x0).save(f'{OUT}/p{i}_{config}.png')
        print(config, i, f'{time.time() - t0:.1f}s', flush=True)
    del tr
    torch.mps.empty_cache()
labels = [c for c, _ in CONFIGS]
sheet = Image.new('RGB', (len(labels) * 256, len(cache) * 256))
for i in range(len(cache)):
    for j, c in enumerate(labels):
        sheet.paste(Image.open(f'{OUT}/p{i}_{c}.png').resize((256, 256)), (j * 256, i * 256))
sheet.save(f'{OUT}/sheet.jpg', quality=88)
print('columns:', labels)
