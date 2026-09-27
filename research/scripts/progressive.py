"""The first step at a quarter of the pixels: does a 2-step painting survive it?

With the few-step LoRA, 2 steps go sigma 1 -> 0.885 -> 0: after the first step the picture is still 88% noise. Here the
first step runs at 256 x 256 (256 patches instead of 1,024), its clean-picture prediction is upsampled to 512 x 512 in
latent space and noised back to sigma 0.885, and the second step runs at full size. Compared with the usual 2 steps at
512 x 512, on cached prompts (stock encoder), the LoRA merged into the ternary DiT.

    python scripts/progressive.py [n_prompts] [sigma1]
"""
import glob
import os
import sys
import time

import torch
import torch.nn.functional as F
from diffusers import Flux2KleinPipeline
from diffusers.models.transformers import transformer_flux2 as tf2
from PIL import Image
from safetensors import safe_open

os.chdir(os.path.join(os.path.dirname(__file__), '..'))
n_prompts = int(sys.argv[1]) if len(sys.argv) > 1 else 8
S1 = float(sys.argv[2]) if len(sys.argv) > 2 else 0.8854396343231201  # the 2-step schedule at 512 x 512
SEED = 7
OUT = f'renders/progressive/s{S1:.3f}'
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


# record the text rows' keys and values in step 1, substitute them in step 2 (see kv_reuse.py)
state = dict(mode='off', nt=0, calls=0, kv=[])
_attn = tf2.dispatch_attention_fn


def attn(query, key, value, *a, **kw):
    nt = state['nt']
    if state['mode'] == 'record':
        state['kv'].append((key[:, :nt].clone(), value[:, :nt].clone()))
    elif state['mode'] == 'replay':
        k0, v0 = state['kv'][state['calls']]
        key, value = torch.cat([k0, key[:, nt:]], dim=1), torch.cat([v0, value[:, nt:]], dim=1)
    state['calls'] += 1
    return _attn(query, key, value, *a, **kw)


tf2.dispatch_attention_fn = attn


def velocity(x_packed, ids, sigma, emb, txt_ids, mode='off'):
    state.update(mode=mode, calls=0, nt=emb.shape[1])
    if mode == 'record':
        state['kv'] = []
    with torch.no_grad():
        v = tr(hidden_states=x_packed.to(dt), timestep=torch.tensor([sigma], device=dev, dtype=dt), guidance=None,
               encoder_hidden_states=emb, txt_ids=txt_ids, img_ids=ids, return_dict=False)[0].float()
    state['mode'] = 'off'
    return v


def packed_noise(side, gen):  # side in patches
    z = torch.randn((1, 128, side, side), generator=gen, dtype=torch.float32).to(dev)
    return pipe._pack_latents(z), pipe._prepare_latent_ids(z).to(dev)


def to_grid(x_packed, side):  # packed [1, side*side, 128] -> [1, 128, side, side]
    return x_packed.permute(0, 2, 1).reshape(1, 128, side, side)


def decode(x0_packed, side):
    lat = to_grid(x0_packed, side)
    mean = pipe.vae.bn.running_mean.view(1, -1, 1, 1).to(dev, torch.float32)
    std = torch.sqrt(pipe.vae.bn.running_var.view(1, -1, 1, 1) + pipe.vae.config.batch_norm_eps).to(dev, torch.float32)
    lat = pipe._unpatchify_latents(lat * std + mean)
    with torch.no_grad():
        img = pipe.vae.decode(lat.to(pipe.vae.dtype), return_dict=False)[0]
    return pipe.image_processor.postprocess(img, output_type='pil')[0]


def two_steps(emb, txt_ids, progressive, reuse=False):
    gen = torch.Generator('cpu').manual_seed(SEED)
    t0 = time.time()
    if progressive:
        x, ids = packed_noise(16, gen)
        x0 = x - velocity(x, ids, 1.0, emb, txt_ids, 'record' if reuse else 'off')
        # upsample the prediction in the unpatchified latent space (32 channels, 32 -> 64 pixels)
        lat = pipe._unpatchify_latents(to_grid(x0, 16))
        lat = F.interpolate(lat, scale_factor=2, mode='bilinear', align_corners=False)
        x0 = pipe._pack_latents(pipe._patchify_latents(lat))
        eps, ids = packed_noise(32, gen)
    else:
        eps, ids = packed_noise(32, gen)  # the Euler step from sigma 1 keeps its noise: x = (1 - S1) x0 + S1 eps
        x0 = eps - velocity(eps, ids, 1.0, emb, txt_ids)
    x = (1 - S1) * x0 + S1 * eps  # (progressive: fresh noise at the full size)
    x0 = x - S1 * velocity(x, ids, S1, emb, txt_ids, 'replay' if reuse else 'off')
    torch.mps.synchronize()
    return decode(x0, 32), time.time() - t0


COLS = [('full', False, False), ('prog', True, False), ('prog+reuse', True, True)]
for i, d in enumerate(cache):
    emb = d['embeds'].to(dev, dt)
    txt_ids = pipe._prepare_text_ids(emb).to(dev)
    for label, prog, reuse in COLS:
        img, secs = two_steps(emb, txt_ids, prog, reuse)
        img.save(f'{OUT}/p{i}_{label}.png')
        print(label, i, f'{secs:.1f}s', flush=True)
sheet = Image.new('RGB', (len(COLS) * 256, len(cache) * 256))
for i in range(len(cache)):
    for j, (label, _, _) in enumerate(COLS):
        sheet.paste(Image.open(f'{OUT}/p{i}_{label}.png').resize((256, 256)), (j * 256, i * 256))
sheet.save(f'{OUT}/sheet.jpg', quality=88)
print('columns: full (2 steps at 512), prog (first step at 256), prog+reuse (and step 2 reuses step 1 text)')
