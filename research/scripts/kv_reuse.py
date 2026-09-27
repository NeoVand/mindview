"""Can the second of 2 steps reuse the first step's text? (If so, step 2 need not compute the prompt's rows at all.)

The image rows see the prompt only through the text rows' keys and values in each block's attention. This records them
in step 1 and substitutes them in step 2, which is exactly what the image rows would see if step 2 skipped the text
rows. Compared with the usual 2 steps (the LoRA merged into the ternary DiT, stock-encoder prompts, 512 x 512).

    python scripts/kv_reuse.py [n_prompts]
"""
import glob
import os
import sys

import torch
from diffusers import Flux2KleinPipeline
from diffusers.models.transformers import transformer_flux2 as tf2
from PIL import Image
from safetensors import safe_open

os.chdir(os.path.join(os.path.dirname(__file__), '..'))
n_prompts = int(sys.argv[1]) if len(sys.argv) > 1 else 8
S1, SEED = 0.8854396343231201, 7
OUT = 'renders/kv_reuse'
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

# ---- the attention wrapper: record or replay the text rows' keys and values, call by call
state = dict(mode='off', nt=0, calls=0, kv=[])
_attn = tf2.dispatch_attention_fn


def attn(query, key, value, *a, **kw):
    nt = state['nt']
    if state['mode'] == 'record':
        state['kv'].append((key[:, :nt].clone(), value[:, :nt].clone()))
    elif state['mode'] == 'replay':
        k0, v0 = state['kv'][state['calls']]
        key = torch.cat([k0, key[:, nt:]], dim=1)
        value = torch.cat([v0, value[:, nt:]], dim=1)
    state['calls'] += 1
    return _attn(query, key, value, *a, **kw)


tf2.dispatch_attention_fn = attn


def velocity(x, ids, sigma, emb, txt_ids, mode):
    state.update(mode=mode, calls=0)
    with torch.no_grad():
        v = tr(hidden_states=x.to(dt), timestep=torch.tensor([sigma], device=dev, dtype=dt), guidance=None,
               encoder_hidden_states=emb, txt_ids=txt_ids, img_ids=ids, return_dict=False)[0].float()
    state['mode'] = 'off'
    return v


def decode(x0):
    lat = x0.permute(0, 2, 1).reshape(1, 128, 32, 32)
    mean = pipe.vae.bn.running_mean.view(1, -1, 1, 1).to(dev, torch.float32)
    std = torch.sqrt(pipe.vae.bn.running_var.view(1, -1, 1, 1) + pipe.vae.config.batch_norm_eps).to(dev, torch.float32)
    lat = pipe._unpatchify_latents(lat * std + mean)
    with torch.no_grad():
        img = pipe.vae.decode(lat.to(pipe.vae.dtype), return_dict=False)[0]
    return pipe.image_processor.postprocess(img, output_type='pil')[0]


for i, d in enumerate(cache):
    emb = d['embeds'].to(dev, dt)
    txt_ids = pipe._prepare_text_ids(emb).to(dev)
    state['nt'] = emb.shape[1]
    for label, reuse in (('full', False), ('reuse', True)):
        z = torch.randn((1, 128, 32, 32), generator=torch.Generator('cpu').manual_seed(SEED)).to(dev)
        eps, ids = pipe._pack_latents(z), pipe._prepare_latent_ids(z).to(dev)
        state['kv'] = []
        x0 = eps - velocity(eps, ids, 1.0, emb, txt_ids, 'record' if reuse else 'off')
        x = (1 - S1) * x0 + S1 * eps
        x0 = x - S1 * velocity(x, ids, S1, emb, txt_ids, 'replay' if reuse else 'off')
        decode(x0).save(f'{OUT}/p{i}_{label}.png')
        print(label, i, flush=True)
sheet = Image.new('RGB', (2 * 256, len(cache) * 256))
for i in range(len(cache)):
    for j, label in enumerate(('full', 'reuse')):
        sheet.paste(Image.open(f'{OUT}/p{i}_{label}.png').resize((256, 256)), (j * 256, i * 256))
sheet.save(f'{OUT}/sheet.jpg', quality=88)
print('columns: full, reuse (step 2 sees step 1 text keys and values)')
