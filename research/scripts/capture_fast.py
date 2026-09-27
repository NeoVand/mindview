"""Capture one real Fast run of mindview-t2i-turbo, for the landing's concept stills (concept_landing.py).

The pipeline as the turbo file runs it, in PyTorch:
    reader     Ternary Bonsai 1.7B (unpacked), the prompt in its chat template padded to 256 rows; its attention in
               layers 1-9 (the ones the file carries) and its states after layers 3, 6 and 9
    cond       the fitted map (fused_final_layers_3_6_9.pt) from those states to the painter's text rows; rows 0-2 are
               the fixed prefix
    painter    the ternary DiT with the 2-step LoRA merged, 2 steps: the first at 256 x 256 with the first 128 text
               rows, its clean-picture guess upsampled and noised back to sigma 0.885, the second at 512 x 512
After every block of the painter: the image rows' attention to each text row (mean over heads), and the picture the
painter has in mind (x0 = x_t - sigma v with v read out of that block's state by the final layer, decoded).

    PROMPT="..." python scripts/capture_fast.py [out_name]
"""
import glob
import json
import math
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import Flux2KleinPipeline, Flux2Transformer2DModel
from diffusers.models.transformers import transformer_flux2 as tf2
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer

os.chdir(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, 'scripts')
torch.set_grad_enabled(False)
PROMPT = os.environ.get('PROMPT', 'a bonsai tree made of glowing circuitry in a dark museum, volumetric light')
NAME = sys.argv[1] if len(sys.argv) > 1 else 'bonsai'
OUT = f'renders/landing_capture/{NAME}'
os.makedirs(OUT, exist_ok=True)
SEED, S1, NT, NT_LOW, TAPS = 7, 0.8854396343231201, 256, 128, (3, 6, 9)
dev, dt = 'mps', torch.bfloat16
HUB = '/Users/neo/.cache/huggingface/hub/'
TERN = glob.glob(HUB + 'models--prism-ml--bonsai-image-ternary-4B-unpacked/snapshots/*/')[0]
READER = glob.glob(HUB + 'models--prism-ml--Ternary-Bonsai-1.7B-unpacked/snapshots/*/')[0]
SANA = glob.glob(HUB + 'models--radames--FLUX.2-klein-Sana-Sprint/snapshots/*/pytorch_lora_weights.safetensors')[0]

# ---- the reader
tok = AutoTokenizer.from_pretrained(TERN + 'tokenizer')
text = tok.apply_chat_template([{'role': 'user', 'content': PROMPT}], tokenize=False, add_generation_prompt=True,
                               enable_thinking=False)
inp = tok(text, return_tensors='pt', padding='max_length', max_length=NT, truncation=True)
n_real = int(inp.attention_mask.sum())
ids = inp.input_ids[0].tolist()
words = [tok.decode([i]) for i in ids[:n_real]]
reader = AutoModelForCausalLM.from_pretrained(READER, dtype=torch.float32, attn_implementation='eager').to(dev).eval()
reader.model.layers = reader.model.layers[:max(TAPS)]  # the file carries 9 layers
out = reader.model(input_ids=inp.input_ids.to(dev), attention_mask=inp.attention_mask.to(dev), output_attentions=True,
                   output_hidden_states=True, use_cache=False)
reader_attn = torch.stack([a[0, :, :n_real, :n_real].float().cpu() for a in out.attentions])  # [9, 16, n, n]
taps = torch.cat([out.hidden_states[k][0].float() for k in TAPS], -1).cpu().double()  # [256, 6144]
reader_norm = torch.stack([h[0, :n_real].float().norm(dim=-1).cpu() for h in out.hidden_states])  # [10, n]
del reader, out
torch.mps.empty_cache()

# ---- the map to the painter's text rows (the adapter and the context embedder in one), and the fixed prefix
import pack_model as pm  # noqa: E402
man = json.load(open(pm.PAINTER + '/manifest.json'))


def dense(name):
    e = man['tensors'][name]
    dtp = np.float16 if e['dtype'] == 'f16' else np.float32
    return np.fromfile(pm.PAINTER + '/' + e['file'], dtype=dtp, count=e['bytes'] // np.dtype(dtp).itemsize,
                       offset=e['offset']).reshape(e['shape']).astype(np.float64)


fused = torch.load('data/adapter/fused_final_layers_3_6_9.pt')
C = torch.from_numpy(dense('context_embedder.weight'))
ctx = taps @ fused['W'].double() + fused['b'].double()
ctx[:3] = torch.from_numpy(dense('adapter.prefix')) @ C.T
ctx = ctx[None].to(dev, dt)  # [1, 256, 3072]

# ---- the painter
pipe = Flux2KleinPipeline.from_pretrained(TERN, text_encoder=None, transformer=None, torch_dtype=dt)
pipe.vae.to(dev)
tr = Flux2Transformer2DModel.from_pretrained(TERN, subfolder='transformer', torch_dtype=dt)
with safe_open(SANA, 'pt') as s:
    for m in sorted({k.rsplit('.lora_', 1)[0] for k in s.keys()}):
        lin = tr.get_submodule(m.removeprefix('transformer.'))
        lin.weight.data = (lin.weight.data.float() + s.get_tensor(f'{m}.lora_B.weight').float()
                           @ s.get_tensor(f'{m}.lora_A.weight').float()).to(dt)
tr.context_embedder = torch.nn.Identity()  # ctx is already in the painter's space
tr.to(dev)

rec = dict(nt=0, attn=[], img=[], temb=None)
_attn = tf2.dispatch_attention_fn


def attn(query, key, value, *a, **kw):
    nt = rec['nt']
    q = query[:, nt:].float()  # image rows [1, ni, H, D]
    k = key.float()
    logits = torch.einsum('bqhd,bkhd->bhqk', q, k) / math.sqrt(q.shape[-1])
    p = torch.softmax(logits, -1)[..., :nt].mean(1)[0]  # [ni, nt]: image rows -> text rows, mean over heads
    rec['attn'].append(p.cpu())
    return _attn(query, key, value, *a, **kw)


tf2.dispatch_attention_fn = attn
for b in tr.transformer_blocks:
    b.register_forward_hook(lambda m, a, o: rec['img'].append(o[1]))
for b in tr.single_transformer_blocks:
    b.register_forward_hook(lambda m, a, o: rec['img'].append(o[:, rec['nt']:]))
tr.time_guidance_embed.register_forward_hook(lambda m, a, o: rec.__setitem__('temb', o))


def decode(x0, side):
    lat = x0.permute(0, 2, 1).reshape(1, 128, side, side)
    mean = pipe.vae.bn.running_mean.view(1, -1, 1, 1).to(dev, torch.float32)
    std = torch.sqrt(pipe.vae.bn.running_var.view(1, -1, 1, 1) + pipe.vae.config.batch_norm_eps).to(dev, torch.float32)
    lat = pipe._unpatchify_latents(lat * std + mean)
    img = pipe.vae.decode(lat.to(pipe.vae.dtype), return_dict=False)[0]
    return pipe.image_processor.postprocess(img, output_type='pil')[0]


def step(s, x, ids, sigma, emb, side):
    """One step; saves the per-block attention and pictures. Returns the velocity."""
    rec.update(nt=emb.shape[1], attn=[], img=[])
    v = tr(hidden_states=x.to(dt), timestep=torch.tensor([sigma], device=dev, dtype=dt), guidance=None,
           encoder_hidden_states=emb, txt_ids=pipe._prepare_text_ids(emb).to(dev), img_ids=ids,
           return_dict=False)[0].float()
    assert len(rec['attn']) == 25 and len(rec['img']) == 25
    np.save(f'{OUT}/attn_s{s}.npy', torch.stack(rec['attn'])[:, :, :n_real].half().numpy())  # [25, ni, n_real]
    for b, h in enumerate(rec['img']):
        vb = tr.proj_out(tr.norm_out(h, rec['temb'])).float()
        decode(x - sigma * vb, side).save(f'{OUT}/lens_s{s}_b{b:02d}.png')
    return v


def packed_noise(side, gen):
    z = torch.randn((1, 128, side, side), generator=gen, dtype=torch.float32).to(dev)
    return pipe._pack_latents(z), pipe._prepare_latent_ids(z).to(dev)


gen = torch.Generator('cpu').manual_seed(SEED)
x, ids = packed_noise(16, gen)
decode(x, 16).save(f'{OUT}/state_s0.png')  # the noise it starts from
v = step(0, x, ids, 1.0, ctx[:, :NT_LOW], 16)
x0 = x - v
decode(x0, 16).save(f'{OUT}/sketch.png')
lat = F.interpolate(pipe._unpatchify_latents(x0.permute(0, 2, 1).reshape(1, 128, 16, 16)), scale_factor=2,
                    mode='bilinear', align_corners=False)
x0 = pipe._pack_latents(pipe._patchify_latents(lat))
eps, ids = packed_noise(32, gen)
x = (1 - S1) * x0 + S1 * eps
decode(x, 32).save(f'{OUT}/state_s1.png')  # the sketch, upsampled and noised back
v = step(1, x, ids, S1, ctx, 32)
decode(x - S1 * v, 32).save(f'{OUT}/final.png')

np.save(f'{OUT}/reader_attn.npy', reader_attn.half().numpy())
np.save(f'{OUT}/reader_norm.npy', reader_norm.numpy())
json.dump(dict(prompt=PROMPT, seed=SEED, words=words, n_real=n_real, nt=NT, nt_low=NT_LOW, taps=TAPS,
               sigmas=[1.0, S1, 0.0], sides=[16, 32]), open(f'{OUT}/meta.json', 'w'), indent=1)
print('saved', OUT, 'words', n_real)
