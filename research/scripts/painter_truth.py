"""Ground truth for painter_reference.py, produced with the stock libraries (HF transformers + diffusers).

  painter_truth.py text ["<prompt>"]          # Ternary Bonsai 1.7B (HF, fp32, CPU): 512 x 6144 tap features (layers 7|14|21)
  painter_truth.py dit  [--dtype fp32|bf16]   # diffusers Flux2KleinPipeline on MPS, conditioned through the ternary adapter

Writes data/painter_truth/text.pt and dit_<dtype>.pt (block outputs of step 1, velocities, latents, TAEF2 + VAE images).
The two modes run in separate processes so the LLM and the DiT are never resident together.
"""
import argparse, glob, os, sys, time
import numpy as np
import torch
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument('mode', choices=['text', 'dit'])
ap.add_argument('prompt', nargs='?', default='a bonsai tree made of glowing circuitry in a dark museum, volumetric light')
ap.add_argument('--dtype', default='fp32', choices=['fp32', 'bf16'])
ap.add_argument('--seed', type=int, default=7)
ap.add_argument('--size', type=int, default=512)
ap.add_argument('--steps', type=int, default=4)
args = ap.parse_args()
ROOT = glob.glob('/Users/neo/.cache/huggingface/hub/models--prism-ml--bonsai-image-ternary-4B-unpacked/snapshots/*/')[0]
LLM = glob.glob('/Users/neo/.cache/huggingface/hub/models--prism-ml--Ternary-Bonsai-1.7B-unpacked/snapshots/*/')[0]
OUT = 'data/painter_truth'
os.makedirs(OUT, exist_ok=True)

if args.mode == 'text':
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(ROOT + 'tokenizer')
    text = tok.apply_chat_template([{'role': 'user', 'content': args.prompt}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
    inp = tok(text, return_tensors='pt', padding='max_length', truncation=True, max_length=512)
    # fp32 on CPU, eager attention: the padding mask then means "pad queries see the real tokens only" (verified bit-exact)
    m = AutoModelForCausalLM.from_pretrained(LLM, dtype=torch.float32, attn_implementation='eager').eval()
    t0 = time.time()
    with torch.no_grad():
        hs = m(**inp, output_hidden_states=True, use_cache=False).hidden_states   # hs[k] = residual after k decoder layers
    taps = torch.cat([hs[k][0] for k in (7, 14, 21)], -1)                        # [512, 6144]
    torch.save(dict(prompt=args.prompt, text=text, ids=inp.input_ids[0], n_real=int(inp.attention_mask.sum()), taps=taps,
                    hs_all=torch.stack([h[0] for h in hs]).half()), f'{OUT}/text.pt')
    print(f'1.7B: {int(inp.attention_mask.sum())} real tokens + pads, {time.time()-t0:.1f}s -> {OUT}/text.pt')
    sys.exit()

# ---------------------------------------------------------------- dit: the diffusers pipeline as the reference
from diffusers import Flux2KleinPipeline
sys.path.insert(0, 'scripts')
from taesd import Decoder
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

dev = 'mps'
dt = torch.float32 if args.dtype == 'fp32' else torch.bfloat16
T = torch.load(f'{OUT}/text.pt')
A, Q = torch.load('data/adapter/adapter_bonsai17t.pt'), torch.load('data/adapter/qwen_extra.pt')
cond = T['taps'].float() @ A['W'] + A['b']        # exactly as export_trace.py (fp32, CPU)
cond[:3] = Q['prefix']
g = torch.Generator('cpu').manual_seed(args.seed)
noise = torch.randn((1, 128, args.size // 16, args.size // 16), generator=g, dtype=torch.float32)   # what the fp32 pipeline draws
g = torch.Generator('cpu').manual_seed(args.seed)
noise_bf16 = torch.randn((1, 128, args.size // 16, args.size // 16), generator=g, dtype=torch.bfloat16)
print('bf16 randn == fp32 randn rounded to bf16:', torch.equal(noise_bf16, noise.bfloat16()))

pipe = Flux2KleinPipeline.from_pretrained(ROOT, text_encoder=None, torch_dtype=dt).to(dev)
pipe.set_progress_bar_config(disable=True)
TR = pipe.transformer
NDB = len(TR.transformer_blocks)
S = dict(step=0, blocks={}, lat=[], vel=[], temb=[], mods=[])
TR.register_forward_pre_hook(lambda m, a, kw: S['lat'].append(kw['hidden_states'][0].float().cpu()), with_kwargs=True)
def on_out(m, a, o):                              # velocity of each step (a hook must return None)
    S['vel'].append(o[0][0].float().cpu()); S['step'] += 1


TR.register_forward_hook(on_out)
TR.time_guidance_embed.register_forward_hook(lambda m, a, o: S['temb'].append(o[0].float().cpu()))
TR.double_stream_modulation_img.register_forward_hook(lambda m, a, o: S['mods'].append(o[0].float().cpu()))
def keep_input(key):                              # embedder outputs of step 1 = the blocks' input
    def f(m, a, o):
        if S['step'] == 0: S['blocks'][key] = o[0].float().cpu()
    return f


TR.x_embedder.register_forward_hook(keep_input('in_img'))
TR.context_embedder.register_forward_hook(keep_input('in_txt'))


def hook(i):
    def f(m, a, kw, o):
        if S['step'] != 0: return
        txt, img = (o[0][0], o[1][0]) if i < NDB else (o[0][:512], o[0][512:])   # single blocks carry [txt; img]
        S['blocks'][i] = (txt.float().cpu(), img.float().cpu())
    return f


for i, b in enumerate(list(TR.transformer_blocks) + list(TR.single_transformer_blocks)):
    b.register_forward_hook(hook(i), with_kwargs=True)
S['after'] = []


def on_step(p, i, t, kw):                         # latents after each Euler update (bn-normalised tokens)
    S['after'].append(kw['latents'][0].float().cpu())
    return kw


t0 = time.time()
with torch.no_grad():
    lat = pipe(prompt_embeds=cond[None].to(dev, dt), height=args.size, width=args.size, num_inference_steps=args.steps,
               latents=noise.to(dev, dt), output_type='latent', callback_on_step_end=on_step).images   # [1, 32, 64, 64] de-normalised
print(f'pipeline {args.dtype}: {time.time()-t0:.1f}s')
sig = pipe.scheduler.sigmas.float().cpu()
final = S['after'][-1]                            # the final bn-normalised token latent (what TAEF2 decodes)

# decoders: TAEF2 (fp32 weights, taesd module, as export_trace.py) on the bn-normalised latent, and the full VAE
hw = args.size // 16
sd = {'.'.join([str(int(k.split('.')[2]) + 1)] + k.split('.')[3:]): v
      for k, v in load_file(hf_hub_download('madebyollin/taef2', 'taef2.safetensors')).items() if k.startswith('decoder.layers.')}
taef2 = Decoder(32, use_midblock_gn=True); taef2.load_state_dict(sd); taef2 = taef2.eval()
with torch.no_grad():
    lt = final.reshape(1, hw, hw, 128).permute(0, 3, 1, 2)
    img_t = taef2(pipe._unpatchify_latents(lt)).clamp(0, 1)[0]                   # fp32 on CPU
    img_v = pipe.vae.decode(lat, return_dict=False)[0]
    img_v = pipe.image_processor.postprocess(img_v, output_type='pt')[0].float().cpu()
Image.fromarray((img_t.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)).save(f'{OUT}/dit_{args.dtype}_taef2.png')
Image.fromarray((img_v.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)).save(f'{OUT}/dit_{args.dtype}_vae.png')
blocks = [S['blocks'][i] for i in range(len(S['blocks']) - 2)]
torch.save(dict(dtype=args.dtype, cond=cond, noise=noise, sigmas=sig, timesteps=pipe.scheduler.timesteps.float().cpu(),
                in_txt=S['blocks']['in_txt'], in_img=S['blocks']['in_img'],
                blk_txt=torch.stack([b[0] for b in blocks]), blk_img=torch.stack([b[1] for b in blocks]),
                lat=torch.stack(S['lat']), vel=torch.stack(S['vel']), final=final, temb=torch.stack(S['temb']),
                mod_img=torch.stack(S['mods']), img_taef2=img_t, img_vae=img_v), f'{OUT}/dit_{args.dtype}.pt')
print('saved', f'{OUT}/dit_{args.dtype}.pt')
