"""Record one complete generation (language stage + painter) as a trace the visual prototype can replay.

  export_trace.py "<prompt>" [--encoder qwen|bonsai4b_adapt|bonsai17_adapt] [--size 512] [--steps 4] [--seed 7] [--name slug]

Writes traces/<name>/manifest.json + raw little-endian float16 .bin arrays + PNGs:
  Act I  (language):  enc_norm [L+1,n], enc_pca [L+1,n,3], enc_threads [L+1,n,256], enc_attn [L,H,n,n], lens words (json)
  Act II (handoff):   ctx_norm [512] (context_embedder output per text token, real + padding)
  Act III (painter):  dit_pca [S,25,N,3], dit_norm_img [S,25,N], dit_norm_txt [S,25,512], dit_attn [S,25,N,n] (image->prompt,
                      mean over heads), dit_attn_mass [S,25,3] (real/pad/image), lens_s*_b*.png (TAEF2), x0_s*.png, xt_s*.png
"""
import argparse, glob, json, os, re, time, gc
import numpy as np
import torch
from PIL import Image
from transformers import AutoTokenizer, AutoModelForCausalLM
from diffusers import Flux2KleinPipeline
import diffusers.models.transformers.transformer_flux2 as tf2

ap = argparse.ArgumentParser()
ap.add_argument('prompt'); ap.add_argument('--encoder', default='qwen'); ap.add_argument('--size', type=int, default=512)
ap.add_argument('--steps', type=int, default=4); ap.add_argument('--seed', type=int, default=7); ap.add_argument('--name')
args = ap.parse_args()
name = args.name or re.sub(r'[^a-z0-9]+', '-', args.prompt.lower())[:40].strip('-') + f'-{args.encoder}-{args.size}-s{args.steps}'
OUT = f'traces/{name}'; os.makedirs(OUT, exist_ok=True)
dev, dt = 'mps', torch.bfloat16
ROOT = glob.glob('/Users/neo/.cache/huggingface/hub/models--prism-ml--bonsai-image-ternary-4B-unpacked/snapshots/*/')[0]
manifest = dict(prompt=args.prompt, encoder=args.encoder, size=args.size, steps=args.steps, seed=args.seed, arrays={})


def save(key, arr):
    arr = np.ascontiguousarray(np.asarray(arr, dtype=np.float16))
    arr.tofile(f'{OUT}/{key}.bin')
    manifest['arrays'][key] = dict(file=f'{key}.bin', dtype='float16', shape=list(arr.shape))


def pca_basis(X, k=3, V0=None, iters=12):
    """Top-k principal directions by subspace iteration (GPU matmuls, tiny CPU QR); warm start keeps signs stable."""
    Xc = X - X.mean(0, keepdim=True)
    V = torch.randn(X.shape[1], k, device=X.device) if V0 is None else V0.clone()
    for _ in range(iters):
        V = Xc.T @ (Xc @ V)
        V = torch.linalg.qr(V.cpu())[0].to(X.device)
    if V0 is not None:  # align signs with the previous basis
        V = V * torch.sign((V * V0).sum(0, keepdim=True) + 1e-9)
    return Xc @ V, V


# ============================================================ Act I: language stage
tok = AutoTokenizer.from_pretrained(ROOT + 'tokenizer')
enc_src = {'qwen': ROOT + 'text_encoder',
           'bonsai4b_adapt': glob.glob('/Users/neo/.cache/huggingface/hub/models--prism-ml--Ternary-Bonsai-4B-unpacked/snapshots/*/')[0],
           'bonsai17_adapt': glob.glob('/Users/neo/.cache/huggingface/hub/models--prism-ml--Ternary-Bonsai-1.7B-unpacked/snapshots/*/')[0]}
enc_src['bonsai17t_adapt'] = enc_src['bonsai17_adapt']  # ternary adapter (adapter_ternarize.py)
enc_src = enc_src[args.encoder]
taps_layers = (7, 14, 21) if args.encoder.startswith('bonsai17') else (9, 18, 27)
t0 = time.time()
enc = AutoModelForCausalLM.from_pretrained(enc_src, dtype=dt, attn_implementation='eager').to(dev).eval()
text = tok.apply_chat_template([{'role': 'user', 'content': args.prompt}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
ids = tok(text, return_tensors='pt').input_ids.to(dev)
n = ids.shape[1]
toks = [tok.decode([i]) for i in ids[0].tolist()]
with torch.no_grad():
    out = enc(input_ids=ids, output_hidden_states=True, output_attentions=True, use_cache=False)
    H = torch.stack(out.hidden_states)[:, 0].float()                        # [L+1, n, d]
    save('enc_norm', H.norm(dim=-1).cpu())
    flat = H[1:, 1:].reshape(-1, H.shape[-1])                                # skip embeddings + attention-sink token 0 for the basis
    _, V = pca_basis(flat)
    save('enc_pca', ((H - flat.mean(0)) @ V).cpu())
    rms = H.pow(2).mean(-1, keepdim=True).sqrt()
    g = torch.Generator().manual_seed(0)
    dims = torch.randperm(H.shape[-1], generator=g)[:256].to(dev)
    save('enc_threads', (H / rms)[:, :, dims].cpu())
    save('enc_attn', torch.stack(out.attentions)[:, 0].float().cpu())       # [L, heads, n, n]
    # logit lens with tied embeddings
    E = enc.get_input_embeddings().weight
    lens = []
    for L in range(H.shape[0]):
        h = enc.model.norm(H[L].to(dt)) if L < H.shape[0] - 1 else H[L].to(dt)
        p = torch.softmax((h @ E.T).float(), -1)
        top = p.topk(5, -1)
        lens.append([[(tok.decode([int(i)]), round(float(v), 4)) for i, v in zip(top.indices[t], top.values[t])] for t in range(n)])
    # conditioning embeddings for the painter (padded to 512)
    inp = tok(text, return_tensors='pt', padding='max_length', max_length=512).to(dev)
    hs = enc(**inp, output_hidden_states=True, use_cache=False).hidden_states
    cond = torch.cat([hs[k] for k in taps_layers], -1).float()[0]            # [512, taps*d]
if args.encoder != 'qwen':
    A = torch.load(f'data/adapter/adapter_{args.encoder.replace("_adapt", "")}.pt')
    Q = torch.load('data/adapter/qwen_extra.pt')
    cond = cond.cpu() @ A['W'] + A['b']
    cond[:3] = Q['prefix']
prompt_embeds = cond[None].to(dev, dt)
manifest['act1'] = dict(tokens=toks, layers=H.shape[0] - 1, hidden=H.shape[-1], taps=list(taps_layers), heads=out.attentions[0].shape[1], lens=lens)
print(f'Act I: {n} tokens through {H.shape[0]-1} layers in {time.time()-t0:.1f}s', flush=True)
del enc, out, hs, H; gc.collect(); torch.mps.empty_cache()

# ============================================================ Act III: the painter
pipe = Flux2KleinPipeline.from_pretrained(ROOT, text_encoder=None, torch_dtype=dt).to(dev)
pipe.set_progress_bar_config(disable=True)
T = pipe.transformer
NB = len(T.transformer_blocks) + len(T.single_transformer_blocks)
N_TXT = 512
S = dict(step=0, block=0, temb=None, V={}, pca={}, nimg={}, ntxt={}, attn={}, mass={}, lensv={}, tunedv={})
# tuned lens (scripts/tuned_lens.py): one affine readout per block from image-token state to this step's velocity
TUNED = torch.load('data/tuned_lens/lens.pt')['W'].to(dev).float() if os.path.exists('data/tuned_lens/lens.pt') else None
manifest['lens_kind'] = 'tuned' if TUNED is not None else 'raw'
T.time_guidance_embed.register_forward_hook(lambda m, a, o: S.__setitem__('temb', o))
T.context_embedder.register_forward_hook(lambda m, a, o: S.__setitem__('ctx', o[0].float().norm(dim=-1).cpu()))

_orig_dispatch = tf2.dispatch_attention_fn


def capturing_dispatch(query, key, value, *a, **kw):
    with torch.no_grad():
        q = query[:, N_TXT:].transpose(1, 2).float()
        k = key.transpose(1, 2).float()
        p = torch.softmax(q @ k.transpose(-1, -2) / q.shape[-1] ** 0.5, -1)   # [1, H, Nimg, Ntot]
        S['attn'][(S['step'], S['block'])] = p[0, :, :, :n].mean(0).cpu()
        S['mass'][(S['step'], S['block'])] = torch.stack([p[..., :n].sum(-1).mean(), p[..., n:N_TXT].sum(-1).mean(), p[..., N_TXT:].sum(-1).mean()]).cpu()
        del p, q, k
    return _orig_dispatch(query, key, value, *a, **kw)


tf2.dispatch_attention_fn = capturing_dispatch


def block_hooks(idx, single):
    def pre(mod, a, kw):
        S['block'] = idx

    def post(mod, a, kw, o):
        h_img, h_txt = (o[:, N_TXT:], o[:, :N_TXT]) if single else (o[1], o[0])
        key = (S['step'], idx)
        with torch.no_grad():
            x = h_img[0].float()
            prev = S['V'].get((S['step'], idx - 1)) if idx > 0 else S['V'].get((S['step'] - 1, NB - 1))
            proj, V = pca_basis(x, V0=prev)
            S['V'][key] = V
            S['pca'][key] = proj.cpu()
            S['nimg'][key] = x.norm(dim=-1).cpu()
            S['ntxt'][key] = h_txt[0].float().norm(dim=-1).cpu()
            S['lensv'][key] = T.proj_out(T.norm_out(h_img, S['temb'])).float()
            if TUNED is not None:
                S['tunedv'][key] = (torch.cat([x, torch.ones(x.shape[0], 1, device=dev)], 1) @ TUNED[idx])[None]
    return pre, post


for i, b in enumerate(list(T.transformer_blocks) + list(T.single_transformer_blocks)):
    pre, post = block_hooks(i, i >= len(T.transformer_blocks))
    b.register_forward_pre_hook(pre, with_kwargs=True)
    b.register_forward_hook(post, with_kwargs=True)
xs, vs, sig = [], [], []
T.register_forward_pre_hook(lambda m, a, kw: xs.append(kw['hidden_states'].float()), with_kwargs=True)
T.register_forward_hook(lambda m, a, o: vs.append(o[0].float()))


def on_step(p, i, t, kw):
    sig.append(float(p.scheduler.sigmas[i])); S['step'] = i + 1
    S['latents'] = kw['latents'].detach().float()
    return kw


t0 = time.time()
with torch.no_grad():
    img = pipe(prompt_embeds=prompt_embeds, height=args.size, width=args.size, num_inference_steps=args.steps,
               generator=torch.Generator('cpu').manual_seed(args.seed), callback_on_step_end=on_step).images[0]
img.save(f'{OUT}/final.png')
print(f'Act III: {args.steps} steps x {NB} blocks in {time.time()-t0:.1f}s', flush=True)

# decoders: full VAE for per-step canvases, TAEF2 for the 25 x steps block lens
hw = args.size // 16
mean = pipe.vae.bn.running_mean.view(1, -1, 1, 1).to(dev, dt)
std = torch.sqrt(pipe.vae.bn.running_var.view(1, -1, 1, 1) + pipe.vae.config.batch_norm_eps).to(dev, dt)
import sys; sys.path.insert(0, 'scripts')
from taesd import Decoder
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
# taef2.safetensors uses diffusers AutoencoderTiny names (decoder.layers.i); taesd's Sequential has a leading Clamp -> i+1
_sd = {'.'.join([str(int(k.split('.')[2]) + 1)] + k.split('.')[3:]): v
       for k, v in load_file(hf_hub_download('madebyollin/taef2', 'taef2.safetensors')).items() if k.startswith('decoder.layers.')}
taef2_dec = Decoder(32, use_midblock_gn=True); taef2_dec.load_state_dict(_sd); taef2_dec = taef2_dec.to(dev).eval()


def tokens_to_latent(tk):
    return tk.to(dt).reshape(1, hw, hw, 128).permute(0, 3, 1, 2)


@torch.no_grad()
def vae_png(tk, path):
    lat = pipe._unpatchify_latents(tokens_to_latent(tk) * std + mean)
    im = pipe.vae.decode(lat, return_dict=False)[0]
    pipe.image_processor.postprocess(im, output_type='pil')[0].save(path)


TAEF2_DENORM = os.environ.get('TAEF2_DENORM', '0') == '1'  # TAEF2 decodes the bn-normalised latents directly (err 0.03 vs 0.11)


@torch.no_grad()
def taef2_png(tk, path, px=256):
    lt = tokens_to_latent(tk)
    lat = pipe._unpatchify_latents(lt * std + mean if TAEF2_DENORM else lt).float()
    im = taef2_dec(lat).clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy()
    Image.fromarray((im * 255).astype(np.uint8)).resize((px, px), Image.LANCZOS).save(path)


t0 = time.time()
for s in range(args.steps):
    vae_png(xs[s], f'{OUT}/xt_s{s}.png')
    vae_png(xs[s] - sig[s] * vs[s], f'{OUT}/x0_s{s}.png')
    for b in range(NB):
        lens_v = S['tunedv'][(s, b)] if TUNED is not None else S['lensv'][(s, b)]
        taef2_png(xs[s] - sig[s] * lens_v, f'{OUT}/lens_s{s}_b{b:02d}.png')
        if TUNED is not None:
            taef2_png(xs[s] - sig[s] * S['lensv'][(s, b)], f'{OUT}/rawlens_s{s}_b{b:02d}.png', px=128)
print(f'decoded {args.steps * (NB + 2)} frames in {time.time()-t0:.1f}s', flush=True)

K = [(s, b) for s in range(args.steps) for b in range(NB)]
save('dit_pca', torch.stack([S['pca'][k] for k in K]).reshape(args.steps, NB, -1, 3))
save('dit_norm_img', torch.stack([S['nimg'][k] for k in K]).reshape(args.steps, NB, -1))
save('dit_norm_txt', torch.stack([S['ntxt'][k] for k in K]).reshape(args.steps, NB, -1))
save('dit_attn', torch.stack([S['attn'][k] for k in K]).reshape(args.steps, NB, hw * hw, n))
save('dit_attn_mass', torch.stack([S['mass'][k] for k in K]).reshape(args.steps, NB, 3))
save('ctx_norm', S['ctx'])
# the true final latent as the VAE decoder receives it, for scripts/decoder_stages.py
with torch.no_grad():
    fl = pipe._unpatchify_latents(tokens_to_latent(S['latents']) * std + mean).float()
np.save(f'{OUT}/final_latent.npy', fl.cpu().numpy())
manifest['act3'] = dict(blocks=NB, double_blocks=len(T.transformer_blocks), grid=hw, sigmas=sig, n_text_tokens=n, text_len=N_TXT)
json.dump(manifest, open(f'{OUT}/manifest.json', 'w'))
print('trace written to', OUT)
