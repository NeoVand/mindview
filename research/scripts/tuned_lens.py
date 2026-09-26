"""Tuned lens for the Bonsai Image DiT: one affine readout per block, fitted across many generations.

The raw lens (block state -> norm_out -> proj_out) decodes to noise until the last few blocks, because
mid-network states are not yet expressed in the output basis. A tuned lens learns, per block b, the affine map
    h_b (3072, image token)  ->  v (128, the velocity this step finally outputs)
with ridge regression over many prompts, steps and tokens. Decoding x0 = x_t - sigma * v_hat then shows what each
block already "knows" about the picture, in pixel terms, without mixing in anything the network did not compute.

  tuned_lens.py fit    [encoder=bonsai17t_adapt] [lambda]   # collect stats over data/prompts/lens_train.txt and solve
  tuned_lens.py render "<prompt>" [encoder]                  # contact sheet: raw lens vs tuned lens, steps x blocks
"""
import sys, glob, os, hashlib, time
import numpy as np
import torch
from PIL import Image
from diffusers import Flux2KleinPipeline

sys.path.insert(0, 'scripts')
from taesd import Decoder
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

dev, dt = 'mps', torch.bfloat16
ROOT = glob.glob('/Users/neo/.cache/huggingface/hub/models--prism-ml--bonsai-image-ternary-4B-unpacked/snapshots/*/')[0]
OUT = 'data/tuned_lens'
os.makedirs(OUT, exist_ok=True)
mode = sys.argv[1]
encoder = 'bonsai17t_adapt'
SIZE, STEPS, N_TXT = 512, 4, 512

pipe = Flux2KleinPipeline.from_pretrained(ROOT, text_encoder=None, torch_dtype=dt).to(dev)
pipe.set_progress_bar_config(disable=True)
T = pipe.transformer
blocks = list(T.transformer_blocks) + list(T.single_transformer_blocks)
NB = len(blocks)
D = T.inner_dim


def embeds(prompt, enc):
    key = hashlib.sha1(f'{enc}|{prompt}'.encode()).hexdigest()[:12]
    return torch.load(f'data/prompt_cache/{key}.pt')['embeds'].to(dev, dt)


hs = []  # per forward: image-token state after each block


def hook(i, single):
    def fn(mod, a, kw, o):
        h = o[:, N_TXT:] if single else o[1]
        hs.append(h[0].detach())
    return fn


for i, b in enumerate(blocks):
    b.register_forward_hook(hook(i, i >= len(T.transformer_blocks)), with_kwargs=True)

if mode == 'fit':
    if len(sys.argv) > 2: encoder = sys.argv[2]
    lam = float(sys.argv[3]) if len(sys.argv) > 3 else 1e-3
    n_prompts = int(os.environ.get('LENS_PROMPTS', '48'))
    prompts = [l.strip() for l in open('data/prompts/lens_train.txt') if l.strip()][:n_prompts]
    XtX = torch.zeros(NB, D + 1, D + 1, device=dev)
    XtY = torch.zeros(NB, D + 1, 128, device=dev)
    n = torch.zeros(NB, device=dev)

    def on_out(m, a, o):
        v = o[0][0].float()                                       # [N, 128] what this step finally outputs
        for b, h in enumerate(hs):
            X = torch.cat([h.float(), torch.ones(h.shape[0], 1, device=dev)], 1)
            XtX[b] += X.T @ X
            XtY[b] += X.T @ v
            n[b] += X.shape[0]
        hs.clear()

    handle = T.register_forward_hook(on_out)
    t0 = time.time()
    for k, p in enumerate(prompts):
        with torch.no_grad():
            pipe(prompt_embeds=embeds(p, encoder), height=SIZE, width=SIZE, num_inference_steps=STEPS,
                 generator=torch.Generator('cpu').manual_seed(1000 + k), output_type='latent')
        torch.mps.synchronize()
        print(f'{k + 1}/{len(prompts)} prompts, {time.time()-t0:.0f}s', flush=True)
    handle.remove()
    # keep the expensive statistics on disk so a failed solve never costs another collection run
    torch.save(dict(XtX=XtX.cpu(), XtY=XtY.cpu(), n=n.cpu()), f'{OUT}/stats.pt')
    W = torch.zeros(NB, D + 1, 128)
    for b in range(NB):
        A = XtX[b].cpu().double() / n[b].item()   # MPS has no float64: move to CPU first
        B = XtY[b].cpu().double() / n[b].item()
        mx = A[:D, D]
        C = A[:D, :D] - torch.outer(mx, mx)
        my = B[D]
        Cxy = B[:D] - torch.outer(mx, my)
        Wb = torch.linalg.solve(C + lam * torch.diag(torch.diag(C)).clamp_min(1e-8), Cxy)
        W[b, :D] = Wb.float()
        W[b, D] = (my - mx @ Wb).float()
    torch.save(dict(W=W.half(), encoder=encoder, lam=lam, prompts=len(prompts)), f'{OUT}/lens.pt')
    print('saved', f'{OUT}/lens.pt', W.shape)

elif mode == 'render':
    prompt = sys.argv[2]
    if len(sys.argv) > 3: encoder = sys.argv[3]
    W = torch.load(f'{OUT}/lens.pt')['W'].to(dev).float()
    sd = {'.'.join([str(int(k.split('.')[2]) + 1)] + k.split('.')[3:]): v
          for k, v in load_file(hf_hub_download('madebyollin/taef2', 'taef2.safetensors')).items() if k.startswith('decoder.layers.')}
    taef2 = Decoder(32, use_midblock_gn=True); taef2.load_state_dict(sd); taef2 = taef2.to(dev).eval()
    per_step, xs, sig = [], [], []

    def on_out(m, a, o):
        per_step.append([h.float() for h in hs]); hs.clear()

    T.register_forward_hook(on_out)
    T.register_forward_pre_hook(lambda m, a, kw: xs.append(kw['hidden_states'][0].float()), with_kwargs=True)
    with torch.no_grad():
        img = pipe(prompt_embeds=embeds(prompt, encoder), height=SIZE, width=SIZE, num_inference_steps=STEPS,
                   generator=torch.Generator('cpu').manual_seed(7),
                   callback_on_step_end=lambda p, i, t, kw: (sig.append(float(p.scheduler.sigmas[i])), kw)[1]).images[0]
    hw = SIZE // 16

    @torch.no_grad()
    def dec(tokens):
        lat = pipe._unpatchify_latents(tokens.to(dt).reshape(1, hw, hw, 128).permute(0, 3, 1, 2)).float()
        im = taef2(lat).clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy()
        return Image.fromarray((im * 255).astype(np.uint8)).resize((128, 128), Image.LANCZOS)

    rows_raw, rows_tuned = [], []
    with torch.no_grad():
        for s in range(STEPS):
            rr, rt = [], []
            for b in range(NB):
                h = per_step[s][b]
                temb = T.time_guidance_embed(torch.tensor([sig[s]], device=dev, dtype=dt), None)
                v_raw = T.proj_out(T.norm_out(h[None].to(dt), temb))[0].float()
                v_tuned = torch.cat([h, torch.ones(h.shape[0], 1, device=dev)], 1) @ W[b]
                rr.append(dec(xs[s] - sig[s] * v_raw)); rt.append(dec(xs[s] - sig[s] * v_tuned))
            rows_raw.append(rr); rows_tuned.append(rt)
    sheet = Image.new('RGB', (128 * NB, 128 * STEPS * 2 + 12), (20, 20, 20))
    for s in range(STEPS):
        for b in range(NB):
            sheet.paste(rows_raw[s][b], (b * 128, s * 128))
            sheet.paste(rows_tuned[s][b], (b * 128, STEPS * 128 + 12 + s * 128))
    name = f'renders/image/tuned_lens_{hashlib.sha1(prompt.encode()).hexdigest()[:6]}.png'
    sheet.save(name)
    img.save(name.replace('.png', '_final.png'))
    print('top half: raw lens, bottom half: tuned lens ->', name)
