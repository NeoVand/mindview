"""Generate with Bonsai Image 4B (FLUX.2 klein, ternary/binary DiT) and capture its internals.

Per denoising step we record:
  - the latent x_t and the model's x0 prediction (decoded to RGB)
  - a "block lens": every one of the 25 blocks' image-token stream pushed through the final
    norm_out + proj_out, i.e. "what image would come out if the network stopped here"
  - PCA(3) of every block's image-token features (the model's own colour-space)
  - attention from image patches to each prompt token, for every single-stream block

Usage: image_capture.py <variant ternary|binary> "<prompt>" [size=512] [steps=4] [seed=7] [tag]
"""
import sys, glob, json, time, gc, os, hashlib
import numpy as np
import torch
from PIL import Image
from diffusers import Flux2KleinPipeline
from diffusers.models.transformers.transformer_flux2 import Flux2ParallelSelfAttnProcessor, apply_rotary_emb, dispatch_attention_fn

variant = sys.argv[1] if len(sys.argv) > 1 else 'ternary'
prompt = sys.argv[2] if len(sys.argv) > 2 else 'a bonsai tree made of glowing circuitry, in a dark museum, volumetric light'
size = int(sys.argv[3]) if len(sys.argv) > 3 else 512
steps = int(sys.argv[4]) if len(sys.argv) > 4 else 4
seed = int(sys.argv[5]) if len(sys.argv) > 5 else 7
tag = sys.argv[6] if len(sys.argv) > 6 else f"{variant}_{os.environ.get('ENCODER', 'qwen')}_{size}_s{steps}_seed{seed}"
out_dir = f'renders/image/{tag}'
os.makedirs(out_dir, exist_ok=True)

root = glob.glob('/Users/neo/.cache/huggingface/hub/models--prism-ml--bonsai-image-ternary-4B-unpacked/snapshots/*/')[0]
dev, dt = 'mps', torch.bfloat16
encoder = os.environ.get('ENCODER', 'qwen')
key = hashlib.sha1(f'{encoder}|{prompt}'.encode()).hexdigest()[:12]
cache = f'data/prompt_cache/{key}.pt'
if not os.path.exists(cache):
    raise SystemExit(f'no cached embeds for this prompt; run: scripts/image_encode_prompts.py --encoder {encoder} "{prompt}"')
prompt_embeds = torch.load(cache)['embeds'].to(dev, dt)
if variant == 'binary':
    from diffusers import Flux2Transformer2DModel
    broot = glob.glob('/Users/neo/.cache/huggingface/hub/models--prism-ml--bonsai-image-binary-4B-unpacked/snapshots/*/')[0]
    pipe = Flux2KleinPipeline.from_pretrained(root, text_encoder=None, torch_dtype=dt,
                                              transformer=Flux2Transformer2DModel.from_pretrained(broot, subfolder='transformer', torch_dtype=dt))
else:
    pipe = Flux2KleinPipeline.from_pretrained(root, text_encoder=None, torch_dtype=dt)

# which of the 512 text positions are real prompt tokens (rest is padding)
tokz = pipe.tokenizer
templ = tokz.apply_chat_template([{'role': 'user', 'content': prompt}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
tok_ids = tokz(templ).input_ids
tok_strs = [tokz.decode([i]) for i in tok_ids]
n_txt_real = len(tok_ids)
print('prompt tokens:', n_txt_real, tok_strs)

pipe.transformer.to(dev); pipe.vae.to(dev)
T = pipe.transformer
N_TXT = prompt_embeds.shape[1]

# ---------------------------------------------------------------- capture state
cap = dict(step=0, temb=None)
lens_v = {}        # (step, block) -> v prediction from block k  [1, N_img, 128]
pca_rgb = {}       # (step, block) -> [N_img, 3]
attn_maps = {}     # (step, single_block) -> [N_img, n_txt_real]
norms = {}         # (step, block) -> mean token norm


def lens(h_img):
    h = T.norm_out(h_img, cap['temb'])
    return T.proj_out(h)


def pca3(h_img):
    # PCA via the 3072x3072 covariance: matmul on GPU, eigendecomposition on CPU (MPS SVD hangs)
    x = h_img[0].float()
    x = x - x.mean(0, keepdim=True)
    C = (x.T @ x).cpu().double()
    evals, evecs = torch.linalg.eigh(C)
    V = evecs[:, -3:].flip(-1).float().to(x.device)
    return (x @ V).cpu().numpy()


def after_block(name, idx, single):
    def fn(mod, args, out):
        if single:
            h_img = out[:, N_TXT:]
        else:
            h_img = out[1]
        key = (cap['step'], idx)
        with torch.no_grad():
            lens_v[key] = lens(h_img).float().cpu()
            pca_rgb[key] = pca3(h_img)
            norms[key] = float(h_img.float().norm(dim=-1).mean())
    return fn


T.time_guidance_embed.register_forward_hook(lambda m, a, o: cap.__setitem__('temb', o))
for i, b in enumerate(T.transformer_blocks):
    b.register_forward_hook(after_block('double', i, False))
for i, b in enumerate(T.single_transformer_blocks):
    b.register_forward_hook(after_block('single', i + len(T.transformer_blocks), True))


class CapturingParallelProcessor(Flux2ParallelSelfAttnProcessor):
    """Same math as the stock processor, plus mean-over-heads attention image->prompt tokens."""

    def __init__(self, idx):
        super().__init__(); self.idx = idx

    def __call__(self, attn, hidden_states, attention_mask=None, image_rotary_emb=None):
        hidden_states = attn.to_qkv_mlp_proj(hidden_states)
        qkv_dim = 3 * attn.inner_dim
        mlp_dim = attn.mlp_hidden_dim * attn.mlp_mult_factor
        local_qkv = hidden_states.shape[-1] * qkv_dim // (qkv_dim + mlp_dim)
        qkv, mlp_hidden_states = torch.split(hidden_states, [local_qkv, hidden_states.shape[-1] - local_qkv], dim=-1)
        query, key, value = qkv.chunk(3, dim=-1)
        query = attn.norm_q(query.unflatten(-1, (-1, attn.head_dim)))
        key = attn.norm_k(key.unflatten(-1, (-1, attn.head_dim)))
        value = value.unflatten(-1, (-1, attn.head_dim))
        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)
            key = apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)
        with torch.no_grad():
            q = query[:, N_TXT:].transpose(1, 2).float()          # [B,H,Nimg,D]
            k = key.transpose(1, 2).float()                        # [B,H,Ntot,D]
            p = torch.softmax(q @ k.transpose(-1, -2) / q.shape[-1] ** 0.5, dim=-1)
            attn_maps[(cap['step'], self.idx)] = p[0, :, :, :n_txt_real].mean(0).cpu().numpy()
            if self.idx == len(T.transformer_blocks):
                # how much attention goes to real prompt vs padding vs image, averaged
                cap.setdefault('mass', []).append([float(p[..., :n_txt_real].sum(-1).mean()),
                                                   float(p[..., n_txt_real:N_TXT].sum(-1).mean()),
                                                   float(p[..., N_TXT:].sum(-1).mean())])
            del p, q, k
        hidden_states = dispatch_attention_fn(query, key, value, attn_mask=attention_mask)
        hidden_states = hidden_states.flatten(2, 3).to(query.dtype)
        mlp_hidden_states = attn.mlp_act_fn(mlp_hidden_states)
        return attn.to_out(torch.cat([hidden_states, mlp_hidden_states], dim=-1))


for i, b in enumerate(T.single_transformer_blocks):
    b.attn.set_processor(CapturingParallelProcessor(i + len(T.transformer_blocks)))

xt_hist, v_hist, sig_hist = [], [], []
T.register_forward_hook(lambda m, a, o: v_hist.append(o[0].float().cpu()))
T.register_forward_pre_hook(lambda m, a, kw: xt_hist.append(kw['hidden_states'].float().cpu()), with_kwargs=True)


def on_step(p, i, t, kw):
    sig_hist.append(float(p.scheduler.sigmas[i]))
    cap['step'] = i + 1
    return kw


# ---------------------------------------------------------------- generate
t0 = time.time()
with torch.no_grad():
    img = pipe(prompt_embeds=prompt_embeds, height=size, width=size, num_inference_steps=steps,
               generator=torch.Generator('cpu').manual_seed(seed), callback_on_step_end=on_step).images[0]
gen_s = time.time() - t0
img.save(f'{out_dir}/final.png')
print(f'generated {size}x{size} in {gen_s:.1f}s ({gen_s/steps:.2f}s/step incl. capture + VAE)')


# ---------------------------------------------------------------- decode x0 predictions
hw = size // 16  # tokens per side


@torch.no_grad()
def decode_tokens(x0_tokens):
    """[1, N, 128] packed latent tokens -> PIL"""
    lat = x0_tokens.to(dev, dt).reshape(1, hw, hw, 128).permute(0, 3, 1, 2)
    mean = pipe.vae.bn.running_mean.view(1, -1, 1, 1).to(dev, dt)
    std = torch.sqrt(pipe.vae.bn.running_var.view(1, -1, 1, 1) + pipe.vae.config.batch_norm_eps).to(dev, dt)
    lat = pipe._unpatchify_latents(lat * std + mean)
    im = pipe.vae.decode(lat, return_dict=False)[0]
    return pipe.image_processor.postprocess(im, output_type='pil')[0]


def rgb_from_pca(p, ref=None):
    p = p.copy()
    lo, hi = np.percentile(p, 1, 0), np.percentile(p, 99, 0)
    p = np.clip((p - lo) / (hi - lo + 1e-6), 0, 1)
    return Image.fromarray((p.reshape(hw, hw, 3) * 255).astype(np.uint8)).resize((size, size), Image.NEAREST)


nb = len(T.transformer_blocks) + len(T.single_transformer_blocks)
t0 = time.time()
sheet_lens, sheet_pca = [], []
for s in range(steps):
    xt = xt_hist[s]
    sig = sig_hist[s]
    decode_tokens(xt).save(f'{out_dir}/step{s}_xt.png')
    decode_tokens(xt - sig * v_hist[s]).save(f'{out_dir}/step{s}_x0.png')
    row_l, row_p = [], []
    for b in range(nb):
        x0_b = xt - sig * lens_v[(s, b)]
        im = decode_tokens(x0_b).resize((size // 4, size // 4), Image.LANCZOS)
        row_l.append(im)
        row_p.append(rgb_from_pca(pca_rgb[(s, b)]).resize((size // 4, size // 4), Image.NEAREST))
    sheet_lens.append(row_l); sheet_pca.append(row_p)
print(f'decoded {steps*(nb+2)} lens images in {time.time()-t0:.1f}s')


def contact(rows, name):
    w, h = rows[0][0].size
    sheet = Image.new('RGB', (w * len(rows[0]), h * len(rows)))
    for r, row in enumerate(rows):
        for c, im in enumerate(row):
            sheet.paste(im, (c * w, r * h))
    sheet.save(f'{out_dir}/{name}.png')


contact(sheet_lens, 'sheet_block_lens')   # rows = steps, cols = blocks 0..24
contact(sheet_pca, 'sheet_block_pca')

# attention: for the last step, per prompt token, averaged over single blocks
s = steps - 1
A = np.mean([attn_maps[(s, b)] for b in range(len(T.transformer_blocks), nb)], 0)  # [Nimg, n_txt]
tiles = []
for j, ts in enumerate(tok_strs):
    m = A[:, j].reshape(hw, hw)
    m = (m - m.min()) / (m.max() - m.min() + 1e-9)
    tiles.append(Image.fromarray((m * 255).astype(np.uint8)).resize((128, 128), Image.BILINEAR).convert('RGB'))
contact([tiles[i:i + 8] for i in range(0, len(tiles) - len(tiles) % 8, 8)] or [tiles], 'attn_tokens_last_step')

json.dump(dict(prompt=prompt, variant=variant, size=size, steps=steps, seed=seed, tokens=tok_strs,
               sigmas=sig_hist, gen_seconds=gen_s, attention_mass_real_pad_img=cap.get('mass'),
               block_norms={f'{k[0]}_{k[1]}': v for k, v in norms.items()}),
          open(f'{out_dir}/meta.json', 'w'), indent=1)
np.savez_compressed(f'{out_dir}/capture.npz', attn=np.stack([attn_maps[(s, b)] for s in range(steps) for b in range(len(T.transformer_blocks), nb)]),
                    pca=np.stack([pca_rgb[(s, b)] for s in range(steps) for b in range(nb)]))
print('saved to', out_dir)
