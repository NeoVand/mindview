"""From-scratch float32 re-implementation of the painter, reading ONLY the exported files (plus the 1.7B tap features).

  painter_reference.py [--model ../static/models/bonsai-image-4b] [--truth data/painter_truth] [--out ../static/lab/painter_ref]

Follows research/PAINTER-SPEC.md op by op: ternary adapter -> DiT (5 double + 20 single blocks) x 4 Euler steps ->
TAEF2 decoder. No diffusers/transformers model classes: plain torch tensor ops, float32 on MPS. Compares every stage
against painter_truth.py (diffusers, fp32 and bf16) and writes browser-sized validation references to --out.
"""
import argparse, json, math, os, time
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument('--model', default='../static/models/bonsai-image-4b')
ap.add_argument('--truth', default='data/painter_truth')
ap.add_argument('--out', default='../static/lab/painter_ref')
args = ap.parse_args()
dev = 'mps'
M = json.load(open(f'{args.model}/manifest.json'))
C = M['config']
D, H, HD, MLP = C['hidden'], C['heads'], C['head_dim'], C['mlp_hidden']
EPS = C['eps']
N_TXT, N_IMG, GRID = M['text']['max_len'], M['image']['tokens'], M['image']['grid']
RAW = {f: np.memmap(f'{args.model}/{f}', dtype=np.uint8, mode='r') for f in M['files']}


# ------------------------------------------------------------------ loading (the manifest is the only index)
def dense(name):
    t = M['tensors'][name]
    a = np.frombuffer(RAW[t['file']], dtype={'f16': np.float16, 'f32': np.float32}[t['dtype']],
                      count=int(np.prod(t['shape'])), offset=t['offset']).reshape(t['shape'])
    return torch.from_numpy(a.astype(np.float32)).to(dev)


class Ternary:
    """Trits (int8) + f32 group scales on the GPU; the f32 weight is rebuilt per use: W[r,c] = trit * scale[r, c/128]."""

    def __init__(self, name):
        t = M['tensors'][name] if name in M['tensors'] else M['views'][name]
        self.rows, self.cols = t['shape']
        words = np.frombuffer(RAW[t['file']], dtype='<u4', count=self.rows * self.cols // 16, offset=t['codes']['offset'])
        words = words.reshape(self.rows, self.cols // 16, 1)
        q = (words >> (2 * np.arange(16, dtype=np.uint32))) & 3           # weight c = 16*word + j sits in bits 2j..2j+1
        self.trit = torch.from_numpy(q.reshape(self.rows, self.cols).astype(np.int8) - 1).to(dev)
        s = np.frombuffer(RAW[t['file']], dtype='<f4', count=self.rows * self.cols // 128, offset=t['scales']['offset'])
        self.scale = torch.from_numpy(s.reshape(self.rows, self.cols // 128).copy()).to(dev)

    def __call__(self, x):                                                  # x [n, cols] -> [n, rows]
        W = (self.trit.float().view(self.rows, -1, 128) * self.scale[..., None]).view(self.rows, self.cols)
        return x @ W.T


t0 = time.time()
DB = [{n: Ternary(f'transformer_blocks.{i}.{n}.weight') for n in
       ['attn.to_q', 'attn.to_k', 'attn.to_v', 'attn.add_q_proj', 'attn.add_k_proj', 'attn.add_v_proj', 'attn.to_out.0',
        'attn.to_add_out', 'ff.linear_in', 'ff.linear_out', 'ff_context.linear_in', 'ff_context.linear_out']}
      for i in range(C['double_blocks'])]
SB = [{n: Ternary(f'single_transformer_blocks.{i}.{n}.weight') for n in ['attn.to_qkv_mlp_proj', 'attn.to_out']}
      for i in range(C['single_blocks'])]
for i, b in enumerate(DB):
    for n in ['norm_q', 'norm_k', 'norm_added_q', 'norm_added_k']: b[n] = dense(f'transformer_blocks.{i}.attn.{n}.weight')
for i, b in enumerate(SB):
    for n in ['norm_q', 'norm_k']: b[n] = dense(f'single_transformer_blocks.{i}.attn.{n}.weight')
W_ctx, W_x, W_out = dense('context_embedder.weight'), dense('x_embedder.weight'), dense('proj_out.weight')
MOD = {k: dense(f'mod.{k}') for k in ['double_img', 'double_txt', 'single', 'norm_out']}   # [step, rows, 3072]
COS, SIN = dense('rope.cos'), dense('rope.sin')                                              # [1536, 64]
ADAPT, A_BIAS, A_PREFIX = Ternary('adapter.weight'), dense('adapter.bias'), dense('adapter.prefix')
SIG = torch.tensor(M['schedule']['sigmas'], dtype=torch.float32)
print(f'loaded exported weights in {time.time()-t0:.0f}s', flush=True)
_v = Ternary('transformer_blocks.0.attn.to_qkv')                   # the fused q|k|v view must equal the three parts
assert all(torch.equal(_v.trit[j * D:(j + 1) * D], DB[0][f'attn.to_{c}'].trit) and torch.equal(_v.scale[j * D:(j + 1) * D], DB[0][f'attn.to_{c}'].scale)
           for j, c in enumerate('qkv'))
del _v


# ------------------------------------------------------------------ ops
def layernorm(x):                                   # no affine, biased variance, eps 1e-6
    mu = x.mean(-1, keepdim=True)
    var = (x - mu).pow(2).mean(-1, keepdim=True)
    return (x - mu) * torch.rsqrt(var + EPS)


def modulate(x, shift, scale):
    return layernorm(x) * (1 + scale) + shift


def rmsnorm(x, w):                                  # over head_dim, learned gain, eps 1e-6
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS) * w


def rope(x):                                        # x [n, H, 128]; pairs (2p, 2p+1) rotate by angle[token, p]
    a, b = x[..., 0::2], x[..., 1::2]
    c, s = COS[:, None, :], SIN[:, None, :]
    return torch.stack([a * c - b * s, b * c + a * s], -1).flatten(-2)


def heads(x):
    return x.view(x.shape[0], H, HD)


def attention(q, k, v):                             # [n, H, 128] each, full (unmasked) softmax attention
    q, k, v = (t.transpose(0, 1) for t in (q, k, v))
    p = torch.softmax(q @ k.transpose(1, 2) / math.sqrt(HD), -1)
    return (p @ v).transpose(0, 1).reshape(q.shape[1], H * HD)


def swiglu(h):                                      # first half gates the second half
    g, u = h.chunk(2, -1)
    return F.silu(g) * u


PROBE_ROWS = torch.cat([torch.arange(8), N_TXT + torch.arange(8)])   # first 8 text + first 8 image tokens (joint order)


def double_block(b, c, x, mi, mt, probe=None):
    """c: text [512, D], x: image [1024, D]; mi/mt: [6, D] = shift, scale, gate (attn) then shift, scale, gate (mlp)."""
    xn, cn = modulate(x, mi[0], mi[1]), modulate(c, mt[0], mt[1])
    q = torch.cat([rmsnorm(heads(b['attn.add_q_proj'](cn)), b['norm_added_q']), rmsnorm(heads(b['attn.to_q'](xn)), b['norm_q'])])
    k = torch.cat([rmsnorm(heads(b['attn.add_k_proj'](cn)), b['norm_added_k']), rmsnorm(heads(b['attn.to_k'](xn)), b['norm_k'])])
    v = torch.cat([heads(b['attn.add_v_proj'](cn)), heads(b['attn.to_v'](xn))])
    q, k = rope(q), rope(k)
    o = attention(q, k, v)
    x = x + mi[2] * b['attn.to_out.0'](o[N_TXT:])
    c = c + mt[2] * b['attn.to_add_out'](o[:N_TXT])
    fx = swiglu(b['ff.linear_in'](modulate(x, mi[3], mi[4])))
    fc = swiglu(b['ff_context.linear_in'](modulate(c, mt[3], mt[4])))
    if probe is not None:
        j = lambda t: t[PROBE_ROWS].reshape(len(PROBE_ROWS), -1)
        probe.update(norm1=j(torch.cat([cn, xn])), q=j(q), k=j(k), v=j(v), attn=j(o), resid_attn=j(torch.cat([c, x])),
                     ff_act=j(torch.cat([fc, fx])))
    x = x + mi[5] * b['ff.linear_out'](fx)
    c = c + mt[5] * b['ff_context.linear_out'](fc)
    return c, x


def single_block(b, h, m, probe=None):
    """h: joint [1536, D] (text first); m: [3, D] = shift, scale, gate. Attention and MLP run in parallel."""
    p = b['attn.to_qkv_mlp_proj'](modulate(h, m[0], m[1]))               # [n, 3*3072 + 2*9216]
    q = rope(rmsnorm(heads(p[:, :D]), b['norm_q']))
    k = rope(rmsnorm(heads(p[:, D:2 * D]), b['norm_k']))
    v = heads(p[:, 2 * D:3 * D])
    o = attention(q, k, v)
    a = swiglu(p[:, 3 * D:])                                              # gate = [9216:18432], up = [18432:27648]
    if probe is not None:
        j = lambda t: t[PROBE_ROWS].reshape(len(PROBE_ROWS), -1)
        probe.update(norm=j(modulate(h, m[0], m[1])), q=j(q), k=j(k), v=j(v), attn=j(o), mlp_act=j(a))
    return h + m[2] * b['attn.to_out'](torch.cat([o, a], -1))


def dit(lat, ctx, s, record=None, probes=None):
    """One velocity evaluation. lat [1024, 128] bn-normalised tokens, ctx [512, 7680]; s = step index."""
    x, c = lat @ W_x.T, ctx @ W_ctx.T
    if record is not None: record.append(torch.cat([c, x]))
    for i, b in enumerate(DB):
        c, x = double_block(b, c, x, MOD['double_img'][s], MOD['double_txt'][s], probes.setdefault(f'double{i}', {}) if probes is not None and i == 0 else None)
        if record is not None: record.append(torch.cat([c, x]))
    h = torch.cat([c, x])
    for i, b in enumerate(SB):
        h = single_block(b, h, MOD['single'][s], probes.setdefault(f'single{i}', {}) if probes is not None and i == 0 else None)
        if record is not None: record.append(h)
    no = MOD['norm_out'][s]                                               # rows: scale, shift
    return (layernorm(h[N_TXT:]) * (1 + no[0]) + no[1]) @ W_out.T         # velocity [1024, 128]


# ------------------------------------------------------------------ TAEF2 decoder from the manifest graph
def taef2(lat_tokens, stages=None):
    z = lat_tokens.view(GRID, GRID, 32, 2, 2).permute(2, 0, 3, 1, 4).reshape(1, 32, 2 * GRID, 2 * GRID)  # unpatchify
    if stages is not None: stages['input'] = z[0]
    T = {k: dense(k) for k in M['tensors'] if k.startswith('taef2.')}
    conv = lambda x, w, b=None: F.conv2d(x, T[w], T[b] if b else None, padding=1)

    def gn(x, pre, groups, eps):
        n, ch, hh, ww = x.shape
        g = x.view(n, groups, -1)
        mu, var = g.mean(-1, keepdim=True), g.var(-1, unbiased=False, keepdim=True)
        return ((g - mu) * torch.rsqrt(var + eps)).view(n, ch, hh, ww) * T[pre + '.weight'][:, None, None] + T[pre + '.bias'][:, None, None]

    x = z
    for op in M['taef2']['graph']:
        if op['op'] == 'tanh_clamp': x = torch.tanh(x / 3) * 3
        elif op['op'] == 'conv3x3':
            if op['cout'] == 3 and stages is not None: stages['res512'] = x[0]   # last feature map before the RGB conv
            x = conv(x, op['w'], op['b'])
        elif op['op'] == 'relu': x = F.relu(x)
        elif op['op'] == 'upsample_nearest_2x':
            if stages is not None: stages[f'res{x.shape[-1]}'] = x[0]      # feature map at 64, 128, 256 before upsampling
            x = x.repeat_interleave(2, -2).repeat_interleave(2, -1)
        elif op['op'] == 'block':
            if 'pool' in op:                                              # x += conv1x1(relu(GN(conv1x1(x))))
                p = op['pool']
                y = F.conv2d(x, T[p['conv_in']])
                y = F.relu(gn(y, p['gn'], p['groups'], p['eps']))
                x = x + F.conv2d(y, T[p['conv_out']])
            y = F.relu(conv(x, op['conv'][0] + '.weight', op['conv'][0] + '.bias'))
            y = F.relu(conv(y, op['conv'][1] + '.weight', op['conv'][1] + '.bias'))
            x = F.relu(conv(y, op['conv'][2] + '.weight', op['conv'][2] + '.bias') + x)
        elif op['op'] == 'clamp01':
            if stages is not None: stages['rgb_raw'] = x[0]
            x = x.clamp(0, 1)
    return x[0]


# ================================================================== run
TR = torch.load(f'{args.truth}/text.pt')
G32 = torch.load(f'{args.truth}/dit_fp32.pt')
G16 = torch.load(f'{args.truth}/dit_bf16.pt') if os.path.exists(f'{args.truth}/dit_bf16.pt') else None
cos = lambda a, b: F.cosine_similarity(a.flatten().cpu().double(), b.flatten().cpu().double(), 0).item()
tokcos = lambda a, b: F.cosine_similarity(a.cpu().double(), b.cpu().double(), -1).min().item()
psnr = lambda a, b: 10 * math.log10(1 / ((a.cpu().double() - b.cpu().double()) ** 2).mean().item())

with torch.no_grad():
    taps = TR['taps'].to(dev)                                             # [512, 6144] = h7 | h14 | h21
    ctx = ADAPT(taps) + A_BIAS
    ctx[:3] = A_PREFIX
    res = dict(adapter=dict(cos=cos(ctx, G32['cond']), max_abs=(ctx.cpu() - G32['cond']).abs().max().item()))
    lat = G32['noise'][0].reshape(128, -1).T.contiguous().to(dev)         # [1024, 128]: token h*32+w, channel c
    noise = lat.clone()
    rec, probes, vel, lats = [], {}, [], []
    t0 = time.time()
    for s in range(M['schedule']['steps']):
        v = dit(lat, ctx, s, rec if s == 0 else None, probes if s == 0 else None)
        lat = lat + (SIG[s + 1] - SIG[s]).item() * v
        vel.append(v); lats.append(lat)
        print(f'step {s+1}: {time.time()-t0:.1f}s  velocity cos vs diffusers fp32 {cos(v, G32["vel"][s]):.6f}', flush=True)
    stages = {}
    img = taef2(lat, stages)

# ------------------------------------------------------------------ compare with diffusers
blocks = torch.stack([r.cpu() for r in rec])                             # [26, 1536, 3072]: input, then 25 blocks
truth = {'fp32': G32, 'bf16': G16} if G16 is not None else {'fp32': G32}
for k, G in truth.items():
    tb = torch.cat([torch.cat([G['in_txt'], G['in_img']])[None], torch.cat([G['blk_txt'], G['blk_img']], 1)])
    res[f'step1_blocks_vs_{k}'] = [dict(block=i - 1, txt_cos=cos(blocks[i, :N_TXT], tb[i, :N_TXT]), img_cos=cos(blocks[i, N_TXT:], tb[i, N_TXT:]),
                                        min_token_cos=tokcos(blocks[i], tb[i])) for i in range(len(blocks))]
    res[f'velocity_cos_vs_{k}'] = [cos(vel[s], G['vel'][s]) for s in range(len(vel))]
    res[f'final_latent_cos_vs_{k}'] = cos(lat, G['final'])
    res[f'image_psnr_vs_{k}_taef2'] = psnr(img, G['img_taef2'])
    res[f'image_psnr_vs_{k}_vae'] = psnr(img, G['img_vae'])
res['diffusers_fp32_taef2_vs_vae_psnr'] = psnr(G32['img_taef2'], G32['img_vae'])
res['temb_max_abs_vs_fp32'] = (dense('temb').cpu() - G32['temb']).abs().max().item()
res['mod_img_max_abs_vs_fp32'] = (MOD['double_img'].cpu().reshape(4, -1) - G32['mod_img']).abs().max().item()
res['mod_img_absmax'] = MOD['double_img'].abs().max().item()
for k in [k for k in res if k.startswith('step1_blocks')]:
    print(k); [print(f"  block {r['block']:3d}  txt {r['txt_cos']:.7f}  img {r['img_cos']:.7f}  min-token {r['min_token_cos']:.6f}") for r in res[k]]
print(json.dumps({k: v for k, v in res.items() if not k.startswith('step1_blocks')}, indent=1))

# ------------------------------------------------------------------ browser-sized references
O = args.out
os.makedirs(O, exist_ok=True)
arrays = {}


def save(name, t, desc):
    a = np.ascontiguousarray(t.detach().float().cpu().numpy().astype('<f4'))
    a.tofile(f'{O}/{name}.bin')
    arrays[name] = dict(file=f'{name}.bin', dtype='f32', shape=list(a.shape), desc=desc)


def rowstats(t):                                                          # per-row checksums [n, 3]: sum, sum of squares, absmax
    t = t.cpu().double()
    return torch.stack([t.sum(-1), (t * t).sum(-1), t.abs().amax(-1)], -1)


n = TR['n_real']
ROWS = list(range(n + 3)) + [N_TXT - 1]                                  # all real tokens, 3 pads, the last pad
save('taps_rows', TR['taps'][ROWS], f'1.7B taps h7|h14|h21 for rows {ROWS[0]}..{n+2} and {N_TXT-1}; [rows, 6144]')
save('taps_stats', rowstats(TR['taps']), 'per-row [sum, sum of squares, absmax] of the taps, all 512 rows')
save('cond_rows', ctx[ROWS], 'adapter output (the DiT conditioning) for the same rows; [rows, 7680]')
save('cond_stats', rowstats(ctx), 'per-row [sum, sum of squares, absmax] of the conditioning, all 512 rows')
save('noise', noise, 'initial latent x_0: [1024 tokens (h*32+w), 128 channels], bn-normalised space')
save('step1_blocks', blocks[:, PROBE_ROWS], 'step 1 residual stream: [26 = embedder output + 25 blocks, 16 rows = text 0..7 + image 0..7, 3072]')
save('step1_block_norms', blocks.norm(dim=-1), 'step 1 L2 norm of every token after each stage: [26, 1536] (joint order)')
for blk, P in probes.items():
    for key, t in P.items():
        save(f'step1_{blk}_{key}', t, f'step 1, {blk} internal "{key}" for the 16 probe rows (heads flattened h*128+d)')
save('velocity', torch.stack(vel), 'velocity v_s = DiT output per step: [4, 1024, 128]')
save('latents', torch.stack(lats), 'latent after each Euler update: [4, 1024, 128]; latents[3] is the final latent')
for k, t in stages.items():
    save(f'taef2_{k}_crop', t[:, :8, :8], f'TAEF2 stage "{k}": top-left 8x8 crop, all channels [{t.shape[0]}, 8, 8]')
    save(f'taef2_{k}_chstats', torch.stack([t.mean((1, 2)), t.std((1, 2), unbiased=False)], -1), f'TAEF2 stage "{k}": per-channel [mean, std]')
Image.fromarray((img.permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)).save(f'{O}/final.png')
Image.fromarray((G32['img_taef2'].permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)).save(f'{O}/diffusers_fp32_taef2.png')
Image.fromarray((G32['img_vae'].permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)).save(f'{O}/diffusers_fp32_vae.png')
json.dump(dict(prompt=TR['prompt'], text=TR['text'], ids=TR['ids'].tolist(), n_real=n, rows=ROWS, probe_rows=PROBE_ROWS.tolist(),
               schedule=M['schedule'], model_manifest='/models/bonsai-image-4b/manifest.json', arrays=arrays,
               metrics=res), open(f'{O}/ref.json', 'w'), indent=1)
tot = sum(os.path.getsize(f'{O}/{f}') for f in os.listdir(O))
print(f'wrote {len(arrays)} arrays + 3 PNGs to {O} ({tot/1e6:.1f} MB)')
