"""Export the painter for the WebGPU runtime: ternary adapter + Bonsai Image 4B DiT + TAEF2 decoder.

  export_painter.py [--size 512] [--steps 4]

Writes ../static/models/bonsai-image-4b/ (git-ignored): manifest.json plus a few little-endian .bin files. Every tensor
starts on a 256-byte boundary, so a file can be uploaded as one GPU buffer and each tensor bound as a sub-range.
  dit_<i>.bin   ternary DiT matrices, block by block; each shard <= 256 MiB (the default WebGPU maxBufferSize)
  dit_misc.bin  fp16 dense weights used per prompt (context/x embedders, proj_out, QK-norm gains), plus f32 tables that
                depend only on the step (modulation vectors) or on the resolution (RoPE cos/sin), and the VAE bn stats
  adapter.bin   ternary adapter (Bonsai 1.7B taps 7|14|21 -> 7680), f32 bias, f32 chat-template prefix rows
  taef2.bin     TAEF2 decoder (madebyollin/taef2), fp16

Ternary layout (identical to the 1.7B runtime): a [rows, cols] matrix (cols % 128 == 0) is
  codes  u32[rows * cols/16]: weight (r, c) in word r*(cols/16) + c/16, bits 2*(c%16)..+1, holding q = trit + 1
  scales f32[rows * cols/128]: one per 128 inputs, index r*(cols/128) + c/128;  weight = (q - 1) * scale
Codes and scales of one matrix live in the same file. The spec is research/PAINTER-SPEC.md.
"""
import argparse, glob, hashlib, json, math, os, re, time
import numpy as np
import torch
from safetensors import safe_open

ap = argparse.ArgumentParser()
ap.add_argument('--size', type=int, default=512)
ap.add_argument('--steps', type=int, default=4)
ap.add_argument('--out', default='../static/models/bonsai-image-4b')
args = ap.parse_args()

ROOT = glob.glob('/Users/neo/.cache/huggingface/hub/models--prism-ml--bonsai-image-ternary-4B-unpacked/snapshots/*/')[0]
TAEF2 = glob.glob('/Users/neo/.cache/huggingface/hub/models--madebyollin--taef2/snapshots/*/taef2.safetensors')[0]
ADAPTER, QEXTRA = 'data/adapter/adapter_bonsai17t.pt', 'data/adapter/qwen_extra.pt'
OUT = args.out
os.makedirs(OUT, exist_ok=True)
ALIGN = 256
SHARD_MAX = 256 * 1024 * 1024
cfg = json.load(open(ROOT + 'transformer/config.json'))
D, H, HD = 3072, cfg['num_attention_heads'], cfg['attention_head_dim']
MLP = int(D * cfg['mlp_ratio'])                                   # 9216
GRID = args.size // 16                                            # 32 latent tokens per side at 512^2
N_IMG, N_TXT = GRID * GRID, 512
DT = safe_open(ROOT + 'transformer/diffusion_pytorch_model.safetensors', 'pt')
report = dict(ternary={}, dense={})


# ------------------------------------------------------------------ file writer with 256-byte alignment
class Bins:
    def __init__(self):
        self.files, self.tensors, self.digests = {}, {}, {}

    def put(self, fname, data):
        """Append raw bytes to a file at the next aligned offset; return (offset, nbytes)."""
        if fname not in self.files:              # (not setdefault: that would re-open, i.e. truncate, the file every call)
            self.files[fname] = dict(fh=open(f'{OUT}/{fname}', 'wb'), size=0, sha=hashlib.sha256())
        f = self.files[fname]
        pad = (-f['size']) % ALIGN
        if pad:
            f['fh'].write(b'\0' * pad); f['sha'].update(b'\0' * pad); f['size'] += pad
        b = data.tobytes() if hasattr(data, 'tobytes') else bytes(data)
        off = f['size']
        f['fh'].write(b); f['sha'].update(b); f['size'] += len(b)
        self.digests[(fname, off)] = (len(b), hashlib.md5(b).hexdigest())
        return off, len(b)

    def dense(self, fname, name, arr, dtype, **meta):
        arr = np.ascontiguousarray(np.asarray(arr, dtype=dtype))
        off, nb = self.put(fname, arr)
        self.tensors[name] = dict(kind='dense', dtype={'float16': 'f16', 'float32': 'f32'}[np.dtype(dtype).name],
                                  shape=list(arr.shape), file=fname, offset=off, bytes=nb, **meta)

    def size(self, fname):
        return self.files[fname]['size'] if fname in self.files else 0

    def close(self):
        out = {}
        for k, f in self.files.items():
            f['fh'].close()
            out[k] = dict(bytes=f['size'], sha256=f['sha'].hexdigest())
            assert os.path.getsize(f'{OUT}/{k}') == f['size'], k
        # read every blob back from disk and compare with what was written
        for (fname, off), (nb, dg) in self.digests.items():
            with open(f'{OUT}/{fname}', 'rb') as fh:
                fh.seek(off)
                assert hashlib.md5(fh.read(nb)).hexdigest() == dg, (fname, off)
        print(f'read back {len(self.digests)} tensor blobs from disk: all identical')
        return out


bins = Bins()


# ------------------------------------------------------------------ ternary packing + exactness check
def ternarize(w):
    """[rows, cols] float weight -> (codes bytes uint8 [rows, cols/4], scales f32 [rows, cols/128], trits int8)."""
    rows, cols = w.shape
    g = w.float().reshape(rows, cols // 128, 128)
    scale = g.abs().amax(-1)                                          # the one nonzero magnitude of each group
    t = torch.where(scale[..., None] > 0, g / scale[..., None].clamp_min(1e-30), torch.zeros(()))
    t = t.round().to(torch.int8).reshape(rows, cols)
    q = (t + 1).to(torch.uint8).numpy().reshape(rows, cols // 4, 4)
    codes = q[..., 0] | (q[..., 1] << 2) | (q[..., 2] << 4) | (q[..., 3] << 6)   # byte k of a u32 = weights 4k..4k+3
    return np.ascontiguousarray(codes), scale.numpy().astype(np.float32), t


def unpack(codes_u8, scales, rows, cols):
    """Inverse of the layout, from the packed bytes (independent of ternarize)."""
    b = codes_u8.reshape(rows, cols // 4)
    q = np.stack([(b >> (2 * j)) & 3 for j in range(4)], -1).reshape(rows, cols).astype(np.int8)
    return torch.from_numpy(q - 1), torch.from_numpy(scales.reshape(rows, cols // 128))


def check_words(codes_u8, scales, w, rows, cols, n=4096, seed=0):
    """Spot-check the u32 formula exactly as the WGSL decoder reads it: word r*(cols/16)+c/16, bits 2*(c%16)."""
    words = codes_u8.reshape(-1).view('<u4')
    rng = np.random.default_rng(seed)
    r, c = rng.integers(0, rows, n), rng.integers(0, cols, n)
    q = (words[r * (cols // 16) + c // 16] >> (2 * (c % 16)).astype(np.uint32)) & 3
    val = (q.astype(np.float32) - 1) * scales.reshape(-1)[r * (cols // 128) + c // 128]
    return float(np.abs(val - w.float().numpy()[r, c]).max())


def put_ternary(fname, items):
    """Pack and write several matrices: all their codes back to back, then all their scales in the same order, so that
    consecutive matrices with equal cols (e.g. to_q, to_k, to_v) can also be bound as one fused matrix."""
    packed, err = [], 0.0
    for name, w, extra in items:
        rows, cols = w.shape
        codes, scales, t = ternarize(w)
        tt, ss = unpack(codes, scales, rows, cols)
        assert torch.equal(tt, t), name
        e = (tt.float().reshape(rows, -1, 128) * ss[..., None] - w.float().reshape(rows, -1, 128)).abs().max().item()
        e = max(e, check_words(codes, scales, w, rows, cols))
        zeros = float((t == 0).float().mean())
        report['ternary'][name] = dict(shape=[rows, cols], max_abs_err=e, zeros=zeros)
        packed.append((name, rows, cols, codes, scales, zeros, extra)); err = max(err, e)
    offs = [bins.put(fname, p[3]) for p in packed] + [bins.put(fname, p[4]) for p in packed]
    for k, (name, rows, cols, _, _, zeros, extra) in enumerate(packed):
        (co, cb), (so, sb) = offs[k], offs[len(packed) + k]
        bins.tensors[name] = dict(kind='ternary', shape=[rows, cols], file=fname, codes=dict(offset=co, bytes=cb),
                                  scales=dict(offset=so, bytes=sb), zeros=round(zeros, 4), **(extra or {}))
    return err


def fused_view(names):
    """A [sum(rows), cols] view over consecutive ternary matrices whose codes and scales are both contiguous."""
    ts = [bins.tensors[n] for n in names]
    for a, b in zip(ts, ts[1:]):
        assert a['file'] == b['file'] and a['shape'][1] == b['shape'][1]
        assert a['codes']['offset'] + a['codes']['bytes'] == b['codes']['offset']
        assert a['scales']['offset'] + a['scales']['bytes'] == b['scales']['offset']
    return dict(kind='ternary', parts=names, shape=[sum(t['shape'][0] for t in ts), ts[0]['shape'][1]], file=ts[0]['file'],
                codes=dict(offset=ts[0]['codes']['offset'], bytes=sum(t['codes']['bytes'] for t in ts)),
                scales=dict(offset=ts[0]['scales']['offset'], bytes=sum(t['scales']['bytes'] for t in ts)))


def fp16_err(w):
    w32 = w.float()
    h = w32.half().float()
    return (h - w32).abs().max().item(), int(((h - w32) != 0).sum()), (h - w32).norm().item() / max(w32.norm().item(), 1e-30)


# ================================================================== 1. schedule (depends only on size and steps)
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.pipelines.flux2.pipeline_flux2_klein import compute_empirical_mu
sched = FlowMatchEulerDiscreteScheduler.from_pretrained(ROOT + 'scheduler')
mu = compute_empirical_mu(image_seq_len=N_IMG, num_steps=args.steps)
sched.set_timesteps(args.steps, 'cpu', sigmas=np.linspace(1.0, 1 / args.steps, args.steps), mu=mu)  # as the pipeline
sigmas = sched.sigmas.float()                                      # [steps+1], terminal 0 appended
timesteps = sched.timesteps.float()                                # sigma * 1000 (float32)
# the fp32 pipeline passes t/1000 and the transformer multiplies by 1000 again: reproduce that round trip bit-exactly
t_model = ((timesteps / 1000).float() * 1000).float()
t_bf16 = ((timesteps.bfloat16() / 1000).bfloat16() * 1000).bfloat16().float()   # what a bf16 pipeline feeds instead
print('mu', mu, 'sigmas', sigmas.tolist(), 'timesteps', timesteps.tolist(), 'bf16 pipeline t', t_bf16.tolist())

# ================================================================== 2. ternary DiT blocks, sharded
BLOCK_ORDER_DOUBLE = ['attn.to_q', 'attn.to_k', 'attn.to_v', 'attn.add_q_proj', 'attn.add_k_proj', 'attn.add_v_proj',
                      'attn.to_out.0', 'attn.to_add_out', 'ff.linear_in', 'ff.linear_out',
                      'ff_context.linear_in', 'ff_context.linear_out']
BLOCK_ORDER_SINGLE = ['attn.to_qkv_mlp_proj', 'attn.to_out']
blocks = [('transformer_blocks', i, BLOCK_ORDER_DOUBLE) for i in range(cfg['num_layers'])] + \
         [('single_transformer_blocks', i, BLOCK_ORDER_SINGLE) for i in range(cfg['num_single_layers'])]
ROW_SPLITS = {  # how the fused matrices' output rows divide (documented in the manifest)
    'ff.linear_in': dict(gate=[0, MLP], up=[MLP, 2 * MLP]),
    'ff_context.linear_in': dict(gate=[0, MLP], up=[MLP, 2 * MLP]),
    'attn.to_qkv_mlp_proj': dict(q=[0, D], k=[D, 2 * D], v=[2 * D, 3 * D], gate=[3 * D, 3 * D + MLP], up=[3 * D + MLP, 3 * D + 2 * MLP]),
}
COL_SPLITS = {'attn.to_out': dict(attn=[0, D], mlp=[D, D + MLP])}   # single block: input = cat(attn_out, swiglu)


def block_bytes(prefix, i, names):
    tot = 0
    for n in names:
        r, c = DT.get_slice(f'{prefix}.{i}.{n}.weight').get_shape()
        tot += r * c // 4 + r * c // 32
    return tot


t0 = time.time()
shard, shard_blocks, views = 0, {}, {}
max_err = 0.0
for prefix, i, names in blocks:
    fname = f'dit_{shard}.bin'
    if bins.size(fname) and bins.size(fname) + block_bytes(prefix, i, names) > SHARD_MAX:
        shard += 1; fname = f'dit_{shard}.bin'
    shard_blocks.setdefault(fname, []).append(f'{prefix}.{i}')
    items = []
    for n in names:
        extra = {}
        if n in ROW_SPLITS: extra['row_splits'] = ROW_SPLITS[n]
        if n in COL_SPLITS and prefix.startswith('single'): extra['col_splits'] = COL_SPLITS[n]
        items.append((f'{prefix}.{i}.{n}.weight', DT.get_tensor(f'{prefix}.{i}.{n}.weight'), extra))
    max_err = max(max_err, put_ternary(fname, items))
    if prefix == 'transformer_blocks':   # fused projections: image q|k|v and text q|k|v, each [9216, 3072]
        views[f'{prefix}.{i}.attn.to_qkv'] = fused_view([f'{prefix}.{i}.attn.to_{c}.weight' for c in 'qkv'])
        views[f'{prefix}.{i}.attn.add_qkv_proj'] = fused_view([f'{prefix}.{i}.attn.add_{c}_proj.weight' for c in 'qkv'])
    print(f'{prefix}.{i} -> {fname} ({bins.size(fname)/2**20:.0f} MiB, {time.time()-t0:.0f}s) max err so far {max_err:.3g}', flush=True)

# which remaining 2-D DiT weights could be ternary? (they are not; this records the evidence)
for key in DT.keys():
    if key in bins.tensors or not key.endswith('weight'): continue
    shp = DT.get_slice(key).get_shape()
    if len(shp) == 2 and shp[1] % 128 == 0:
        g = DT.get_tensor(key).float().reshape(shp[0], -1, 128).abs()
        ok = ((g == g.amax(-1, keepdim=True)) | (g == 0)).all(-1).float().mean().item()
        report['dense'].setdefault(key, {})['groups_that_look_ternary'] = ok

# ================================================================== 3. dense per-prompt weights (fp16) + f32 tables
MISC = 'dit_misc.bin'
for key in ['context_embedder.weight', 'x_embedder.weight', 'proj_out.weight']:
    w = DT.get_tensor(key)
    e, n, rel = fp16_err(w)
    report['dense'].setdefault(key, {}).update(shape=list(w.shape), fp16_max_abs_err=e, fp16_inexact=n, fp16_rel_err=rel)
    bins.dense(MISC, key, w.float().half().numpy(), np.float16, layout='row-major [out, in] (torch Linear), y = x @ W^T, no bias')
for key in sorted(k for k in DT.keys() if re.search(r'\.norm_(added_)?[qk]\.weight$', k)):
    w = DT.get_tensor(key)
    e, n, _ = fp16_err(w)
    report['dense'].setdefault('qk_norm_gains', dict(max_fp16_err=0.0, count=0))
    report['dense']['qk_norm_gains']['max_fp16_err'] = max(report['dense']['qk_norm_gains']['max_fp16_err'], e)
    report['dense']['qk_norm_gains']['count'] += 1
    bins.dense(MISC, key, w.float().half().numpy(), np.float16, layout='RMSNorm gain over head_dim 128, eps 1e-6')

# modulation: depends only on the timestep, so precompute it with diffusers' own modules (fp32, CPU)
from diffusers.models.transformers.transformer_flux2 import Flux2TimestepGuidanceEmbeddings, Flux2Modulation
from diffusers.models.normalization import AdaLayerNormContinuous


def load_module(mod, prefix):
    mod.load_state_dict({k: DT.get_tensor(f'{prefix}.{k}').float() for k in mod.state_dict()})
    return mod.float().eval()


temb_mod = load_module(Flux2TimestepGuidanceEmbeddings(cfg['timestep_guidance_channels'], D, bias=False, guidance_embeds=False), 'time_guidance_embed')
m_img = load_module(Flux2Modulation(D, 2), 'double_stream_modulation_img')
m_txt = load_module(Flux2Modulation(D, 2), 'double_stream_modulation_txt')
m_single = load_module(Flux2Modulation(D, 1), 'single_stream_modulation')
norm_out = load_module(AdaLayerNormContinuous(D, D, elementwise_affine=False, eps=cfg['eps'], bias=False), 'norm_out')
with torch.no_grad():
    temb = temb_mod(t_model, None)                                         # [steps, 3072]
    mods = dict(double_img=m_img(temb).reshape(-1, 6, D), double_txt=m_txt(temb).reshape(-1, 6, D),
                single=m_single(temb).reshape(-1, 3, D), norm_out=norm_out.linear(norm_out.silu(temb)).reshape(-1, 2, D))
    # independent float64 re-derivation of the same vectors (the formulas in PAINTER-SPEC.md)
    half = 128
    freqs = torch.exp(-math.log(10000) * torch.arange(half, dtype=torch.float32) / half)
    arg = t_model[:, None].float() * freqs[None]
    sinus = torch.cat([torch.cos(arg), torch.sin(arg)], -1).double()       # flip_sin_to_cos: cos first
    W1 = DT.get_tensor('time_guidance_embed.timestep_embedder.linear_1.weight').double()
    W2 = DT.get_tensor('time_guidance_embed.timestep_embedder.linear_2.weight').double()
    te = torch.nn.functional.silu(torch.nn.functional.silu(sinus @ W1.T) @ W2.T)
    chk = dict(double_img=te @ DT.get_tensor('double_stream_modulation_img.linear.weight').double().T,
               double_txt=te @ DT.get_tensor('double_stream_modulation_txt.linear.weight').double().T,
               single=te @ DT.get_tensor('single_stream_modulation.linear.weight').double().T,
               norm_out=te @ DT.get_tensor('norm_out.linear.weight').double().T)
    report['modulation_check_max_abs_diff_vs_float64'] = {k: (mods[k].reshape(len(t_model), -1).double() - chk[k]).abs().max().item() for k in mods}
ROWS = dict(double_img=['shift_msa', 'scale_msa', 'gate_msa', 'shift_mlp', 'scale_mlp', 'gate_mlp'],
            double_txt=['shift_msa', 'scale_msa', 'gate_msa', 'shift_mlp', 'scale_mlp', 'gate_mlp'],
            single=['shift', 'scale', 'gate'], norm_out=['scale', 'shift'])
for k, v in mods.items():
    bins.dense(MISC, f'mod.{k}', v.numpy(), np.float32, rows=ROWS[k], layout='[step, row, 3072]; x_mod = (1 + scale) * LayerNorm(x) + shift')
bins.dense(MISC, 'temb', temb.numpy(), np.float32, layout='[step, 3072] timestep embedding (for reference only; mods are derived from it)')

# RoPE: 4 axes x 32 dims, theta 2000; joint sequence = 512 text tokens (ids (0,0,0,l)) then 1024 image tokens (ids (0,h,w,0))
ids = torch.cat([torch.stack([torch.zeros(N_TXT), torch.zeros(N_TXT), torch.zeros(N_TXT), torch.arange(N_TXT)], -1),
                 torch.stack([torch.zeros(N_IMG), torch.arange(N_IMG) // GRID, torch.arange(N_IMG) % GRID, torch.zeros(N_IMG)], -1)]).double()
inv = 1.0 / (cfg['rope_theta'] ** (torch.arange(0, 32, 2, dtype=torch.float64) / 32))    # 16 frequencies per axis
ang = torch.cat([ids[:, a:a + 1] * inv[None] for a in range(4)], -1)                    # [1536, 64] one angle per dim pair
bins.dense(MISC, 'rope.cos', torch.cos(ang).numpy(), np.float32, layout='[token, pair]; pair p rotates head dims (2p, 2p+1)')
bins.dense(MISC, 'rope.sin', torch.sin(ang).numpy(), np.float32, layout='[token, pair]; out[2p] = x[2p]*cos - x[2p+1]*sin; out[2p+1] = x[2p+1]*cos + x[2p]*sin')

# VAE latent batch-norm stats (patchified 128-ch space): x_vae = x * std + mean (TAEF2 does NOT need this)
V = safe_open(ROOT + 'vae/diffusion_pytorch_model.safetensors', 'pt')
vcfg = json.load(open(ROOT + 'vae/config.json'))
bins.dense(MISC, 'vae.bn_mean', V.get_tensor('bn.running_mean').float().numpy(), np.float32)
bins.dense(MISC, 'vae.bn_std', torch.sqrt(V.get_tensor('bn.running_var').float() + vcfg['batch_norm_eps']).numpy(), np.float32)

# ================================================================== 4. adapter (ternary) + bias + prefix
A, Q = torch.load(ADAPTER), torch.load(QEXTRA)
Wl = A['W'].T.contiguous()                                   # x @ W  ==  Linear with weight W^T [7680 out, 6144 in]
aerr = put_ternary('adapter.bin', [('adapter.weight', Wl, dict(layout='cond = concat(h7, h14, h21) @ W^T + bias'))])
bins.dense('adapter.bin', 'adapter.bias', A['b'].numpy(), np.float32)
bins.dense('adapter.bin', 'adapter.prefix', Q['prefix'].numpy(), np.float32, layout='rows 0..2 of the 512x7680 conditioning, copied verbatim (Qwen3-4B taps of <|im_start|>, user, \\n)')

# ================================================================== 5. TAEF2 decoder (fp16) + its layer graph
T2 = safe_open(TAEF2, 'pt')
tmax = 0.0
for k in sorted(k for k in T2.keys() if k.startswith('decoder.layers.')):
    w = T2.get_tensor(k)
    tmax = max(tmax, fp16_err(w)[0])
    bins.dense('taef2.bin', 'taef2.' + k[len('decoder.layers.'):], w.half().numpy(), np.float16)
report['dense']['taef2'] = dict(fp16_max_abs_err=tmax)
graph = [dict(op='tanh_clamp', note='x = 3 * tanh(x / 3)'), dict(op='conv3x3', w='taef2.0.weight', b='taef2.0.bias', cin=32, cout=64), dict(op='relu')]
for i in range(2, 18):
    kind = {5: 'up', 10: 'up', 15: 'up', 6: 'conv', 11: 'conv', 16: 'conv'}.get(i, 'block')
    if kind == 'up': graph.append(dict(op='upsample_nearest_2x'))
    elif kind == 'conv': graph.append(dict(op='conv3x3', w=f'taef2.{i}.weight', b=None, cin=64, cout=64))
    else:
        blk = dict(op='block', conv=[f'taef2.{i}.conv.{j}' for j in (0, 2, 4)])
        if f'decoder.layers.{i}.pool.0.weight' in T2.keys():
            blk['pool'] = dict(conv_in=f'taef2.{i}.pool.0.weight', gn=f'taef2.{i}.pool.1', groups=4, eps=1e-5, conv_out=f'taef2.{i}.pool.3.weight')
        graph.append(blk)
graph += [dict(op='conv3x3', w='taef2.18.weight', b='taef2.18.bias', cin=64, cout=3), dict(op='clamp01')]

# ================================================================== 6. manifest
files = bins.close()
manifest = dict(
    format='mindview-painter-v1',
    source=dict(dit='prism-ml/bonsai-image-ternary-4B-unpacked@' + ROOT.rstrip('/').split('/')[-1],
                adapter=ADAPTER, taef2='madebyollin/taef2', text_encoder='prism-ml/Ternary-Bonsai-1.7B (GGUF in the browser)'),
    spec='research/PAINTER-SPEC.md',
    ternary_layout=dict(group=128, codes='u32 row-major, 16 weights per word, weight (r,c) at word r*(cols/16)+c/16, bits 2*(c%16)..+1 hold q=trit+1',
                        scales='f32 row-major, index r*(cols/128)+c/128', value='(q-1)*scale', byte_order='little-endian', align=ALIGN),
    config=dict(hidden=D, heads=H, head_dim=HD, mlp_hidden=MLP, double_blocks=cfg['num_layers'], single_blocks=cfg['num_single_layers'],
                in_channels=cfg['in_channels'], joint_attention_dim=cfg['joint_attention_dim'], eps=cfg['eps'], qk_norm_eps=cfg['eps'],
                rope=dict(theta=cfg['rope_theta'], axes_dims=cfg['axes_dims_rope'], text_ids='(0,0,0,l)', image_ids='(0,h,w,0)'),
                attention_scale=1 / math.sqrt(HD), sequence='text (512) then image (1024)', guidance_embeds=False),
    text=dict(template='<|im_start|>user\\n{prompt}<|im_end|>\\n<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n',
              max_len=N_TXT, pad_id=151643, padding='right', llm_layers_run=21, taps=[7, 14, 21], tap_dim=2048,
              pad_attention='pad query at position p attends to the real tokens only (not itself, not other pads); RoPE position p',
              prefix_rows=3, adapter_in=6144, adapter_out=7680),
    image=dict(size=args.size, grid=GRID, tokens=N_IMG, token_order='t = h*grid + w', channels=128,
               channel_order='c128 = 4*c32 + 2*dy + dx (latent pixel (2h+dy, 2w+dx) of VAE channel c32)', latent_space='bn-normalised'),
    schedule=dict(steps=args.steps, mu=mu, sigmas=sigmas.tolist(), timesteps=timesteps.tolist(), t_model=t_model.tolist(),
                  t_bf16_pipeline=t_bf16.tolist(), update='x_{i+1} = x_i + (sigma_{i+1} - sigma_i) * v_i'),
    taef2=dict(input='bn-normalised latent, unpatchified to [32, 64, 64]', output='[3, 512, 512] RGB in [0,1]', graph=graph),
    shards={k: v for k, v in shard_blocks.items()},
    files=files,
    tensors=bins.tensors,
    views=views,
    validation=dict(ternary_max_abs_err=max_err, adapter_max_abs_err=aerr, **{k: v for k, v in report['dense'].items()},
                    modulation_check=report['modulation_check_max_abs_diff_vs_float64']),
)
json.dump(manifest, open(f'{OUT}/manifest.json', 'w'), indent=1)
json.dump(report, open(f'{OUT}/export_report.json', 'w'), indent=1)

tern = [v for k, v in report['ternary'].items() if k != 'adapter.weight']
print(f'\nDiT ternary matrices: {len(tern)}, params {sum(v["shape"][0]*v["shape"][1] for v in tern)/1e9:.3f}B, '
      f'max |trit*scale - w| = {max(v["max_abs_err"] for v in tern):.3g}, zeros {np.mean([v["zeros"] for v in tern]):.3f}')
print(f'adapter: max |trit*scale - w| = {aerr:.3g}, zeros {report["ternary"]["adapter.weight"]["zeros"]:.3f}')
for k, v in report['dense'].items(): print('dense', k, v)
print('modulation check vs float64:', report['modulation_check_max_abs_diff_vs_float64'])
tot = 0
for k, v in files.items():
    print(f'{k:14s} {v["bytes"]/1e6:10.2f} MB'); tot += v['bytes']
print(f'total {tot/1e6:.1f} MB ({tot/2**30:.3f} GiB), {len(bins.tensors)} tensors, {time.time()-t0:.0f}s')
