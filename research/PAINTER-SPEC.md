# Painter forward spec: Bonsai 1.7B → ternary adapter → Bonsai Image 4B DiT → TAEF2 (512², 4 steps)

This is an op-by-op description of the whole image path, detailed enough to re-implement it from scratch (the target is a
hand-written WGSL runtime).

**How it was checked**
- Every claim was checked against the installed sources: diffusers 0.40.0 (`transformer_flux2.py`,
  `pipeline_flux2_klein.py`, `scheduling_flow_match_euler_discrete.py`, `embeddings.py`, `normalization.py`,
  `attention_dispatch.py`), transformers 5.17.0 (`modeling_qwen3.py`, `output_capturing.py`) and torch 2.14.
- It was then checked numerically: `scripts/painter_reference.py` implements exactly this document from the exported
  files and matches diffusers fp32 to within 1 − cos ≈ 1e-11 (see §7).

**Scripts**

| script | what it does |
|---|---|
| `scripts/export_painter.py` | writes `static/models/bonsai-image-4b/` (weights + `manifest.json`) |
| `scripts/painter_truth.py` | produces the ground truth with the stock libraries: the 1.7B via HF, the DiT via the diffusers pipeline, fp32 and bf16 |
| `scripts/painter_reference.py` | the from-scratch f32 implementation; writes `static/lab/painter_ref/` (see its README) |

**Notation**
- `x @ Wᵀ` is a torch `Linear` with weight `W` stored `[out, in]`. No layer in the DiT, the adapter or TAEF2's 1×1
  convolutions has a bias unless stated.
- All shapes are for one image at 512² (batch 1).
- `LN` is LayerNorm without affine parameters. `RMS(x, g)` is `x · rsqrt(mean(x²) + 1e-6) · g`.
- `silu(x) = x · sigmoid(x)`.

## 0. Shapes at a glance

| stage | tensor | shape |
|---|---|---|
| tokens | prompt + template, right-padded | 512 ids (29 real for the reference prompt) |
| 1.7B taps | residual after layers 7, 14, 21 | 3 × [512, 2048] → concat [512, 6144] |
| adapter | conditioning `cond` | [512, 7680] |
| DiT text stream | `c` | [512, 3072] |
| DiT image stream | `x` | [1024, 3072] (32×32 latent tokens) |
| joint sequence | `[c; x]` | [1536, 3072], **text first** |
| latent tokens | `lat` | [1024, 128] (bn-normalised, 2×2-patchified 32-ch VAE latent) |
| TAEF2 input | unpatchified `lat` | [32, 64, 64] |
| image | RGB | [3, 512, 512] in [0, 1] |

## 1. Text side

### 1.1 Template, tokenisation, padding
- **Tokenizer:** the Qwen2 tokenizer from the image repo (`tokenizer/`). It is the same vocabulary as the 1.7B GGUF
  tokenizer, which the browser already matches exactly.
- **Template:** the chat template with `add_generation_prompt=True, enable_thinking=False`. It gives exactly this string:
  ```
  <|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n
  ```
  - For the reference prompt that is 29 tokens: `151644, 872, 198, 64, 81034, 2143, 4916, 1865, 315, 48492, 16224, 884,
    304, 264, 6319, 23971, 11, 62820, 16340, 3100, 151645, 198, 151644, 77091, 198, 151667, 271, 151668, 271`.
  - There is no BOS token.
- **Padding and truncation:**
  - `padding='max_length', max_length=512, truncation=True`.
  - Padding is on the **right**, with `<|endoftext|>` (id **151643**).
  - `attention_mask` is 1 for the `n` real tokens and 0 for the pads.

### 1.2 The 1.7B pass over all 512 positions
**All 512 positions go through the LLM.** This is how the pipeline and the adapter were run: HF forward with that
attention mask. Position ids are simply `0..511` (transformers ignores the mask for positions).

Attention rule per layer, for query position `i` and key position `j`:
```
allowed(i, j) = (j <= i) and (j < n)          # n = number of real tokens
```
- **Real tokens:** ordinary causal attention. They never see a pad, so their states are bit-identical to an unpadded
  run (verified: max |diff| = 0).
- **Pad tokens:** a pad at position `p ≥ n` attends to the **n real tokens only**. It does **not** attend to itself or
  to any other pad.
- **What a pad's state depends on:** only three things, namely the pad embedding, its own RoPE position `p` (the YaRN
  RoPE the runtime already has), and the real tokens' K/V.
- **How to compute them:** after the real prefill, run the 512 − n pads as one batch per layer against the cached
  real-token K/V. Pads need no K/V from each other.
- **Verified in HF fp32:** recomputing the pad at p = 29, 79 and 511 with an explicit mask reproduces the full padded
  run bit-exactly. Letting a pad attend to itself changes its taps by up to 530 in absolute value, so the self-exclusion
  matters.
- **Cost:** only layers 0..20 are needed (the last tap is after 21 layers). That is 512 × 21 layers of the 1.7B, about
  1.1 TFLOP.

### 1.3 Taps
- **Index semantics:** `output.hidden_states[k]` is the residual stream **after k decoder layers**, before any norm.
  - `hidden_states[0]` is the token embeddings.
  - `hidden_states[k]` is the output of `model.layers[k-1]` (0-based).
  - Only `hidden_states[28]` has the final RMSNorm applied, which is irrelevant here.
  - This was verified with forward hooks in transformers 5.17.
- **Taps used:** k = **7, 14, 21**.
- **Concatenation:** `X = concat(h7, h14, h21, dim=-1)`, giving [512, 6144] in f32.

### 1.4 Adapter
The adapter maps the 1.7B taps into Qwen3-4B's layer 9/18/27 space. It is a plain affine map with **no
normalisation**:
```
cond = X @ Wᵀ + b          # W ternary [7680, 6144], group-128 f32 scales along the 6144 inputs; b f32 [7680]
cond[0:3] = prefix         # f32 [3, 7680]: the Qwen3-4B taps of "<|im_start|>", "user", "\n" (identical for every prompt)
```
- **dtype:** f32 throughout. The bf16 pipeline then casts `cond` to bf16; the fp32 path keeps f32.
- **Where the weights come from:** W and b come from `data/adapter/adapter_bonsai17t.pt`, and the prefix from
  `data/adapter/qwen_extra.pt`.
  - W is exactly `trit × scale`, with 44.3% zeros.
  - The scales are **f32**, not fp16 as the ternarize docstring says. They are therefore exact in this layout.
- **Rows 0..2:** the 1.7B taps of these rows are discarded; the rows still matter inside the LLM as K/V.
- **Magnitudes:** row 0 is Qwen's massive activation, with norm 29,403 and absmax 16,384. Real rows have norm ≈ 153,
  pad rows ≈ 113.
- **Pads are not masked anywhere downstream.** All 512 rows condition the DiT.

## 2. DiT (Flux2Transformer2DModel, FLUX.2 klein 4B config)
- **Width:** hidden 3072 = 24 heads × 128.
- **MLP:** hidden 9216 (mlp_ratio 3), SwiGLU.
- **Blocks:** 5 double-stream blocks, then 20 single-stream blocks.
- **Norm eps:** 1e-6 everywhere.
- **Parameters:** no biases anywhere. No guidance embedding (`guidance_embeds: false`; the pipeline has
  `is_distilled: true`, so there is no CFG and `guidance_scale` is ignored). `patch_size` = 1.

### 2.1 Embedders (dense, fp16 in the export)
```
x = lat  @ W_xᵀ        # x_embedder.weight [3072, 128]
c = cond @ W_ctxᵀ      # context_embedder.weight [3072, 7680]: NOT ternary. c does not depend on the step: compute once per prompt
```

### 2.2 Timestep embedding and modulation (depend only on the step, precomputed)
- **Timestep:** `t = σ·1000` in f32. The pipeline passes `t/1000` and the model multiplies by 1000 again; in f32 the
  values come back unchanged: 1000, 958.08539, 883.98187, 717.49658.
- **Formulas:**
  ```
  f_i  = exp(-ln(10000) · i / 128),  i = 0..127
  e    = [cos(t·f_0..127), sin(t·f_0..127)]              # 256, cos FIRST (flip_sin_to_cos=True), computed in f32
  temb = W2 @ silu(W1 @ e)                                 # time_guidance_embed.timestep_embedder.linear_1 [3072,256], linear_2 [3072,3072]
  mod_double_img = W_mi @ silu(temb)   → 6 × 3072: shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp
  mod_double_txt = W_mt @ silu(temb)   → 6 × 3072 (same order)
  mod_single     = W_ms @ silu(temb)   → 3 × 3072: shift, scale, gate
  mod_norm_out   = W_no @ silu(temb)   → 2 × 3072: SCALE, SHIFT   (note: reversed order)
  ```
- **Sharing:** one modulation set per step is shared by all 5 double blocks (per stream) and by all 20 single blocks.
- **Export:** these are `mod.double_img`, `mod.double_txt`, `mod.single` and `mod.norm_out`, with shape
  `[4 steps, rows, 3072]` in f32. `temb` is also included for reference.
- **What this saves:** the 170.6M bf16 parameters behind them (time MLP and modulation linears) never reach the browser.
- **Magnitudes:** modulation values reach |18.6|.

### 2.3 Norms
- **Activation norms:** `LN(x) = (x − mean) · rsqrt(var + 1e-6)`, with biased variance and no affine. This covers
  `norm1`, `norm1_context`, `norm2`, `norm2_context`, the single block's `norm`, and the LN inside `norm_out`.
- **Modulated input:** `LN(x) · (1 + scale) + shift`.
- **QK norm:** RMSNorm over each 128-dim head vector, eps 1e-6, with a learned gain [128].
  - Gains: `norm_q` and `norm_k` for the image stream (and in single blocks), `norm_added_q` and `norm_added_k` for the
    text stream in double blocks.
  - Applied to q and k **before** RoPE. v is not normalised.

### 2.4 RoPE (Flux2PosEmbed)
- **Axes:** 4 axes of 32 dims, theta **2000**.
- **Position ids** (t, h, w, l):
  - text token `l` (0..511, pads included): `(0, 0, 0, l)`
  - image token at row `h`, column `w` (0..31): `(0, h, w, 0)`, with token index `h·32 + w`
- **Angles:** axis `a` owns head dims `32a .. 32a+31`. Within it, pair `i` (dims `32a+2i`, `32a+2i+1`, i = 0..15) turns
  by `pos_a · 2000^(−2i/32)`.
- **Rotation of one pair** (interleaved, `repeat_interleave_real=True`):
  ```
  out[2p]   = x[2p]·cos − x[2p+1]·sin
  out[2p+1] = x[2p+1]·cos + x[2p]·sin                   # p = 16a + i, 64 pairs per head, same angles for all 24 heads
  ```
- **What rotates:** text rotates only dims 96..127 (axis l). Image rotates only dims 32..95 (axes h, w). Dims 0..31
  never rotate.
- **Export:** `rope.cos` and `rope.sin` [1536, 64] f32, in joint order (text rows 0..511, then image rows).
  - diffusers computes the angles in f64 on CPU and f32 on MPS; the difference is negligible.

### 2.5 Attention
- **Kind:** full bidirectional softmax attention over all 1536 tokens, per head, with scale `1/√128`.
- **No mask:** pads are attended. With the stock encoder they took 21–28% of the image's attention mass.
- **Order:** the joint order is always `[text; image]`.
- The native SDPA backend is used (`dispatch_attention_fn`, `scale=None`).

### 2.6 Double-stream block (× 5), step s
```
mi = mod.double_img[s]; mt = mod.double_txt[s]               # rows as named in 2.2
xn = LN(x)·(1+mi.scale_msa) + mi.shift_msa                   # image [1024,3072]
cn = LN(c)·(1+mt.scale_msa) + mt.shift_msa                   # text  [512,3072]
qi = RMS(heads(xn @ to_qᵀ), norm_q);        ki = RMS(heads(xn @ to_kᵀ), norm_k);        vi = heads(xn @ to_vᵀ)
qt = RMS(heads(cn @ add_q_projᵀ), norm_added_q); kt = RMS(heads(cn @ add_k_projᵀ), norm_added_k); vt = heads(cn @ add_v_projᵀ)
q = RoPE([qt; qi]);  k = RoPE([kt; ki]);  v = [vt; vi]       # [1536, 24, 128]
o = softmax(q·kᵀ/√128)·v  → [1536, 3072]                     # heads flattened h·128+d
x = x + mi.gate_msa · (o[512:] @ to_out.0ᵀ)
c = c + mt.gate_msa · (o[:512] @ to_add_outᵀ)
x = x + mi.gate_mlp · (SwiGLU((LN(x)·(1+mi.scale_mlp)+mi.shift_mlp) @ ff.linear_inᵀ) @ ff.linear_outᵀ)
c = c + mt.gate_mlp · (SwiGLU((LN(c)·(1+mt.scale_mlp)+mt.shift_mlp) @ ff_context.linear_inᵀ) @ ff_context.linear_outᵀ)
SwiGLU(h) = silu(h[:, 0:9216]) · h[:, 9216:18432]            # linear_in [18432, 3072], linear_out [3072, 9216]
```
- **Fused q|k|v:** in the export, the codes and the scales of `to_q | to_k | to_v` (and of `add_q | add_k | add_v`)
  are each contiguous. They can therefore be bound as one [9216, 3072] matrix (`views` in the manifest).
- **Reading the gates:** `gate_*` multiplies the branch output. The `1 +` applies to scale only.

### 2.7 Single-stream block (× 20), step s
After double block 4, `h = [c; x]` [1536, 3072].
```
m  = mod.single[s]
p  = (LN(h)·(1+m.scale) + m.shift) @ to_qkv_mlp_projᵀ        # [1536, 27648] = q 0:3072 | k 3072:6144 | v 6144:9216 | gate 9216:18432 | up 18432:27648
q  = RoPE(RMS(heads(p[:, 0:3072]), norm_q));  k = RoPE(RMS(heads(p[:, 3072:6144]), norm_k));  v = heads(p[:, 6144:9216])
o  = softmax(q·kᵀ/√128)·v   → [1536, 3072]
a  = silu(p[:, 9216:18432]) · p[:, 18432:27648]              # [1536, 9216]
h  = h + m.gate · ([o | a] @ to_outᵀ)                        # to_out [3072, 12288]: columns 0:3072 attention, 3072:12288 MLP
```
- This is a "parallel" block: one fused input projection and one fused output projection, with a single gate and a
  single residual add.
- **Optional saving:** in the last block the 512 text rows need only k and v. Their `to_out`/MLP outputs are discarded.

### 2.8 Output
```
y = h[512:]                                                  # image rows [1024, 3072]
y = LN(y)·(1 + mod.norm_out[s].scale) + mod.norm_out[s].shift
v = y @ proj_outᵀ                                            # proj_out [128, 3072] → velocity [1024, 128]
```

## 3. Sampler

### 3.1 Schedule (FlowMatchEulerDiscreteScheduler, exponential dynamic shift)
```
s   = linspace(1, 1/steps, steps) as float32          = [1, 0.75, 0.5, 0.25]
mu  = compute_empirical_mu(image_seq_len=1024, steps=4) = 2.0306897
        (a1,b1 = 8.73809524e-5, 1.89833333;  a2,b2 = 1.6927e-4, 0.45666666;
         m200 = a2·L + b2, m10 = a1·L + b1, a = (m200 − m10)/190, b = m200 − 200a, mu = a·steps + b;  L = 1024 ≤ 4300)
σ_i = e^mu / (e^mu + (1/s_i − 1))                    = [1.0, 0.95808536, 0.88398188, 0.71749657], then σ_4 = 0
t_i = σ_i · 1000                                     = [1000, 958.08539, 883.98187, 717.49658]
```
- Sigmas are computed in float32 (numpy) and timesteps in float32 (torch).
- **The bf16 pipeline** (the old `export_trace.py` path) casts the timestep to bf16 before embedding it. The model then
  sees t = **1000, 960, 884, 716**, and temb differs by up to 0.08.
- The export uses the exact f32 values, which is the fp32 pipeline's behaviour. The two paths end 42 dB apart (§7).

### 3.2 Loop
```
lat = x_0 (noise, [1024,128])
for i in 0..3:  v_i = DiT(lat, c, step i);   lat = lat + (σ_{i+1} − σ_i) · v_i        # f32 (the scheduler upcasts)
```
- **x0 prediction** (useful for previews): `x0 = lat_i − σ_i · v_i`.
- **Why the order matters:** only the modulation depends on the step. The text stream is still recomputed every step,
  because its modulation changes; only `c = cond @ W_ctxᵀ` is shared.

### 3.3 Noise, packing, token order
- **Noise:** the pipeline draws `torch.randn((1, 128, 32, 32), generator=CPU seed)` directly in the **patchified**
  128-channel space.
  - A bf16 draw equals the fp32 draw rounded to bf16 (verified), so older bf16 traces used the same noise.
  - The browser cannot reproduce torch's RNG stream cheaply. For validation it loads `painter_ref/noise.bin`; for live
    use any N(0, 1) source works.
- **Packing:**
  - `tokens[t, ch] = noise[ch, t // 32, t % 32]`, i.e. `reshape(128, 1024).T`.
  - Token `t = h·32 + w` is row-major over the 32×32 grid, `h` = row (image y).
- **Channel meaning:** channel `ch = 4·c + 2·dy + dx` is VAE latent channel `c` (0..31) at latent pixel `(2h+dy, 2w+dx)`.
  - Patchify: `view(32, H/2, 2, W/2, 2).permute(0, 2, 4, 1, 3)`.
  - Unpatchify: `view(32, 2, 2, H, W).permute(0, 3, 1, 4, 2)`, giving 64×64.
  - As a token view: `lat.view(32, 32, 32, 2, 2).permute(2, 0, 3, 1, 4).reshape(32, 64, 64)`.

### 3.4 Batch-norm normalisation and the decoder input
- The DiT lives in **bn-normalised** latent space: `z = (vae_latent_patchified − mean) / std` per 128 channels.
  - `mean = vae.bn.running_mean`, `std = sqrt(running_var + 1e-4)`.
  - Exported as `vae.bn_mean` and `vae.bn_std`.
- **Full VAE path:** `lat · std + mean`, then unpatchify to [32, 64, 64], then `AutoencoderKLFlux2.decode`. This path is
  not ported.
- **TAEF2 path (ported):** unpatchify the **bn-normalised** final latent directly to [32, 64, 64] and decode. Do **not**
  de-normalise: that is what TAEF2 expects, with error 0.03 vs 0.11 if de-normalised first.
- TAEF2 vs full VAE on the reference: 30.3 dB PSNR.

## 4. TAEF2 decoder (madebyollin/taef2, `use_midblock_gn=True`)
- **Size:** 1,338,499 parameters, exported fp16. The source f32 weights are exactly representable in fp16.
- **Convolutions:** 3×3 with padding 1 (zero padding).
- **Upsampling:** nearest-neighbour ×2.
- **Naming:** the index `n` below is `decoder.layers.n` in `taef2.safetensors` (= `taef2.n.*` in the export; the taesd
  `Sequential` index is n+1).

| n | op | output |
|---|---|---|
| – | `x = 3·tanh(x/3)` | 32×64×64 |
| 0, 1 | conv3×3 32→64 (+bias), ReLU | 64×64×64 |
| 2, 3, 4 | Block(64) **with** GN pool | 64×64×64 |
| 5 | upsample ×2 | 64×128×128 |
| 6 | conv3×3 64→64, no bias | 64×128×128 |
| 7, 8, 9 | Block(64) | 64×128×128 |
| 10, 11 | upsample ×2, conv3×3 no bias | 64×256×256 |
| 12, 13, 14 | Block(64) | 64×256×256 |
| 15, 16 | upsample ×2, conv3×3 no bias | 64×512×512 |
| 17 | Block(64) | 64×512×512 |
| 18 | conv3×3 64→3 (+bias) | 3×512×512 |
| – | `clamp(0, 1)` | RGB |

```
Block(x):
  if pool:  x = x + P3( relu( GroupNorm(4 groups, 256 ch, eps 1e-5, affine)( P0(x) ) ) )   # P0 1×1 64→256, P3 1×1 256→64, no bias
  y = C4( relu( C2( relu( C0(x) ) ) ) )                                                   # 3×3 64→64 with bias
  return relu(y + x)                                                                      # skip is identity (64 = 64)
```
- The GroupNorm statistics cover each group's 64 channels × the **whole** 64×64 map, so they need a global reduction.
- Cost is about 71 GMAC, 55% of it at 512².
- The manifest's `taef2.graph` lists these ops with tensor names.

## 5. Weights

### 5.1 Exactly ternary (exported in the 2-bit layout)
All **100** DiT block matrices, 3,680,501,760 weights.
- **Exactness:** every group of 128 inputs holds {0, ±s} with one magnitude s, so `trit × scale` rebuilds the bf16
  weights with **max abs error 0** for every tensor. This was checked from the packed files through the u32 decode
  formula.
- **Sparsity:** zeros are 29.7–47.1% per matrix, mean 34.4%.

| per block | tensor | shape [out, in] |
|---|---|---|
| double ×5 | `attn.to_q`, `attn.to_k`, `attn.to_v` (image) | [3072, 3072] each |
| | `attn.add_q_proj`, `attn.add_k_proj`, `attn.add_v_proj` (text) | [3072, 3072] each |
| | `attn.to_out.0` (image), `attn.to_add_out` (text) | [3072, 3072] each |
| | `ff.linear_in`, `ff_context.linear_in` (gate ∣ up) | [18432, 3072] each |
| | `ff.linear_out`, `ff_context.linear_out` | [3072, 9216] each |
| single ×20 | `attn.to_qkv_mlp_proj` (q ∣ k ∣ v ∣ gate ∣ up) | [27648, 3072] |
| | `attn.to_out` (in = attn 3072 ∣ mlp 9216) | [3072, 12288] |
| adapter | `adapter.weight` | [7680, 6144] (44.3% zeros, error 0) |

### 5.2 Not ternary (bf16 in the source, 195,042,816 parameters)
None of these has a ternary group structure: 0% of their groups look ternary.

| tensor | shape | in the export |
|---|---|---|
| `context_embedder.weight` | [3072, 7680] | **fp16**. Max abs error 3.0e-8, relative 2.6e-8; 5,273 tiny entries (down to 1.6e-10) fall below fp16's normal range |
| `x_embedder.weight` | [3072, 128] | fp16 (max abs error 3e-8) |
| `proj_out.weight` | [128, 3072] | fp16 (max abs error 3e-8) |
| `*.attn.norm_{q,k}`, `*.attn.norm_added_{q,k}` | 60 × [128] | fp16 (exact) |
| `time_guidance_embed.timestep_embedder.linear_1 / linear_2` | [3072, 256] / [3072, 3072] | folded into `temb`, `mod.*` |
| `double_stream_modulation_img / _txt.linear` | [18432, 3072] each | folded into `mod.double_img` / `mod.double_txt` |
| `single_stream_modulation.linear` | [9216, 3072] | folded into `mod.single` |
| `norm_out.linear` | [6144, 3072] | folded into `mod.norm_out` |

### 5.3 Export (`static/models/bonsai-image-4b/`, `manifest.json` indexes everything)
- **Layout:** all files are little-endian, and every tensor starts at a 256-byte aligned offset.
- **Shards:** each DiT shard is ≤ 256 MiB, so it fits the default WebGPU `maxBufferSize` as one buffer.
- **Ternary tensors:** in the manifest they carry `codes` and `scales` offsets. The layout matches the 1.7B runtime:
  - codes: u32, 16 weights per word; weight (r, c) is at word `r·(cols/16) + c/16`, bits `2·(c%16)..+1`, and holds
    `q = trit + 1`
  - scales: f32 at `r·(cols/128) + c/128`

| file | contents | size |
|---|---|---|
| `dit_0.bin` | double blocks 0–2 | 207.0 MB |
| `dit_1.bin` | double 3–4, single 0–2 | 241.5 MB |
| `dit_2.bin` | single 3–9 | 241.5 MB |
| `dit_3.bin` | single 10–16 | 241.5 MB |
| `dit_4.bin` | single 17–19 | 103.5 MB |
| `dit_misc.bin` | fp16 embedders, `proj_out`, QK gains; f32 `mod.*` [4,·,3072], `temb`, `rope.cos/sin` [1536,64], `vae.bn_*` | 50.4 MB |
| `adapter.bin` | ternary adapter, f32 bias [7680], f32 prefix [3, 7680] | 13.4 MB |
| `taef2.bin` | TAEF2 decoder fp16 | 2.7 MB |
| **total** | + `manifest.json` (82 kB) | **1101.7 MB (1.03 GiB)** |

**Extra manifest fields**
- `row_splits` (fused `linear_in` / `to_qkv_mlp_proj`) and `col_splits` (single `to_out`).
- `views`: fused `to_qkv` and `add_qkv_proj` per double block.
- `schedule`, `text`, `image` and `config` blocks mirroring this document.
- `files` with a sha256 for each file.
- The exporter reads every blob back from disk and checks it.

## 6. Port notes
- **Compute per step**
  - Ternary matmuls: every block applies 122.7M weights to 1536 rows (double blocks: 1024 image rows and 512 text rows
    through separate weights). That is 4.71 TMAC.
  - Attention: 0.36 TMAC.
  - In total about 10 TFLOP per step and 40 TFLOP for 4 steps.
  - Once per prompt: `context_embedder` (12 GMAC), the adapter (24 GMAC) and the 1.7B over 512 × 21 layers (0.54 TMAC).
- **Precision: keep activations in f32.**
  - The text residual stream reaches |x| ≈ 28,700 and `cond` row 0 reaches 16,384.
  - f16 would overflow in LayerNorm's sum of squares and lose the large prefix row.
  - bf16 activations (diffusers bf16) already drift to per-token cosine 0.92 in late blocks.
- **Largest intermediates**
  - `to_qkv_mlp_proj` output: 1536 × 27648 f32 = 170 MB, above the default 128 MiB `maxStorageBufferBindingSize`.
    Raise the limit or split by rows or columns.
  - Materialised attention: 24 × 1536² f32 = 226 MB per block. Do it per head, or flash-style with a separate
    probability pass for the visuals.
- **Step-invariant pieces:** `c = cond @ W_ctxᵀ`, RoPE and the adapter output.
- **Everything with a step index** is in `mod.*`.

## 7. Validation (`painter_reference.py`)
**Test case:** prompt "a bonsai tree made of glowing circuitry in a dark museum, volumetric light", seed 7.

**Setup**
- Ground truth comes from `painter_truth.py`: the 1.7B in HF fp32 on CPU (eager), then the diffusers
  `Flux2KleinPipeline` on MPS with the same `cond` and noise.
- The reference itself runs f32 on MPS, loads only the exported files plus `text.pt` (the taps), and rebuilds each
  weight from its trits.

**Results**

| check | vs diffusers **fp32** | vs diffusers **bf16** (old export_trace path) |
|---|---|---|
| adapter output | bit-identical (max abs 0) | (same input) |
| step 1, every block, text & image (flattened cos) | 1 − cos ≤ **8.6e-12** | 0.99991 – 0.999999 |
| step 1, worst single token | 1 − cos ≤ 3.4e-9 | 0.92 (text pad tokens 141 and 337, single blocks 13–18) |
| velocity cos, steps 1–4 | 1 − 3e-12 … 1 − 3e-11 | 0.99996, 0.99968, 0.99953, 0.99860 |
| final latent cos | 1 − 2.5e-11 | 0.99889 |
| final image PSNR (both TAEF2) | **114.6 dB** | 42.0 dB |
| `temb` / modulation (exported vs diffusers) | 1.3e-5 / 3.8e-5 max abs | – |

- **Other image comparisons:** TAEF2 vs the full VAE is 30.3 dB (diffusers fp32 latent).
- **Timing on the M4 Air:** diffusers fp32 takes 21 s and bf16 16 s for 4 steps; the reference takes about 5.3 s/step.

## 8. Surprises and traps
- **Pads are real computation.** All 512 positions run through 21 layers of the 1.7B with an unusual mask: a pad sees
  the real tokens but **not itself** (§1.2). The current browser runtime is plain causal and needs this pad mode. Every
  pad row differs, through its RoPE position.
- **`norm_out` order is (scale, shift).** The blocks use (shift, scale, gate).
- **Timestep dtype matters.** The bf16 pipeline embeds t = 960/884/716 instead of 958.09/883.98/717.50, and the
  sinusoid's highest frequency is 1 rad per unit of t. The export follows fp32.
- **`context_embedder` is dense** (23.6M bf16, the largest non-ternary matrix). It is exported fp16 (47 MB), but it is
  used once per prompt.
- **Massive activations:** `cond` row 0 (norm 29k) and the text stream (|x| up to 28.7k); see §6.
- **The adapter's scales are f32.** A strict fp16-scale layout would not be exact for it; the DiT scales are
  bf16-exact.
- **RoPE splits the head.** Text rotates only dims 96..127 and image only dims 32..95; dims 0..31 are position-free.
- **Hook pitfall (PyTorch):** a forward hook that returns anything (even `False`) replaces the module output.
  `painter_truth.py` uses def-hooks that return None.
- **Noise:** torch's CPU RNG stream is not practical to reproduce in the browser; validate from `noise.bin`.
