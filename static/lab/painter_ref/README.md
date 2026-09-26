# painter_ref: validation references for the WebGPU painter

Written by `research/scripts/painter_reference.py`. That script is a from-scratch float32 PyTorch implementation of
`research/PAINTER-SPEC.md`, and it reads only `/models/bonsai-image-4b/` plus the 1.7B tap features. It matches
diffusers fp32 to within 1 − cos ≤ 1e-11 per block, and its final TAEF2 image matches at 114.6 dB PSNR.

- **Case:** prompt "a bonsai tree made of glowing circuitry in a dark museum, volumetric light", seed 7, 512², 4 steps.
- **Precision:** every tensor was computed in f32, with weights taken exactly from the exported files.
- **Files:** every `.bin` is raw little-endian **float32**, row-major, with no header. Shapes and descriptions are in
  `ref.json` → `arrays`.

## Conventions

- **Joint sequence:** 1536 rows. Rows 0..511 are text tokens and rows 512..1535 are image tokens.
- **Image token order:** image token `t` sits at latent-grid cell `(h, w) = (t / 32, t % 32)`.
- **Probe rows** (`ref.json` → `probe_rows`): text tokens 0..7 and image tokens 0..7, which are joint rows `0..7, 512..519`.
- **Head layout:** head vectors are flattened as `head * 128 + d`.
- **Latents:** `[1024 tokens, 128 channels]` in the bn-normalised space the DiT works in. Channel `4*c + 2*dy + dx`
  holds VAE channel `c` at pixel `(2h + dy, 2w + dx)`.
- **Text rows** (`ref.json` → `rows`): all real tokens (0..28), three pads (29..31) and the last pad (511).
- **Row checksums** (`*_stats`): `[sum, sum of squares, absmax]` for all 512 rows.

## Files

| file | shape | what |
|---|---|---|
| `ref.json` | | prompt, the 512 token ids, `n_real` (29), schedule, array index, and the metrics vs diffusers |
| `taps_rows.bin` | [33, 6144] | 1.7B residual stream after layers 7, 14, 21, concatenated `h7 \| h14 \| h21` (fp32 run, pad masking as in the spec) |
| `taps_stats.bin` | [512, 3] | row checksums of the taps |
| `cond_rows.bin` | [33, 7680] | adapter output, i.e. the DiT conditioning (rows 0..2 are the copied prefix) |
| `cond_stats.bin` | [512, 3] | row checksums of the conditioning |
| `noise.bin` | [1024, 128] | initial latent x₀ (`torch.randn` seed 7, fp32, CPU generator) |
| `step1_blocks.bin` | [26, 16, 3072] | step 1 residual stream for the probe rows. Index 0 is the embedder outputs (`context_embedder` for text, `x_embedder` for image), 1..5 are after double blocks 0..4, 6..25 after single blocks 0..19 |
| `step1_block_norms.bin` | [26, 1536] | L2 norm of every token at the same 26 points (checks all rows cheaply) |
| `step1_double0_norm1.bin` | [16, 3072] | double block 0: `(1+scale)·LN(x)+shift`, the attention input (text rows use the txt modulation) |
| `step1_double0_{q,k}.bin` | [16, 3072] | double block 0: q and k after QK RMSNorm **and** RoPE |
| `step1_double0_v.bin` | [16, 3072] | double block 0: v |
| `step1_double0_attn.bin` | [16, 3072] | double block 0: attention output before `to_out.0` / `to_add_out` |
| `step1_double0_resid_attn.bin` | [16, 3072] | double block 0: residual after the gated attention update |
| `step1_double0_ff_act.bin` | [16, 9216] | double block 0: SwiGLU output, the input of `linear_out` |
| `step1_single0_norm.bin` | [16, 3072] | single block 0: modulated LayerNorm input |
| `step1_single0_{q,k,v,attn}.bin` | [16, 3072] | single block 0: q and k after norm and RoPE, v, and the attention output |
| `step1_single0_mlp_act.bin` | [16, 9216] | single block 0: SwiGLU of the MLP part of `to_qkv_mlp_proj` |
| `velocity.bin` | [4, 1024, 128] | DiT output v for each step |
| `latents.bin` | [4, 1024, 128] | latent after each Euler update; `latents[3]` is the final latent that TAEF2 decodes |
| `taef2_input_crop.bin` | [32, 8, 8] | unpatchified TAEF2 input, top-left 8×8 |
| `taef2_res{64,128,256}_crop.bin` | [64, 8, 8] | TAEF2 feature map just before each 2× upsample (after that stage's blocks) |
| `taef2_res512_crop.bin` | [64, 8, 8] | TAEF2 feature map before the final RGB conv |
| `taef2_rgb_raw_crop.bin` | [3, 8, 8] | final conv output before `clamp(0, 1)` |
| `taef2_*_chstats.bin` | [C, 2] | per-channel `[mean, std]` of the full map at each of those points |
| `final.png` | 512² | this reference's image (TAEF2) |
| `diffusers_fp32_taef2.png` | 512² | diffusers fp32 pipeline, final latent decoded by TAEF2 |
| `diffusers_fp32_vae.png` | 512² | diffusers fp32 pipeline decoded by the full FLUX.2 VAE (TAEF2 vs VAE: 30.3 dB) |

## Suggested tolerances for the WGSL port

These tolerances assume f32 activations with a different summation order:

- **Cosine:** ≥ 0.99999 for every row of `step1_blocks` and the internals, and ≥ 0.9999 for velocity and latents.
- **Relative L2:** ≤ 1e-3.
- **Absolute errors:** the text rows reach |x| ≈ 28,700 (row 0, the `<|im_start|>` prefix), so compare relative
  error, not absolute error.
- **Final image:** expect > 40 dB PSNR.

For scale, diffusers in bf16 already drifts to a per-block cosine of 0.9999 and a final image of 42 dB.
