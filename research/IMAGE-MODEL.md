# Bonsai Image 4B: study log (2026-09-25)

## What it is
- **FLUX.2 [klein] 4B** (Black Forest Labs, Apache 2.0) with its diffusion transformer (DiT) made **ternary** ({−1,0,+1}) or **binary** ({−1,+1}), FP16 scale per group of 128. Released May 2026. White paper: `papers/bonsai-image-4b-whitepaper.pdf`.
- **DiT:** 25 blocks.
  - 5 double-stream blocks: image and text keep separate weights and meet in joint attention.
  - 20 single-stream blocks: one sequence, with attention and MLP running in parallel from one fused `to_qkv_mlp_proj` (3072→27648).
  - Hidden size 3072, 24 heads × 128.
  - 4-step distilled FlowMatch-Euler sampler, no CFG.
  - Modulation depends only on the timestep, never on the prompt, and is shared by all blocks of each type.
- **Verified** (`scripts/image_weight_stats.py`):
  - 3.68B DiT parameters (95%) are exactly ternary, about 34% zeros (30–47% per matrix). The binary variant is exactly ±1.
  - The remaining 195M stay bf16: modulation, embedders, `norm_out`, `proj_out`.
- **Text encoder:** stock **Qwen3-4B** in bf16, *not* ternary (the MLX/CUDA packs use 4-bit). The prompt is padded to 512 tokens. Hidden states from layers **9, 18, 27** are concatenated into a 7680-dim embedding per token. Pad tokens are *not* masked in the DiT, and they take about 21–28% of image→text attention.
- **VAE:** FLUX.2, 32 latent channels, 2×2 patchify, so a 512² image is 32×32 = 1024 image tokens and 1024² is 64×64 = 4096.

## Experiments (`scripts/image_*.py`, outputs in `renders/image/`)
All runs used diffusers 0.40 on PyTorch MPS in bf16, with the unpacked weights (no low-bit kernels).

| | 512², 4 steps | 1024², 4 steps |
|---|---|---|
| Plain generation (incl. VAE) | **15 s** | **55 s** |
| With full capture (attention + PCA every block) | ~260 s | – |

- **Memory:** text encoder and DiT together at bf16 (16 GB) pushed the 32 GB Air into swap. Prompts are now encoded once and cached (`image_encode_prompts.py` → `data/prompt_cache/`), and only the DiT and VAE stay resident.
- **Ternary vs binary** (`compare/grid_ternary_top_binary_bottom.png`): both look excellent at 512². Ternary follows prompts somewhat better ("frozen in glass").
- **x0 trajectory** (`traj_qwen_512_s16_x0sheet.png`): the sigma schedule stays near 1 until the end (4 steps: 1, .958, .884, .718). So the noisy latent x_t looks like static almost throughout, while the **x0 prediction** already shows the full composition after step 1 and then sharpens. At 16 steps the trajectory is smooth (blob → fox → fur detail) with no artifacts.
- **Block lens** (`ternary_qwen_512_s4_seed7/sheet_block_lens.png`): each block's image tokens are pushed through `norm_out` + `proj_out` and decoded. Decoded to pixels, the image only appears in the **last 3–5 blocks**. The token norm jumps from about 600 to 4284 in blocks 22–24.
- **PCA feature fields** (`sheet_block_pca.png`, `pca_film.mp4`): each block's 3072-dim features are projected to 3 colours. The **layout is already visible internally from about block 5**, long before it can be decoded. The result is vivid false-colour fields. Aligning component signs from block to block keeps the colours from flickering.
- **Words on the image** (`attention_words.png`): attention from image patches to each prompt token, averaged over the 20 single blocks. Each word lands on what it paints: "dark" and "museum" on the room, "bons" on the pot, "ai" on the canopy, "circuit" on the trunk core, "glowing" on the sparks, "light" on the whole tree.
- **Ternary LLM as text encoder** (`compare/grid_qwen_top_bonsai4b_bottom.png`):
  - Swapping Qwen3-4B for **Ternary-Bonsai-4B** (a ternary Qwen3-4B) keeps the *meaning*: tree, fisherman, glass wave, fox in snow.
  - But it produces an uncanny, fever-dream style: cracked faces, a red melting fox.
  - Naive recalibration failed. Per-channel affine collapsed every prompt to one generic image; norm-only scaling made it more ghostly.
  - A clean version would need a small trained adapter. The raw version is an aesthetic in its own right.

## Browser feasibility
- **Proof it runs:** Xenova's `webml-community/bonsai-image-webgpu` runs Bonsai Image fully in WebGPU from the MLX packs. It is a 3.3 GB download for the ternary variant and is closed source; its runtime is module-scoped and exposes no globals.
- **Measured on this M4 Air** (Chrome 152 in-app pane, ternary, seed 7):

| Run | Time |
|---|---|
| First load (download + GPU upload) | 97 s |
| 512², 4 steps | 13.1 s cold, **11.9 s warm** |
| 512², 16 steps | 40.6 s (about 2.3 s/step + 3.5 s for text encoding and VAE) |
| 768², 4 steps | 27.1 s |
| 1024², 4 steps | **52.4 s** |

- In-browser WebGPU with low-bit kernels is slightly *faster* than our unpacked PyTorch/MPS path (15 s / 55 s).
- **What we'd have to build:** our own WGSL for the Qwen3-4B encoder (4-bit), the DiT (ternary, 2-bit slots) and the VAE decoder. Capturing attention needs our own attention kernel, because flash attention never materialises the probabilities.
  - Readable references: `R4ZZ3/flux2-webgpu` and `madebyollin/taef2` (a tiny 10.7 MB decoder for cheap live previews).

## The adapter: a ternary LLM as the encoder (`scripts/adapter.py`, `adapter_ternarize.py`)

**Setup**
- **Method:** ridge regression, per token, from a ternary LLM's hidden states to Qwen3-4B's layers 9/18/27.
- **Training data:** 1,500 prompts (PartiPrompts + SD prompts) giving 109k token rows: all real tokens plus 32 padding positions per prompt. Forward passes stop at the last sampled pad, since the model is causal.
- **Prefix:** the 3 chat-template prefix tokens are identical for every prompt, so they are copied verbatim instead of regressed.
- **Held-out prompts:** 12, not in training.

| Encoder | Held-out cosine | Adapter size | Result (`compare/grid_adapters.png`, `compare_hard/`) |
|---|---|---|---|
| Ternary Bonsai 4B, raw | 0.54–0.73 | – | right subjects, uncanny look |
| Ternary Bonsai 4B + float adapter | 0.941 | 59M params, 118 MB fp16 | clean |
| **Ternary Bonsai 1.7B + float adapter** | 0.931 | 47M params, 94 MB fp16 | clean, close to stock |
| **Ternary Bonsai 1.7B + ternary adapter** | 0.900 | ~10 MB (g128, 44% zeros) | visually the same as the float adapter |

- **Harder prompts** (counting, text, style, small objects): the adapters match the stock encoder. All three make the same painter-level mistakes: 3 apples + 1 pear, "OPIEN".
- **Result:** the whole pipeline can be ternary. That is the 1.7B LLM (0.46 GB), a ternary adapter (10 MB), the ternary DiT (~1.2–1.4 GB) and the FP16 VAE (0.17 GB), about 2 GB total, against 3.3–3.9 GB with the stock encoder. The adapter is just one more ternary matmul in the runtime.
