# mindview: combined piece, working plan (draft 2026-09-25)

**One ternary mind reads, another paints.** A visitor types a phrase. A language model reads it, token by token, through all of its layers. Three of those layers are lifted out and handed to the painter, the Bonsai Image DiT, which builds the picture over 25 blocks and a few denoising steps. Everything shown is the real computation, read straight from GPU buffers.

## Three acts, grounded in what the experiments showed

**I · Reading.** The language model processes the prompt.
- Each token is a column. Its residual stream of 2048–2560 threads rises through the layers (the "loom").
- Logit-lens ghost words drift through other languages before they settle.
- The three tap layers (9 / 18 / 27 for Qwen3-4B) glow as they are sampled.

**II · Handoff.** 512 text tokens enter the painter.
- The real words (about 20–40) are bright.
- The ~480 **padding tokens** are a dim "choir of silence". They are never masked, and they still absorb 21–28% of the image's attention.

**III · Painting.** The 1024 image patches (32×32 at 512²) form a lattice.
- The 25 blocks become a tower of translucent planes, each coloured by its own PCA feature field. The layout is visible internally from about block 5, while the pixels only appear in the last 3–5 blocks.
- Attention beams run from each word to the patches it paints ("museum" onto the walls, "circuit" onto the trunk).
- The canvas shows the x0 prediction: the full composition after step 1, sharpening afterwards.
- Behind everything sits the static of 3.68B trits, about 34% of them zero.

## Engineering shape
- **Our own WebGPU runtime (WGSL).** There is no licensable runtime we could instrument: Xenova's is closed and exposes nothing. It has three parts:
  - **Encoder:** see "encoder decision" below.
  - **DiT:** ternary in 2-bit slots, with fused qkv+mlp for the single-stream blocks.
  - **VAE decoder:** FP16, plus **TAEF2** (10.7 MB) for cheap per-block and per-step previews.
- **Target speed**, set by Xenova's runtime on this Air: 512²/4 steps in about 12 s, 1024² in about 52 s. Being 2–3× slower is fine for the art; the runtime is a *stepper* the renderer can pause and slow down.
- **Viz compute on GPU:**
  - PCA colours via power iteration seeded from the previous block. This is cheap, and it also stabilises the colours across blocks.
  - Image→prompt-token attention computed as a separate small pass (1024 × ~30 × 24 heads), because flash attention hides the probabilities.
  - Everything else comes from binding the runtime's buffers directly.
- **Verification:** every kernel is checked against the PyTorch captures in `research/`, using the same seeds and the cached prompt embeddings.

## Milestones
1. **Visual studies on recorded traces.** Build the three acts in SvelteKit + raw WebGPU, fed by trace files exported from the Python captures. This designs the look without waiting for the runtime.
2. **DiT + decoder runtime in WGSL**, fed by cached embeddings, validated block by block.
3. **Language-stage runtime**, then the live pipeline (visitor input → image), pacing controls, and kiosk mode.
4. **Video export:** deterministic frame stepping and WebCodecs, at 4K.

## Encoder decision: settled by the adapter experiment
**Chosen: ternary Bonsai 1.7B + ternary linear adapter** (held-out cosine 0.90, images on par with the stock encoder; see IMAGE-MODEL.md). The whole piece is then ternary except the VAE, about 2 GB. Its Act I shows the 1.7B's 28 layers; the taps are layers 7/14/21.

Options that were considered:
- (a) **Stock Qwen3-4B, 4-bit:** exact image quality, 2.3 GB, not ternary.
- (b) **Ternary Bonsai 4B + linear adapter:** everything ternary, about 1 GB.
- (c) **Ternary Bonsai 1.7B + adapter:** the LLM you chose. It could also *answer or elaborate* before painting. About 0.5 GB, making the whole piece roughly 2.1 GB.

## First trace (`traces/bonsai-museum-ternary/`, from `scripts/export_trace.py`)
- **Pipeline:** all ternary. Bonsai 1.7B, then the ternary adapter, then the DiT (512², 4 steps). The files are 26 MB: `manifest.json` (shapes, tokens, logit-lens words, sigmas), raw float16 `.bin` arrays and PNGs.
- **Timing:** full capture takes 5 s for Act I, 23 s for Act III, and 14 s to decode 108 frames. PCA runs as warm-started subspace iteration on the GPU, 10× faster than the eigh version.
- **Act I logit lens on the 1.7B** (the visitor's words drifting through layers 0→28):
  - " tree" → 在日本 ("in Japan") → 在深圳 → " planted"
  - " glowing" → " Flame" → 灯光 ("lamplight") → 灯 → " moss"
  - " museum" → " halls" → 钟 ("bell") → 保存 ("preserve")
- **TAEF2 block lens:** TAEF2 expects the *bn-normalised* latents after unpatchify (error 0.03 vs 0.11 when denormalised first). Decoding 100 lens frames with it is cheap.

## M1 status (2026-09-25): first visual study running
- **Where it lives:** `src/lib/engine/` (TypeScript + WGSL, no 3D library) and `src/routes/+page.svelte`. It replays `static/traces/bonsai-museum-ternary/` in a loop of about 110 s:
  - intro: the prompt, large
  - Act I: the loom of 29 token columns × 256 threads × 28 layers, with the input words and logit-lens words
  - Act II: particles from layers 7/14/21, plus padding dust, fly to the painter
  - Act III: a tower of 25 planes lit by the block scan, beams from the focused word to the patches it attends to, and a canvas showing TAEF2 previews over the previous step's x0
  - finale: the final image alone
- **Rendering:** HDR target, 13-tap bloom, ACES tonemap. Colours follow the ternary duotone (glacier −1, ember +1); C toggles to false-colour PCA.
- **Performance:** 60 fps in every act at 1808×1130 on the M4 Air. `pnpm build` writes a static site of 26 MB, trace included.
- **Controls:** space pauses, ←/→ seek (shift for bigger jumps), C switches colours, F goes fullscreen, R restarts. In dev, `window.mindview` exposes the engine.

## Honest image formation (after review, 2026-09-25)
**The feedback:** the canvas felt synthetic. It stayed noise until the last blocks and then snapped into a sharp image. There were three causes:
1. The **raw lens** (block state → `norm_out` → `proj_out`) can't read mid-network states.
2. I mixed the previous step's picture under the noise, a compositing trick.
3. I switched from the TAEF2 preview to the full VAE mid-way.

**The fixes:** every frame on the canvas is now computed by the model or read out by a fixed linear map.
- **Tuned lens** (`scripts/tuned_lens.py`):
  - one affine readout per block, from the 3072-d image-token state to that step's velocity
  - fitted by ridge regression over 16 other prompts × 4 steps × 1024 tokens
  - the bonsai prompt is not in the training set
  - result: step 1 block 1 is already a blurry earth-toned "tree"; around block 12 the prompt's style takes over (brown → glowing blue); by block 20 the tree is crisp; later steps refine from the first block on
  - comparison sheet: `renders/image/tuned_lens_0d0bf2.png`
- **The canvas uses one decoder (TAEF2) throughout the painting**, with no blending across steps.
- **Decoder stages** (`scripts/decoder_stages.py`) run on the true final latent (`final_latent.npy`), with per-stage linear RGB readouts fitted on 40 images. The finale walks through them: the latent the painter hands over (64 × 64 × 32, false colour) → 64² → 128² → 256² → 512² → the image.
- **The finished image** appears only through the decoder's own resolution stages.

## Second review: "it feels gimmicky, not like looking inside" (2026-09-25)
**The feedback:**
- The image waits at the end of the tower like a destination.
- Nothing causally connects billions of weights × activations to the result.
- Act I is strings with no sense of words mixing or combining.
- The demo should be live (type → watch), not a recording.

**The direction taken:** show the computation's three kinds of work, each where it happens.
1. **Mixing (attention):** words read each other.
2. **Tallying (ternary matmul):** each neuron sums thousands of +x / −x / skip votes.
3. **Accumulating (residual):** each block adds its change, and that change is shown on the block itself.

Concept stills from real data are in `research/renders/concepts/` (A mixing, B tally, C blocks paint), generated by `scripts/concepts.py`.

**Live runtime, milestone 2a, done:** `src/lib/runtime/`.
- Contents: GGUF reader, byte-level BPE tokenizer, and WGSL kernels (ternary matmul, RMSNorm, YaRN RoPE, causal attention that keeps its probabilities, SwiGLU) for Ternary Bonsai 1.7B.
- **Checked against PyTorch** (`/lab/llm`):
  - tokenizer identical on all tests, including unicode and emoji
  - all 29 hidden states match with per-token cosine 1.00000
  - next-token top-5 identical
- **Speed:** prefill 0.5–0.7 s per prompt with naive kernels.
- **What stays on the GPU:** every layer's residual, the post-attention residual, the MLP activations and the attention probabilities.

**Live study, `/lab/assembly`:** type a prompt, and the 1.7B reads it in the tab.
- Each layer, tokens are placed by 3D MDS of their hidden-state directions, aligned layer to layer.
- They are linked by the attention that produced that layer (strongest head, attention sinks excluded), with particles flowing from source to reader.
- Phrases visibly assemble:
  - bonsai prompt: "glowing circuit-ry", "volum-etric light", "dark museum", "bons-ai tree"
  - lighthouse prompt: "old lighthouse keeper", "feeding cats", "thunder storm"

## Third review: "rectangular, white noise; find a genius idea" (2026-09-25)
**The feedback:**
- The slab study (every weight drawn as a cell, lit by weight × input) read as rectangles of static.
- The assembly study stalled after 30 s.
- There was no image generation in either.
- The user asked for an inventive form, wants to orbit the view, and suggested flow and vector fields ("dot products are like flow").

**What the experiments showed** (`scripts/flow_field.py`, `concept_*.py`, `renders/flow/`):
- **Literal vector field:** the real MLP field sampled on a plane through a word is almost uniform (parallel streamlines). In 2048-d the variation happens off any 2D slice, so it's a dead end.
- **Atlas:** a plane through the words at one layer, each point named by argmax over the vocabulary's dot products, gives a legible "map of meaning" with the neuron membranes as contour lines and attention as routes. Promising as a zoomed-in view.
- **Threads, chosen:**
  - Each word is a thread falling through the layers.
  - Each layer's step is drawn as the real running sum of its terms: attention (P[h,t,s] · Wo_h v_s, coloured by source word) then 6,144 neuron pushes (act_j · Wdown[:, j]).
  - Everything is projected onto 2 fixed PCA directions, with a per-layer rescale.
  - Each step lands exactly on the next layer's state.
  - The words start apart and weave together.
- **Live study `/lab/threads`:**
  - The 1.7B runs in the tab. Wo and Wdown columns are projected on the GPU (kernel `PROJECT_WGSL`).
  - About 3M vertices are drawn as additive screen-space line quads (`viz/lines.ts`) at 58 fps on the M4 Air.
  - Orbit and zoom, names riding the tips, and a live neuron counter in layer 1.
- **Next:** the painter as the continuation. The 1024 image patches become threads (grid position plus wander), coloured by the tuned lens and lent light by the word threads through joint attention. The canvas forms where they land. This needs the DiT + VAE ported to WGSL.

## The whole journey is live (2026-09-26)
**The user's direction:**
- Horizontal threads, with the camera riding alongside as the computation happens, then going back through the past.
- The threads then feed the diffusion painter, which forms its own maps and images.

**Built:**
- **`/lab/threads`, one continuous ride:**
  - Reader: 28 layers of word threads, with the model's ghost words at every layer (logit lens on the GPU).
  - Handoff: lines from layers 7, 14 and 21.
  - Painter, computed live block by block:
    - 1,024 patch fibres coloured by each block's guess
    - the bundle's front face showing that guess (tuned lens → x0 → TAEF2 first stage → probe64)
    - word strands above, with attention lines into the patches they paint
    - each pass ends in a full TAEF2 decode left standing as a canvas
    - the finished picture
  - Then a finale, an overview and a rewind. The timeline scrubber covers the whole journey.
- **Runtime pieces, all validated:**
  - `gemm.ts`: tiled ternary GEMM, about 1.2 TFLOP/s
  - `bonsai-llm.ts`: `encodeLong` (512 tokens, pad masking) and `readout` (logit lens)
  - `taef2.ts`: max error 0.5/255
  - `painter.ts`: the DiT, exact against PyTorch, 8.5 s per step
  - `/lab/painter` holds the checks
