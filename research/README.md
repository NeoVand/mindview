# mindview: research log

Phase 0 is research and experiments; nothing here is product code yet. This log was written 2026-09-25 on an M4 MacBook Air (32 GB, macOS 26.3, Chrome 152).

## 1. The PrismML Bonsai family

PrismML (Caltech spin-out) publishes open-weight models under Apache 2.0 at https://huggingface.co/prism-ml, 36 repos in total. Every model keeps its **base model's architecture unchanged**: only the weight representation changes. The compression method itself is proprietary ("Caltech IP"), and the white papers (`papers/`) cover deployment and benchmarks, with no training details.

| Release | Models | Base | Weights | Notes |
|---|---|---|---|---|
| 1-bit Bonsai (Mar 2026) | 1.7B / 4B / 8B | Qwen3 dense | {−1,+1} × FP16 scale per 128 (Q1_0, 1.125 bpw) | 1.7B = 0.24 GB |
| Ternary Bonsai (Apr 2026) | 1.7B / 4B / 8B | Qwen3 dense | {−1,0,+1} × FP16 scale per 128 (1.71 bpw ideal; packed as 2-bit) | 1.7B = 0.46 GB, 8B = 2.2 GB |
| Bonsai 27B (Jul 2026) | 1-bit + ternary 27B | Qwen3.6-27B hybrid | same, **unrotated** basis; `-unpacked` BF16 repos exist | 5.75 GB ternary |
| **Bonsai 2 27B** (Sep 16 2026) | ternary 27B | Qwen3.8-27B hybrid (`qwen35`) | ternary g128 in a **Hadamard-rotated basis** | PTQ1_0 5.95 GB / PQ2_0 7.2 GB, 98.2% of FP16 |
| Bonsai Image | 4B binary / ternary | diffusion | MLX / gemlite | text-to-image |

Every large matrix is low-bit, including the embeddings, attention, MLP and LM head. Only the norms (and in Bonsai 2, the DeltaNet gate/conv parameters, 0.1%) stay in higher precision.

### Architectures

**Dense Qwen3 (1.7B / 4B / 8B).**

| | Layers | Hidden | FFN | Heads Q/KV | head_dim | Vocab | Embeddings |
|---|---|---|---|---|---|---|---|
| 1.7B | 28 | 2048 | 6144 | 16 / 8 | 128 | 151,669 | tied |
| 4B | 36 | 2560 | 9728 | 32 / 8 | 128 | 151,669 | tied |
| 8B | 36 | 4096 | 12288 | 32 / 8 | 128 | 151,669 | untied |

All three use RoPE (YaRN ×4), RMSNorm, SwiGLU, and per-head q/k norms.

**Hybrid 27B (Bonsai 27B, Bonsai 2 27B).**
- 64 blocks, hidden 5120, FFN 17408, vocab 248,320, 262K context.
- Layers 3, 7, 11, …, 63 (16 layers) are **full attention**: 24 Q heads, 4 KV heads, head_dim 256, a sigmoid output gate, partial RoPE (64 of 256 dims, MRoPE).
- The other 48 layers are **Gated DeltaNet** linear attention. Each has a causal conv1d (kernel 4) over q/k/v with 16 key heads and 48 value heads (128 dims each), and **a 128×128 recurrent state matrix per value head**. That is 48 × 48 × 128 × 128 ≈ **37.7M floats of evolving "working memory"**, updated by the delta rule every token.
- Bonsai 2's rotation: `f(x) = W·(H₁₀₂₄·S·x)`, a sign flip followed by a blockwise Walsh–Hadamard transform applied to activations before each folded matmul. The sign vectors live in GGUF metadata under `prism.hadamard.*`.
- The best compact spec for building our own runtime is `reference/bonsai2-mlx-runtime/runtime.py`: a roughly 270-line reference forward pass with a lossless PQ2_0/PTQ1_0 codec, from the prism-ml HF repo (Apache 2.0).

### Packed formats (llama.cpp fork `PrismML-Eng/llama.cpp`, branch `prism`)

| Type | Id | Block (per 128 weights) | Decode |
|---|---|---|---|
| Q1_0 | 41 | fp16 d + 16 B of bits | `bit ? d : −d` (LSB-first) |
| PQ2_0 | 142 | fp16 d + 32 B of 2-bit codes | `(q−1)·d`, q ∈ {0,1,2} |
| PTQ1_0 | 143 | 24 B (5 trits per byte, base 3) + 2 B (4 trits per byte) + fp16 d at the end | `((byte·3ⁿ)&255)·3>>8`, in a non-positional element order (see the fork's `ptq1_0.glsl`) |
| Q2_0 | 42 | group **64** in the current fork. Older ternary GGUFs stored g128 under id 42 and are now "legacy". | |

The fork's Vulkan GLSL shaders (`dequant_*.comp`, `mul_mat_vecq_ptq1_0.comp`) are the easiest code to port to WGSL.

## 2. Experiments run

Scripts are in `scripts/`, outputs in `data/` and `renders/`. The Python env is `research/.venv` (uv, torch 2.14 with MPS, transformers 5.17).

### E1. Are the weights really ternary? (`weight_stats.py`)
- **Yes.** `Ternary-Bonsai-1.7B-unpacked` rounds losslessly to {−1,0,+1} × a per-128-group scale. `Bonsai-1.7B-unpacked` is exactly ±scale.
- The trits are **≈39% zero** (range 31–44% per matrix), with −1 and +1 almost perfectly balanced at about 30% each. About 40% of the network's connections are simply absent.
- The embedding matrix is 31% zero.

### E2. Full activation access (`capture.py`)
- Hooking the FP16-unpacked model in PyTorch on MPS captures everything: residual stream per layer, q/k/v, attention probabilities for every head, the 6144 MLP neurons per layer, and logits.
- The data volume is small: **about 1M floats (4 MB) per token** for the 1.7B model, plus attention over the context. Generation ran at 2.7 tok/s with full capture and readback to CPU.
- Sample output (ternary 1.7B, prompt *"What does it feel like to be a thought?"*): *"Being a thought is a paradoxical experience: it is both a sensation and an absence. It is the feeling of being aware, yet the feeling of being nothing…"*

### E3. What the internals look like (`render_weights.py`, `render_activations.py`)
- **Weights alone look like TV static.** Rendered pixel-for-pixel (cyan −1 / black 0 / amber +1), the trits are maximum-entropy noise. The only visible structure is in the **group scales**, which show bands that line up with the 128-row attention heads. Bonsai 2's Hadamard rotation makes the static even more uniform, by design. The meaning is not in the weights; it appears when activations flow through them.
- **The residual stream is a set of threads.** Its 2048 dimensions keep their identity through the depth of the network, showing up as vertical coherence across layers. A few **"massive" dimensions** dominate: dim 1999 has mean |h| ≈ 1352 against a median of 5. The stream's norm grows from 34 to 9000 across the 28 layers.
- **MLP neurons fire sparsely.** Only 1–3% of the 6144 neurons in a layer exceed 10% of that layer's maximum, so each token lights a sparse "starfield".
- **The logit lens is poetic.** Decoding each layer's residual stream as if it were the last layer, the middle layers wander through multilingual and associative fragments ("存在", "实体", "état", " living", " fascinating") before the answer settles at layers 21–27. You can watch a word form.

### E4. WebGPU on this machine (`webgpu-bench/index.html`)
- The Chrome 152 adapter reports `apple metal-3` with `shader-f16`, `subgroups` (size 32) and `timestamp-query`, and 4 GB max buffer and storage binding.
- A first, naive WGSL ternary matvec (2-bit slots, one subgroup per row) sustains **40–61 GB/s** on 400 MB of resident weights. The M4's peak is about 120 GB/s.
- Weights-only upper bounds for decode: **27B ≈ 8 tok/s**, 8B ≈ 28 tok/s, 1.7B ≈ 130 tok/s. For an art piece this is plenty, since we want it slow.
- Prior proof that the 27B runs in a browser: Xenova's `webml-community/ternary-bonsai-2-webgpu-kernels` Space runs the full Bonsai 2 27B in WebGPU. It is minified, has no license, and exposes no internals.

## 3. Landscape

- **Runtimes that can run Bonsai in a browser:** transformers.js v4 with onnx-community ONNX (logits only, fused GQA), `bitgpu` (MIT raw WGSL, 1-bit only, returns hidden states, the most forkable), llama.cpp WebGPU (q1_0), and Xenova's custom runtime. **None expose full internals, so we build our own**, which is the point anyway: every buffer the renderer needs is already on the GPU.
- **Prior art:**
  - bbycroft.net/llm: 3D, but an 85k-parameter toy model.
  - Transformer Explainer: SvelteKit, GPT-2, 2D, educational.
  - Anthropic circuit-tracer / Neuronpedia: offline analysis.
  - Refik Anadol: uses model outputs as material rather than showing the computation.
  - **Nothing renders a billion-parameter model's real computation live on-device.**
- **Bend:** Bend 2 launched 2026-09-17 (bendlang/bend) and is a new language with dependent types and proofs. It targets C, Metal, CUDA and JS. It has **no WebGPU or WASM**, supports F32/U32 only, and has tree-backed arrays, so it doesn't fit inference or rendering. It could be a native side piece for recursive or procedural graphics fed by token statistics over a socket.

## 4. Open questions / next experiments
1. Write a minimal WGSL forward pass for Ternary-Bonsai-1.7B, validated token-for-token against `data/capture_*.npz`, with every intermediate tensor kept in named GPU buffers.
2. Build visual studies against those live buffers: the residual "loom", the lit weight-static field, the rising logit-lens words, attention constellations, and (for 27B) the DeltaNet memory tiles.
3. Try the Xenova Space on this Air to measure real Bonsai 2 27B tok/s and memory headroom.
4. Build an offline 4K/8K video path: deterministic frame stepping plus WebCodecs, or Python renders from captures.
