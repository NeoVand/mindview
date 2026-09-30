# How much faster can the painter go, and by what mathematics

A night of measuring and reading, 30 September 2026. The question: is there a way, from the mathematics of ternary
matrices, information theory or a new representation, to make the painter tens of times faster on an ordinary laptop
without changing its pictures. The short answer: the arithmetic savings exist and are about 5×, but the hardware in a
laptop cannot cash them; the compute that can is the Neural Engine, which nobody is using; and the 100× is three
different kinds of work, only one of which is ours to write.

## 1. What was measured (M4 MacBook Air, Chrome 154 unless said otherwise)

| Quantity                                                          | Value                                                                     | How                                                                         |
| ----------------------------------------------------------------- | ------------------------------------------------------------------------- | --------------------------------------------------------------------------- |
| GPU f16 peak, scalar FMA                                          | **3.61 TFLOP/s**                                                          | 16 independent chains, 2,048 iterations, timestamps                         |
| GPU f32 peak                                                      | 3.45 TFLOP/s                                                              | same                                                                        |
| Our painter, effective                                            | 2.2 TFLOP/s (Best: 32 TFLOP in 15 s)                                      | mode timings                                                                |
| Subgroup-matrix (Metal `simdgroup_matrix`) MMA rate               | 3.6–3.7 TFLOP/s with data-dependent inputs                                | 12–29 "TFLOP/s" with loop-invariant inputs: fast-math folding, not hardware |
| A staged subgroup-matrix GEMM, 1024 × 3072 × 3072                 | 2.26 TFLOP/s, rel. error 1e-3 with f32 sums every 64 of K                 | pure f16 sums: 6e-3, too lossy                                              |
| Ternary dot product as AND + popcount                             | 3.1 TMAC/s **per activation bit-plane**                                   | loses at ≥ 2-bit activations (FMA: 1.8 TMAC/s)                              |
| Integer and float streams together                                | 3.08 Gop/s vs 3.6 float alone                                             | no dual issue                                                               |
| Workgroup-memory gather vs FMA                                    | ~4× slower per element                                                    | Apple GPU: 32 banks/cycle/core vs 128 FMAs                                  |
| Activations at the big matrices' inputs, int8 per token           | cosine ≥ 0.998 (most ≥ 0.9995)                                            | one Best step, 4 prompts, every Linear                                      |
| int4 per token, clipped at the 99.9th percentile                  | cosine 0.79–0.98                                                          | not usable                                                                  |
| Near-zero activations                                             | 2–7%                                                                      | no sparsity to mine                                                         |
| **Neural Engine**, 8 painter-sized MLP blocks, 1024 rows, Core ML | fp16 3.4 TOPS; int8 weights 6.4; **int8 weights + activations 12.9 TOPS** | same framework on the GPU: 2.0–3.0                                          |

So the browser painter runs at 59% of the GPU's real ceiling (my first "85%" used a latency-bound peak). Kernel work
can win at most 1.7×, and the subgroup-matrix path is a way to approach that ceiling, not to exceed it: the M4 has
no matrix hardware in the GPU. The M5 does (see §5).

## 2. The arithmetic of ternary matrices

A ternary matrix product is additions only: y_i = Σ_{j∈P_i} x_j − Σ_{j∈N_i} x_j. Different rows share partial sums,
and the question of how many additions are needed has a classical answer.

**Mailman** (Liberty & Zucker): any m × n matrix over a finite alphabet Σ can be preprocessed in O(mn) so that applying
it to any _real_ vector costs O(mn log|Σ| / log max{m, n}) operations. The construction: A = U P, where U holds every
possible column pattern once and P routes each input to its pattern's bin. **Williams** (2007): with O(n^{2+ε})
preprocessing, matrix-vector products over any finite semiring in O(n² / (ε log n)²). **Savage** (1974): n²/log_s n
arithmetic operations without preprocessing. Winograd's Ω(n²) holds only for unbounded algebras. **Linear computation
coding** (Müller, Gäde, Bereyhi 2021) states the same idea as rate-distortion theory for functions: a fixed 512 × 4096
real matrix at 16-bit precision needs 1.5 adders per entry instead of 7.5, with no multipliers. Ternary weights are the
extreme point of that curve, one adder per entry, and the results above push below one.

For our shapes the useful variant groups the _activations_: split the K = 3,072 inputs into groups of k, tabulate all
3^k signed sums of each group (a Gray-code walk costs one addition per entry, mirror symmetry halves the table), then
every output row does one lookup and one addition per group. Per token, for m output rows:

    cost ≈ (K/k) · 3^k / 2  +  m · K / k        versus     m · K

With 3^k ≈ m the table cost is amortised over the rows and the ratio is k / 1.5: for m = 3,072, k ≈ 7, about **4.9×
fewer additions**; for the MLP's 12,288 rows, about 5.7×. That matches what T-MAC and bitnet.cpp measure on CPUs
(3–6×), where the lookup instruction (`tbl`) is the cheap operation.

Why a GPU cannot cash it, in numbers: the table for k = 7 is 2,187 entries × 439 groups × 2 bytes = 1.9 MB _per token_;
even k = 4 is 124 KB against 32 KB of workgroup memory, so tables must be tiled and rebuilt; and each lookup is a
gather, which the M4 executes ~4× slower per element than an FMA, before bank conflicts. The ~5× fewer operations
become ~1× or worse. The same holds for the bit-plane trick: popcount buys 1.7× per plane, and no activation of
ours survives below 6 bits. Both measured, both closed.

Where the mathematics _is_ being cashed: LUT Tensor Core (ISCA 2025) replaces the MMA instruction with a table-lookup
MMA; TENET (2025) is an FPGA and ASIC for ternary LLMs reporting 2.7× lower latency and 21× better energy than an A100;
TerEffic on FPGA. These exist because commodity MAC hardware cannot exploit ternary weights beyond bandwidth. On a
laptop, dense MAC engines (GPU, Neural Engine, AMX) win.

## 3. Caching

Caching pays where computation recurs. In diffusion the recurrence is across steps: FORA, Δ-DiT and TeaCache reuse
block outputs; ToCa caches per token; DiffSparse learns which tokens to recompute (1.81× on FLUX.1-schnell at 4 steps
with FID preserved, a learned predictor trained in 4–10 GPU-hours); RAS updates only the regions the model is focused
on (training-free, 1.6–2.4× at more steps). All of it needs step-to-step similarity: at 4 steps (Best) it is worth
1.3–1.8×; at 2 (Fast) less, since the sketch changes resolution; at 1 (Instant) nothing recurs.

Across prompts the computation diverges in the first block, because joint attention mixes the text into every image
token immediately, and nonlinearities destroy any low-rank structure of the difference (the GELU's Jacobian is
diagonal and dense). What is prompt-independent we already cache: the modulation vectors per step (`schedules`),
and one could add block 0's image-side projections at a fixed seed (~1%). Text K/V reuse across steps was tried and
broke lettering. Caching is exhausted at one step.

## 4. Representation

- **int8 activations** are safe (measured, §1). They are the precondition for every integer engine: the Neural Engine
  (int8 direct hand-off since macOS 26), the M5 GPU's neural accelerators (int8 ≈ 2× fp16), NVIDIA's int8 tensor cores.
  And ternary weights are _exactly_ representable in int8, with no dequantisation error at all.
- **int4 activations** fail: cosine ≤ 0.98 per token even clipped. SVDQuant makes W4A4 work on FLUX only with a rank-32
  high-precision branch and channel smoothing, and still lands at LPIPS 0.29 to bf16 on schnell (W8A8: 0.12). Visible.
- **2-bit palettised weights on the Neural Engine.** Core ML supports 1–8-bit lookup-table weights with per-group
  tables (macOS 15+). A ternary matrix with one scale per 128 weights _is_ a 2-bit palette {−s, 0, +s} per group:
  4× less weight traffic than int8, and a 3,072 × 12,288 matrix becomes 9.4 MB, under the engine's 32 MB SRAM, where
  the literature says it runs at its compute roofline instead of 30% below it. Draw Things reports ~22 TFLOPS on M4
  with int8; a 2-bit ternary block should do at least as well. This is the one place where being ternary is a
  structural advantage on hardware a laptop already has, and I found no one who has done it for a DiT.

## 5. The compute a laptop hides

- **M4 Neural Engine**: 19 TFLOPS fp16 true peak (the marketed 38 TOPS counts int8, which is dequantised to fp16
  before the MACs). 5.7 TFLOPS on a single 2048² matmul, 94% of peak in deep graphs; 1 × 1 convolutions in the
  (B, C, 1, S) layout run ~3× faster than matmuls; 32 MB SRAM cliff; 0.1 ms dispatch. Reachable only through Core ML
  (Draw Things treats it as an accelerator inside their own stack, per-matmul programs, 1.8× end-to-end for image
  generation on M4). Our own 12.9 TOPS on the first try, before any of these fits.
- **M5**: every GPU core gains a neural accelerator: 1,024 fp16 FLOP per cycle per core; the 10-core M5 lands near
  15–17 TFLOPS, the M5 Max at ~70 (130 TOPS int8). Apple measures FLUX-dev 3.8× faster on M5 than M4 in MLX.
  Reachable through Metal 4 tensor ops / MPP; not from WebGPU's subgroup matrices yet.
- **AMX** on the CPU: ~2 TFLOPS through Accelerate, native only.

## 6. Ceilings, composed

For the same pictures, on this M4 Air:

| Path                                                                                  | Best (15 s) | Fast (6 s) | Instant (4 s) |
| ------------------------------------------------------------------------------------- | ----------- | ---------- | ------------- |
| Browser, kernels to the ceiling                                                       | 9 s         | 3.4 s      | 2.2 s         |
| Native: Neural Engine (2-bit palette, int8 activations, 1 × 1 layout) + GPU alongside | ~2.5 s      | ~1 s       | ~0.7 s        |
| plus learned token sparsity (DiffSparse/RAS), quality held to the held-out LPIPS      | ~1.5 s      | ~0.8 s     | ~0.7 s        |

And on an M5 Max, natively, another ~4× from the hardware. The 100× decomposes as roughly 8× native software on the
M4 (ours to write), 4× one hardware generation (Apple's), and 3× a smaller distilled painter (a different network,
the only step that touches quality, and the very step that produced klein from FLUX.2 32B).

## 7. What this means for mindview

The site runs in the browser by design. The browser's share of the above is 1.5–1.7×, from kernel work. Everything
larger needs a native engine (Swift, Core ML, Metal 4), which would be a second runtime for the same one-file model,
with the same pictures. The first thing to build, if we go there, is the smallest thing that settles it: one single
block of the painter as a Core ML program with 2-bit palettised ternary weights and int8 activations, timed on the
Neural Engine against the 25 ms it takes on the GPU today, and checked against PyTorch to the cosine we hold the
browser to.

## References

- Liberty, Zucker. _The Mailman algorithm: a note on matrix vector multiplication._ Yale TR-1402. https://edoliberty.com/papers/mailmanAlgorithm.pdf
- Williams. _Matrix-vector multiplication in sub-quadratic time (some preprocessing required)._ SODA 2007. https://people.csail.mit.edu/rrw/mat-vec3.pdf
- Müller, Gäde, Bereyhi. _Linear computation coding._ 2021. https://arxiv.org/abs/2102.00398
- Wei et al. _T-MAC: CPU renaissance via table lookup for low-bit LLM deployment on edge._ https://arxiv.org/abs/2407.00088
- Wang et al. _Bitnet.cpp: efficient edge inference for ternary LLMs._ https://arxiv.org/abs/2502.11880
- Mo et al. _LUT Tensor Core._ ISCA 2025. https://arxiv.org/abs/2408.06003
- _TENET: a sparsity-aware LUT-centric architecture for ternary LLM inference on edge._ https://arxiv.org/abs/2509.13765
- Li et al. _SVDQuant: absorbing outliers by low-rank components for 4-bit diffusion models._ ICLR 2025. https://arxiv.org/abs/2411.05007
- _DiffSparse: accelerating diffusion transformers with learned token sparsity._ https://arxiv.org/abs/2604.03674
- Liu et al. _RAS: region-adaptive sampling for diffusion transformers._ https://arxiv.org/abs/2502.10389
- Zou et al. _ToCa: token-wise feature caching._ ICLR 2025. https://arxiv.org/abs/2409.18523
- Apple. _Deploying transformers on the Apple Neural Engine._ https://machinelearning.apple.com/research/neural-engine-transformers
- Apple. _Exploring LLMs with MLX and the neural accelerators in the M5 GPU._ https://machinelearning.apple.com/research/exploring-llms-mlx-m5
- Bryngelson. _Apple Neural Engine: architecture, programming, and performance._ 2026. https://arxiv.org/abs/2606.22283
- _Orion: characterizing and programming Apple's Neural Engine._ https://arxiv.org/abs/2603.06728
- _Inside the M4 Apple Neural Engine, part 2: benchmarks._ https://maderix.substack.com/p/inside-the-m4-apple-neural-engine-615
- Zakharko. _Investigating the GPU neural accelerators on Apple A19/M5._ https://tzakharko.github.io/apple-neural-accelerators-benchmark/
- Draw Things. _Making Apple Neural Engine work in a custom inference stack._ https://engineering.drawthings.ai/p/making-apple-neural-engine-work-in
- Core ML Tools, _Palettization overview._ https://apple.github.io/coremltools/docs-guides/source/opt-palettization-overview.html

## 8. The pyramid test (30 September, evening): what the painter's depth actually does

`research/scripts/pyramid_probe.py` runs Best on the 12 held-out prompts and reads every block through the tuned lens;
`dictionary_probe.py` measures how many MLP units a token uses; `pyramid_steps.py` runs Best with its first steps at
16 × 16 tokens. Results in `research/renders/pyramid/`.

- **Depth is not coarse-to-fine.** Within a step, the lens picture's error at 128 px tracks its error at 512 px almost
  exactly until the last steps; early blocks do not hold the coarse picture, and most of a step's commitment happens in
  its last ten blocks. Coarse-before-fine exists only across steps (end of step 2: LPIPS 0.24 at 128 px vs 0.32 at 512;
  step 3: 0.09 vs 0.15). Average-pooling a block's update to 16 × 16 keeps 65–85% of its energy, to 8 × 8 only 50–70%;
  the last double block and the last single block add updates 2–4× the state's size, 95% coarse. A single-pass
  coarse-to-fine network built from these blocks would have to be relearned, not rearranged.
- **The dictionary is dense.** Per token, 90% of an MLP's output contribution needs ~4,900 of the 9,216 SwiGLU units,
  95% ~6,000, 99% ~7,500; keeping the top half leaves 5–13% output error, the top quarter 26–36%. No sparsity for a
  router to find; MoE-fication would create it by retraining.
- **Coarse steps hold.** Three steps at 16 × 16 then one at 32 × 32 (44% of Best's work), with no training, paints
  nine of twelve prompts as well as Best by eye (the astronaut, the lighthouse and the kite lose something). LPIPS to
  Best is 0.63 because the coarse noise differs; a faithful student would take the teacher's noise, pool it 2 × 2
  (rescaled by 2) for the coarse steps and inject the high-frequency remainder at the enlargement, so that it can be
  distilled to Best's own pictures per sample with the existing pipeline (`distill_steps.py`, a `pyr` mode).

So on this network, software alone gives ~2.3× on Best (coarse steps, to be distilled), plus P-frame caching for
interactive editing; the realtime target needs the native engine and, beyond that, a different network.
