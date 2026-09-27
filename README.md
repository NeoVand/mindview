# mindview

Two ternary neural networks, one reading a prompt and one painting it, shown while they compute. Everything runs in
your browser with WebGPU, and every number on screen is one the models computed. The landing opens with a recording of
one real run, so that it starts without a download; change the words and the models run in your tab.

**Live:** [neovand.github.io/mindview](https://neovand.github.io/mindview/) ·
[the labs](https://neovand.github.io/mindview/lab) · [paint](https://neovand.github.io/mindview/paint)

![The painter's attention: each of 24 heads' shares over the picture, for one patch of the trunk](docs/where-it-looks.png)

## The pipeline

Every weight that does the work is ternary (−1, 0 or +1, with one scale per 128 weights). The models never multiply
by their weights: each weight adds its input, subtracts it or skips it.

1. **The reader:** [Ternary Bonsai 1.7B](https://huggingface.co/prism-ml/Ternary-Bonsai-1.7B-gguf) by PrismML,
   28 layers. It reads the prompt.
2. **The adapter:** a ternary linear map we trained. It takes each token's state from reader layers 7, 14 and 21 and
   turns it into the painter's conditioning, in place of the 4B text encoder the painter was trained with.
3. **The painter:** the diffusion transformer of Bonsai Image 4B, which is FLUX.2 [klein] 4B with ternary weights.
   It has 5 double blocks and 20 single blocks and runs over 1,536 rows (512 for the prompt, 1,024 patches of the
   picture), four steps from noise.
4. **The decoder:** [TAEF2](https://huggingface.co/madebyollin/taef2), a tiny autoencoder that turns the latent into
   a 512 × 512 picture.

After every block, a per-block lens (a small readout fitted with `research/scripts/tuned_lens.py`) shows the
picture the painter has in mind at that point.

## One file, about a gigabyte

The labs run the whole pipeline above, because they show all of it. Painting alone needs less.
[mindview-t2i-turbo](https://huggingface.co/mohsenvand/mindview-t2i-turbo) is the same kind of pipeline packed into one
file of 1,208 MB, and [`/paint`](https://neovand.github.io/mindview/paint) and the landing run it. On a MacBook Air (M4)
in Chrome a picture takes about 4 s in one step (Instant), 6 s in two (Fast) and 15 s in four (Best).
[mindview-t2i](https://huggingface.co/mohsenvand/mindview-t2i) is the earlier file, without the trained branches.

- **The reader keeps 9 of its 28 layers.** A sweep over which reader layers to tap
  (`research/scripts/adapter_layers.py`) found that shallow layers condition the painter about as well as deep ones.
  The map from layers 3, 6 and 9 scores a per-token cosine of 0.959 against the stock encoder, measured in the
  painter's input space on held-out prompts. The site's adapter, which reads layers 7, 14 and 21, scores 0.952.
  Every layer the map does not read is a layer the file does not carry.
- **The map and the painter's context embedder are one matrix** (6,144 → 3,072, f16), because both are linear.
- **Two steps, from a borrowed LoRA.** radames'
  [SANA-Sprint LoRA](https://huggingface.co/radames/FLUX.2-klein-Sana-Sprint), trained on the original klein, carries
  over to the ternary painter; its best rank-8 approximation paints like the full rank 256
  (`research/scripts/lora_steps.py`, `lora_compress.py`). It is the start of the 2-step branch (below).
- **Two steps, the first at a quarter of the size.** Fast sketches its first step at 256 × 256 with the first 128 text
  rows, upsamples that step's guess of the clean picture in latent space, noises it back and paints the second step at
  512 × 512: a third less time, the same pictures (`research/scripts/progressive.py`, `progressive_rows.py`).
- **One step, trained on the ternary painter.** [epfl-vita/flux2-klein-1step-rdm](https://huggingface.co/epfl-vita/flux2-klein-1step-rdm)
  distils klein to one step with a full fine-tune. Its change carries over to the ternary painter at rank 4, with the
  modulation recomputed at σ = 1, but softly and with defects (`rdm_delta.py`, `rdm_test.py`, `rdm_rank.py`). So a
  rank-32 branch is trained on the ternary painter with RDM on klein as the teacher: LPIPS on TAEF2-decoded pictures,
  2,000 prompts, 3,000 steps on one H100, about 25 minutes (`research/scripts/distill_1step.py`,
  `research/cloud/distill_modal.py`). Plain mean squared error made it blurrier.
- **Two and four steps, trained the same way.** A rank-32 branch for two steps (started from the LoRA's rank-8
  approximation) learns from klein with the LoRA, run exactly as Fast runs; another for four steps learns from klein
  itself. Both learn offsets on each step's modulation, and at every step the student starts from the teacher's own
  noisy picture. 2,000 prompts, 3,000 steps each on one H100, about 20 minutes each; on 12 held-out prompts the LPIPS
  distance to the teacher went from 0.496 to 0.419 (two steps) and from 0.448 to 0.370 (four)
  (`research/scripts/distill_steps.py`).
- **Ternary weights are stored 5 to a byte** (1.6 bits each). Tensors are deflated wherever that saves anything, and
  the browser inflates them. The file is read with range requests, in parts, so it is never all in memory at once.

| Part                               | Stored   |
| ---------------------------------- | -------- |
| The reader: 9 layers and tokenizer | 158.4 MB |
| The map                            | 35.3 MB  |
| The painter                        | 761.3 MB |
| The decoder                        | 2.5 MB   |
| The 1-step branch (rank 32)        | 82.7 MB  |
| The 2-step branch (rank 32)        | 82.6 MB  |
| The 4-step branch (rank 32)        | 82.6 MB  |
| Schedules for 1–4 steps and RDM's  | 2.1 MB   |

To rebuild it (after `research/scripts/adapter.py targets`, which runs the stock encoder over the prompts):

```sh
research/.venv/bin/python research/scripts/adapter_layers.py stats
research/.venv/bin/python research/scripts/adapter_layers.py fit
research/.venv/bin/python research/scripts/adapter_layers.py final 3 6 9
research/.venv/bin/python research/scripts/pack_model.py --fused research/data/adapter/fused_final_layers_3_6_9.pt --float \
  --lora research/data/distill/fast_trained_r32.safetensors --lora-rank 32 \
  --fast-offsets research/data/distill/fast_offsets.pt \
  --best research/data/distill/best_trained_r32.safetensors --best-rank 32 \
  --best-offsets research/data/distill/best_offsets.pt \
  --rdm research/data/distill/rdm_trained_r32.safetensors --rdm-rank 32
```

The 1-step branch needs, first: `research/scripts/rdm_delta.py` (RDM minus klein, as rank-256 SVDs), the training
(`distill_1step.py prepare`, then `modal run research/cloud/distill_modal.py --step upload|fetch|targets|train|pull`,
then `distill_1step.py export 32`), and `RDM_STUDENT=research/data/distill/student_r32.pt
export_painter_schedules.py` (the `1r` schedule with the trained offsets). The 2- and 4-step branches: `modal run
research/cloud/distill_modal.py --script steps --mode fast|best --step targets|train`, then `--step pull --only fast_`
(or `best_`) and `distill_steps.py export fast|best 32`.

The landing's recorded run is made with `node research/scripts/record_threads.mjs` (with `pnpm dev` running): it runs the
opening prompt live on the local file in headless Chrome and writes `static/recordings/bonsai-museum`.

What makes it fast in the browser: an f16 ternary GEMM (tiles staged and multiplied in f16, summed in f32), attention in
f16, the text stream cut to what the prompt needs, and a register-tiled decoder. `/dev/gemm`, `/dev/profile` and
`/dev/rows` (development only) measure them.

The pages `/lab/pack` and `/lab/pack/grid` compare the file's pictures with the reader's and painter's own files.

## What you can open

| Page                                                             | What it is                                                                                                                                                                                                                                                                                                                                           |
| ---------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [`/paint`](https://neovand.github.io/mindview/paint)             | Type a prompt and get a picture, from the one-file model.                                                                                                                                                                                                                                                                                            |
| [`/`](https://neovand.github.io/mindview/)                       | The landing: your words as threads through the reader's layers, then into the painter, a sketch and then the picture. The opening prompt plays a recording of a real run (5 MB); any other prompt runs the one-file model live in your tab.                                                                                                          |
| [`/lab/threads`](https://neovand.github.io/mindview/lab/threads) | Each word is a thread through the reader's layers, then into the painter as it paints, block by block, with where the picture looks at each word.                                                                                                                                                                                                    |
| [`/lab/layer`](https://neovand.github.io/mindview/lab/layer)     | Operation by operation, every number readable. Pick one layer of the reader, or one block of the painter (for a word or a patch of the picture). You see the norms, the projections with their running sums, the rotary turns, attention (in the painter, 24 heads over the prompt and all 1,024 patches), the neurons, and the gated residual adds. |
| [`/lab/machine`](https://neovand.github.io/mindview/lab/machine) | All 5.1 billion weights (the reader's 1.4 billion, the painter's 3.7 billion) on one wall, one cell each, lit by what they multiply for the word or patch you follow. The picture each block has in mind sits above it, filling in as it paints.                                                                                                     |
| [`/lab/cube`](https://neovand.github.io/mindview/lab/cube)       | Every multiplication as a voxel: the reading, then the painting below it, which takes hundreds of times more operations.                                                                                                                                                                                                                             |

## Running it

You need:

- A browser with WebGPU: a recent Chrome or Edge. It is developed and tested in Chrome on an Apple M4.
- A GPU with about 3 GB to spare.
- Room for about 1.6 GB of downloads on the first visit to the labs, or 0.96 GB for `/paint`. The browser keeps them
  in Cache Storage for later visits.

To run it locally:

```sh
pnpm install
pnpm dev
```

By default the models come from Hugging Face, the same as on the live site. To work from local copies (faster reloads,
no network), put them in `static/models` and they are used instead:

```sh
hf download prism-ml/Ternary-Bonsai-1.7B-gguf Ternary-Bonsai-1.7B-Q2_0.gguf --local-dir /tmp/reader
mkdir -p static/models/ternary-bonsai-1.7b
mv /tmp/reader/Ternary-Bonsai-1.7B-Q2_0.gguf static/models/ternary-bonsai-1.7b/model.gguf
hf download mohsenvand/mindview-painter --local-dir static/models/bonsai-image-4b
hf download mohsenvand/mindview-t2i-turbo mindview-t2i-turbo.gguf --local-dir /tmp/t2i
mkdir -p static/models/mindview-t2i
mv /tmp/t2i/mindview-t2i-turbo.gguf static/models/mindview-t2i/model.gguf
```

`MODELS=hub` or `MODELS=local` forces one or the other (see `vite.config.ts` and `src/lib/models.ts`).

Other commands:

- `pnpm check`: type-check.
- `pnpm lint`: formatting and lint.
- `pnpm build`: build the static site into `build/`. Set `BASE_PATH=/mindview` to serve it from a folder.

## How it is built

- `src/lib/runtime/`: the WebGPU runtime. It holds a GGUF reader, the ternary matrix kernels, the reader, the
  painter, TAEF2, and one GPU scheduler that feeds work to the GPU a few milliseconds per frame, so the page stays
  smooth while the models run.
- `src/lib/lab/`: the labs. They share one stage (camera, bloom) and one panel renderer that draws matrices and
  vectors straight from GPU buffers at any zoom, down to a single printed number. They also share one painting,
  which keeps every matrix input for the prompt's words and 17 patches.
- `src/lib/engine/`, `src/lib/viz/`: the first piece and the threads.
- `research/`: the Python side, which:
  - exports the painter's files (`export_painter.py`, `export_painter_schedules.py`, `export_painter_viz2.py`);
  - trains the adapter (`adapter.py`, `adapter_ternarize.py`);
  - fits the map for the one-file model and packs it (`adapter_layers.py`, `pack_model.py`);
  - fits the lens (`tuned_lens.py`);
  - checks the browser runtime against PyTorch (`painter_reference.py`, `painter_truth.py`).

  The painter in the browser matches PyTorch block for block, with cosine 1.000000.

## Deploying

`.github/workflows/deploy.yml` builds the site on every push to `main` and publishes it on GitHub Pages
(Settings → Pages → Source: GitHub Actions). The site is static; the models are fetched from Hugging Face.

## Credits

- [PrismML](https://huggingface.co/prism-ml): the Bonsai models. Ternary Bonsai 1.7B and Bonsai Image 4B are
  Apache 2.0.
- [Black Forest Labs](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B): FLUX.2 [klein] 4B, the painter's
  architecture (Apache 2.0).
- [Ollin Boer Bohan](https://huggingface.co/madebyollin/taef2): TAEF2 (MIT).
- The painter bundle on Hugging Face ([mohsenvand/mindview-painter](https://huggingface.co/mohsenvand/mindview-painter))
  repackages those weights for the browser, together with the adapter and the lens trained here.
  [mohsenvand/mindview-t2i-turbo](https://huggingface.co/mohsenvand/mindview-t2i-turbo) packs them, with the map
  trained here, into one file.
- [radames](https://huggingface.co/radames/FLUX.2-klein-Sana-Sprint): the 2-step LoRA (Apache 2.0).
- [EPFL VITA](https://huggingface.co/epfl-vita/flux2-klein-1step-rdm): RDM, the 1-step fine-tune the 1-step branch comes
  from (Apache 2.0).
