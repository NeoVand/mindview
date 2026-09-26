# mindview

Two ternary neural networks, one reading a prompt and one painting it, shown while they compute. Everything runs in
your browser with WebGPU, and every number on screen is one the models just computed in that tab.

**Live:** [neovand.github.io/mindview](https://neovand.github.io/mindview/) ·
[the labs](https://neovand.github.io/mindview/lab)

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

## What you can open

| Page                                                             | What it is                                                                                                                                                                                                                                                                                                                                           |
| ---------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [`/`](https://neovand.github.io/mindview/)                       | The first piece: a recorded run of one prompt, replayed as light.                                                                                                                                                                                                                                                                                    |
| [`/lab/threads`](https://neovand.github.io/mindview/lab/threads) | Each word is a thread through the reader's layers, then into the painter as it paints, block by block, with where the picture looks at each word.                                                                                                                                                                                                    |
| [`/lab/layer`](https://neovand.github.io/mindview/lab/layer)     | Operation by operation, every number readable. Pick one layer of the reader, or one block of the painter (for a word or a patch of the picture). You see the norms, the projections with their running sums, the rotary turns, attention (in the painter, 24 heads over the prompt and all 1,024 patches), the neurons, and the gated residual adds. |
| [`/lab/machine`](https://neovand.github.io/mindview/lab/machine) | All 5.1 billion weights (the reader's 1.4 billion, the painter's 3.7 billion) on one wall, one cell each, lit by what they multiply for the word or patch you follow. The picture each block has in mind sits above it, filling in as it paints.                                                                                                     |
| [`/lab/cube`](https://neovand.github.io/mindview/lab/cube)       | Every multiplication as a voxel: the reading, then the painting below it, which takes hundreds of times more operations.                                                                                                                                                                                                                             |

## Running it

You need:

- A browser with WebGPU: a recent Chrome or Edge. It is developed and tested in Chrome on an Apple M4.
- A GPU with about 3 GB to spare.
- Room for about 1.6 GB of downloads on the first visit. The browser keeps them in Cache Storage for later visits.

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
hf download NeoVand/mindview-painter --local-dir static/models/bonsai-image-4b
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
- The painter bundle on Hugging Face ([NeoVand/mindview-painter](https://huggingface.co/NeoVand/mindview-painter))
  repackages those weights for the browser, together with the adapter and the lens trained here.
