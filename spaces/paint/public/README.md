---
title: mindview-t2i-turbo
emoji: 🌳
colorFrom: gray
colorTo: yellow
sdk: static
app_file: index.html
pinned: false
license: apache-2.0
short_description: Text to image in your browser, on WebGPU, ternary weights
models:
  - mohsenvand/mindview-t2i-turbo
tags:
  - webgpu
  - text-to-image
  - ternary
---

# mindview-t2i-turbo, in your browser

Type something and it paints it, in one, two or four steps, on your own graphics card. Nothing runs on a server: this
Space is a static page, and the model, [mindview-t2i-turbo](https://huggingface.co/mohsenvand/mindview-t2i-turbo), runs
in your browser with WebGPU.

- **You need** a desktop browser with WebGPU (a recent Chrome or Edge) and a GPU with about 3 GB to spare.
- **The first visit downloads the model** (1.2 GB, one file). The browser keeps it for later visits.
- **On a MacBook Air (M4)** a first picture takes about 4 s in one step (Instant), 6 s in two (Fast) and 15 s in four
  (Best). Quick Best runs Best's first three steps at a quarter of the size, in about half the time.
- **On a phone** it would need more graphics memory (about 2.8 GB) than a phone gives a web page, so the page says so
  before it downloads anything.

The model is one file: the first 9 layers of PrismML's
[Ternary Bonsai 1.7B](https://huggingface.co/prism-ml/Ternary-Bonsai-1.7B-gguf) read the prompt, a linear map turns
them into the painter's conditioning, the diffusion transformer of
[Bonsai Image 4B](https://huggingface.co/prism-ml/bonsai-image-ternary-4B-unpacked) (FLUX.2 [klein] 4B with ternary
weights) paints, and [TAEF2](https://huggingface.co/madebyollin/taef2) turns the result into pixels. Nearly every weight
is −1, 0 or +1. Small side branches, trained on the ternary painter, let it paint in fewer steps. The
[model card](https://huggingface.co/mohsenvand/mindview-t2i-turbo) has the details and the sources.

This is the paint page of [mindview](https://neovand.github.io/mindview/), an art installation that shows these models
computing, layer by layer. The code is at [github.com/NeoVand/mindview](https://github.com/NeoVand/mindview).
