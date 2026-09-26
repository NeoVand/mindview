"""Plain generation timing (no capture) for Bonsai Image unpacked in diffusers/MPS, plus seed-matched outputs."""
import sys, glob, time, os, hashlib
import torch
from diffusers import Flux2KleinPipeline, Flux2Transformer2DModel
variant = sys.argv[1]; size = int(sys.argv[2]); prompts = sys.argv[3:]
encoder = os.environ.get('ENCODER', 'qwen'); steps = int(os.environ.get('STEPS', 4)); seed = int(os.environ.get('SEED', 7))
root = glob.glob('/Users/neo/.cache/huggingface/hub/models--prism-ml--bonsai-image-ternary-4B-unpacked/snapshots/*/')[0]
kw = {}
if variant == 'binary':
    broot = glob.glob('/Users/neo/.cache/huggingface/hub/models--prism-ml--bonsai-image-binary-4B-unpacked/snapshots/*/')[0]
    kw['transformer'] = Flux2Transformer2DModel.from_pretrained(broot, subfolder='transformer', torch_dtype=torch.bfloat16)
pipe = Flux2KleinPipeline.from_pretrained(root, text_encoder=None, torch_dtype=torch.bfloat16, **kw).to('mps')
pipe.set_progress_bar_config(disable=True)
os.makedirs('renders/image/compare', exist_ok=True)
for i, p in enumerate(prompts):
    emb = torch.load(f"data/prompt_cache/{hashlib.sha1(f'{encoder}|{p}'.encode()).hexdigest()[:12]}.pt")['embeds'].to('mps')
    t0 = time.time()
    with torch.no_grad():
        img = pipe(prompt_embeds=emb, height=size, width=size, num_inference_steps=steps, generator=torch.Generator('cpu').manual_seed(seed)).images[0]
    torch.mps.synchronize(); dt = time.time() - t0
    name = f'renders/image/compare/{variant}_{encoder}_{size}_s{steps}_seed{seed}_p{i}.png'
    img.save(name)
    print(f'{variant} {encoder} {size}² {steps} steps: {dt:.1f}s ({dt/steps:.2f}s/step incl VAE)  {name}', flush=True)
