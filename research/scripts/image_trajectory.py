"""Decode the model's x0 prediction after every denoising step (cheap: no attention/PCA capture)."""
import sys, glob, os, hashlib, time
import torch
from PIL import Image
from diffusers import Flux2KleinPipeline
prompt, size, steps, seed = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]) if len(sys.argv) > 4 else 7
encoder = os.environ.get('ENCODER', 'qwen')
root = glob.glob('/Users/neo/.cache/huggingface/hub/models--prism-ml--bonsai-image-ternary-4B-unpacked/snapshots/*/')[0]
pipe = Flux2KleinPipeline.from_pretrained(root, text_encoder=None, torch_dtype=torch.bfloat16).to('mps')
pipe.set_progress_bar_config(disable=True)
emb = torch.load(f"data/prompt_cache/{hashlib.sha1(f'{encoder}|{prompt}'.encode()).hexdigest()[:12]}.pt")['embeds'].to('mps')
xs, vs = [], []
pipe.transformer.register_forward_pre_hook(lambda m, a, kw: xs.append(kw['hidden_states'].clone()), with_kwargs=True)
pipe.transformer.register_forward_hook(lambda m, a, o: vs.append(o[0].clone()))
sig = []
t0 = time.time()
with torch.no_grad():
    pipe(prompt_embeds=emb, height=size, width=size, num_inference_steps=steps, generator=torch.Generator('cpu').manual_seed(seed),
         callback_on_step_end=lambda p, i, t, kw: (sig.append(float(p.scheduler.sigmas[i])), kw)[1]).images[0].save(
        out := f'renders/image/traj_{encoder}_{size}_s{steps}.png')
print(f'{size}² {steps} steps generated in {time.time()-t0:.1f}s; sigmas', [round(s, 3) for s in sig])
hw = size // 16; dev = 'mps'; dt = torch.bfloat16
mean = pipe.vae.bn.running_mean.view(1, -1, 1, 1).to(dev, dt); std = torch.sqrt(pipe.vae.bn.running_var.view(1, -1, 1, 1) + pipe.vae.config.batch_norm_eps).to(dev, dt)
frames = []
with torch.no_grad():
    for x, v, s in zip(xs, vs, sig):
        lat = (x - s * v).to(dt).reshape(1, hw, hw, 128).permute(0, 3, 1, 2)
        im = pipe.vae.decode(pipe._unpatchify_latents(lat * std + mean), return_dict=False)[0]
        frames.append(pipe.image_processor.postprocess(im, output_type='pil')[0].resize((256, 256)))
cols = min(8, len(frames)); rows = (len(frames) + cols - 1) // cols
sheet = Image.new('RGB', (256 * cols, 256 * rows))
for i, f in enumerate(frames): sheet.paste(f, ((i % cols) * 256, (i // cols) * 256))
sheet.save(f'renders/image/traj_{encoder}_{size}_s{steps}_x0sheet.png')
