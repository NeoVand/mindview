"""The 1-step branch's training (scripts/distill_1step.py) on one H100 at Modal.

    modal run research/cloud/distill_modal.py --step upload     # cond.pt, init.pt (make them with `prepare`)
    modal run research/cloud/distill_modal.py --step fetch      # the source models, into the volume (CPU only)
    modal run research/cloud/distill_modal.py --step targets --n 2000 --k 3   # 3 noises per prompt
    modal run research/cloud/distill_modal.py --step train --steps 3000 --rank 32 --batch 4
    modal run research/cloud/distill_modal.py --step pull       # checkpoint and held-out pictures, back here

Fast and Best (scripts/distill_steps.py): --script steps --mode fast|best with --step targets / train / pull.

Everything lives in the volume `mindview-distill`: the Hugging Face cache (/vol/hf) and the work (/vol/work, laid out
like research/: data/distill, renders/distill). The models are public; no token is needed.
"""
import pathlib

import modal

RESEARCH = pathlib.Path(__file__).resolve().parent.parent
app = modal.App('mindview-distill')
vol = modal.Volume.from_name('mindview-distill', create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version='3.12')
    .pip_install('torch==2.14.0', 'torchvision==0.29.0', 'diffusers==0.40.0', 'transformers==5.17.0', 'accelerate',
                 'safetensors', 'lpips', 'numpy', 'pillow', 'huggingface_hub[hf_transfer]')
    .env({'HF_HOME': '/vol/hf', 'HUB': '/vol/hf/hub', 'WORK': '/vol/work', 'HF_HUB_ENABLE_HF_TRANSFER': '1',
          'PYTHONUNBUFFERED': '1'})
    .add_local_file(RESEARCH / 'scripts/distill_1step.py', '/root/scripts/distill_1step.py')
    .add_local_file(RESEARCH / 'scripts/distill_steps.py', '/root/scripts/distill_steps.py')
    .add_local_file(RESEARCH / 'scripts/rdm_delta_map.py', '/root/scripts/rdm_delta_map.py')
    .add_local_file(RESEARCH / 'scripts/taesd.py', '/root/scripts/taesd.py')
    .add_local_file(RESEARCH / 'data/prompts/adapter_train.txt', '/root/data/prompts/adapter_train.txt')
    .add_local_file(RESEARCH / 'data/calibration_prompts.txt', '/root/data/calibration_prompts.txt')
)


@app.function(image=image, volumes={'/vol': vol}, timeout=3600, cpu=8, memory=16384)
def fetch():
    from huggingface_hub import snapshot_download
    snapshot_download('black-forest-labs/FLUX.2-klein-4B', allow_patterns=['transformer/*'])
    snapshot_download('prism-ml/bonsai-image-ternary-4B-unpacked',
                      allow_patterns=['transformer/*', 'vae/*', 'tokenizer/*', 'scheduler/*', 'model_index.json'])
    snapshot_download('prism-ml/Ternary-Bonsai-1.7B-unpacked')
    snapshot_download('epfl-vita/flux2-klein-1step-rdm', allow_patterns=['model.safetensors'])
    snapshot_download('madebyollin/taef2', allow_patterns=['taef2.safetensors'])
    snapshot_download('radames/FLUX.2-klein-Sana-Sprint', allow_patterns=['pytorch_lora_weights.safetensors'])
    vol.commit()


@app.function(image=image, volumes={'/vol': vol}, gpu='H100', timeout=6 * 3600, memory=65536)
def run(args: list[str], script: str = 'distill_1step.py'):
    import subprocess
    try:
        subprocess.run(['python', f'/root/scripts/{script}', *args], check=True, cwd='/root')
    finally:
        vol.commit()


@app.local_entrypoint()
def main(step: str, n: int = 2000, k: int = 1, steps: int = 3000, rank: int = 32, batch: int = 4, lr: float = 5e-5,
         script: str = '1step', mode: str = ''):
    if script == 'steps' and step in ('targets', 'train'):
        args = ['targets', mode, str(n)] if step == 'targets' else ['train', mode, str(steps), str(rank), str(batch), str(lr)]
        run.remote(args, 'distill_steps.py')
        return
    if step == 'upload':
        with vol.batch_upload(force=True) as up:
            for f in ('cond.pt', 'init.pt'):
                up.put_file(RESEARCH / 'data/distill' / f, f'/work/data/distill/{f}')
    elif step == 'fetch':
        fetch.remote()
    elif step == 'targets':
        run.remote(['targets', str(n), str(k)])
    elif step == 'train':
        run.remote(['train', str(steps), str(rank), str(batch), str(lr)])
    elif step == 'pull':
        out = RESEARCH / 'renders/distill_cloud'
        out.mkdir(parents=True, exist_ok=True)
        for e in vol.listdir('/work/renders/distill'):
            (out / pathlib.Path(e.path).name).write_bytes(b''.join(vol.read_file(e.path)))
        ck = RESEARCH / 'data/distill/cloud'
        ck.mkdir(parents=True, exist_ok=True)
        for e in vol.listdir('/work/data/distill'):
            if e.path.endswith('.pt') and ('student' in e.path or '_r' in pathlib.Path(e.path).name):
                (ck / pathlib.Path(e.path).name).write_bytes(b''.join(vol.read_file(e.path)))
        print('pulled into', out, 'and', ck)
    else:
        raise SystemExit(f'unknown step {step}')
