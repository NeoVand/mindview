"""Encode prompts with the Qwen3-4B text encoder once and cache the 512x7680 embeddings (layers 9/18/27).

Usage: image_encode_prompts.py [--encoder qwen|bonsai4b] "<prompt>" ["<prompt>" ...]
The 'bonsai4b' encoder swaps in PrismML's ternary Qwen3-4B LLM (Ternary-Bonsai-4B-unpacked).
"""
import sys, glob, hashlib, os, time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

args = sys.argv[1:]
encoder = 'qwen'
if args and args[0] == '--encoder':
    encoder, args = args[1], args[2:]
root = glob.glob('/Users/neo/.cache/huggingface/hub/models--prism-ml--bonsai-image-ternary-4B-unpacked/snapshots/*/')[0]
enc_path = root + 'text_encoder' if encoder == 'qwen' else glob.glob('/Users/neo/.cache/huggingface/hub/models--prism-ml--Ternary-Bonsai-4B-unpacked/snapshots/*/')[0]
tok = AutoTokenizer.from_pretrained(root + 'tokenizer')
t0 = time.time()
model = AutoModelForCausalLM.from_pretrained(enc_path, dtype=torch.bfloat16).to('mps').eval()
print(f'{encoder} encoder loaded in {time.time()-t0:.1f}s')
os.makedirs('data/prompt_cache', exist_ok=True)
for prompt in args:
    text = tok.apply_chat_template([{'role': 'user', 'content': prompt}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
    inp = tok(text, return_tensors='pt', padding='max_length', truncation=True, max_length=512).to('mps')
    with torch.no_grad():
        out = model(**inp, output_hidden_states=True, use_cache=False)
    emb = torch.stack([out.hidden_states[k] for k in (9, 18, 27)], 1)  # [1,3,512,2560]
    emb = emb.permute(0, 2, 1, 3).reshape(1, 512, 3 * 2560).to(torch.bfloat16).cpu()
    key = hashlib.sha1(f'{encoder}|{prompt}'.encode()).hexdigest()[:12]
    torch.save(dict(prompt=prompt, encoder=encoder, embeds=emb, n_real=int(inp.attention_mask.sum())), f'data/prompt_cache/{key}.pt')
    print(key, encoder, int(inp.attention_mask.sum()), 'tokens:', prompt)
