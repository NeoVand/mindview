"""Pack the whole text-to-image pipeline into one GGUF file: mindview-t2i.

    reader    Ternary Bonsai 1.7B (PrismML), cut to the layers whose states the conditioning reads (up to 21 for the
              adapter's taps 7, 14, 21; 9 for the fitted map's 3, 6, 9), with its tokenizer
    cond      one linear map from the reader's taps to the DiT's text stream (3,072 x 6,144), with its bias and the 3
              fixed prefix rows: the adapter (ternary, 6,144 -> 7,680) times the DiT's context embedder (dense,
              7,680 -> 3,072), or a map fitted straight into the DiT's input space (adapter_layers.py)
    dit       the ternary diffusion transformer of Bonsai Image 4B (FLUX.2 [klein] 4B), with the per-step modulation for
              the 4-step schedule precomputed
    decoder   TAEF2

Every ternary weight is stored as a trit, 5 to a byte (1.6 bits), with its group's scale (one per 128 inputs) in a
companion f16 tensor `<name>.scale`; see TRIT_DOC. Everything else keeps its dtype. Each tensor is then deflated where
that saves anything (browsers inflate natively): the trits by about 1%, the scales (stored as two byte planes) by more
than half; the tokenizer goes into one deflated JSON blob. See DEFLATE_DOC. The file is checked on the way out: every
tensor reads back exactly.

    python research/scripts/pack_model.py [--out ...] [--fused data/adapter/fused_layers_a_b_c.pt [--float]]

--fused takes a conditioning map fitted by adapter_layers.py (its taps decide how many reader layers are kept); it is
stored ternary (its least-squares scales), or f16 with --float. Without it, the map is the adapter times the context
embedder, f16, and the reader keeps 21 layers.
"""
import argparse
import json
import os
import struct
import sys
import zlib

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
READER = os.path.join(ROOT, 'static/models/ternary-bonsai-1.7b/model.gguf')
PAINTER = os.path.join(ROOT, 'static/models/bonsai-image-4b')

F32, F16, TRIT = 0, 1, 200
DEFLATE, F16_PLANES, JSON_BLOB = 202, 203, 204
Q2_TYPES = (42, 142)  # PrismML ternary g128: per 128 weights an f16 scale, then 32 bytes of 2-bit codes (q = trit + 1)
TRIT_DOC = (
    'type 200: trits of a row-major matrix, 5 per byte as q0 + 3 q1 + 9 q2 + 27 q3 + 81 q4 with q = trit + 1, '
    'padded with q = 1 to a multiple of 80 trits (16 bytes); the value is trit * scale, the scale being '
    '<name>.scale (f16, [cols / 128, rows]: one per 128 inputs of each row)'
)
DEFLATE_DOC = (
    'type 202: a deflated tensor: u32 inner type, u32 inflated bytes, u32 deflated bytes, u32 0, then the raw deflate '
    'stream; type 203: f16 values as two byte planes (every high byte, then every low byte); type 204: UTF-8 JSON; '
    'the stored size of each tensor is in mindview.stored'
)
ALIGN = 32

# ---------------------------------------------------------------------------------------------------------------- GGUF


class Reader:
    """GGUF v3, keeping each metadata value's type (so it can be written back as it was)."""

    SCALAR = {0: 'B', 1: 'b', 2: 'H', 3: 'h', 4: 'I', 5: 'i', 6: 'f', 7: '?', 10: 'Q', 11: 'q', 12: 'd'}

    def __init__(self, path):
        self.f = open(path, 'rb')
        f = self.f
        assert f.read(4) == b'GGUF'
        (self.version,) = self.rd('I')
        (nt,) = self.rd('Q')
        (nkv,) = self.rd('Q')
        self.kv = {}
        for _ in range(nkv):
            k = self.rstr()
            (t,) = self.rd('I')
            self.kv[k] = (t, self.rval(t))
        self.tensors = {}
        for _ in range(nt):
            name = self.rstr()
            (nd,) = self.rd('I')
            dims = self.rd('Q' * nd)
            (ty,) = self.rd('I')
            (off,) = self.rd('Q')
            self.tensors[name] = dict(dims=dims, type=ty, offset=off)
        align = self.kv.get('general.alignment', (4, 32))[1]
        self.data = (f.tell() + align - 1) // align * align

    def rd(self, fmt):
        return struct.unpack('<' + fmt, self.f.read(struct.calcsize(fmt)))

    def rstr(self):
        (n,) = self.rd('Q')
        return self.f.read(n).decode('utf-8')

    def rval(self, t):
        if t == 8:
            return self.rstr()
        if t == 9:
            (at,) = self.rd('I')
            (n,) = self.rd('Q')
            if at == 8:
                return (at, [self.rstr() for _ in range(n)])
            fmt = self.SCALAR[at]
            return (at, list(struct.unpack('<' + fmt * n, self.f.read(struct.calcsize(fmt) * n))))
        return self.rd(self.SCALAR[t])[0]

    def raw(self, name, nbytes):
        self.f.seek(self.data + self.tensors[name]['offset'])
        return self.f.read(nbytes)


class Writer:
    def __init__(self):
        self.kv = []  # (key, type, value)
        self.tensors = []  # (name, dims (ne0 first), type, bytes)

    def meta(self, key, t, value):
        self.kv.append((key, t, value))

    def string(self, key, value):
        self.meta(key, 8, value)

    def u32(self, key, value):
        self.meta(key, 4, int(value))

    def add(self, name, dims, ty, data: bytes, deflate=True):
        """Store a tensor, deflated when that saves at least 1% (inner type ty)."""
        if deflate:
            c = zlib.compressobj(9, zlib.DEFLATED, -15)
            z = c.compress(data) + c.flush()
            if len(z) + 16 < 0.99 * len(data):
                assert zlib.decompress(z, -15) == data
                data = struct.pack('<IIII', ty, len(data), len(z), 0) + z
                ty = DEFLATE
        self.tensors.append((name, tuple(int(d) for d in dims), ty, data))

    def write(self, path):
        def s(x):
            b = x.encode('utf-8')
            return struct.pack('<Q', len(b)) + b

        def val(t, v):
            if t == 8:
                return s(v)
            if t == 9:
                at, items = v
                out = struct.pack('<IQ', at, len(items))
                if at == 8:
                    return out + b''.join(s(x) for x in items)
                return out + struct.pack('<' + Reader.SCALAR[at] * len(items), *items)
            return struct.pack('<' + Reader.SCALAR[t], v)

        head = b'GGUF' + struct.pack('<IQQ', 3, len(self.tensors), len(self.kv))
        for k, t, v in self.kv:
            head += s(k) + struct.pack('<I', t) + val(t, v)
        offs, o = [], 0
        for name, dims, ty, data in self.tensors:
            offs.append(o)
            o += (len(data) + ALIGN - 1) // ALIGN * ALIGN
        for (name, dims, ty, data), off in zip(self.tensors, offs):
            head += s(name) + struct.pack('<I', len(dims)) + struct.pack('<' + 'Q' * len(dims), *dims)
            head += struct.pack('<IQ', ty, off)
        head += b'\0' * ((ALIGN - len(head) % ALIGN) % ALIGN)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            f.write(head)
            for name, dims, ty, data in self.tensors:
                f.write(data)
                f.write(b'\0' * ((ALIGN - len(data) % ALIGN) % ALIGN))
        return len(head) + o


# ------------------------------------------------------------------------------------------------------------- trits

POW3 = np.array([1, 3, 9, 27, 81], dtype=np.uint16)


def pack_trits(q: np.ndarray) -> bytes:
    """q (uint8, values 0..2, row-major) -> 5 trits per byte, padded to 80 trits."""
    q = q.reshape(-1)
    assert q.max() <= 2, 'a code of 3: not ternary'
    n = (q.size + 79) // 80 * 80
    p = np.ones(n, dtype=np.uint16)
    p[: q.size] = q
    return (p.reshape(-1, 5) * POW3).sum(axis=1).astype(np.uint8).tobytes()


def unpack_trits(b: bytes, n: int) -> np.ndarray:
    v = np.frombuffer(b, dtype=np.uint8).astype(np.uint16)
    out = np.empty((v.size, 5), dtype=np.uint8)
    for i in range(5):
        out[:, i] = v % 3
        v //= 3
    return out.reshape(-1)[:n]


def codes_from_u32(words: np.ndarray, n: int) -> np.ndarray:
    """16 two-bit codes per little-endian u32 (the painter's layout) -> q per weight."""
    w = words.astype(np.uint32)
    q = np.stack([(w >> (2 * i)) & 3 for i in range(16)], axis=1)
    return q.reshape(-1)[:n].astype(np.uint8)


def add_ternary(w: Writer, name, rows, cols, q: np.ndarray, scales: np.ndarray, report):
    """q: (rows * cols,) uint8; scales: (rows * cols / 128,) float (exactly representable in f16)."""
    s16 = scales.astype(np.float16)
    bad = np.count_nonzero(s16.astype(np.float64) != scales.astype(np.float64))
    assert bad == 0, f'{name}: {bad} scales are not exact in f16'
    trits = pack_trits(q)
    # the round trip, exactly
    assert np.array_equal(unpack_trits(trits, q.size), q), f'{name}: trits do not read back'
    w.add(name, (cols, rows), TRIT, trits)
    u = s16.view(np.uint16)
    w.add(name + '.scale', (cols // 128, rows), F16_PLANES,
          (u >> 8).astype(np.uint8).tobytes() + (u & 0xFF).astype(np.uint8).tobytes())
    report['trits'] += len(w.tensors[-2][3])
    report['scales'] += len(w.tensors[-1][3])
    report['params'] += q.size
    report['zeros'] += int(np.count_nonzero(q == 1))


# ------------------------------------------------------------------------------------------------------------- main


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=os.path.join(ROOT, 'static/models/mindview-t2i/model.gguf'))
    ap.add_argument('--fused', help='a conditioning map from adapter_layers.py (fused_layers_*.pt)')
    ap.add_argument('--float', action='store_true', help='keep the fitted map f16 instead of ternary')
    ap.add_argument('--lora', help='a few-step LoRA for the DiT (diffusers safetensors), stored as a side branch')
    ap.add_argument('--lora-source', default='radames/FLUX.2-klein-Sana-Sprint', help='where the LoRA comes from')
    ap.add_argument('--lora-rank', type=int, default=64, help="keep the LoRA's best rank-r approximation")
    ap.add_argument('--schedules', default='1,2,3,4', help="other step counts to carry (from the bundle's schedules)")
    args = ap.parse_args()
    fused = None
    if args.fused:
        import torch
        fused = torch.load(args.fused)
    taps = list(fused['taps']) if fused else [7, 14, 21]
    layers = max(taps)
    w = Writer()
    sizes = {}

    # ---- the reader: metadata and tokenizer as they are, the layers up to the last tap
    r = Reader(READER)
    # not 'qwen3': the file is not a Qwen3 model (GGUF tools and the Hub take the architecture at its word). The reader's
    # settings keep their qwen3.* keys, and the runtime loads that part as the Qwen3 it is.
    w.meta('general.architecture', 8, 'mindview-t2i')
    w.string('general.name', 'mindview-t2i')
    w.string(
        'general.description',
        f'Text to image in one file: Ternary Bonsai 1.7B ({layers} layers) reads the prompt, a linear map turns its '
        f'layers {", ".join(map(str, taps))} into conditioning, the ternary DiT of Bonsai Image 4B paints in 4 steps, '
        'TAEF2 decodes 512 x 512.',
    )
    w.string('general.license', 'apache-2.0')
    w.u32('general.alignment', ALIGN)
    blob = {}
    for k, (t, v) in r.kv.items():
        if k in ('tokenizer.ggml.tokens', 'tokenizer.ggml.merges', 'tokenizer.ggml.token_type'):
            blob[k] = v[1]  # into the deflated blob below
        elif k == 'tokenizer.chat_template':
            blob[k] = v  # the prompt goes through it, but at the top level it would advertise a chat model
        elif k.startswith(('qwen3.', 'tokenizer.')):
            w.meta(k, t, layers if k == 'qwen3.block_count' else v)
    tok_json = json.dumps(blob).encode('utf-8')
    w.add('tokenizer.blob', (len(tok_json),), JSON_BLOB, tok_json)
    rep = dict(trits=0, scales=0, params=0, zeros=0, dense=0)
    for name, t in r.tensors.items():
        if name.startswith('blk.') and int(name.split('.')[1]) >= layers:
            continue
        K, M = t['dims'][0], t['dims'][1] if len(t['dims']) > 1 else 1
        if t['type'] == F32:
            data = r.raw(name, 4 * K * M)
            w.add(name, t['dims'], F32, data)
            rep['dense'] += len(data)
        elif t['type'] in Q2_TYPES:
            blocks = np.frombuffer(r.raw(name, M * K // 128 * 34), dtype=np.uint8).reshape(-1, 34)
            scales = blocks[:, :2].copy().view(np.float16).reshape(-1).astype(np.float32)
            b = blocks[:, 2:]
            q = np.stack([(b >> (2 * i)) & 3 for i in range(4)], axis=2).reshape(-1)
            add_ternary(w, name, M, K, q, scales, rep)
        else:
            sys.exit(f'{name}: type {t["type"]} is not handled')
    sizes['reader'] = dict(rep)
    print(f"reader: {rep['params'] / 1e6:.1f} M ternary weights, trits {rep['trits'] / 1e6:.1f} MB, "
          f"scales {rep['scales'] / 1e6:.1f} MB, norms {rep['dense'] / 1e6:.2f} MB, zeros {rep['zeros'] / rep['params']:.3f}")

    # ---- the painter's files
    man = json.load(open(os.path.join(PAINTER, 'manifest.json')))
    files = {}

    def blob(fn):
        if fn not in files:
            files[fn] = open(os.path.join(PAINTER, fn), 'rb').read()
        return files[fn]

    def dense(name) -> np.ndarray:
        e = man['tensors'][name]
        dt = np.float16 if e['dtype'] == 'f16' else np.float32
        return np.frombuffer(blob(e['file']), dtype=dt, count=e['bytes'] // np.dtype(dt).itemsize,
                             offset=e['offset']).reshape(e['shape'])

    def ternary(name):
        e = man['tensors'][name]
        rows, cols = e['shape']
        words = np.frombuffer(blob(e['file']), dtype=np.uint32, count=e['codes']['bytes'] // 4,
                              offset=e['codes']['offset'])
        scales = np.frombuffer(blob(e['file']), dtype=np.float32, count=e['scales']['bytes'] // 4,
                               offset=e['scales']['offset'])
        return rows, cols, codes_from_u32(words, rows * cols), scales

    # ---- cond: the map from the taps to the DiT's text stream (the adapter and the context embedder in one)
    C = dense('context_embedder.weight').astype(np.float64)  # 3072 x 7680
    prefix = dense('adapter.prefix').astype(np.float64)  # 3 x 7680: the stock encoder's rows for the fixed prefix
    n0 = len(w.tensors)
    if fused is not None and not args.float:
        t = fused['t'].numpy().astype(np.int8)  # 3072 x G x 128
        out, G, _ = t.shape
        s_ls = fused['s_ls'].numpy().astype(np.float64)
        # scales go through f16 (as stored); the bias is re-solved for those
        s16 = s_ls.astype(np.float16).astype(np.float32).reshape(-1)
        add_ternary(w, 'cond.weight', out, G * 128, (t + 1).astype(np.uint8).reshape(-1), s16,
                    dict(trits=0, scales=0, params=0, zeros=0))
        bias = fused['b_ls'].numpy().astype(np.float64)
        what = f'ternary, taps {taps}'
        cond_source = ('mindview: fitted by ridge regression to the stock encoder (Qwen3-4B) in the DiT input space '
                       '(adapter_layers.py), ternary with least-squares scales')
    else:
        if fused is not None:
            Wc = fused['W'].numpy().astype(np.float64).T  # 3072 x 6144
            bias = fused['b'].numpy().astype(np.float64)
        else:
            rows, cols, q, sc = ternary('adapter.weight')  # 7680 x 6144
            A = (q.reshape(rows, cols).astype(np.float64) - 1) * np.repeat(sc.reshape(rows, cols // 128), 128, axis=1)
            Wc = C @ A  # 3072 x 6144
            bias = C @ dense('adapter.bias').astype(np.float64)
        Wc16 = Wc.astype(np.float16)
        assert np.isfinite(Wc16).all()
        w.add('cond.weight', (Wc.shape[1], Wc.shape[0]), F16, Wc16.tobytes())
        what = f'f16, taps {taps}'
        cond_source = ('mindview: fitted by ridge regression to the stock encoder (Qwen3-4B) in the DiT input space '
                       '(adapter_layers.py), f16' if fused is not None else
                       'the mindview adapter (ternary, ridge regression to the stock encoder) times the DiT context '
                       'embedder, f16')
    w.add('cond.bias', (C.shape[0],), F32, bias.astype(np.float32).tobytes())
    w.add('cond.prefix', (C.shape[0], 3), F32, (prefix @ C.T).astype(np.float32).tobytes())
    sizes['cond'] = sum(len(x[3]) for x in w.tensors[n0:])
    print(f"cond: {what}: {sizes['cond'] / 1e6:.1f} MB stored")

    # ---- the DiT and the decoder
    rep = dict(trits=0, scales=0, params=0, zeros=0, dense=0)
    skip = {'adapter.weight', 'adapter.bias', 'adapter.prefix', 'context_embedder.weight'}
    for name, e in man['tensors'].items():
        if name in skip:
            continue
        if e['kind'] == 'ternary':
            rows, cols, q, sc = ternary(name)
            add_ternary(w, name, rows, cols, q, sc, rep)
        else:
            a = dense(name)
            w.add(name, tuple(reversed(a.shape)), F16 if a.dtype == np.float16 else F32, a.tobytes())
            rep['dense'] += a.nbytes
    sizes['dit'] = dict(rep)
    print(f"dit + decoder: {rep['params'] / 1e6:.1f} M ternary weights, trits {rep['trits'] / 1e6:.1f} MB, "
          f"scales {rep['scales'] / 1e6:.1f} MB, dense {rep['dense'] / 1e6:.1f} MB, zeros {rep['zeros'] / rep['params']:.3f}")

    # ---- other step counts: sigmas and modulation per step (export_painter_schedules.py), and a few-step LoRA
    sched_meta = {}
    sj = os.path.join(PAINTER, 'schedules.json')
    if args.schedules and os.path.exists(sj):
        allm = json.load(open(sj))['512']
        data = np.fromfile(os.path.join(PAINTER, 'schedules.bin'), dtype=np.float32)
        parts_s, at = [], 0
        for n in [int(x) for x in args.schedules.split(',')]:
            m = allm[str(n)]
            parts_s.append(data[m['offset']:m['offset'] + n * m['rows'] * 3072])
            sched_meta[str(n)] = dict(sigmas=m['sigmas'], mu=m['mu'], rows=m['rows'], offset=at)
            at += n * m['rows'] * 3072
        w.add('sched.mod', (3072, at // 3072), F32, np.concatenate(parts_s).tobytes())
    lora_meta = None
    if args.lora:
        import torch
        from safetensors import safe_open
        r = args.lora_rank
        n_lora = 0
        with safe_open(args.lora, 'pt') as f:
            meta = f.metadata() or {}
            scale = float(meta.get('alpha', 1)) / float(meta.get('rank', 1)) if 'alpha' in meta else 1.0
            for m in sorted({k.rsplit('.lora_', 1)[0] for k in f.keys()}):
                name = m.removeprefix('transformer.')
                assert f'{name}.weight' in man['tensors'], f'no DiT weight for LoRA module {name}'
                A = f.get_tensor(f'{m}.lora_A.weight').double()  # [R, in]
                B = f.get_tensor(f'{m}.lora_B.weight').double() * scale  # [out, R]
                # the best rank-r approximation of B A, split evenly (sqrt of the singular values on each side)
                Qb, Rb = torch.linalg.qr(B)
                Qa, Ra = torch.linalg.qr(A.T)
                U, S, Vh = torch.linalg.svd(Rb @ Ra.T)
                sq = S[:r].sqrt()
                Ar = (sq[:, None] * (Vh[:r] @ Qa.T)).numpy()  # [r, in]
                Bt = ((Qb @ U[:, :r]) * sq[None, :]).numpy().T  # [r, out]
                for suffix, x in (('a', Ar), ('bt', Bt)):
                    h = np.ascontiguousarray(x).astype(np.float16)
                    assert np.isfinite(h).all()
                    u = h.view(np.uint16)
                    w.add(f'lora.{name}.{suffix}', (x.shape[1], x.shape[0]), F16_PLANES,
                          (u >> 8).astype(np.uint8).tobytes() + (u & 0xFF).astype(np.uint8).tobytes())
                n_lora += 1
        lora_meta = dict(rank=r, modules=n_lora, steps=[1, 2], scale=1.0,
                         source=args.lora_source,
                         method=meta.get('method', ''), license='apache-2.0',
                         note=f'the LoRA\'s best rank-{r} approximation (SVD of B A per module); '
                              'y = W x + B (A x), on for 1 and 2 steps')
        print(f'lora: {n_lora} modules at rank {r}')

    # ---- what the runtime needs to know
    painter = {k: man[k] for k in ('text', 'image', 'schedule', 'taef2', 'config') if k in man}
    painter['text'] = dict(painter['text'], taps=taps, llm_layers_run=layers)
    painter['cond'] = {
        'weight': 'cond.weight',
        'maps': f'the taps [{" | ".join(f"h{t}" for t in taps)}] of text rows 3.. to the DiT stream (3072), with cond.bias',
        'prefix': 'cond.prefix: text rows 0..2, the same for every prompt',
    }
    w.string('mindview.painter', json.dumps(painter))
    if sched_meta:
        w.string('mindview.schedules', json.dumps({'512': sched_meta}))
    if lora_meta:
        w.string('mindview.lora', json.dumps(lora_meta))
    w.string('mindview.trits', TRIT_DOC)
    w.string('mindview.deflate', DEFLATE_DOC)
    w.string('mindview.stored', json.dumps({n: len(d) for n, _, ty, d in w.tensors if ty == DEFLATE}))
    w.string('mindview.sources', json.dumps({
        'reader': f'prism-ml/Ternary-Bonsai-1.7B-gguf (Ternary-Bonsai-1.7B-Q2_0.gguf), layers 0-{layers - 1}',
        'dit': man.get('source', {}).get('dit'),
        'cond': cond_source,
        'decoder': 'madebyollin/taef2',
        **({'lora': f'{args.lora_source} (Apache 2.0), rank {args.lora_rank} of 256, for 1 and 2 steps'}
           if args.lora else {}),
    }))

    total = w.write(args.out)
    parts = {}
    for n, _, ty, d in w.tensors:
        k = 'reader' if n.startswith(('token_embd', 'blk.', 'output_norm', 'tokenizer')) else 'cond' if n.startswith('cond.') \
            else 'decoder' if n.startswith('taef2') else 'lora' if n.startswith('lora.') else 'dit'
        parts[k] = parts.get(k, 0) + len(d)
    print('stored: ' + ', '.join(f'{k} {v / 1e6:.1f} MB' for k, v in parts.items()))
    print(f'wrote {args.out}: {total / 1e6:.1f} MB ({total / 2**30:.3f} GiB), {len(w.tensors)} tensors')


if __name__ == '__main__':
    main()
