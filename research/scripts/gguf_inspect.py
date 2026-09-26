"""Minimal GGUF header/tensor-table reader that tolerates PrismML's custom quant types."""
import struct, sys

GGUF_SCALAR = {0: 'B', 1: 'b', 2: 'H', 3: 'h', 4: 'I', 5: 'i', 6: 'f', 7: '?', 10: 'Q', 11: 'q', 12: 'd'}


def read_gguf(path, verbose=True):
    f = open(path, 'rb')

    def rd(fmt):
        return struct.unpack('<' + fmt, f.read(struct.calcsize(fmt)))

    def rstr():
        (n,) = rd('Q')
        return f.read(n).decode('utf-8', 'replace')

    def rval(t):
        if t == 8:
            return rstr()
        if t == 9:
            (at,) = rd('I'); (n,) = rd('Q')
            if at == 8:
                return [rstr() for _ in range(n)]
            fmt = GGUF_SCALAR[at]
            return list(struct.unpack('<' + fmt * n, f.read(struct.calcsize(fmt) * n)))
        return rd(GGUF_SCALAR[t])[0]

    magic = f.read(4); (ver,) = rd('I'); (nt,) = rd('Q'); (nkv,) = rd('Q')
    kv = {}
    for _ in range(nkv):
        k = rstr(); (t,) = rd('I'); kv[k] = rval(t)
    tensors = []
    for _ in range(nt):
        name = rstr(); (nd,) = rd('I'); dims = rd('Q' * nd); (ty,) = rd('I'); (off,) = rd('Q')
        tensors.append(dict(name=name, dims=dims, type=ty, offset=off))
    align = kv.get('general.alignment', 32)
    data_start = (f.tell() + align - 1) // align * align
    return dict(magic=magic, version=ver, kv=kv, tensors=tensors, data_start=data_start)


if __name__ == '__main__':
    g = read_gguf(sys.argv[1])
    print(g['magic'], 'v', g['version'], 'tensors', len(g['tensors']), 'data_start', g['data_start'])
    for k, v in g['kv'].items():
        if isinstance(v, list):
            v = f'<list {len(v)}> {v[:4]}'
        elif isinstance(v, str) and len(v) > 120:
            v = v[:120] + '...'
        print(f'  {k} = {v}')
    by_type = {}
    for t in g['tensors']:
        by_type.setdefault(t['type'], []).append(t)
    for ty, ts in sorted(by_type.items()):
        print(f'type {ty}: {len(ts)} tensors, e.g. ' + ', '.join(f"{t['name']}{t['dims']}" for t in ts[:3]))
    # infer bytes per tensor from offsets
    ts = sorted(g['tensors'], key=lambda t: t['offset'])
    for a, b in zip(ts[:12], ts[1:13]):
        n = 1
        for d in a['dims']: n *= d
        print(f"  {a['name']:36s} {str(a['dims']):16s} type={a['type']:3d} bytes={b['offset']-a['offset']:10d} bytes/elem={(b['offset']-a['offset'])/n:.4f}")
