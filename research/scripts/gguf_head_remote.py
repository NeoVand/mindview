import struct, sys, urllib.request, collections, re
url = sys.argv[1]
CH = 4*1024*1024
class R:
    def __init__(s, url): s.url=url; s.buf=b''; s.base=0; s.pos=0
    def _fill(s, need):
        while s.pos+need > s.base+len(s.buf):
            start = s.base+len(s.buf)
            req = urllib.request.Request(s.url, headers={'Range': f'bytes={start}-{start+CH-1}'})
            data = urllib.request.urlopen(req).read()
            # drop consumed prefix
            keep = s.pos - s.base
            s.buf = s.buf[keep:] + data; s.base = s.pos
    def read(s, n):
        s._fill(n); o=s.pos-s.base; s.pos+=n; return s.buf[o:o+n]
    def u32(s): return struct.unpack('<I', s.read(4))[0]
    def u64(s): return struct.unpack('<Q', s.read(8))[0]
    def str(s): n=s.u64(); return s.read(n).decode('utf-8','replace')
r = R(url)
assert r.read(4)==b'GGUF'
ver=r.u32(); nt=r.u64(); nkv=r.u64()
print('version',ver,'n_tensors',nt,'n_kv',nkv)
SC={0:('B',1),1:('b',1),2:('H',2),3:('h',2),4:('I',4),5:('i',4),6:('f',4),7:('?',1),10:('Q',8),11:('q',8),12:('d',8)}
def val(t):
    if t in SC:
        f,n=SC[t]; return struct.unpack('<'+f, r.read(n))[0]
    if t==8: return r.str()
    if t==9:
        at=r.u32(); n=r.u64()
        if at in SC:
            f,sz=SC[at]; data=r.read(sz*n); return ('arr',at,n,list(struct.unpack('<'+f*n, data)) if n<200000 else None)
        out=[val(at) for _ in range(n)]; return ('arr',at,n,out)
    raise Exception(t)
kvs={}
for i in range(nkv):
    k=r.str(); t=r.u32(); v=val(t); kvs[k]=v
    if k.startswith('tokenizer.ggml.') and isinstance(v,tuple): print(k, 'array', v[1], v[2]); continue
    if k=='tokenizer.chat_template': print(k,'<len',len(v),'>'); continue
    if isinstance(v,tuple):
        a=v[3]
        if k=='prism.hadamard.sign_values':
            print(k,'int array len',v[2],'first 32:',a[:32]); continue
        if v[2]>80:
            import collections as C; print(k,"array",v[2], C.Counter(re.sub(r"blk\.\d+\.","",x) for x in a)); continue
    print(k,'=',v if not isinstance(v,tuple) else v[3])
TN={0:'F32',1:'F16',2:'Q4_0',8:'Q8_0',30:'BF16',34:'TQ1_0',35:'TQ2_0',41:'Q1_0',42:'Q2_0',142:'PQ2_0',143:'PTQ1_0'}
tensors=[]
offs={}
for i in range(nt):
    n=r.str(); nd=r.u32(); dims=[r.u64() for _ in range(nd)]; t=r.u32(); off=r.u64()
    tensors.append((n,dims,TN.get(t,t))); offs[n]=off
import re
seen=set()
for n,d,t in tensors:
    key=re.sub(r'blk\.\d+\.','blk.N.',n)
    m=re.match(r'blk\.(\d+)\.',n)
    if m and int(m.group(1)) not in (0,3,64): continue
    print(f'  {n:45s} {str(d):22s} {t}')
print('types histogram', collections.Counter(t for _,_,t in tensors))
print('header bytes read', r.pos)
