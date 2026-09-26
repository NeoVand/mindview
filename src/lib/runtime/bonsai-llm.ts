// Live WebGPU runtime for PrismML's Ternary Bonsai (Qwen3 dense, ternary g128 weights).
// Every intermediate the visuals need stays on the GPU: the residual stream after every layer, the post-attention
// residual, every MLP activation, and every attention probability. Nothing is recorded ahead of time.
import {
	GGML_F32,
	GGML_PQ2_0,
	GGML_Q2_0_G128_LEGACY,
	halfToFloat,
	parseGGUF,
	type GGUF,
	GGML_F16,
	GGML_TRIT5,
	unpackTrits,
	type GGUFTensor
} from './gguf';
import { fetchModelFile } from './cache';
import { TernaryGemm, type GemmJob } from './gemm';
import { gemmMs, type GpuScheduler, type GpuTask } from './scheduler';
import { Tokenizer } from './tokenizer';

export const MAX_TOKENS = 128;
/** The longest sequence encodeLong accepts (the painter reads a 512-token prompt). */
export const LONG_TOKENS = 512;

export interface LLMConfig {
	layers: number;
	dim: number;
	ffn: number;
	heads: number;
	kvHeads: number;
	headDim: number;
	vocab: number;
	eps: number;
	ropeTheta: number;
	ropeFactor: number;
	ropeOrigCtx: number;
}

export interface Ternary {
	rows: number; // M (outputs)
	cols: number; // K (inputs)
	codes: number; // offset in u32 words
	scales: number; // offset in floats
	meanScale: number; // average group scale (for display normalisation)
}

/** Where each activation lives inside the single arena buffer (offsets in floats). */
export interface ArenaLayout {
	resid: number; // [layers + 1][MAX_TOKENS][dim]   residual stream entering each layer (last = output of the last layer)
	mid: number; // [layers][MAX_TOKENS][dim]         residual after attention, before the MLP
	act: number; // [layers][MAX_TOKENS][ffn]         SwiGLU activations fed to down_proj (the MLP 'neurons')
	probs: number; // [layers][heads][MAX_TOKENS][MAX_TOKENS] attention probabilities
	final: number; // [MAX_TOKENS][dim]               final-norm output
	logits: number; // [vocab]                        next-token logits for the last position
	// per layer ([layers][MAX_TOKENS][width]) so any layer's computation can be shown after the pass
	xn: number; // attention input after RMSNorm
	qraw: number; // queries straight out of the q projection (before q-norm and rotary)
	kraw: number; // keys straight out of the k projection
	q: number; // queries after q-norm and rotary
	k: number; // keys after k-norm and rotary
	v: number;
	attn: number; // attention output, before the output projection
	xn2: number; // MLP input after RMSNorm
	gate: number;
	up: number;
	size: number;
}

const WGSL = /* wgsl */ `
struct P { a: u32, b: u32, c: u32, d: u32, e: u32, f: u32, g: u32, h: u32, i: u32, j: u32, k: u32, l: u32, fa: f32, fb: f32, fc: f32, fd: f32 };
@group(0) @binding(0) var<uniform> U: P;
@group(0) @binding(1) var<storage, read> CODES: array<u32>;
@group(0) @binding(2) var<storage, read> SCALES: array<f32>;
@group(0) @binding(3) var<storage, read> NORMW: array<f32>;
@group(0) @binding(4) var<storage, read_write> A: array<f32>;
@group(0) @binding(5) var<storage, read> TOK: array<u32>;
@group(0) @binding(6) var<storage, read> INVF: array<f32>;

var<workgroup> red: array<f32, 256>;
var<workgroup> part: array<f32, 512>;
var<workgroup> probs: array<f32, 128>;

// a=n, b=dim, c=codes, d=scales, e=out
@compute @workgroup_size(256)
fn embed(@builtin(global_invocation_id) g: vec3u) {
  let k = g.x; let t = g.y;
  if (k >= U.b || t >= U.a) { return; }
  let id = TOK[t];
  let w = CODES[U.c + id * (U.b / 16u) + k / 16u];
  let code = (w >> ((k % 16u) * 2u)) & 3u;
  A[U.e + t * U.b + k] = SCALES[U.d + id * (U.b / 128u) + k / 128u] * (f32(code) - 1.0);
}

// a=rows, b=len, c=in, d=out, e=weight offset, fa=eps
@compute @workgroup_size(256)
fn rmsnorm(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_id) li: vec3u) {
  let base = U.c + wg.x * U.b;
  var ss = 0.0;
  for (var i = li.x; i < U.b; i += 256u) { let v = A[base + i]; ss += v * v; }
  red[li.x] = ss;
  workgroupBarrier();
  for (var s = 128u; s > 0u; s >>= 1u) { if (li.x < s) { red[li.x] += red[li.x + s]; } workgroupBarrier(); }
  let inv = inverseSqrt(red[0] / f32(U.b) + U.fa);
  let ob = U.d + wg.x * U.b;
  for (var i = li.x; i < U.b; i += 256u) { A[ob + i] = A[base + i] * inv * NORMW[U.e + i]; }
}

// y[t][m] = (base[t][m] if flag) + sum_k x[t][k] * W[m][k];  W ternary, 16 codes per u32, one scale per 128
// a=n, b=K, c=M, d=codes, e=scales, f=x, g=y, h=base, i=flags
@compute @workgroup_size(64)
fn matmul(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_id) li: vec3u) {
  let m = wg.x + wg.z * 65535u;
  if (m >= U.c) { return; }
  let t0 = wg.y * 8u;
  let nt = min(8u, U.a - t0);
  let K = U.b;
  let wpr = K / 16u;
  let rowW = U.d + m * wpr;
  let rowS = U.e + m * (K / 128u);
  var acc = array<f32, 8>(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
  for (var w = li.x; w < wpr; w += 64u) {
    let bits = CODES[rowW + w];
    let s = SCALES[rowS + w / 8u];
    for (var q = 0u; q < 4u; q++) {
      let sh = q * 8u;
      let tv = vec4f(f32((bits >> sh) & 3u), f32((bits >> (sh + 2u)) & 3u), f32((bits >> (sh + 4u)) & 3u), f32((bits >> (sh + 6u)) & 3u)) - vec4f(1.0);
      let kk = w * 16u + q * 4u;
      for (var j = 0u; j < nt; j++) {
        let xb = U.f + (t0 + j) * K + kk;
        acc[j] += s * dot(tv, vec4f(A[xb], A[xb + 1u], A[xb + 2u], A[xb + 3u]));
      }
    }
  }
  for (var j = 0u; j < 8u; j++) { part[j * 64u + li.x] = acc[j]; }
  workgroupBarrier();
  for (var s = 32u; s > 0u; s >>= 1u) {
    if (li.x < s) { for (var j = 0u; j < 8u; j++) { part[j * 64u + li.x] += part[j * 64u + li.x + s]; } }
    workgroupBarrier();
  }
  if (li.x < nt) {
    let t = t0 + li.x;
    var y = part[li.x * 64u];
    if ((U.i & 1u) == 1u) { y += A[U.h + t * U.c + m]; }
    A[U.g + t * U.c + m] = y;
  }
}

// rotary (Qwen rotate_half convention, YaRN frequencies, cos/sin scaled by the attention factor)
// a=n, b=q heads, c=kv heads, d=q, e=k, f=position of token 0, fa=attention scaling
@compute @workgroup_size(64)
fn rope(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_id) li: vec3u) {
  let t = wg.x; let h = wg.y; let i = li.x;
  var base = U.d + (t * U.b + h) * 128u;
  if (h >= U.b) { base = U.e + (t * U.c + (h - U.b)) * 128u; }
  let ang = f32(t + U.f) * INVF[i];
  let c = cos(ang) * U.fa;
  let s = sin(ang) * U.fa;
  let x1 = A[base + i];
  let x2 = A[base + i + 64u];
  A[base + i] = x1 * c - x2 * s;
  A[base + i + 64u] = x2 * c + x1 * s;
}

// causal attention for one (query token, head); keeps the probabilities for the visuals
// a=n, b=q heads, c=kv heads, d=q, e=k, f=v, g=out, h=probs (this layer), fa=1/sqrt(head_dim)
@compute @workgroup_size(128)
fn attention(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_id) li: vec3u) {
  let t = wg.x; let h = wg.y; let s = li.x;
  let kvh = h / (U.b / U.c);
  let qb = U.d + (t * U.b + h) * 128u;
  let live = s <= t && s < U.a;
  var score = -1e30;
  if (live) {
    let kb = U.e + (s * U.c + kvh) * 128u;
    var d = 0.0;
    for (var j = 0u; j < 128u; j++) { d += A[qb + j] * A[kb + j]; }
    score = d * U.fa;
  }
  red[s] = score;
  workgroupBarrier();
  for (var st = 64u; st > 0u; st >>= 1u) { if (s < st) { red[s] = max(red[s], red[s + st]); } workgroupBarrier(); }
  let mx = red[0];
  workgroupBarrier();
  var e = 0.0;
  if (live) { e = exp(score - mx); }
  red[s] = e;
  workgroupBarrier();
  for (var st = 64u; st > 0u; st >>= 1u) { if (s < st) { red[s] += red[s + st]; } workgroupBarrier(); }
  let p = e / red[0];
  probs[s] = p;
  A[U.h + (h * ${MAX_TOKENS}u + t) * ${MAX_TOKENS}u + s] = p;
  workgroupBarrier();
  var o = 0.0;
  for (var j = 0u; j <= t; j++) { o += probs[j] * A[U.f + (j * U.c + kvh) * 128u + s]; }
  A[U.g + (t * U.b + h) * 128u + s] = o;
}

// causal attention for up to ${LONG_TOKENS} tokens, without keeping the probabilities (the lean encoder pass)
// a=n, b=q heads, c=kv heads, d=q, e=k, f=v, g=out, h=real tokens (0 = all; padding queries see only real keys),
// fa=1/sqrt(head_dim)
var<workgroup> sc: array<f32, ${LONG_TOKENS}>;
@compute @workgroup_size(128)
fn attention_long(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_id) li: vec3u) {
  let t = wg.x; let h = wg.y; let i = li.x;
  let kvh = h / (U.b / U.c);
  let qb = U.d + (t * U.b + h) * 128u;
  let last = select(t, min(t, U.h - 1u), U.h > 0u);
  var mx = -1e30;
  for (var s = i; s <= last; s += 128u) {
    let kb = U.e + (s * U.c + kvh) * 128u;
    var d = 0.0;
    for (var j = 0u; j < 128u; j++) { d += A[qb + j] * A[kb + j]; }
    sc[s] = d * U.fa;
    mx = max(mx, d * U.fa);
  }
  red[i] = mx;
  workgroupBarrier();
  for (var st = 64u; st > 0u; st >>= 1u) { if (i < st) { red[i] = max(red[i], red[i + st]); } workgroupBarrier(); }
  let m = red[0];
  workgroupBarrier();
  var sum = 0.0;
  for (var s = i; s <= last; s += 128u) { let e = exp(sc[s] - m); sc[s] = e; sum += e; }
  red[i] = sum;
  workgroupBarrier();
  for (var st = 64u; st > 0u; st >>= 1u) { if (i < st) { red[i] += red[i + st]; } workgroupBarrier(); }
  let tot = red[0];
  var o = 0.0;
  for (var s = 0u; s <= last; s++) { o += sc[s] * A[U.f + (s * U.c + kvh) * 128u + i]; }
  A[U.g + (t * U.b + h) * 128u + i] = o / tot;
}

// running argmax and log-sum-exp over vocabulary chunks (the model's reading of a vector, one chunk at a time)
// a=rows, b=chunk width, c=logits [rows][b], d=state [rows][3] = (max, index, sum exp), e=first vocab id, f=first chunk
@compute @workgroup_size(256)
fn argmax_chunk(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_id) li: vec3u) {
  let r = wg.x; let i = li.x;
  let base = U.c + r * U.b;
  var m = -1e30; var at = 0u;
  for (var j = i; j < U.b; j += 256u) { let v = A[base + j]; if (v > m) { m = v; at = j; } }
  red[i] = m; part[i] = f32(at);
  workgroupBarrier();
  for (var st = 128u; st > 0u; st >>= 1u) {
    if (i < st && red[i + st] > red[i]) { red[i] = red[i + st]; part[i] = part[i + st]; }
    workgroupBarrier();
  }
  let cm = red[0]; let ci = part[0];
  workgroupBarrier();
  var se = 0.0;
  for (var j = i; j < U.b; j += 256u) { se += exp(A[base + j] - cm); }
  red[i] = se;
  workgroupBarrier();
  for (var st = 128u; st > 0u; st >>= 1u) { if (i < st) { red[i] += red[i + st]; } workgroupBarrier(); }
  if (i == 0u) {
    let sb = U.d + r * 3u;
    if (U.f == 1u) { A[sb] = cm; A[sb + 1u] = f32(U.e) + ci; A[sb + 2u] = red[0]; }
    else {
      let M = A[sb]; let nm = max(M, cm);
      A[sb + 2u] = A[sb + 2u] * exp(M - nm) + red[0] * exp(cm - nm);
      if (cm > M) { A[sb + 1u] = f32(U.e) + ci; }
      A[sb] = nm;
    }
  }
}

// a=count, b=gate, c=up, d=out
@compute @workgroup_size(256)
fn swiglu(@builtin(global_invocation_id) g: vec3u) {
  let i = g.x;
  if (i >= U.a) { return; }
  let x = A[U.b + i];
  A[U.d + i] = x / (1.0 + exp(-x)) * A[U.c + i];
}
`;

type Op =
	| { k: 'kernel'; pipe: GPUComputePipeline; p: number[]; wg: [number, number, number] }
	| { k: 'gemm'; job: GemmJob }
	| { k: 'copy'; from: number; to: number; count: number };

export interface PrefillResult {
	ids: number[];
	tokens: string[];
	ms: number;
}

export class BonsaiLLM {
	readonly config: LLMConfig;
	readonly tokenizer: Tokenizer;
	readonly arena: GPUBuffer;
	readonly layout: ArenaLayout;
	readonly attentionScaling: number;
	readonly invFreq: Float32Array;
	private tensors = new Map<string, Ternary>();
	private norms = new Map<string, number>();
	private codes: GPUBuffer;
	private scales: GPUBuffer;
	private normw: GPUBuffer;
	private normCpu: Float32Array;
	private tokBuf: GPUBuffer;
	private invBuf: GPUBuffer;
	private params: GPUBuffer;
	private bind: GPUBindGroup;
	private pipes: Record<
		| 'embed'
		| 'rmsnorm'
		| 'matmul'
		| 'rope'
		| 'attention'
		| 'attention_long'
		| 'argmax_chunk'
		| 'swiglu',
		GPUComputePipeline
	>;
	private bindLayout: GPUBindGroupLayout;
	private long?: {
		arena: GPUBuffer;
		taps: GPUBuffer;
		tok: GPUBuffer;
		bind: GPUBindGroup;
		gemm: TernaryGemm;
	};
	n = 0;

	private constructor(
		private device: GPUDevice,
		gguf: GGUF
	) {
		const m = gguf.meta;
		const arch = String(m['general.architecture']);
		if (arch !== 'qwen3') throw new Error(`This runtime runs Qwen3 dense models, not ${arch}.`);
		const embd = gguf.tensors.get('token_embd.weight')!;
		this.config = {
			layers: Number(m['qwen3.block_count']),
			dim: Number(m['qwen3.embedding_length']),
			ffn: Number(m['qwen3.feed_forward_length']),
			heads: Number(m['qwen3.attention.head_count']),
			kvHeads: Number(m['qwen3.attention.head_count_kv']),
			headDim: Number(m['qwen3.attention.key_length'] ?? 128),
			vocab: embd.dims[1],
			eps: Number(m['qwen3.attention.layer_norm_rms_epsilon']),
			ropeTheta: Number(m['qwen3.rope.freq_base']),
			ropeFactor: Number(m['qwen3.rope.scaling.factor'] ?? 1),
			ropeOrigCtx: Number(m['qwen3.rope.scaling.original_context_length'] ?? 0)
		};
		const c = this.config;
		if (c.headDim !== 128) throw new Error('Only head_dim 128 is supported.');
		this.tokenizer = new Tokenizer(
			m['tokenizer.ggml.tokens'] as string[],
			m['tokenizer.ggml.merges'] as string[],
			(m['tokenizer.ggml.token_type'] as number[]) ?? []
		);
		({ inv: this.invFreq, scaling: this.attentionScaling } = yarn(c));

		// ---- weights: split each g128 block (fp16 scale + 32 bytes of 2-bit codes) into aligned code / scale arrays
		let codeWords = 0,
			scaleCount = 0,
			normCount = 0;
		// the scales of trit-packed tensors (GGML_TRIT5) sit beside them as <name>.scale
		const isScale = (t: GGUFTensor) =>
			t.type === GGML_F16 &&
			t.name.endsWith('.scale') &&
			gguf.tensors.get(t.name.slice(0, -6))?.type === GGML_TRIT5;
		for (const t of gguf.tensors.values()) {
			if (isScale(t)) continue;
			if (t.type === GGML_Q2_0_G128_LEGACY || t.type === GGML_PQ2_0 || t.type === GGML_TRIT5) {
				const [K, M] = t.dims;
				this.tensors.set(t.name, {
					rows: M,
					cols: K,
					codes: codeWords,
					scales: scaleCount,
					meanScale: 0
				});
				codeWords += (M * K) / 16;
				scaleCount += (M * K) / 128;
			} else if (t.type === GGML_F32) {
				this.norms.set(t.name, normCount);
				normCount += t.dims.reduce((a, b) => a * b, 1);
			} else throw new Error(`Tensor ${t.name} has unsupported type ${t.type}.`);
		}
		const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
		// the weights can be copied back (to show a single weight exactly when it is pointed at)
		const weightUsage = usage | GPUBufferUsage.COPY_SRC;
		this.codes = device.createBuffer({
			label: 'ternary codes',
			size: codeWords * 4,
			usage: weightUsage,
			mappedAtCreation: true
		});
		this.scales = device.createBuffer({
			label: 'group scales',
			size: scaleCount * 4,
			usage: weightUsage,
			mappedAtCreation: true
		});
		this.normw = device.createBuffer({
			label: 'norm weights',
			size: Math.max(16, normCount * 4),
			usage,
			mappedAtCreation: true
		});
		const codeBytes = new Uint8Array(this.codes.getMappedRange());
		const codeU32 = new Uint32Array(codeBytes.buffer, codeBytes.byteOffset, codeBytes.length / 4);
		const scaleF = new Float32Array(this.scales.getMappedRange());
		const normF = new Float32Array(this.normw.getMappedRange());
		const src = new Uint8Array(gguf.buffer);
		const dv = new DataView(gguf.buffer);
		for (const t of gguf.tensors.values()) {
			if (isScale(t)) continue;
			if (t.type === GGML_TRIT5) {
				const info = this.tensors.get(t.name)!;
				const n = info.rows * info.cols;
				unpackTrits(
					new Uint8Array(gguf.buffer, t.offset, Math.ceil(n / 80) * 16),
					n,
					codeU32,
					info.codes
				);
				const st = gguf.tensors.get(t.name + '.scale')!;
				const h = new Uint16Array(gguf.buffer.slice(st.offset, st.offset + (n / 128) * 2));
				let sum = 0;
				for (let i = 0; i < h.length; i++) {
					const v = halfToFloat(h[i]);
					scaleF[info.scales + i] = v;
					sum += Math.abs(v);
				}
				info.meanScale = sum / h.length;
				continue;
			}
			if (t.type === GGML_F32) {
				const n = t.dims.reduce((a, b) => a * b, 1);
				normF.set(
					new Float32Array(gguf.buffer.slice(t.offset, t.offset + n * 4)),
					this.norms.get(t.name)!
				);
				continue;
			}
			const info = this.tensors.get(t.name)!;
			const blocks = (info.rows * info.cols) / 128;
			let cOut = info.codes * 4;
			let sum = 0;
			for (let b = 0, off = t.offset; b < blocks; b++, off += 34, cOut += 32) {
				const sc = halfToFloat(dv.getUint16(off, true));
				scaleF[info.scales + b] = sc;
				sum += Math.abs(sc);
				codeBytes.set(src.subarray(off + 2, off + 34), cOut);
			}
			info.meanScale = sum / blocks;
		}
		this.normCpu = normF.slice();
		this.codes.unmap();
		this.scales.unmap();
		this.normw.unmap();

		// ---- activations arena
		const N = MAX_TOKENS,
			L = c.layers,
			D = c.dim,
			F = c.ffn,
			H = c.heads,
			KV = c.kvHeads * c.headDim;
		let o = 0;
		const take = (n: number) => ((o += n), o - n);
		this.layout = {
			resid: take((L + 1) * N * D),
			mid: take(L * N * D),
			act: take(L * N * F),
			probs: take(L * H * N * N),
			final: take(N * D),
			logits: take(c.vocab),
			xn: take(L * N * D),
			qraw: take(L * N * H * c.headDim),
			kraw: take(L * N * KV),
			q: take(L * N * H * c.headDim),
			k: take(L * N * KV),
			v: take(L * N * KV),
			attn: take(L * N * H * c.headDim),
			xn2: take(L * N * D),
			gate: take(L * N * F),
			up: take(L * N * F),
			size: o
		};
		this.arena = device.createBuffer({
			label: 'activations',
			size: o * 4,
			usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
		});
		this.tokBuf = device.createBuffer({ size: N * 4, usage });
		this.invBuf = device.createBuffer({ size: 64 * 4, usage });
		device.queue.writeBuffer(this.invBuf, 0, this.invFreq);
		this.params = device.createBuffer({
			size: 256 * 1024,
			usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
		});

		const layout = device.createBindGroupLayout({
			entries: [
				{
					binding: 0,
					visibility: GPUShaderStage.COMPUTE,
					buffer: { type: 'uniform', hasDynamicOffset: true, minBindingSize: 64 }
				},
				{ binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
				{ binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
				{ binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
				{ binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
				{ binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
				{ binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }
			]
		});
		const module = device.createShaderModule({ label: 'bonsai-llm', code: WGSL });
		const pl = device.createPipelineLayout({ bindGroupLayouts: [layout] });
		const mk = (entryPoint: string) =>
			device.createComputePipeline({ layout: pl, compute: { module, entryPoint } });
		this.pipes = {
			embed: mk('embed'),
			rmsnorm: mk('rmsnorm'),
			matmul: mk('matmul'),
			rope: mk('rope'),
			attention: mk('attention'),
			attention_long: mk('attention_long'),
			argmax_chunk: mk('argmax_chunk'),
			swiglu: mk('swiglu')
		};
		this.bindLayout = layout;
		this.bind = device.createBindGroup({
			layout,
			entries: [
				{ binding: 0, resource: { buffer: this.params, size: 64 } },
				{ binding: 1, resource: { buffer: this.codes } },
				{ binding: 2, resource: { buffer: this.scales } },
				{ binding: 3, resource: { buffer: this.normw } },
				{ binding: 4, resource: { buffer: this.arena } },
				{ binding: 5, resource: { buffer: this.tokBuf } },
				{ binding: 6, resource: { buffer: this.invBuf } }
			]
		});
	}

	static async load(
		device: GPUDevice,
		url: string,
		onProgress?: (fraction: number) => void
	): Promise<BonsaiLLM> {
		// kept on this computer after the first download
		return new BonsaiLLM(device, parseGGUF(await fetchModelFile(url, onProgress)));
	}

	/** Free the reader's GPU memory (it cannot be used afterwards). */
	destroy() {
		for (const b of [
			this.codes,
			this.scales,
			this.normw,
			this.arena,
			this.tokBuf,
			this.invBuf,
			this.params
		])
			b.destroy();
		this.long?.arena.destroy();
		this.long?.taps.destroy();
	}

	/** From a parsed GGUF already in memory (e.g. the reader part of the one-file model, see packed.ts). */
	static fromGGUF(device: GPUDevice, gguf: GGUF): BonsaiLLM {
		return new BonsaiLLM(device, gguf);
	}

	/** GPU buffers holding every ternary weight (for drawing the real weights) and where a tensor lives in them. */
	get weightBuffers() {
		return { codes: this.codes, scales: this.scales };
	}

	tensor(name: string): Ternary {
		return this.t(name);
	}

	/** The names of every ternary tensor. */
	get tensorNames(): string[] {
		return [...this.tensors.keys()];
	}

	/** A norm's gain vector (a copy). */
	normWeight(name: string): Float32Array {
		const at = this.norms.get(name);
		if (at === undefined) throw new Error(`Missing norm ${name}`);
		const n =
			name.includes('q_norm') || name.includes('k_norm') ? this.config.headDim : this.config.dim;
		return this.normCpu.slice(at, at + n);
	}

	/**
	 * One weight exactly as the model holds it: its code (-1, 0 or +1) and its group's scale (one per 128 weights
	 * of a row). row is the output, col the input.
	 */
	async weight(name: string, row: number, col: number): Promise<{ code: number; scale: number }> {
		const t = this.t(name);
		const dev = this.device;
		const buf = dev.createBuffer({
			size: 8,
			usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
		});
		const enc = dev.createCommandEncoder();
		enc.copyBufferToBuffer(
			this.codes,
			(t.codes + row * (t.cols / 16) + Math.floor(col / 16)) * 4,
			buf,
			0,
			4
		);
		enc.copyBufferToBuffer(
			this.scales,
			(t.scales + row * (t.cols / 128) + Math.floor(col / 128)) * 4,
			buf,
			4,
			4
		);
		dev.queue.submit([enc.finish()]);
		await buf.mapAsync(GPUMapMode.READ);
		const u = new Uint32Array(buf.getMappedRange().slice(0));
		buf.destroy();
		const code = ((u[0] >>> ((col % 16) * 2)) & 3) - 1;
		return { code, scale: new Float32Array(u.buffer)[1] };
	}

	private t(name: string): Ternary {
		const t = this.tensors.get(name);
		if (!t) throw new Error(`Missing tensor ${name}`);
		return t;
	}

	/** Run the whole prompt through every layer at once; all intermediates stay in the arena. */
	/**
	 * Run the prompt through every layer, keeping everything in the arena. With a scheduler, the work goes through its
	 * queue in slices of a few milliseconds (frames keep flowing); otherwise it is one submission. `layers` stops the
	 * pass early (then there are no logits: only the layers that ran are in the arena).
	 */
	async prefill(
		ids: number[],
		scheduler?: GpuScheduler,
		layers = this.config.layers
	): Promise<PrefillResult> {
		const c = this.config,
			L = this.layout,
			dev = this.device;
		const n = ids.length;
		if (n < 1 || n > MAX_TOKENS)
			throw new Error(`Prompts must be 1 to ${MAX_TOKENS} tokens long (this one is ${n}).`);
		this.n = n;
		dev.queue.writeBuffer(this.tokBuf, 0, new Uint32Array(ids));
		const D = c.dim,
			F = c.ffn,
			H = c.heads,
			KVH = c.kvHeads,
			HD = c.headDim,
			N = MAX_TOKENS;
		const cmds: { pipe: GPUComputePipeline; p: number[]; wg: [number, number, number] }[] = [];
		const push = (pipe: GPUComputePipeline, p: number[], wg: [number, number, number]) =>
			cmds.push({ pipe, p, wg });
		const mm = (w: Ternary, x: number, y: number, rows: number, base = -1) =>
			push(
				this.pipes.matmul,
				[rows, w.cols, w.rows, w.codes, w.scales, x, y, Math.max(0, base), base >= 0 ? 1 : 0],
				[Math.min(w.rows, 65535), Math.ceil(rows / 8), Math.ceil(w.rows / 65535)]
			);
		const norm = (rows: number, len: number, x: number, y: number, weight: string) =>
			push(
				this.pipes.rmsnorm,
				[rows, len, x, y, this.norms.get(weight)!, 0, 0, 0, 0, 0, 0, 0, c.eps],
				[rows, 1, 1]
			);

		const embd = this.t('token_embd.weight');
		push(this.pipes.embed, [n, D, embd.codes, embd.scales, L.resid], [Math.ceil(D / 256), n, 1]);
		for (let l = 0; l < Math.min(layers, c.layers); l++) {
			const resid = L.resid + l * N * D,
				mid = L.mid + l * N * D,
				act = L.act + l * N * F;
			const xn = L.xn + l * N * D,
				xn2 = L.xn2 + l * N * D,
				q = L.q + l * N * H * HD,
				k = L.k + l * N * KVH * HD,
				v = L.v + l * N * KVH * HD,
				attn = L.attn + l * N * H * HD,
				gate = L.gate + l * N * F,
				up = L.up + l * N * F;
			const b = `blk.${l}.`;
			norm(n, D, resid, xn, b + 'attn_norm.weight');
			const qraw = L.qraw + l * N * H * HD,
				kraw = L.kraw + l * N * KVH * HD;
			mm(this.t(b + 'attn_q.weight'), xn, qraw, n);
			mm(this.t(b + 'attn_k.weight'), xn, kraw, n);
			mm(this.t(b + 'attn_v.weight'), xn, v, n);
			norm(n * H, HD, qraw, q, b + 'attn_q_norm.weight');
			norm(n * KVH, HD, kraw, k, b + 'attn_k_norm.weight');
			push(
				this.pipes.rope,
				[n, H, KVH, q, k, 0, 0, 0, 0, 0, 0, 0, this.attentionScaling],
				[n, H + KVH, 1]
			);
			push(
				this.pipes.attention,
				[n, H, KVH, q, k, v, attn, L.probs + l * H * N * N, 0, 0, 0, 0, 1 / Math.sqrt(HD)],
				[n, H, 1]
			);
			mm(this.t(b + 'attn_output.weight'), attn, mid, n, resid);
			norm(n, D, mid, xn2, b + 'ffn_norm.weight');
			mm(this.t(b + 'ffn_gate.weight'), xn2, gate, n);
			mm(this.t(b + 'ffn_up.weight'), xn2, up, n);
			push(this.pipes.swiglu, [n * F, gate, up, act], [Math.ceil((n * F) / 256), 1, 1]);
			mm(this.t(b + 'ffn_down.weight'), act, L.resid + (l + 1) * N * D, n, mid);
		}
		if (layers >= c.layers) {
			norm(n, D, L.resid + c.layers * N * D, L.final, 'output_norm.weight');
			const head = this.tensors.get('output.weight') ?? embd; // tied embeddings
			mm(head, L.final + (n - 1) * D, L.logits, 1);
		}

		const stride = 256;
		const data = new ArrayBuffer(cmds.length * stride);
		cmds.forEach((cmd, i) => {
			const u = new Uint32Array(data, i * stride, 12);
			const f = new Float32Array(data, i * stride + 48, 4);
			cmd.p.slice(0, 12).forEach((v, k) => (u[k] = v));
			cmd.p.slice(12).forEach((v, k) => (f[k] = v));
		});
		const t0 = performance.now();
		const result = () => ({
			ids,
			tokens: ids.map((i) => this.tokenizer.decode([i])),
			ms: performance.now() - t0
		});
		if (scheduler) {
			// the uniforms go in with the first slice; the slices follow one another in the queue
			const cost = (cmd: (typeof cmds)[number]) =>
				cmd.pipe === this.pipes.matmul ? (2 * cmd.p[0] * cmd.p[1] * cmd.p[2]) / 1e8 : 0.3;
			const tasks: GpuTask[] = [];
			let group: number[] = [],
				acc = 0;
			const flush = () => {
				if (!group.length) return;
				const g = group,
					first = tasks.length === 0;
				tasks.push({
					cost: acc,
					record: (enc) => {
						if (first) dev.queue.writeBuffer(this.params, 0, data);
						const pass = enc.beginComputePass();
						for (const i of g) {
							pass.setPipeline(cmds[i].pipe);
							pass.setBindGroup(0, this.bind, [i * stride]);
							pass.dispatchWorkgroups(...cmds[i].wg);
						}
						pass.end();
					}
				});
				group = [];
				acc = 0;
			};
			cmds.forEach((cmd, i) => {
				const c = cost(cmd);
				if (acc + c > scheduler.slice) flush();
				group.push(i);
				acc += c;
			});
			flush();
			return new Promise((resolve) => {
				tasks[tasks.length - 1].done = () => resolve(result());
				scheduler.push(...tasks);
			});
		}
		dev.queue.writeBuffer(this.params, 0, data);
		const enc = dev.createCommandEncoder();
		const pass = enc.beginComputePass();
		cmds.forEach((cmd, i) => {
			pass.setPipeline(cmd.pipe);
			pass.setBindGroup(0, this.bind, [i * stride]);
			pass.dispatchWorkgroups(...cmd.wg);
		});
		pass.end();
		dev.queue.submit([enc.finish()]);
		await dev.queue.onSubmittedWorkDone();
		return result();
	}

	/** Scratch space for long sequences: [N][dim] residual and normed copy, q, k, v, attention, gate, up. */
	private longSpace(tapCount: number) {
		const c = this.config,
			dev = this.device,
			N = LONG_TOKENS;
		const D = c.dim,
			F = c.ffn,
			Q = c.heads * c.headDim,
			KV = c.kvHeads * c.headDim;
		let o = 0;
		const take = (k: number) => ((o += k), o - k);
		const at = {
			R: take(N * D),
			XN: take(N * D),
			QO: take(N * Q),
			KO: take(N * KV),
			VO: take(N * KV),
			AT: take(N * Q),
			GA: take(N * F),
			UP: take(N * F)
		};
		const tapBytes = Math.max(16, tapCount * N * D * 4);
		if (!this.long || this.long.arena.size < o * 4 || this.long.taps.size < tapBytes) {
			this.long?.arena.destroy();
			this.long?.taps.destroy();
			const arena = dev.createBuffer({
				label: 'long encoder',
				size: o * 4,
				usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
			});
			const tok =
				this.long?.tok ??
				dev.createBuffer({ size: N * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
			const bind = dev.createBindGroup({
				layout: this.bindLayout,
				entries: [
					{ binding: 0, resource: { buffer: this.params, size: 64 } },
					{ binding: 1, resource: { buffer: this.codes } },
					{ binding: 2, resource: { buffer: this.scales } },
					{ binding: 3, resource: { buffer: this.normw } },
					{ binding: 4, resource: { buffer: arena } },
					{ binding: 5, resource: { buffer: tok } },
					{ binding: 6, resource: { buffer: this.invBuf } }
				]
			});
			const taps = dev.createBuffer({
				label: 'long encoder taps',
				size: tapBytes,
				usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
			});
			this.long = {
				arena,
				taps,
				tok,
				bind,
				gemm: new TernaryGemm(dev, this.codes, this.scales, arena, 256)
			};
		}
		return { ...this.long, at };
	}

	private slotSub = -1;
	private slotAt = 0;
	private gemmAt = 0;

	/** Encode kernels, GEMMs and copies (arena -> taps buffer) in order; within one scheduler submission (`sub`)
	 * each call gets uniform slots of its own. */
	private runOps(ops: Op[], enc: GPUCommandEncoder, sub?: number) {
		const { arena, taps, bind, gemm } = this.long!;
		const stride = 256;
		if (sub === undefined || sub !== this.slotSub) {
			this.slotSub = sub ?? -1;
			this.slotAt = 0;
			this.gemmAt = 0;
		}
		const kernels = ops.filter((x) => x.k === 'kernel');
		const jobs = ops.flatMap((x) => (x.k === 'gemm' ? [x.job] : []));
		const k0 = this.slotAt,
			g0 = this.gemmAt;
		if (k0 + kernels.length > 1024 || g0 + jobs.length > 256)
			throw new Error('Too many kernels for one submission.');
		this.slotAt += kernels.length;
		this.gemmAt += jobs.length;
		const data = new ArrayBuffer(Math.max(1, kernels.length) * stride);
		kernels.forEach((cmd, i) => {
			const u = new Uint32Array(data, i * stride, 12);
			const f = new Float32Array(data, i * stride + 48, 4);
			cmd.p.slice(0, 12).forEach((v, k) => (u[k] = v));
			cmd.p.slice(12).forEach((v, k) => (f[k] = v));
		});
		if (kernels.length) this.device.queue.writeBuffer(this.params, k0 * stride, data);
		if (jobs.length) gemm.prepare(jobs, g0);
		let pass: GPUComputePassEncoder | null = null;
		let ki = k0,
			gi = g0;
		for (const op of ops) {
			if (op.k === 'copy') {
				pass?.end();
				pass = null;
				enc.copyBufferToBuffer(arena, op.from * 4, taps, op.to * 4, op.count * 4);
				continue;
			}
			pass ??= enc.beginComputePass();
			if (op.k === 'kernel') {
				pass.setPipeline(op.pipe);
				pass.setBindGroup(0, bind, [ki++ * stride]);
				pass.dispatchWorkgroups(...op.wg);
			} else gemm.dispatch(pass, gi++, op.job);
		}
		pass?.end();
	}

	/**
	 * Run up to 512 tokens through the first layers and return the residual stream after each tapped layer
	 * (tap k = the state after k layers, i.e. HF hidden_states[k]), as [taps][n][dim]. A lean pass: nothing else is
	 * kept, and the fast tiled GEMM does the matrix work. Causal, so its first tokens agree with prefill().
	 * With `real` set, tokens from that position on are padding: they attend to the real tokens only.
	 */
	private longOps(ids: number[], taps: number[], real: number): Op[] {
		const c = this.config,
			dev = this.device,
			n = ids.length;
		if (n < 1 || n > LONG_TOKENS) throw new Error(`encodeLong takes 1 to ${LONG_TOKENS} tokens.`);
		const D = c.dim,
			F = c.ffn,
			H = c.heads,
			KVH = c.kvHeads,
			HD = c.headDim,
			N = LONG_TOKENS;
		const { tok, at } = this.longSpace(taps.length);
		const { R, XN, QO, KO, VO, AT, GA, UP } = at;
		dev.queue.writeBuffer(tok, 0, new Uint32Array(ids));
		const ops: Op[] = [];
		const kern = (pipe: GPUComputePipeline, p: number[], wg: [number, number, number]) =>
			ops.push({ k: 'kernel', pipe, p, wg });
		const mm = (name: string, x: number, y: number, accumulate = false) => {
			const w = this.t(name);
			ops.push({
				k: 'gemm',
				job: { M: n, N: w.rows, K: w.cols, codes: w.codes, scales: w.scales, x, y, accumulate }
			});
		};
		const norm = (rows: number, len: number, x: number, y: number, weight: string) =>
			kern(
				this.pipes.rmsnorm,
				[rows, len, x, y, this.norms.get(weight)!, 0, 0, 0, 0, 0, 0, 0, c.eps],
				[rows, 1, 1]
			);
		const embd = this.t('token_embd.weight');
		kern(this.pipes.embed, [n, D, embd.codes, embd.scales, R], [Math.ceil(D / 256), n, 1]);
		const last = Math.max(...taps);
		const tapAt = (l: number) =>
			taps.forEach(
				(tp, i) => tp === l && ops.push({ k: 'copy', from: R, to: i * N * D, count: n * D })
			);
		tapAt(0);
		for (let l = 0; l < last; l++) {
			const b = `blk.${l}.`;
			norm(n, D, R, XN, b + 'attn_norm.weight');
			mm(b + 'attn_q.weight', XN, QO);
			mm(b + 'attn_k.weight', XN, KO);
			mm(b + 'attn_v.weight', XN, VO);
			norm(n * H, HD, QO, QO, b + 'attn_q_norm.weight');
			norm(n * KVH, HD, KO, KO, b + 'attn_k_norm.weight');
			kern(
				this.pipes.rope,
				[n, H, KVH, QO, KO, 0, 0, 0, 0, 0, 0, 0, this.attentionScaling],
				[n, H + KVH, 1]
			);
			kern(
				this.pipes.attention_long,
				[n, H, KVH, QO, KO, VO, AT, real, 0, 0, 0, 0, 1 / Math.sqrt(HD)],
				[n, H, 1]
			);
			mm(b + 'attn_output.weight', AT, R, true);
			norm(n, D, R, XN, b + 'ffn_norm.weight');
			mm(b + 'ffn_gate.weight', XN, GA);
			mm(b + 'ffn_up.weight', XN, UP);
			kern(this.pipes.swiglu, [n * F, GA, UP, GA], [Math.ceil((n * F) / 256), 1, 1]);
			mm(b + 'ffn_down.weight', GA, R, true);
			tapAt(l + 1);
		}
		return ops;
	}

	async encodeLong(ids: number[], taps: number[], real = 0): Promise<Float32Array[]> {
		const dev = this.device,
			n = ids.length,
			D = this.config.dim,
			N = LONG_TOKENS;
		const ops = this.longOps(ids, taps, real);
		const tapBuf = this.long!.taps;
		const enc = dev.createCommandEncoder();
		this.runOps(ops, enc);
		const out = dev.createBuffer({
			size: taps.length * n * D * 4,
			usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
		});
		taps.forEach((_, i) =>
			enc.copyBufferToBuffer(tapBuf, i * N * D * 4, out, i * n * D * 4, n * D * 4)
		);
		dev.queue.submit([enc.finish()]);
		await out.mapAsync(GPUMapMode.READ);
		const all = new Float32Array(out.getMappedRange().slice(0));
		out.destroy();
		return taps.map((_, i) => all.subarray(i * n * D, (i + 1) * n * D));
	}

	/**
	 * encodeLong as scheduler tasks (a few milliseconds each). When they have run, the taps sit in `longTaps`
	 * as [taps][512][dim] (rows beyond the sequence are stale).
	 */
	encodeLongTasks(ids: number[], taps: number[], real = 0, budget = 9): GpuTask[] {
		return this.opTasks(this.longOps(ids, taps, real), budget);
	}

	/** Group ops into scheduler tasks of about `budget` ms each. */
	private opTasks(ops: Op[], budget: number): GpuTask[] {
		const cost = (o: Op) =>
			o.k === 'gemm'
				? gemmMs(o.job.M, o.job.N, o.job.K)
				: o.k === 'copy'
					? 0.2
					: o.pipe === this.pipes.attention_long
						? 6
						: 0.4;
		const tasks: GpuTask[] = [];
		let group: Op[] = [],
			acc = 0;
		const flush = () => {
			if (!group.length) return;
			const g = group;
			tasks.push({ cost: acc, record: (enc, sub) => this.runOps(g, enc, sub) });
			group = [];
			acc = 0;
		};
		for (const o of ops) {
			const c = cost(o);
			if (acc + c > budget) flush();
			group.push(o);
			acc += c;
		}
		flush();
		return tasks;
	}

	/** Where encodeLong leaves its taps: [taps][512][dim] floats. */
	get longTaps(): GPUBuffer {
		if (!this.long) throw new Error('encodeLong has not run.');
		return this.long.taps;
	}

	/**
	 * The model's own reading of arbitrary residual-stream vectors (the logit lens): final norm, then a dot product
	 * with every vocabulary row. Returns the top token and its probability for each of the `rows` vectors.
	 */
	async readout(
		vectors: Float32Array,
		rows: number,
		scheduler?: GpuScheduler
	): Promise<{ ids: Uint32Array; probs: Float32Array }> {
		const c = this.config,
			dev = this.device,
			D = c.dim;
		const ids = new Uint32Array(rows),
			probs = new Float32Array(rows);
		const { arena, at } = this.longSpace(0);
		const embd = this.tensors.get('output.weight') ?? this.t('token_embd.weight');
		const room = 2 * LONG_TOKENS * c.ffn; // gate + up scratch holds one chunk of logits
		for (let r0 = 0; r0 < rows; r0 += LONG_TOKENS) {
			const m = Math.min(LONG_TOKENS, rows - r0);
			const ops: Op[] = [];
			const state = at.QO; // [m][3]
			ops.push({
				k: 'kernel',
				pipe: this.pipes.rmsnorm,
				p: [m, D, at.R, at.XN, this.norms.get('output_norm.weight')!, 0, 0, 0, 0, 0, 0, 0, c.eps],
				wg: [m, 1, 1]
			});
			const width = Math.floor(room / m / 64) * 64;
			for (let v0 = 0, first = 1; v0 < c.vocab; v0 += width, first = 0) {
				const w = Math.min(width, c.vocab - v0);
				ops.push({
					k: 'gemm',
					job: {
						M: m,
						N: w,
						K: D,
						codes: embd.codes + v0 * (D / 16),
						scales: embd.scales + v0 * (D / 128),
						x: at.XN,
						y: at.GA
					}
				});
				ops.push({
					k: 'kernel',
					pipe: this.pipes.argmax_chunk,
					p: [m, w, at.GA, state, v0, first],
					wg: [m, 1, 1]
				});
			}
			const out = dev.createBuffer({
				size: m * 12,
				usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
			});
			if (scheduler) {
				const tasks = this.opTasks(ops, scheduler.slice);
				const chunk = vectors.subarray(r0 * D, (r0 + m) * D);
				const record = tasks[0].record;
				tasks[0].record = (enc, sub) => {
					dev.queue.writeBuffer(arena, at.R * 4, chunk);
					record(enc, sub);
				};
				await new Promise<void>((resolve) =>
					scheduler.push(...tasks, {
						cost: 0.1,
						record: (enc) => enc.copyBufferToBuffer(arena, state * 4, out, 0, m * 12),
						done: resolve
					})
				);
			} else {
				dev.queue.writeBuffer(arena, at.R * 4, vectors, r0 * D, m * D);
				const enc = dev.createCommandEncoder();
				this.runOps(ops, enc);
				enc.copyBufferToBuffer(arena, state * 4, out, 0, m * 12);
				dev.queue.submit([enc.finish()]);
			}
			await out.mapAsync(GPUMapMode.READ);
			const st = new Float32Array(out.getMappedRange().slice(0));
			out.destroy();
			for (let i = 0; i < m; i++) {
				ids[r0 + i] = st[i * 3 + 1];
				probs[r0 + i] = 1 / st[i * 3 + 2]; // exp(max - lse)
			}
		}
		return { ids, probs };
	}

	/** Copy many small regions of the arena back in one round trip. */
	async readMany(regions: { offset: number; count: number }[]): Promise<Float32Array[]> {
		const total = regions.reduce((a, r) => a + r.count, 0);
		const buf = this.device.createBuffer({
			size: total * 4,
			usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
		});
		const enc = this.device.createCommandEncoder();
		let p = 0;
		for (const r of regions) {
			enc.copyBufferToBuffer(this.arena, r.offset * 4, buf, p * 4, r.count * 4);
			p += r.count;
		}
		this.device.queue.submit([enc.finish()]);
		await buf.mapAsync(GPUMapMode.READ);
		const all = new Float32Array(buf.getMappedRange().slice(0));
		buf.destroy();
		p = 0;
		return regions.map((r) => all.subarray(p, (p += r.count)));
	}

	/** Copy a region of the arena back to the CPU (for validation and small CPU-side summaries). */
	async read(offset: number, count: number): Promise<Float32Array> {
		const buf = this.device.createBuffer({
			size: count * 4,
			usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
		});
		const enc = this.device.createCommandEncoder();
		enc.copyBufferToBuffer(this.arena, offset * 4, buf, 0, count * 4);
		this.device.queue.submit([enc.finish()]);
		await buf.mapAsync(GPUMapMode.READ);
		const out = new Float32Array(buf.getMappedRange().slice(0));
		buf.destroy();
		return out;
	}
}

/** YaRN rotary frequencies exactly as transformers computes them for Qwen3 (beta_fast 32, beta_slow 1). */
export function yarn(c: LLMConfig): { inv: Float32Array; scaling: number } {
	const dim = c.headDim,
		base = c.ropeTheta,
		factor = c.ropeFactor || 1,
		orig = c.ropeOrigCtx;
	const half = dim / 2;
	const inv = new Float32Array(half);
	const extrap = (i: number) => 1 / base ** ((2 * i) / dim);
	if (factor <= 1 || !orig) {
		for (let i = 0; i < half; i++) inv[i] = extrap(i);
		return { inv, scaling: 1 };
	}
	const corr = (rot: number) => (dim * Math.log(orig / (rot * 2 * Math.PI))) / (2 * Math.log(base));
	const low = Math.max(Math.floor(corr(32)), 0);
	let high = Math.min(Math.ceil(corr(1)), dim - 1);
	if (low === high) high += 0.001;
	for (let i = 0; i < half; i++) {
		const ramp = Math.min(1, Math.max(0, (i - low) / (high - low)));
		const extrapFactor = 1 - ramp;
		inv[i] = (extrap(i) / factor) * (1 - extrapFactor) + extrap(i) * extrapFactor;
	}
	return { inv, scaling: 0.1 * Math.log(factor) + 1 };
}
