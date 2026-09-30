// Live WebGPU runtime for the painter: Bonsai Image 4B (FLUX.2 klein architecture, ternary DiT), fed by Ternary
// Bonsai 1.7B through the ternary adapter, 4 flow-matching Euler steps at 512 x 512, decoded by TAEF2.
// Follows research/PAINTER-SPEC.md and research/scripts/painter_reference.py op by op:
//   text:  taps h7|h14|h21 of the 1.7B over 512 tokens -> adapter (ternary 6144 -> 7680, + bias; rows 0..2 fixed)
//          (in the one-file model, mindview-t2i, the adapter and W_ctx are one map, 6144 -> 3072: see fromPacked)
//   DiT:   x = lat W_x^T, c = ctx W_ctx^T; 5 double blocks (text and image weights, joint attention), then 20 single
//          blocks over the joint sequence (text first); velocity = (LN(h_img)(1 + scale) + shift) W_out^T
//   step:  lat += (sigma_{s+1} - sigma_s) v
// Everything is f32 on the GPU; ternary matrices use the tiled ternary GEMM, dense ones a strided batched GEMM.
import type { BonsaiLLM } from './bonsai-llm';
import { fetchModelFile, fetchModelJson } from './cache';
import { halfToFloat, unpackTrits, GGML_TRIT5, type GGUFTensor } from './gguf';
import { PackedModel } from './packed';
import { TernaryGemm, type GemmJob } from './gemm';
import { DenseGemm, denseOk, type DenseJob } from './dense';
import { PainterFiles } from './painter-files';
import { gemmMs, type GpuTask } from './scheduler';
import { Taef2 } from './taef2';

const D = 3072,
	H = 24,
	HD = 128,
	MLP = 9216,
	NT = 512, // text tokens
	NI = 1024, // image tokens
	NJ = NT + NI,
	ROPE_IMG = NT, // the rope table's first image row (text rows are 0 .. 511)
	CIN = 128,
	CTX = 7680,
	TAP = 6144;

const WGSL = /* wgsl */ `
struct P { a: u32, b: u32, c: u32, d: u32, e: u32, f: u32, g: u32, h: u32, i: u32, j: u32, k: u32, l: u32, m: u32, n: u32, o: u32, p: u32, fa: f32, fb: f32, fc: f32, fd: f32 };
@group(0) @binding(0) var<uniform> U: P;
@group(0) @binding(1) var<storage, read> WD: array<f32>;
@group(0) @binding(2) var<storage, read_write> A: array<f32>;

var<workgroup> red: array<f32, 512>;

// LayerNorm (no affine, eps fa) then modulation: y = LN(x) * (1 + scale) + shift
// a=rows, b=width, c=x, d=y, e=shift, f=scale, g=x row stride, h=y row stride
@compute @workgroup_size(256)
fn modulate(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_id) li: vec3u) {
  let r = wg.x; let i = li.x;
  let xb = U.c + r * U.g;
  var s = 0.0;
  for (var k = i; k < U.b; k += 256u) { s += A[xb + k]; }
  red[i] = s;
  workgroupBarrier();
  for (var st = 128u; st > 0u; st >>= 1u) { if (i < st) { red[i] += red[i + st]; } workgroupBarrier(); }
  let mean = red[0] / f32(U.b);
  workgroupBarrier();
  var v = 0.0;
  for (var k = i; k < U.b; k += 256u) { let d = A[xb + k] - mean; v += d * d; }
  red[i] = v;
  workgroupBarrier();
  for (var st = 128u; st > 0u; st >>= 1u) { if (i < st) { red[i] += red[i + st]; } workgroupBarrier(); }
  let inv = inverseSqrt(red[0] / f32(U.b) + U.fa);
  let yb = U.d + r * U.h;
  for (var k = i; k < U.b; k += 256u) {
    A[yb + k] = (A[xb + k] - mean) * inv * (1.0 + A[U.f + k]) + A[U.e + k];
  }
}

// per-head RMSNorm (learned gain, eps fa) then rotary embedding on interleaved pairs (2p, 2p+1)
// a=tokens, b=src, c=src stride, d=dst, e=dst stride, f=gain (WD), g=cos (WD) [tokens][64], h=sin (WD), i=first rope row
@compute @workgroup_size(64)
fn qknorm_rope(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_id) li: vec3u) {
  let t = wg.x; let hh = wg.y; let p = li.x;
  let sb = U.b + t * U.c + hh * 128u + 2u * p;
  let x0 = A[sb]; let x1 = A[sb + 1u];
  red[p] = x0 * x0 + x1 * x1;
  workgroupBarrier();
  for (var st = 32u; st > 0u; st >>= 1u) { if (p < st) { red[p] += red[p + st]; } workgroupBarrier(); }
  let inv = inverseSqrt(red[0] / 128.0 + U.fa);
  let a = x0 * inv * WD[U.f + 2u * p];
  let b = x1 * inv * WD[U.f + 2u * p + 1u];
  let row = (U.i + t) * 64u + p;
  let c = WD[U.g + row]; let s = WD[U.h + row];
  let db = U.d + t * U.e + hh * 128u + 2u * p;
  A[db] = a * c - b * s;
  A[db + 1u] = b * c + a * s;
}

// softmax over each row, in place: a=rows, b=width, c=offset
@compute @workgroup_size(256)
fn softmax(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_id) li: vec3u) {
  let r = wg.x + wg.y * 65535u;
  if (r >= U.a) { return; }
  let i = li.x;
  let base = U.c + r * U.b;
  var m = -1e30;
  for (var k = i; k < U.b; k += 256u) { m = max(m, A[base + k]); }
  red[i] = m;
  workgroupBarrier();
  for (var st = 128u; st > 0u; st >>= 1u) { if (i < st) { red[i] = max(red[i], red[i + st]); } workgroupBarrier(); }
  let mx = red[0];
  workgroupBarrier();
  var s = 0.0;
  for (var k = i; k < U.b; k += 256u) { let e = exp(A[base + k] - mx); A[base + k] = e; s += e; }
  red[i] = s;
  workgroupBarrier();
  for (var st = 128u; st > 0u; st >>= 1u) { if (i < st) { red[i] += red[i + st]; } workgroupBarrier(); }
  let inv = 1.0 / red[0];
  for (var k = i; k < U.b; k += 256u) { A[base + k] *= inv; }
}

// SwiGLU: out[r][j] = silu(in[r][j]) * in[r][half + j]
// a=rows, b=half, c=in, d=in stride, e=out, f=out stride
@compute @workgroup_size(256)
fn swiglu(@builtin(global_invocation_id) g: vec3u) {
  let j = g.x; let r = g.y;
  if (j >= U.b || r >= U.a) { return; }
  let x = A[U.c + r * U.d + j];
  A[U.e + r * U.f + j] = x / (1.0 + exp(-x)) * A[U.c + r * U.d + U.b + j];
}

// The painting this block has in mind: x0 = x_t - sigma * v_lens, written in TAEF2's layout [32][64][64].
// a=values (1024 * 128), b=v_lens [1024][128] (no bias), c=lens bias (WD), d=lat, e=x0 out, fa=sigma
@compute @workgroup_size(256)
fn lens_x0(@builtin(global_invocation_id) g: vec3u) {
  let i = g.x;
  if (i >= U.a) { return; }
  let t = i / 128u; let ch = i % 128u;
  let h = t / 32u; let w = t % 32u;
  let c = ch / 4u; let dy = (ch % 4u) / 2u; let dx = ch % 2u;
  A[U.e + (c * 64u + 2u * h + dy) * 64u + 2u * w + dx] = A[U.d + i] - U.fa * (A[U.b + i] + WD[U.c + ch]);
}

// After a step: its clean guess x0 = lat - sigma' v (lat already moved on to sigma'), unpatchified into TAEF2's layout
// [32][2G][2G]. a=values (G * G * 128), b=vel, d=lat, e=x0 out, f=G (patches a side), fa=sigma'
@compute @workgroup_size(256)
fn x0_unpatch(@builtin(global_invocation_id) g: vec3u) {
  let i = g.x;
  if (i >= U.a) { return; }
  let t = i / 128u; let ch = i % 128u;
  let h = t / U.f; let w = t % U.f;
  let c = ch / 4u; let dy = (ch % 4u) / 2u; let dx = ch % 2u;
  let side = 2u * U.f;
  A[U.e + (c * side + 2u * h + dy) * side + 2u * w + dx] = A[U.d + i] - U.fa * A[U.b + i];
}

// After a first step at 16 x 16 patches: its clean guess (b: [32][32][32], from x0_unpatch) enlarged to 32 x 32 patches,
// bilinear per channel as torch's interpolate(scale 2, align_corners=False), into e ([32][64][64], TAEF2's layout: the
// sketch) and, mixed back with noise c ([1024][128]) to sigma fa, into the latent d ([1024][128]). a=1024 * 128
@compute @workgroup_size(256)
fn upsample_sketch(@builtin(global_invocation_id) g: vec3u) {
  let i = g.x;
  if (i >= U.a) { return; }
  let t = i / 128u; let ch = i % 128u;
  let c = ch / 4u;
  let y = 2u * (t / 32u) + (ch % 4u) / 2u; let x = 2u * (t % 32u) + ch % 2u;
  let sy = (f32(y) + 0.5) * 0.5 - 0.5; let sx = (f32(x) + 0.5) * 0.5 - 0.5;
  let y0 = floor(sy); let x0 = floor(sx);
  let fy = sy - y0; let fx = sx - x0;
  let ya = u32(clamp(y0, 0.0, 31.0)); let yb = u32(clamp(y0 + 1.0, 0.0, 31.0));
  let xa = u32(clamp(x0, 0.0, 31.0)); let xb = u32(clamp(x0 + 1.0, 0.0, 31.0));
  let base = U.b + c * 1024u;
  let v = (1.0 - fy) * ((1.0 - fx) * A[base + ya * 32u + xa] + fx * A[base + ya * 32u + xb])
        + fy * ((1.0 - fx) * A[base + yb * 32u + xa] + fx * A[base + yb * 32u + xb]);
  A[U.e + (c * 64u + y) * 64u + x] = v;
  A[U.d + i] = (1.0 - U.fa) * v + U.fa * A[U.c + i];
}

// How much each image patch attends to each prompt word (mean over heads) in the last attention.
// a=image tokens, b=words, c=probabilities [heads][joint][joint], d=joint index of the first word, e=out [a][b],
// f=joint index of the first image token, g=joint length, h=heads
@compute @workgroup_size(64)
fn attn_words(@builtin(global_invocation_id) g: vec3u) {
  let i = g.x; let j = g.y;
  if (i >= U.a || j >= U.b) { return; }
  var s = 0.0;
  for (var hh = 0u; hh < U.h; hh++) { s += A[U.c + (hh * U.g + U.f + i) * U.g + U.d + j]; }
  A[U.e + i * U.b + j] = s / f32(U.h);
}

// For drawing the computation as threads: the mean of a block of rows (a = rows, b = width, c = rows, d = mean out)
@compute @workgroup_size(256)
fn row_mean(@builtin(global_invocation_id) g: vec3u) {
  let k = g.x;
  if (k >= U.b) { return; }
  var s = 0.0;
  for (var r = 0u; r < U.a; r++) { s += A[U.c + r * U.b + k]; }
  A[U.d + k] = s / f32(U.a);
}

// ... and each row, less that mean, projected on three fixed directions (WD at f, [3][b]), with its length:
// out at d, 4 per row (b = width, c = rows, e = mean)
var<workgroup> red4: array<vec4f, 128>;
@compute @workgroup_size(128)
fn row_project(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_id) li: vec3u) {
  let r = wg.x; let i = li.x;
  var p = vec4f(0.0);
  for (var k = i; k < U.b; k += 128u) {
    let v = A[U.c + r * U.b + k] - A[U.e + k];
    p += vec4f(v * WD[U.f + k], v * WD[U.f + U.b + k], v * WD[U.f + 2u * U.b + k], v * v);
  }
  red4[i] = p;
  workgroupBarrier();
  for (var st = 64u; st > 0u; st >>= 1u) { if (i < st) { red4[i] += red4[i + st]; } workgroupBarrier(); }
  if (i == 0u) {
    let t = red4[0];
    A[U.d + r * 4u] = t.x; A[U.d + r * 4u + 1u] = t.y; A[U.d + r * 4u + 2u] = t.z; A[U.d + r * 4u + 3u] = sqrt(t.w);
  }
}

// The attention of a block at a glance: mean over heads, pooled b x b (each pooled row still sums to 1).
// a=joint length, b=pool, c=probabilities [heads][a][a], d=out [a/b][a/b], e=heads
@compute @workgroup_size(8, 8)
fn attn_pool(@builtin(global_invocation_id) g: vec3u) {
  let n = U.a / U.b;
  if (g.x >= n || g.y >= n) { return; }
  var s = 0.0;
  for (var hh = 0u; hh < U.e; hh++) {
    for (var y = 0u; y < U.b; y++) {
      let base = U.c + (hh * U.a + g.y * U.b + y) * U.a + g.x * U.b;
      for (var x = 0u; x < U.b; x++) { s += A[base + x]; }
    }
  }
  A[U.d + g.y * n + g.x] = s / f32(U.e * U.b);
}

// Euler step: lat += fa * vel.  a=count, b=lat, c=vel
// copy floats within the activations: a=count, b=from, c=to
@compute @workgroup_size(256)
fn copy_rows(@builtin(global_invocation_id) g: vec3u) {
  let i = g.x + g.y * 65535u * 256u;
  if (i >= U.a) { return; }
  A[U.c + i] = A[U.b + i];
}

@compute @workgroup_size(256)
fn euler(@builtin(global_invocation_id) g: vec3u) {
  if (g.x >= U.a) { return; }
  A[U.b + g.x] += U.fa * A[U.c + g.x];
}

// Strided batched GEMM (f32): C[z][m][n] = fa * sum_k A[z][m][k] B[z](k, n)  (+ C if accumulate)
// a=M, b=N, c=K, d=A, e=lda, f=A batch stride, g=B, h=ldb, i=B batch stride, j=C, k=ldc, l=C batch stride,
// m=flags: 1 B from the weights buffer, 2 B stored [k][n] (else [n][k]), 4 accumulate into C
var<workgroup> As: array<f32, 2048>;
var<workgroup> Bs: array<f32, 2048>;
fn bval(o: u32) -> f32 { if ((U.m & 1u) != 0u) { return WD[o]; } return A[o]; }
@compute @workgroup_size(16, 16)
fn dgemm(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_id) li: vec3u, @builtin(local_invocation_index) lid: u32) {
  let m0 = wg.y * 64u; let n0 = wg.x * 64u; let z = wg.z;
  let ab = U.d + z * U.f; let bb = U.g + z * U.i; let cb = U.j + z * U.l;
  let nn = (U.m & 2u) != 0u;
  var acc0 = vec4f(0.0); var acc1 = vec4f(0.0); var acc2 = vec4f(0.0); var acc3 = vec4f(0.0);
  for (var k0 = 0u; k0 < U.c; k0 += 32u) {
    for (var i = 0u; i < 8u; i++) {
      let idx = lid + i * 256u;
      let r = idx / 32u; let kk = idx % 32u;
      var v = 0.0;
      if (m0 + r < U.a && k0 + kk < U.c) { v = A[ab + (m0 + r) * U.e + k0 + kk]; }
      As[kk * 64u + r] = v;
    }
    for (var i = 0u; i < 8u; i++) {
      let idx = lid + i * 256u;
      var c = 0u; var kk = 0u;
      if (nn) { kk = idx / 64u; c = idx % 64u; } else { c = idx / 32u; kk = idx % 32u; }
      var v = 0.0;
      if (n0 + c < U.b && k0 + kk < U.c) {
        if (nn) { v = bval(bb + (k0 + kk) * U.h + n0 + c); } else { v = bval(bb + (n0 + c) * U.h + k0 + kk); }
      }
      Bs[kk * 64u + c] = v;
    }
    workgroupBarrier();
    for (var kk = 0u; kk < 32u; kk++) {
      let a = vec4f(As[kk * 64u + li.y * 4u], As[kk * 64u + li.y * 4u + 1u], As[kk * 64u + li.y * 4u + 2u], As[kk * 64u + li.y * 4u + 3u]);
      let b = vec4f(Bs[kk * 64u + li.x * 4u], Bs[kk * 64u + li.x * 4u + 1u], Bs[kk * 64u + li.x * 4u + 2u], Bs[kk * 64u + li.x * 4u + 3u]);
      acc0 += a.x * b; acc1 += a.y * b; acc2 += a.z * b; acc3 += a.w * b;
    }
    workgroupBarrier();
  }
  let accs = array<vec4f, 4>(acc0, acc1, acc2, acc3);
  for (var i = 0u; i < 4u; i++) {
    let m = m0 + li.y * 4u + i;
    if (m >= U.a) { continue; }
    for (var j = 0u; j < 4u; j++) {
      let n = n0 + li.x * 4u + j;
      if (n >= U.b) { continue; }
      let o = cb + m * U.k + n;
      var v = U.fa * accs[i][j];
      if ((U.m & 4u) != 0u) { v += A[o]; }
      A[o] = v;
    }
  }
}`;

type Kernel =
	| 'modulate'
	| 'qknorm_rope'
	| 'softmax'
	| 'swiglu'
	| 'euler'
	| 'dgemm'
	| 'lens_x0'
	| 'x0_unpatch'
	| 'upsample_sketch'
	| 'attn_words'
	| 'row_mean'
	| 'row_project'
	| 'attn_pool'
	| 'copy_rows';
/** Seeded Gaussian noise (LCG + Box-Muller), as setNoise and upsampleLatent make it. */
function gaussian(n: number, seed: number): Float32Array {
	const z = new Float32Array(n);
	let r = seed >>> 0 || 1;
	const rnd = () => ((r = (r * 1664525 + 1013904223) >>> 0) + 0.5) / 4294967296;
	for (let i = 0; i < n; i += 2) {
		const m = Math.sqrt(-2 * Math.log(rnd())),
			th = 2 * Math.PI * rnd();
		z[i] = m * Math.cos(th);
		z[i + 1] = m * Math.sin(th);
	}
	return z;
}

/** The rope table's rows for a 16 x 16 patch grid (a first step at 256 x 256): rows 512 + 32 h + w of the full one. */
function rope16(full: Float32Array): Float32Array {
	const out = new Float32Array(256 * 64);
	for (let h = 0; h < 16; h++)
		for (let w = 0; w < 16; w++)
			out.set(
				full.subarray((ROPE_IMG + 32 * h + w) * 64, (ROPE_IMG + 32 * h + w + 1) * 64),
				(16 * h + w) * 64
			);
	return out;
}

/**
 * Side branches of the DiT's ternary matrices (y = W x + B (A x)), one set per prefix: `lora` (the few-step LoRA, for 1
 * and 2 steps) and `rdm` (EPFL's 1-step fine-tune, with schedule '1r'). A [rank][in] is in the dense buffer
 * (`<prefix>.<module>.a`), B transposed [rank][out] in a buffer of its own at the offsets in `at` (`<prefix>.<module>.bt`).
 * Ranks are padded to multiples of 32.
 */
export interface PainterLora {
	buf: GPUBuffer;
	at: Map<string, number>;
	ranks: Partial<Record<Branch, number>>;
}
type Branch = 'lora' | 'rdm' | 'best';
/** A schedule: a step count, or '1r' (1 step with the `rdm` branch). */
export type Schedule = number | '1r';

type Op =
	| { k: 'kernel'; name: Kernel; p: number[]; f?: number[]; wg: [number, number, number] }
	| { k: 'gemm'; job: GemmJob }
	| { k: 'dense'; job: DenseJob } // f16 dense GEMM (attention)
	| { k: 'copy'; from: number; to: GPUBuffer; at: number; count: number }; // arena floats -> buffer floats

/** Where the numbers of one row live in a stream at a tap point: the region, its row width, its first row. */
type Tap = (
	ops: Op[],
	name: string,
	region: number,
	width: number,
	rowBase: number,
	rows: 'txt' | 'img' | 'all',
	block: number
) => void;

const POOL = 8; // the pooled attention map: 1536 / 8 = 192 on a side

/**
 * What a painting keeps for the labs, filled in while it paints: for the rows followed (the prompt's words and some
 * image patches, by joint index: text 0..511, image 512..1535), every step and block, the input of every matrix
 * ('light'); for a few rows, everything the block computes ('full'); per block, the attention pooled to 192 x 192.
 *   double block: n1 (q, k, v input), o (output projection input), n2 (MLP input), cat (MLP output input);
 *                 full: h_in, q, k, v (raw), qr, kr (normed and turned), h_mid, p (MLP gate | up), h_out, attn
 *   single block: n1 (fused input), cat (output input: attention | MLP); full: h_in, p (q | k | v | gate | up), qr,
 *                 kr, h_out, attn
 *   attn: the row's attention over all 1536 rows, per head [24][1536]
 *   per step (block -1): lat (the patch's latent, before the step); (block 25): nout (input of the output
 *   projection), vel (the patch's velocity)
 *   per word (step -1, block -1): taps (the reader's layers 7 | 14 | 21), ctx (the adapter's output)
 */
export class PainterCapture {
	readonly buffer: GPUBuffer;
	readonly size: number;
	private map = new Map<string, number>();

	constructor(
		device: GPUDevice,
		readonly steps: number,
		readonly words: number[],
		readonly patches: number[],
		readonly full: number[],
		/** The steps at which the full rows keep everything (the others keep what every row keeps). */
		readonly fullSteps: number[] = Array.from({ length: steps }, (_, i) => i)
	) {
		let o = 0;
		const put = (s: number, b: number, r: number, name: string, n: number) => {
			this.map.set(`${s}|${b}|${r}|${name}`, o);
			o += n;
		};
		const rows = [...words, ...patches];
		for (const r of words) {
			put(-1, -1, r, 'taps', TAP);
			put(-1, -1, r, 'ctx', CTX);
		}
		for (let s = 0; s < steps; s++) {
			for (let b = 0; b < 25; b++) put(s, b, -1, 'pool', (NJ / POOL) ** 2);
			for (const r of rows) {
				const isFull = full.includes(r) && fullSteps.includes(s);
				if (r >= NT) put(s, -1, r, 'lat', CIN);
				for (let b = 0; b < 25; b++) {
					const dbl = b < 5;
					put(s, b, r, 'n1', D);
					put(s, b, r, 'cat', dbl ? MLP : D + MLP);
					if (dbl) {
						put(s, b, r, 'o', D);
						put(s, b, r, 'n2', D);
					}
					if (!isFull) continue;
					put(s, b, r, 'h_in', D);
					put(s, b, r, 'h_out', D);
					put(s, b, r, 'qr', D);
					put(s, b, r, 'kr', D);
					put(s, b, r, 'attn', H * NJ);
					if (dbl) {
						put(s, b, r, 'q', D);
						put(s, b, r, 'k', D);
						put(s, b, r, 'v', D);
						put(s, b, r, 'h_mid', D);
						put(s, b, r, 'p', 2 * MLP);
					} else put(s, b, r, 'p', 3 * D + 2 * MLP);
				}
				if (r >= NT) {
					put(s, 25, r, 'nout', D);
					put(s, 25, r, 'vel', CIN);
				}
			}
		}
		this.size = o;
		this.buffer = device.createBuffer({
			label: 'painter capture',
			size: Math.max(16, o * 4),
			usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
		});
	}

	/** Where a vector is kept (floats into buffer), or undefined if it is not. */
	at(step: number, block: number, row: number, name: string): number | undefined {
		return this.map.get(`${step}|${block}|${row}|${name}`);
	}

	destroy() {
		this.buffer.destroy();
	}
}

export interface PainterProgress {
	stage: string;
	fraction: number;
}

/** Where the painter reads its settings and constants: its own files (PainterFiles) or the one-file model. */
export interface PainterSource {
	manifest: PainterFiles['manifest'];
	dense(name: string): Float32Array;
	names(prefix: string): string[];
}

/** Called after every block of every step with where things stand (for the visuals). */
export type BlockHook = (step: number, block: number) => void | Promise<void>;

/**
 * The painter's states after one block, on the drawing space: per image patch and per prompt row (from row 3), the
 * projections on three fixed directions and the length of the state less the group's mean (4 floats each); and the
 * block's quick picture (64 x 64 rgba floats), for the patches' colours.
 */
export interface BlockSpace {
	img: Float32Array;
	txt: Float32Array;
	colours: Float32Array | null;
}

export class Painter {
	readonly taef2: Taef2;
	/** Real (unpadded) rows of the prompt the painter last read. */
	private textRows = 4;
	/**
	 * How many text rows the DiT runs over: 512 as the model was trained, or 'auto': the prompt and at least 8 pads,
	 * in steps of 64 (the pads after the first few change little; each row saved is work saved in every block).
	 */
	textLength: number | 'auto' = NT;
	/** Text rows of the current prompt's stream (joint rows: text 0 .. nt - 1, then the 1024 image rows). */
	private nt = NT;
	private get nj() {
		return this.nt + this.ni;
	}

	/** A step's shape: image rows (256 for the coarse first steps at 256 x 256, see setNoise), text rows, joint rows. */
	private dims(s: number) {
		const ni = s < this.low ? 256 : NI;
		const nt = ni === NI ? this.nt : Math.min(this.nt, this.lowTextRows);
		return { ni, nt, nj: nt + ni, grid: ni === NI ? 32 : 16 };
	}
	/** How many of the painting's first steps run at 256 x 256 (setNoise with low). */
	private low = 0;
	private seed = 7;
	/** Image rows of the current step: 1024 (32 x 32 patches, 512 x 512) or 256 (16 x 16, a first step at 256 x 256). */
	private ni = NI;
	/** Text rows for a first step at 256 x 256 (setNoise with low): a multiple of 64. */
	lowTextRows = 128;
	private pickRows(tokens: number) {
		const want = this.textLength === 'auto' ? Math.ceil((tokens + 8) / 64) * 64 : this.textLength;
		return Math.min(NT, Math.max(64, Math.ceil(want / 64) * 64));
	}
	readonly arena: GPUBuffer;
	private dense: GPUBuffer;
	private codesBuf: GPUBuffer;
	private scalesBuf: GPUBuffer;
	private gemm: TernaryGemm;
	private dense16?: DenseGemm; // attention's matrix products in f16 (with precision 'f16')
	private pipes: Record<Kernel, GPUComputePipeline>;
	private bind: GPUBindGroup;
	private params: GPUBuffer;
	private tern = new Map<string, { rows: number; cols: number; codes: number; scales: number }>();
	private denseAt = new Map<string, number>();
	readonly at: Record<string, number>;
	sigmas: number[];
	steps = 4;
	/** A few-step LoRA as side branches of the DiT's ternary matrices (from the one-file model), and whether it is on. */
	private lora?: PainterLora;
	/** The ternary GEMMs' arithmetic: 'f16' (the default where the GPU has it: about 1.4x) or 'f32'. */
	get precision() {
		return this.gemm.precision;
	}
	set precision(p: 'f32' | 'f16') {
		this.gemm.precision = p;
	}
	/**
	 * The side branch in use (setSteps): the few-step LoRA (or the trained Fast branch) with 1 or 2 steps, RDM with
	 * '1r', the trained Best branch with 4; none otherwise.
	 */
	private branch?: Branch;
	get hasLora() {
		return !!this.lora?.ranks.lora;
	}
	/** Whether the file carries the 1-step fine-tune (schedule '1r'). */
	get hasOneStep() {
		return !!this.lora?.ranks.rdm && !!this.schedules?.meta['1r'];
	}
	private schedules?: {
		meta: Record<string, { sigmas: number[]; offset: number }>;
		data: Float32Array;
	};
	readonly manifest: PainterFiles['manifest'];
	/** The adapter and the context embedder are one map (the one-file model): text rows are made once per prompt. */
	readonly fused: boolean;
	private ctxInit?: Float32Array;

	private constructor(
		private device: GPUDevice,
		files: PainterSource,
		codes: GPUBuffer,
		scales: GPUBuffer,
		dense: GPUBuffer,
		tern: Map<string, { rows: number; cols: number; codes: number; scales: number }>,
		denseAt: Map<string, number>,
		probes: Map<string, Float32Array>, // colour probes for the decoder's stages (probe64, probe128, probe256)
		lora?: PainterLora
	) {
		this.lora = lora;
		this.manifest = files.manifest;
		this.dense = dense;
		this.codesBuf = codes;
		this.scalesBuf = scales;
		this.tern = tern;
		this.denseAt = denseAt;
		this.sigmas = (files.manifest.schedule as { sigmas: number[] }).sigmas;
		// activations
		let o = 0;
		const regions: [string, number][] = [
			['lat', NI * CIN],
			['noise', NI * CIN],
			['vel', NI * CIN],
			['taps', NT * TAP],
			['ctx', NT * CTX],
			['ctxd', NT * D], // the text rows of the stream, as the fused map makes them (fused only)
			['h', NJ * D],
			['n1', NJ * D],
			['q', NJ * D],
			['k', NJ * D],
			['v', NJ * D],
			['o', NJ * D],
			['p', NJ * (3 * D + 2 * MLP)],
			['cat', NJ * (D + MLP)],
			['s', H * NJ * NJ],
			['mod', 12 * 17 * D],
			['lt', NJ * Math.max(0, ...Object.values(lora?.ranks ?? {}))], // a side branch's A x (see stepOps)
			['abias', CTX],
			['vl', NI * CIN],
			['x0', 32 * 64 * 64],
			['aw', NI * 128],
			['awt', 128 * 128], // the prompt's words reading each other (joint attention, text rows)
			['mu', 2 * D], // the mean image row and mean text row after a block
			['proj', NJ * 4], // each row's projection on the drawing space (image rows, then text rows)
			['pool', (NJ / POOL) ** 2] // a block's attention pooled (see attn_pool)
		];
		const at: Record<string, number> = {};
		for (const [name, n] of regions) {
			at[name] = o;
			o += Math.ceil(n / 64) * 64;
		}
		this.at = at;
		this.arena = device.createBuffer({
			label: 'painter activations',
			size: o * 4,
			usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
		});
		// timestep-only vectors and the adapter's constants live in the arena, where the GEMM epilogues can read them
		const mods = ['double_img', 'double_txt', 'single', 'norm_out'].map((k) =>
			files.dense(`mod.${k}`)
		);
		for (let s = 0; s < 4; s++) {
			const row = new Float32Array(17 * D);
			row.set(mods[0].subarray(s * 6 * D, (s + 1) * 6 * D), 0);
			row.set(mods[1].subarray(s * 6 * D, (s + 1) * 6 * D), 6 * D);
			row.set(mods[2].subarray(s * 3 * D, (s + 1) * 3 * D), 12 * D);
			row.set(mods[3].subarray(s * 2 * D, (s + 1) * 2 * D), 15 * D);
			device.queue.writeBuffer(this.arena, (at.mod + s * 17 * D) * 4, row);
		}
		this.fused = denseAt.has('cond.weight') || tern.has('cond.weight');
		if (this.fused) {
			// rows 0..2: the fixed prefix; rows 3..: the bias, to which the map adds each row's taps
			const init = new Float32Array(NT * D);
			init.set(files.dense('cond.prefix'), 0);
			const bias = files.dense('cond.bias');
			for (let r = 3; r < NT; r++) init.set(bias, r * D);
			this.ctxInit = init;
			// a ternary map adds its bias in the GEMM's epilogue
			if (tern.has('cond.weight')) device.queue.writeBuffer(this.arena, at.abias * 4, bias);
		} else {
			device.queue.writeBuffer(this.arena, at.abias * 4, files.dense('adapter.bias'));
			device.queue.writeBuffer(this.arena, at.ctx * 4, files.dense('adapter.prefix'));
		}

		this.gemm = new TernaryGemm(device, codes, scales, this.arena, 1024, 16, lora?.buf);
		this.dense16 = device.features.has('shader-f16')
			? new DenseGemm(device, this.arena, 256)
			: undefined;
		const module = device.createShaderModule({ label: 'painter', code: WGSL });
		const layout = device.createBindGroupLayout({
			entries: [
				{
					binding: 0,
					visibility: GPUShaderStage.COMPUTE,
					buffer: { type: 'uniform', hasDynamicOffset: true, minBindingSize: 80 }
				},
				{ binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
				{ binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }
			]
		});
		const pl = device.createPipelineLayout({ bindGroupLayouts: [layout] });
		const names: Kernel[] = [
			'modulate',
			'qknorm_rope',
			'softmax',
			'swiglu',
			'euler',
			'dgemm',
			'lens_x0',
			'x0_unpatch',
			'upsample_sketch',
			'attn_words',
			'row_mean',
			'row_project',
			'attn_pool',
			'copy_rows'
		];
		this.pipes = Object.fromEntries(
			names.map((e) => [
				e,
				device.createComputePipeline({ layout: pl, compute: { module, entryPoint: e } })
			])
		) as Record<Kernel, GPUComputePipeline>;
		this.params = device.createBuffer({
			size: 1024 * 256,
			usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
		});
		this.bind = device.createBindGroup({
			layout,
			entries: [
				{ binding: 0, resource: { buffer: this.params, size: 80 } },
				{ binding: 1, resource: { buffer: dense } },
				{ binding: 2, resource: { buffer: this.arena } }
			]
		});
		const tw = new Map(files.names('taef2.').map((k) => [k.slice(6), files.dense(k)]));
		for (const [k, v] of probes) tw.set(k, v);
		this.taef2 = new Taef2(device, tw, 512);
	}

	/** Download the painter (about 1.1 GB) and put it on the GPU. */
	static async load(
		device: GPUDevice,
		base: string,
		onProgress?: (p: PainterProgress) => void
	): Promise<Painter> {
		const files = await PainterFiles.open(base);
		const names = Object.keys(files.manifest.files as Record<string, unknown>);
		const tensors = files.manifest.tensors;
		// one codes buffer and one scales buffer for every ternary tensor (DiT and adapter)
		const tern = new Map<string, { rows: number; cols: number; codes: number; scales: number }>();
		let nc = 0,
			ns = 0;
		for (const [k, e] of Object.entries(tensors)) {
			if (e.kind !== 'ternary') continue;
			tern.set(k, { rows: e.shape[0], cols: e.shape[1], codes: nc, scales: ns });
			nc += e.codes.bytes / 4;
			ns += e.scales.bytes / 4;
		}
		// (copyable, so a lab can read back one weight exactly)
		const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
		const codes = device.createBuffer({ label: 'painter codes', size: nc * 4, usage });
		const scales = device.createBuffer({ label: 'painter scales', size: ns * 4, usage });
		// dense weights the DiT reads: embedders, projection out, QK gains, rope tables
		const denseNames = Object.keys(tensors).filter(
			(k) =>
				tensors[k].kind === 'dense' &&
				!k.startsWith('taef2.') &&
				!k.startsWith('mod.') &&
				!k.startsWith('adapter.') &&
				k !== 'temb'
		);
		const denseAt = new Map<string, number>();
		let nd = 0;
		for (const k of denseNames) {
			denseAt.set(k, nd);
			nd += (tensors[k] as { shape: number[] }).shape.reduce((a, b) => a * b, 1);
		}
		for (const k of ['rope.cos16', 'rope.sin16']) {
			denseAt.set(k, nd);
			nd += 256 * 64;
		}
		// optional readouts for the visuals: the tuned lens and the latent colour probe (viz.json / viz.bin)
		const viz = await fetchModelJson<
			Record<
				'lens' | 'probe' | 'probe64' | 'probe128' | 'probe256' | 'space_img' | 'space_txt',
				{ shape: number[]; offset: number; bytes: number }
			>
		>(`${base}/viz.json`).catch(() => null);
		if (viz) {
			denseAt.set('viz.lens', nd);
			nd += viz.lens.shape.reduce((a, b) => a * b, 1);
			denseAt.set('viz.probe', nd);
			nd += 99;
			for (const k of ['space_img', 'space_txt'] as const)
				if (viz[k]) {
					denseAt.set(`viz.${k}`, nd);
					nd += 3 * D;
				}
		}
		const dense = device.createBuffer({ label: 'painter dense', size: nd * 4, usage });
		const probes = new Map<string, Float32Array>();
		if (viz) {
			const bin = await fetchModelFile(`${base}/viz.bin`);
			const h = new Uint16Array(bin, viz.lens.offset, viz.lens.bytes / 2);
			const lens = new Float32Array(h.length);
			for (let i = 0; i < h.length; i++) lens[i] = halfToFloat(h[i]);
			device.queue.writeBuffer(dense, denseAt.get('viz.lens')! * 4, lens);
			device.queue.writeBuffer(
				dense,
				denseAt.get('viz.probe')! * 4,
				new Float32Array(bin, viz.probe.offset, 99)
			);
			for (const k of ['space_img', 'space_txt'] as const)
				if (viz[k])
					device.queue.writeBuffer(
						dense,
						denseAt.get(`viz.${k}`)! * 4,
						new Float32Array(bin.slice(viz[k].offset, viz[k].offset + 3 * D * 4))
					);
			for (const k of ['probe64', 'probe128', 'probe256'] as const)
				if (viz[k])
					probes.set(k, new Float32Array(bin.slice(viz[k].offset, viz[k].offset + 195 * 4)));
		}
		// fetch file by file, uploading as each arrives
		const total = names.reduce(
			(a, n) => a + (files.manifest.files as Record<string, { bytes: number }>)[n].bytes,
			0
		);
		let done = 0;
		for (const n of names) {
			await files.fetch([n], (f) =>
				onProgress?.({
					stage: 'Downloading the painter',
					fraction:
						(done + f * (files.manifest.files as Record<string, { bytes: number }>)[n].bytes) /
						total
				})
			);
			done += (files.manifest.files as Record<string, { bytes: number }>)[n].bytes;
			for (const [k, e] of Object.entries(tensors)) {
				if (e.file !== n) continue;
				if (e.kind === 'ternary') {
					const t = files.ternary(k),
						where = tern.get(k)!;
					device.queue.writeBuffer(codes, where.codes * 4, t.codes);
					device.queue.writeBuffer(scales, where.scales * 4, t.scales);
				} else if (denseAt.has(k)) {
					device.queue.writeBuffer(dense, denseAt.get(k)! * 4, files.dense(k));
					if (k === 'rope.cos' || k === 'rope.sin')
						device.queue.writeBuffer(dense, denseAt.get(`${k}16`)! * 4, rope16(files.dense(k)));
				}
			}
			if (n.startsWith('dit_') && n !== 'dit_misc.bin') files.release(n);
		}
		onProgress?.({ stage: 'Ready', fraction: 1 });
		const painter = new Painter(device, files, codes, scales, dense, tern, denseAt, probes);
		// other step counts (optional): sigmas and modulation per step, see export_painter_schedules.py
		const sj = await fetchModelJson<
			Record<string, Record<string, { sigmas: number[]; offset: number }>>
		>(`${base}/schedules.json`).catch(() => null);
		if (sj) {
			const data = new Float32Array(await fetchModelFile(`${base}/schedules.bin`));
			painter.schedules = { meta: sj['512'], data };
		}
		return painter;
	}

	/**
	 * The painter from the one-file model (see packed.ts): the fused conditioning map, the DiT and TAEF2, read in parts
	 * and put on the GPU as they arrive (the ternary weights unpacked from trits).
	 */
	static async fromPacked(
		device: GPUDevice,
		model: PackedModel,
		onProgress?: (p: PainterProgress) => void
	): Promise<Painter> {
		const ts = model.tensors('painter');
		// a ternary tensor is the one with a `.scale` beside it (its stored type may be a deflate wrapper)
		const isTernary = (t: GGUFTensor) => model.tensor(t.name + '.scale') !== undefined;
		const isScale = (t: GGUFTensor) =>
			t.name.endsWith('.scale') && model.tensor(t.name.slice(0, -6)) !== undefined;
		// kept on the CPU (the constructor reads them): the step modulation, the decoder, the fixed text rows
		const onCpu = (k: string) =>
			k.startsWith('taef2.') ||
			k.startsWith('mod.') ||
			k === 'sched.mod' ||
			k === 'temb' ||
			k.startsWith('cond.b') ||
			k === 'cond.prefix';
		const tern = new Map<string, { rows: number; cols: number; codes: number; scales: number }>();
		const denseAt = new Map<string, number>();
		const loraAt = new Map<string, number>(); // the side branches' B (transposed), in a buffer of their own
		// the GEMM's side branch runs in chunks of 32: a lower rank is padded with zero rows (buffers start zeroed)
		const ranks: Partial<Record<Branch, number>> = {},
			stored: Partial<Record<Branch, number>> = {};
		for (const b of ['lora', 'rdm', 'best'] as const) {
			const meta = model.meta[`mindview.${b}`];
			if (!meta) continue;
			stored[b] = (JSON.parse(String(meta)) as { rank: number }).rank;
			ranks[b] = Math.ceil(stored[b]! / 32) * 32;
		}
		const branchOf = (name: string) => {
			const b = name.slice(0, name.indexOf('.')) as Branch;
			return stored[b] ? b : undefined;
		};
		let nc = 0,
			ns = 0,
			nd = 0,
			nl = 0;
		for (const t of ts) {
			const b = branchOf(t.name);
			if (b && t.name.endsWith('.bt')) {
				loraAt.set(t.name, nl);
				nl += Math.ceil(((PackedModel.count(t) / stored[b]!) * ranks[b]!) / 4) * 4;
			} else if (b && t.name.endsWith('.a')) {
				denseAt.set(t.name, nd);
				nd += Math.ceil(((PackedModel.count(t) / stored[b]!) * ranks[b]!) / 4) * 4;
			} else if (isTernary(t)) {
				const [K, M] = t.dims;
				tern.set(t.name, { rows: M, cols: K, codes: nc, scales: ns });
				nc += (M * K) / 16;
				ns += (M * K) / 128;
			} else if (!isScale(t) && !onCpu(t.name)) {
				denseAt.set(t.name, nd);
				nd += PackedModel.count(t);
			}
		}
		for (const k of ['rope.cos16', 'rope.sin16']) {
			denseAt.set(k, nd);
			nd += 256 * 64;
		}
		const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
		const codes = device.createBuffer({ label: 'painter codes', size: nc * 4, usage });
		const scales = device.createBuffer({ label: 'painter scales', size: ns * 4, usage });
		const dense = device.createBuffer({ label: 'painter dense', size: nd * 4, usage });
		const loraBuf = nl
			? device.createBuffer({ label: 'painter lora', size: nl * 4, usage })
			: undefined;
		const kept = new Map<string, Float32Array>();
		await model.read(
			ts,
			(part) => {
				for (const [name, p] of part) {
					const t = model.tensor(name)!;
					if (p.type === GGML_TRIT5) {
						const where = tern.get(name)!;
						const n = where.rows * where.cols;
						const words = new Uint32Array(n / 16);
						unpackTrits(p.data, n, words, 0);
						device.queue.writeBuffer(codes, where.codes * 4, words);
						const sc = part.get(name + '.scale')!;
						device.queue.writeBuffer(scales, where.scales * 4, PackedModel.floats(sc, name));
					} else if (loraAt.has(name)) {
						device.queue.writeBuffer(loraBuf!, loraAt.get(name)! * 4, PackedModel.floats(p, name));
					} else if (!isScale(t)) {
						const v = PackedModel.floats(p, name);
						if (onCpu(name)) kept.set(name, v);
						else device.queue.writeBuffer(dense, denseAt.get(name)! * 4, v);
						if (name === 'rope.cos' || name === 'rope.sin')
							device.queue.writeBuffer(dense, denseAt.get(`${name}16`)! * 4, rope16(v));
					}
				}
			},
			(f) => onProgress?.({ stage: 'Downloading the painter', fraction: f })
		);
		onProgress?.({ stage: 'Ready', fraction: 1 });
		const source: PainterSource = {
			manifest: { ...model.painterConfig, tensors: {} } as PainterSource['manifest'],
			dense: (name) => {
				const v = kept.get(name);
				if (!v) throw new Error(`The model has no ${name}.`);
				return v;
			},
			names: (prefix) => [...kept.keys()].filter((k) => k.startsWith(prefix))
		};
		const painter = new Painter(
			device,
			source,
			codes,
			scales,
			dense,
			tern,
			denseAt,
			new Map(),
			loraBuf ? { buf: loraBuf, at: loraAt, ranks } : undefined
		);
		// other step counts (sigmas and modulation per step), when the file carries them
		const sched = model.meta['mindview.schedules'];
		if (sched && kept.has('sched.mod'))
			painter.schedules = { meta: JSON.parse(String(sched))['512'], data: kept.get('sched.mod')! };
		return painter;
	}

	private t(name: string) {
		const t = this.tern.get(name);
		if (!t) throw new Error(`The painter has no ternary ${name}.`);
		return t;
	}

	private w(name: string) {
		const o = this.denseAt.get(name);
		if (o === undefined) throw new Error(`The painter has no dense ${name}.`);
		return o;
	}

	/** Free the painter's GPU memory (it cannot be used afterwards). */
	destroy() {
		for (const b of [this.arena, this.codesBuf, this.scalesBuf, this.dense, this.params])
			b.destroy();
		this.lora?.buf.destroy();
	}

	/** The reader's layers whose states are the conditioning (7, 14, 21 unless the model says otherwise). */
	get taps(): number[] {
		return (this.manifest.text as { taps?: number[] }).taps ?? [7, 14, 21];
	}

	/** Text side: the 1.7B reads the prompt padded to the text rows (512, or fewer: textLength); its tapped layers
	 * become the conditioning. */
	async encode(llm: BonsaiLLM, prompt: string) {
		const text = (this.manifest.text as { template: string }).template
			.replaceAll('\\n', '\n')
			.replace('{prompt}', prompt);
		const all = llm.tokenizer.encode(text);
		const nt = (this.nt = this.pickRows(all.length));
		const real = all.slice(0, nt);
		const n = real.length;
		const ids = real.concat(
			new Array(nt - n).fill((this.manifest.text as { pad_id: number }).pad_id)
		);
		const taps = await llm.encodeLong(ids, this.taps, n);
		const rows = new Float32Array(nt * TAP);
		for (let t = 0; t < nt; t++)
			for (let k = 0; k < 3; k++)
				rows.set(taps[k].subarray(t * 2048, (t + 1) * 2048), t * TAP + k * 2048);
		this.device.queue.writeBuffer(this.arena, this.at.taps * 4, rows);
		if (this.fused)
			this.device.queue.writeBuffer(
				this.arena,
				this.at.ctxd * 4,
				this.ctxInit!.subarray(0, nt * D)
			);
		this.run([this.condOp()]);
		await this.device.queue.onSubmittedWorkDone();
		return { ids, real: n };
	}

	/**
	 * Start from the given noise (image rows x 128, bn-normalised latent space) or from seeded Gaussian noise; with
	 * low (true, or how many), the first step or steps run at 256 x 256 (see upsampleLatent).
	 */
	setNoise(noise?: Float32Array, seed = 7, low: boolean | number = false) {
		this.low = typeof low === 'number' ? low : low ? 1 : 0;
		this.ni = this.low ? 256 : NI;
		this.seed = seed;
		let z = noise;
		if (!z) {
			z = new Float32Array(this.ni * CIN);
			let s = seed >>> 0 || 1;
			const rnd = () => ((s = (s * 1664525 + 1013904223) >>> 0) + 0.5) / 4294967296;
			for (let i = 0; i < z.length; i += 2) {
				const r = Math.sqrt(-2 * Math.log(rnd())),
					th = 2 * Math.PI * rnd();
				z[i] = r * Math.cos(th);
				z[i + 1] = r * Math.sin(th);
			}
		}
		this.device.queue.writeBuffer(this.arena, this.at.lat * 4, z);
		this.device.queue.writeBuffer(this.arena, this.at.noise * 4, z);
	}

	/**
	 * After the last coarse step at 256 x 256: the clean picture it predicts (x0 = latent - sigma v), upsampled to
	 * 512 x 512 in latent space (bilinear, per channel) and noised back to the next step's sigma with fresh noise, for
	 * the steps at full size. Returns the upsampled prediction (for decode).
	 */
	async upsampleLatent(seed = 7): Promise<Float32Array> {
		const s1 = this.sigmas[this.low],
			n = 256;
		const lat = await this.read(this.at.lat, n * CIN),
			vel = await this.read(this.at.vel, n * CIN);
		// token (h, w), channel 4c + 2dy + dx <-> latent pixel (2h + dy, 2w + dx) of channel c (32 channels)
		const small = new Float32Array(32 * 32 * 32); // [c][y][x], 32 x 32 latent pixels
		for (let t = 0; t < n; t++)
			for (let k = 0; k < CIN; k++) {
				const c = k >> 2,
					dy = (k >> 1) & 1,
					dx = k & 1;
				small[(c * 32 + 2 * (t >> 4) + dy) * 32 + 2 * (t & 15) + dx] =
					lat[t * CIN + k] - s1 * vel[t * CIN + k];
			}
		const at = (c: number, y: number, x: number) =>
			small[(c * 32 + Math.min(31, Math.max(0, y))) * 32 + Math.min(31, Math.max(0, x))];
		const z = new Float32Array(NI * CIN),
			clean = new Float32Array(NI * CIN);
		let r = seed >>> 0 || 1;
		const rnd = () => ((r = (r * 1664525 + 1013904223) >>> 0) + 0.5) / 4294967296;
		const noise = new Float32Array(NI * CIN);
		for (let i = 0; i < noise.length; i += 2) {
			const m = Math.sqrt(-2 * Math.log(rnd())),
				th = 2 * Math.PI * rnd();
			noise[i] = m * Math.cos(th);
			noise[i + 1] = m * Math.sin(th);
		}
		for (let t = 0; t < NI; t++)
			for (let k = 0; k < CIN; k++) {
				const c = k >> 2,
					y = 2 * (t >> 5) + ((k >> 1) & 1),
					x = 2 * (t & 31) + (k & 1);
				// bilinear, as torch's interpolate(scale 2, align_corners=False): source (y + 0.5) / 2 - 0.5
				const sy = (y + 0.5) / 2 - 0.5,
					sx = (x + 0.5) / 2 - 0.5;
				const y0 = Math.floor(sy),
					x0 = Math.floor(sx),
					fy = sy - y0,
					fx = sx - x0;
				const v =
					(1 - fy) * ((1 - fx) * at(c, y0, x0) + fx * at(c, y0, x0 + 1)) +
					fy * ((1 - fx) * at(c, y0 + 1, x0) + fx * at(c, y0 + 1, x0 + 1));
				clean[t * CIN + k] = v;
				z[t * CIN + k] = (1 - s1) * v + s1 * noise[t * CIN + k];
			}
		this.ni = NI;
		this.device.queue.writeBuffer(this.arena, this.at.lat * 4, z);
		this.device.queue.writeBuffer(this.arena, this.at.noise * 4, noise);
		return clean;
	}

	/** One denoising step (0..3): the DiT's 25 blocks, then the Euler update. Calls onBlock after each block. */
	async step(s: number, onBlock?: BlockHook) {
		const { ops, hooks } = this.stepOps(s);
		if (!onBlock) {
			// nothing to look at in between: one submission, one wait
			this.run(ops);
			await this.device.queue.onSubmittedWorkDone();
			return;
		}
		// submit block by block so the visuals can look in between
		let from = 0;
		for (const [to, b] of hooks) {
			this.run(ops.slice(from, to));
			from = to;
			await this.device.queue.onSubmittedWorkDone();
			await onBlock?.(s, b);
		}
	}

	/** The ops of one step and where its blocks end: [op index after which, block] (-1 embedded, 25 done). */
	private stepOps(s: number, tap?: Tap, pool?: (b: number, ops: Op[]) => void) {
		const at = this.at,
			mod = at.mod + s * 17 * D,
			// a first step at 256 x 256 reads a prefix of the text rows (the reader is causal: the same rows)
			{ ni, nt, nj } = this.dims(s);
		// image rows turn by the rope table's rows for (h, w): the full table's 512.. at 32 x 32, a 16 x 16 copy at 256
		const ropeImg: [string, string, number] =
			ni === NI ? ['rope.cos', 'rope.sin', ROPE_IMG] : ['rope.cos16', 'rope.sin16', 0];
		const MI = (r: number) => mod + r * D,
			MT = (r: number) => mod + (6 + r) * D,
			MS = (r: number) => mod + (12 + r) * D,
			MO = (r: number) => mod + (15 + r) * D;
		const T0 = 0,
			I0 = nt * D; // row offsets of the text and image parts of the joint stream
		const ops: Op[] = [];
		const K = (name: Kernel, p: number[], wg: [number, number, number], f: number[] = []) =>
			ops.push({ k: 'kernel', name, p, f, wg });
		const G = (w: string, M: number, x: number, y: number, extra: Partial<GemmJob> = {}) => {
			const t = this.t(w);
			// the few-step LoRA: T = X A^T first, then the ternary GEMM adds T B^T before its bias and gate
			const mod = w.replace(/\.weight$/, '');
			const br = this.branch;
			const lb = br ? this.lora?.at.get(`${br}.${mod}.bt`) : undefined;
			let lora: GemmJob['lora'];
			if (br && lb !== undefined) {
				const r = this.lora!.ranks[br]!;
				K(
					'dgemm',
					[M, r, t.cols, x, t.cols, 0, this.w(`${br}.${mod}.a`), t.cols, 0, at.lt, r, 0, 1],
					[Math.ceil(r / 64), Math.ceil(M / 64), 1],
					[1]
				);
				lora = { t: at.lt, b: lb, rank: r };
			}
			ops.push({
				k: 'gemm',
				job: { M, N: t.rows, K: t.cols, codes: t.codes, scales: t.scales, x, y, ...extra, lora }
			});
		};
		const modulate = (rows: number, x: number, y: number, shift: number, scale: number) =>
			K('modulate', [rows, D, x, y, shift, scale, D, D], [rows, 1, 1], [1e-6]);
		const qk = (
			tokens: number,
			src: number,
			stride: number,
			dst: number,
			gain: string,
			first: number,
			table: [string, string] = ['rope.cos', 'rope.sin']
		) =>
			K(
				'qknorm_rope',
				[tokens, src, stride, dst, D, this.w(gain), this.w(table[0]), this.w(table[1]), first],
				[tokens, H, 1],
				[1e-6]
			);
		const attention = (v: number, vStride: number, out: number, outStride: number) => {
			const qk: DenseJob = {
				M: nj,
				N: nj,
				K: HD,
				a: at.q,
				lda: D,
				sa: HD,
				b: at.k,
				ldb: D,
				sb: HD,
				c: at.s,
				ldc: nj,
				sc: nj * nj,
				batch: H,
				nn: false,
				scale: 1 / Math.sqrt(HD)
			};
			const pv: DenseJob = {
				M: nj,
				N: HD,
				K: nj,
				a: at.s,
				lda: nj,
				sa: nj * nj,
				b: v,
				ldb: vStride,
				sb: HD,
				c: out,
				ldc: outStride,
				sc: HD,
				batch: H,
				nn: true,
				scale: 1
			};
			if (this.dense16 && this.gemm.precision === 'f16' && denseOk(qk) && denseOk(pv)) {
				ops.push({ k: 'dense', job: qk });
				K('softmax', [H * nj, nj, at.s], [Math.min(H * nj, 65535), Math.ceil((H * nj) / 65535), 1]);
				ops.push({ k: 'dense', job: pv });
				return;
			}
			K(
				'dgemm',
				[nj, nj, HD, at.q, D, HD, at.k, D, HD, at.s, nj, nj * nj, 0],
				[nj / 64, nj / 64, H],
				[1 / Math.sqrt(HD)]
			);
			K('softmax', [H * nj, nj, at.s], [Math.min(H * nj, 65535), Math.ceil((H * nj) / 65535), 1]);
			K(
				'dgemm',
				[nj, HD, nj, at.s, nj, nj * nj, v, vStride, HD, out, outStride, HD, 2],
				[HD / 64, nj / 64, H],
				[1]
			);
		};

		const hooks: [number, number][] = []; // [op index after which, block] (-1 = embedded, 25 = step done)
		const T = (
			name: string,
			region: number,
			width: number,
			rowBase: number,
			rows: 'txt' | 'img' | 'all',
			b: number
		) => tap?.(ops, name, region, width, rowBase, rows, b);
		T('lat', at.lat, CIN, nt, 'img', -1);
		// embed: image tokens from the latent, text tokens from the conditioning
		{
			K(
				'dgemm',
				[ni, D, CIN, at.lat, CIN, 0, this.w('x_embedder.weight'), CIN, 0, at.h + I0, D, 0, 1],
				[D / 64, ni / 64, 1],
				[1]
			);
			if (this.fused)
				K('copy_rows', [nt * D, at.ctxd, at.h + T0], [Math.ceil((nt * D) / 256), 1, 1]);
			else
				K(
					'dgemm',
					[
						nt,
						D,
						CTX,
						at.ctx,
						CTX,
						0,
						this.w('context_embedder.weight'),
						CTX,
						0,
						at.h + T0,
						D,
						0,
						1
					],
					[D / 64, nt / 64, 1],
					[1]
				);
		}
		hooks.push([ops.length, -1]);
		for (let b = 0; b < 5; b++) {
			const P = `transformer_blocks.${b}.`;
			T('h_in', at.h, D, 0, 'all', b);
			modulate(nt, at.h + T0, at.n1 + T0, MT(0), MT(1));
			modulate(ni, at.h + I0, at.n1 + I0, MI(0), MI(1));
			T('n1', at.n1, D, 0, 'all', b);
			G(P + 'attn.add_q_proj.weight', nt, at.n1 + T0, at.q + T0);
			G(P + 'attn.add_k_proj.weight', nt, at.n1 + T0, at.k + T0);
			G(P + 'attn.add_v_proj.weight', nt, at.n1 + T0, at.v + T0);
			G(P + 'attn.to_q.weight', ni, at.n1 + I0, at.q + I0);
			G(P + 'attn.to_k.weight', ni, at.n1 + I0, at.k + I0);
			G(P + 'attn.to_v.weight', ni, at.n1 + I0, at.v + I0);
			T('q', at.q, D, 0, 'all', b);
			T('k', at.k, D, 0, 'all', b);
			T('v', at.v, D, 0, 'all', b);
			qk(nt, at.q + T0, D, at.q + T0, P + 'attn.norm_added_q.weight', 0);
			qk(ni, at.q + I0, D, at.q + I0, P + 'attn.norm_q.weight', ropeImg[2], [
				ropeImg[0],
				ropeImg[1]
			]);
			qk(nt, at.k + T0, D, at.k + T0, P + 'attn.norm_added_k.weight', 0);
			qk(ni, at.k + I0, D, at.k + I0, P + 'attn.norm_k.weight', ropeImg[2], [
				ropeImg[0],
				ropeImg[1]
			]);
			T('qr', at.q, D, 0, 'all', b);
			T('kr', at.k, D, 0, 'all', b);
			attention(at.v, D, at.o, D);
			T('attn', at.s, nj, 0, 'all', b);
			pool?.(b, ops);
			T('o', at.o, D, 0, 'all', b);
			G(P + 'attn.to_out.0.weight', ni, at.o + I0, at.h + I0, { bias: MI(2), gated: true });
			G(P + 'attn.to_add_out.weight', nt, at.o + T0, at.h + T0, { bias: MT(2), gated: true });
			T('h_mid', at.h, D, 0, 'all', b);
			modulate(ni, at.h + I0, at.n1 + I0, MI(3), MI(4));
			modulate(nt, at.h + T0, at.n1 + T0, MT(3), MT(4));
			T('n2', at.n1, D, 0, 'all', b);
			G(P + 'ff.linear_in.weight', ni, at.n1 + I0, at.p);
			T('p', at.p, 2 * MLP, nt, 'img', b);
			K('swiglu', [ni, MLP, at.p, 2 * MLP, at.cat, MLP], [MLP / 256, ni, 1]);
			T('cat', at.cat, MLP, nt, 'img', b);
			G(P + 'ff.linear_out.weight', ni, at.cat, at.h + I0, { bias: MI(5), gated: true });
			G(P + 'ff_context.linear_in.weight', nt, at.n1 + T0, at.p);
			T('p', at.p, 2 * MLP, 0, 'txt', b);
			K('swiglu', [nt, MLP, at.p, 2 * MLP, at.cat, MLP], [MLP / 256, nt, 1]);
			T('cat', at.cat, MLP, 0, 'txt', b);
			G(P + 'ff_context.linear_out.weight', nt, at.cat, at.h + T0, { bias: MT(5), gated: true });
			T('h_out', at.h, D, 0, 'all', b);
			hooks.push([ops.length, b]);
		}
		const PW = 3 * D + 2 * MLP,
			CW = D + MLP;
		for (let b = 0; b < 20; b++) {
			const P = `single_transformer_blocks.${b}.`;
			const bb = 5 + b;
			T('h_in', at.h, D, 0, 'all', bb);
			modulate(nj, at.h, at.n1, MS(0), MS(1));
			T('n1', at.n1, D, 0, 'all', bb);
			G(P + 'attn.to_qkv_mlp_proj.weight', nj, at.n1, at.p);
			T('p', at.p, PW, 0, 'all', bb);
			// text rows turn by rope rows 0.., image rows by rope rows 512.. (whatever the text length)
			qk(nt, at.p, PW, at.q, P + 'attn.norm_q.weight', 0);
			qk(ni, at.p + nt * PW, PW, at.q + nt * D, P + 'attn.norm_q.weight', ropeImg[2], [
				ropeImg[0],
				ropeImg[1]
			]);
			qk(nt, at.p + D, PW, at.k, P + 'attn.norm_k.weight', 0);
			qk(ni, at.p + D + nt * PW, PW, at.k + nt * D, P + 'attn.norm_k.weight', ropeImg[2], [
				ropeImg[0],
				ropeImg[1]
			]);
			T('qr', at.q, D, 0, 'all', bb);
			T('kr', at.k, D, 0, 'all', bb);
			attention(at.p + 2 * D, PW, at.cat, CW);
			T('attn', at.s, nj, 0, 'all', bb);
			pool?.(bb, ops);
			K('swiglu', [nj, MLP, at.p + 3 * D, PW, at.cat + D, CW], [MLP / 256, nj, 1]);
			T('cat', at.cat, CW, 0, 'all', bb);
			G(P + 'attn.to_out.weight', nj, at.cat, at.h, { bias: MS(2), gated: true });
			T('h_out', at.h, D, 0, 'all', bb);
			hooks.push([ops.length, bb]);
		}
		// velocity and the Euler update
		modulate(ni, at.h + I0, at.n1 + I0, MO(1), MO(0));
		T('nout', at.n1, D, 0, 'img', 25);
		K(
			'dgemm',
			[ni, CIN, D, at.n1 + I0, D, 0, this.w('proj_out.weight'), D, 0, at.vel, CIN, 0, 1],
			[CIN / 64, ni / 64, 1],
			[1]
		);
		T('vel', at.vel, CIN, nt, 'img', 25);
		K(
			'euler',
			[ni * CIN, at.lat, at.vel],
			[Math.ceil((ni * CIN) / 256), 1, 1],
			[this.sigmas[s + 1] - this.sigmas[s]]
		);

		hooks.push([ops.length, 25]);
		return { ops, hooks };
	}

	// ---- scheduled work: the same computation cut into slices of a few milliseconds

	private cost(o: Op): number {
		if (o.k === 'copy') return 0.002;
		if (o.k === 'gemm') return gemmMs(o.job.M, o.job.N, o.job.K);
		if (o.k === 'dense') return (2 * o.job.M * o.job.N * o.job.K * o.job.batch) / 1.6e9;
		const p = o.p;
		switch (o.name) {
			case 'dgemm':
				return (2 * p[0] * p[1] * p[2] * o.wg[2]) / 0.8e9;
			case 'softmax':
				return p[0] * p[1] * 1.5e-7;
			case 'modulate':
				return p[0] * p[1] * 2e-7;
			case 'swiglu':
				return p[0] * p[1] * 1.5e-7;
			case 'qknorm_rope':
				return p[0] * 0.0004;
			default:
				return 0.2;
		}
	}

	/** Cut an op into row slices that each cost at most `budget` ms (rows in multiples of 64 where it matters). */
	private slice(o: Op, budget: number): Op[] {
		const c = this.cost(o);
		if (c <= budget || o.k === 'copy') return [o];
		const rows = o.k === 'gemm' || o.k === 'dense' ? o.job.M : o.p[0];
		let R = Math.max(64, Math.floor((rows * budget) / c / 64) * 64);
		if (o.k === 'kernel' && (o.name === 'softmax' || o.name === 'modulate' || o.name === 'swiglu'))
			R = Math.max(1, Math.floor((rows * budget) / c));
		if (o.k === 'kernel' && !['dgemm', 'softmax', 'modulate', 'swiglu'].includes(o.name))
			return [o];
		const out: Op[] = [];
		for (let r0 = 0; r0 < rows; r0 += R) {
			const n = Math.min(R, rows - r0);
			if (o.k === 'gemm') {
				const j = o.job;
				// a side branch's A x ([M][rank]) moves with the rows too
				const lora = j.lora && { ...j.lora, t: j.lora.t + r0 * j.lora.rank };
				out.push({ k: 'gemm', job: { ...j, M: n, x: j.x + r0 * j.K, y: j.y + r0 * j.N, lora } });
				continue;
			}
			if (o.k === 'dense') {
				const j = o.job;
				out.push({ k: 'dense', job: { ...j, M: n, a: j.a + r0 * j.lda, c: j.c + r0 * j.ldc } });
				continue;
			}
			const p = [...o.p];
			let wg: [number, number, number];
			if (o.name === 'dgemm') {
				p[0] = n;
				p[3] += r0 * p[4];
				p[9] += r0 * p[10];
				wg = [o.wg[0], Math.ceil(n / 64), o.wg[2]];
			} else if (o.name === 'softmax') {
				p[0] = n;
				p[2] += r0 * p[1];
				wg = [Math.min(n, 65535), Math.ceil(n / 65535), 1];
			} else if (o.name === 'modulate') {
				p[0] = n;
				p[2] += r0 * p[6];
				p[3] += r0 * p[7];
				wg = [n, 1, 1];
			} else {
				p[0] = n;
				p[2] += r0 * p[3];
				p[4] += r0 * p[5];
				wg = [o.wg[0], n, 1];
			}
			out.push({ ...o, p, wg });
		}
		return out;
	}

	/** Group ops into tasks of about `budget` ms; `done` runs after the last. */
	private tasks(ops: Op[], budget: number, done?: () => void | Promise<void>): GpuTask[] {
		const tasks: GpuTask[] = [];
		let group: Op[] = [],
			acc = 0;
		const flush = () => {
			if (!group.length) return;
			const g = group;
			tasks.push({ cost: acc, record: (enc, sub) => this.record(g, enc, sub) });
			group = [];
			acc = 0;
		};
		for (const big of ops)
			for (const o of this.slice(big, budget)) {
				const c = this.cost(o);
				if (acc + c > budget) flush();
				group.push(o);
				acc += c;
			}
		flush();
		if (done) tasks.push({ cost: 0, record: () => {}, done });
		return tasks;
	}

	/** Text side as tasks: the 1.7B re-reads the prompt padded to 512 tokens, then the adapter. */
	encodeTasks(llm: BonsaiLLM, prompt: string, budget = 9): { tasks: GpuTask[]; real: number } {
		const tpl = (this.manifest.text as { template: string }).template;
		const text = tpl.replaceAll('\\n', '\n').replace('{prompt}', prompt);
		const all = llm.tokenizer.encode(text);
		const nt = (this.nt = this.pickRows(all.length));
		const real = all.slice(0, nt);
		const n = real.length;
		this.textRows = n;
		const ids = real.concat(
			new Array(nt - n).fill((this.manifest.text as { pad_id: number }).pad_id)
		);
		const tasks = llm.encodeLongTasks(ids, this.taps, n, budget);
		// taps [3][512][2048] in the reader -> [512][h7 | h14 | h21] here, then the adapter (rows 3.. only)
		tasks.push({
			cost: 1,
			record: (enc) => {
				for (let t = 0; t < nt; t++)
					for (let k = 0; k < 3; k++)
						enc.copyBufferToBuffer(
							llm.longTaps,
							(k * NT + t) * 2048 * 4, // the reader keeps [taps][512][dim]
							this.arena,
							(this.at.taps + t * TAP + k * 2048) * 4,
							2048 * 4
						);
			}
		});
		if (this.fused)
			tasks.push({
				cost: 0.5,
				record: () =>
					this.device.queue.writeBuffer(
						this.arena,
						this.at.ctxd * 4,
						this.ctxInit!.subarray(0, nt * D)
					)
			});
		tasks.push(...this.tasks([this.condOp()], budget));
		return { tasks, real: n };
	}

	/**
	 * The conditioning from the taps, rows 3.. (rows 0..2 are the fixed prefix, already in place): the ternary adapter
	 * into ctx (7,680 wide, embedded each step), or the fused map straight into ctxd (3,072 wide, added to the bias).
	 */
	private condOp(): Op {
		if (this.fused && this.tern.has('cond.weight')) {
			const c = this.t('cond.weight');
			return {
				k: 'gemm',
				job: {
					M: this.nt - 3,
					N: D,
					K: TAP,
					codes: c.codes,
					scales: c.scales,
					x: this.at.taps + 3 * TAP,
					y: this.at.ctxd + 3 * D,
					bias: this.at.abias
				}
			};
		}
		if (this.fused)
			return {
				k: 'kernel',
				name: 'dgemm',
				p: [
					this.nt - 3,
					D,
					TAP,
					this.at.taps + 3 * TAP,
					TAP,
					0,
					this.w('cond.weight'),
					TAP,
					0,
					this.at.ctxd + 3 * D,
					D,
					0,
					1 | 4
				],
				f: [1],
				wg: [D / 64, Math.ceil((this.nt - 3) / 64), 1]
			};
		const a = this.t('adapter.weight');
		return {
			k: 'gemm',
			job: {
				M: this.nt - 3,
				N: CTX,
				K: TAP,
				codes: a.codes,
				scales: a.scales,
				x: this.at.taps + 3 * TAP,
				y: this.at.ctx + 3 * CTX,
				bias: this.at.abias
			}
		};
	}

	/**
	 * One step as tasks. After every block a readout: the painting it has in mind (TAEF2's quick picture, straight
	 * into taef2.earlyTexture) and how much each patch attends to each prompt word, handed to onBlock. After the last
	 * block, its guess of the finished picture is decoded in full, and `stage` hears of each of the decoder's stages
	 * (128, 256: taef2.stageTextures; 512: taef2.texture), see Taef2.decodeTasks. onPicture is called while the
	 * commands are recorded, right after each quick picture is made (to copy it before the next block replaces it).
	 */
	stepTasks(
		s: number,
		words: { first: number; count: number },
		onBlock: (b: number, attn: Float32Array, space?: BlockSpace, words?: Float32Array) => void,
		stage: { record?: (enc: GPUCommandEncoder, res: number) => void; done?: (res: number) => void },
		budget = 9,
		onPicture?: (enc: GPUCommandEncoder, b: number) => void,
		onInput?: (space: BlockSpace) => void
	): GpuTask[] {
		const { ops, hooks } = this.stepOps(s);
		const tasks: GpuTask[] = [];
		const lens = this.hasLens && this.taef2.hasEarly;
		let from = 0;
		for (const [to, b] of hooks) {
			tasks.push(...this.tasks(ops.slice(from, to), budget));
			from = to;
			if (b < 0 && onInput && this.hasSpace) tasks.push(this.inputTask(onInput));
			// without a lens there is no picture per block: when the step is done, its own clean guess is decoded (the
			// last coarse step at 256 x 256 is enlarged and noised back for the next, see sketchTasks; earlier coarse
			// steps show their enlarged guess, and keep their latent)
			if (b === 25 && !lens)
				tasks.push(
					...(s < this.low
						? this.sketchTasks(stage, budget, s === this.low - 1)
						: this.guessTasks(s, stage, budget))
				);
			if (b < 0 || b > 24) continue;
			tasks.push(
				this.readoutTask(
					s,
					b,
					words,
					(attn, space, w) => onBlock(b, attn, space, w),
					onPicture && ((enc) => onPicture(enc, b))
				)
			);
			if (b === 24 && lens) tasks.push(...this.taef2.decodeTasks(budget, stage));
		}
		return tasks;
	}

	/** After step s (full size): its clean guess, x0 = lat - sigma' v, through TAEF2 (stages as in stepTasks). */
	private guessTasks(
		s: number,
		stage: { record?: (enc: GPUCommandEncoder, res: number) => void; done?: (res: number) => void },
		budget = 9
	): GpuTask[] {
		const at = this.at;
		const op: Op = {
			k: 'kernel',
			name: 'x0_unpatch',
			p: [NI * CIN, at.vel, 0, at.lat, at.x0, 32],
			f: [this.sigmas[s + 1] ?? 0],
			wg: [(NI * CIN) / 256, 1, 1]
		};
		return [
			{
				cost: 1,
				record: (enc, sub) => {
					this.record([op], enc, sub);
					this.taef2.copyLatent(enc, this.arena, at.x0);
				}
			},
			...this.taef2.decodeTasks(budget, stage)
		];
	}

	/**
	 * After a coarse step at 256 x 256 (setNoise with low): its clean guess enlarged to 512 x 512 in the latent space,
	 * the sketch, goes through TAEF2 (stages as in stepTasks). After the last coarse step (last), the enlarged guess is
	 * also mixed with fresh noise back to the next step's sigma, as the next step's latent (all on the GPU, as
	 * upsampleLatent does on the CPU); earlier coarse steps keep their latent.
	 */
	sketchTasks(
		stage: { record?: (enc: GPUCommandEncoder, res: number) => void; done?: (res: number) => void },
		budget = 9,
		last = true
	): GpuTask[] {
		const at = this.at,
			s = this.sigmas[this.low];
		const ops: Op[] = [
			{
				k: 'kernel',
				name: 'x0_unpatch',
				p: [256 * CIN, at.vel, 0, at.lat, at.vl, 16],
				f: [s],
				wg: [(256 * CIN) / 256, 1, 1]
			},
			{
				k: 'kernel',
				name: 'upsample_sketch',
				// the enlarged guess into x0; with last, the noised latent into lat (else into vl, thrown away)
				p: [NI * CIN, at.vl, at.noise, last ? at.lat : at.vl, at.x0],
				f: [s],
				wg: [(NI * CIN) / 256, 1, 1]
			}
		];
		return [
			{
				cost: 1,
				record: (enc, sub) => {
					if (last)
						this.device.queue.writeBuffer(
							this.arena,
							at.noise * 4,
							gaussian(NI * CIN, this.seed + 1)
						);
					this.record(ops, enc, sub);
					this.taef2.copyLatent(enc, this.arena, at.x0);
				}
			},
			...this.taef2.decodeTasks(budget, stage)
		];
	}

	/** Read the painter's states back on the drawing space after each block (see BlockSpace); off unless wanted. */
	drawSpace = false;

	/** Whether the drawing space for the painter's states is loaded (viz.bin space_img / space_txt) and wanted. */
	get hasSpace() {
		return this.drawSpace && this.denseAt.has('viz.space_img') && this.denseAt.has('viz.space_txt');
	}

	/** The image patches' states as the pass begins (the latent, embedded), on the drawing space. */
	private inputTask(onSpace: (space: BlockSpace) => void): GpuTask {
		const at = this.at,
			img = at.h + this.nt * D;
		const ops: Op[] = [
			{ k: 'kernel', name: 'row_mean', p: [NI, D, img, at.mu], wg: [D / 256, 1, 1] },
			{
				k: 'kernel',
				name: 'row_project',
				p: [0, D, img, at.proj, at.mu, this.w('viz.space_img')],
				wg: [NI, 1, 1]
			}
		];
		let out: GPUBuffer;
		return {
			cost: 0.5,
			record: (enc, sub) => {
				this.record(ops, enc, sub);
				out = this.device.createBuffer({
					size: NI * 16,
					usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
				});
				enc.copyBufferToBuffer(this.arena, at.proj * 4, out, 0, NI * 16);
			},
			done: async () => {
				await out.mapAsync(GPUMapMode.READ);
				const img = new Float32Array(out.getMappedRange().slice(0));
				out.destroy();
				onSpace({ img, txt: new Float32Array(0), colours: null });
			}
		};
	}

	private readoutTask(
		s: number,
		b: number,
		words: { first: number; count: number },
		onData: (attn: Float32Array, space?: BlockSpace, words?: Float32Array) => void,
		onPicture?: (enc: GPUCommandEncoder) => void
	): GpuTask {
		const at = this.at,
			count = words.count,
			{ ni, nt: ntj, nj } = this.dims(s);
		const ops: Op[] = [];
		// the states after this block, projected on the drawing space: every image patch, and the prompt's real rows
		// (from row 3; each group less its own mean)
		const space = this.hasSpace;
		const nt = Math.max(1, Math.min(this.textRows, ntj) - 3);
		if (space) {
			const img = at.h + ntj * D,
				txt = at.h + 3 * D;
			ops.push(
				{ k: 'kernel', name: 'row_mean', p: [ni, D, img, at.mu], wg: [D / 256, 1, 1] },
				{
					k: 'kernel',
					name: 'row_project',
					p: [0, D, img, at.proj, at.mu, this.w('viz.space_img')],
					wg: [ni, 1, 1]
				},
				{ k: 'kernel', name: 'row_mean', p: [nt, D, txt, at.mu + D], wg: [D / 256, 1, 1] },
				{
					k: 'kernel',
					name: 'row_project',
					p: [0, D, txt, at.proj + ni * 4, at.mu + D, this.w('viz.space_txt')],
					wg: [nt, 1, 1]
				}
			);
		}
		const lens = this.hasLens && this.taef2.hasEarly;
		if (lens) {
			const L = this.w('viz.lens') + b * 3073 * CIN;
			ops.push({
				k: 'kernel',
				name: 'dgemm',
				p: [ni, CIN, D, at.h + ntj * D, D, 0, L, CIN, 0, at.vl, CIN, 0, 3],
				f: [1],
				wg: [CIN / 64, ni / 64, 1]
			});
			ops.push({
				k: 'kernel',
				name: 'lens_x0',
				p: [ni * CIN, at.vl, L + 3072 * CIN, at.lat, at.x0],
				f: [this.sigmas[s]],
				wg: [(ni * CIN) / 256, 1, 1]
			});
		}
		ops.push({
			k: 'kernel',
			name: 'attn_words',
			p: [ni, count, at.s, words.first, at.aw, ntj, nj, H],
			wg: [ni / 64, count, 1]
		});
		// and how the words read each other in this block: [word][word], averaged over the heads
		const nW = Math.min(count, 128);
		ops.push({
			k: 'kernel',
			name: 'attn_words',
			p: [nW, nW, at.s, words.first, at.awt, words.first, nj, H],
			wg: [Math.ceil(nW / 64), nW, 1]
		});
		let out: GPUBuffer;
		// read back: attention [NI][count], then (with the space) projections [NI + nt][4], then the quick picture
		// as rgba floats [64 * 64][4] (the patches' colours)
		const nA = ni * count,
			nP = space ? (ni + nt) * 4 : 0,
			nC = space && lens ? 64 * 64 * 4 : 0,
			nT = nW * nW;
		return {
			cost: 4,
			record: (enc, sub) => {
				this.record(ops, enc, sub);
				out = this.device.createBuffer({
					size: (nA + nP + nC + nT) * 4,
					usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
				});
				enc.copyBufferToBuffer(this.arena, at.aw * 4, out, 0, nA * 4);
				if (nP) enc.copyBufferToBuffer(this.arena, at.proj * 4, out, nA * 4, nP * 4);
				enc.copyBufferToBuffer(this.arena, at.awt * 4, out, (nA + nP + nC) * 4, nT * 4);
				if (lens) {
					this.taef2.copyLatent(enc, this.arena, at.x0);
					this.taef2.decodeEarly(enc);
					if (nC) this.taef2.copyEarly(enc, out, (nA + nP) * 4);
					onPicture?.(enc);
				}
			},
			done: async () => {
				await out.mapAsync(GPUMapMode.READ);
				const all = new Float32Array(out.getMappedRange().slice(0));
				out.destroy();
				onData(
					all.subarray(0, nA),
					nP
						? {
								img: all.subarray(nA, nA + ni * 4),
								txt: all.subarray(nA + ni * 4, nA + nP),
								colours: nC ? all.subarray(nA + nP, nA + nP + nC) : null
							}
						: undefined,
					all.subarray(nA + nP + nC)
				);
			}
		};
	}

	/**
	 * A whole painting that keeps what `cap` asks for (see PainterCapture), as tasks (after encodeTasks). After every
	 * block its quick picture is made in taef2.earlyTexture (onPicture is called while that is recorded, to copy it)
	 * and onBlock(s, b) runs once it is done; onStep(s) after each step; the finished picture lands in taef2.texture,
	 * then onDone.
	 */
	captureTasks(
		cap: PainterCapture,
		opts: {
			budget?: number;
			onPicture?: (enc: GPUCommandEncoder, s: number, b: number) => void;
			onBlock?: (s: number, b: number) => void;
			onStep?: (s: number) => void;
			onDone?: () => void;
			/** The words' inputs to the adapter and the embedder are kept. */
			onWords?: () => void;
		} = {}
	): GpuTask[] {
		const at = this.at,
			budget = opts.budget ?? 9;
		const rows = [...cap.words, ...cap.patches];
		const tasks: GpuTask[] = [];
		// the words as the adapter takes them, and as it gives them
		const first: Op[] = [];
		for (const r of cap.words) {
			const t = cap.at(-1, -1, r, 'taps'),
				c = cap.at(-1, -1, r, 'ctx');
			if (t !== undefined)
				first.push({ k: 'copy', from: at.taps + r * TAP, to: cap.buffer, at: t, count: TAP });
			if (c !== undefined)
				first.push({ k: 'copy', from: at.ctx + r * CTX, to: cap.buffer, at: c, count: CTX });
		}
		tasks.push(...this.tasks(first, budget));
		if (opts.onWords) tasks.push({ cost: 0, record: () => {}, done: opts.onWords });
		for (let s = 0; s < this.steps; s++) {
			const tap: Tap = (ops, name, region, width, rowBase, which, b) => {
				for (const r of rows) {
					if (which === 'txt' && r >= this.nt) continue;
					if (which === 'img' && r < this.nt) continue;
					const dst = cap.at(s, b, r, name);
					if (dst === undefined) continue;
					if (name === 'attn') {
						for (let h = 0; h < H; h++)
							ops.push({
								k: 'copy',
								from: at.s + (h * this.nj + r) * this.nj,
								to: cap.buffer,
								at: dst + h * this.nj,
								count: this.nj
							});
					} else
						ops.push({
							k: 'copy',
							from: region + (r - rowBase) * width,
							to: cap.buffer,
							at: dst,
							count: width
						});
				}
			};
			const pool = (b: number, ops: Op[]) => {
				const dst = cap.at(s, b, -1, 'pool');
				if (dst === undefined) return;
				const n = this.nj / POOL;
				ops.push({
					k: 'kernel',
					name: 'attn_pool',
					p: [this.nj, POOL, at.s, at.pool, H],
					wg: [n / 8, n / 8, 1]
				});
				ops.push({ k: 'copy', from: at.pool, to: cap.buffer, at: dst, count: n * n });
			};
			const { ops, hooks } = this.stepOps(s, tap, pool);
			let from = 0;
			for (const [to, b] of hooks) {
				tasks.push(...this.tasks(ops.slice(from, to), budget));
				from = to;
				if (b >= 0 && b < 25) tasks.push(this.pictureTask(s, b, opts.onPicture, opts.onBlock));
				if (b === 25 && opts.onStep)
					tasks.push({ cost: 0, record: () => {}, done: () => opts.onStep!(s) });
			}
		}
		tasks.push(...this.finalTasks(() => opts.onDone?.(), budget));
		return tasks;
	}

	/** The painting block b of step s has in mind, in taef2.earlyTexture (the tuned lens, then the decoder's first stage). */
	private pictureTask(
		s: number,
		b: number,
		onPicture?: (enc: GPUCommandEncoder, s: number, b: number) => void,
		onBlock?: (s: number, b: number) => void
	): GpuTask {
		const at = this.at;
		const lens = this.hasLens && this.taef2.hasEarly;
		const ops: Op[] = [];
		if (lens) {
			const L = this.w('viz.lens') + b * 3073 * CIN;
			ops.push({
				k: 'kernel',
				name: 'dgemm',
				p: [NI, CIN, D, at.h + this.nt * D, D, 0, L, CIN, 0, at.vl, CIN, 0, 3],
				f: [1],
				wg: [CIN / 64, NI / 64, 1]
			});
			ops.push({
				k: 'kernel',
				name: 'lens_x0',
				p: [NI * CIN, at.vl, L + 3072 * CIN, at.lat, at.x0],
				f: [this.sigmas[s]],
				wg: [(NI * CIN) / 256, 1, 1]
			});
		}
		return {
			cost: 2,
			record: (enc, sub) => {
				if (!lens) return;
				this.record(ops, enc, sub);
				this.taef2.copyLatent(enc, this.arena, at.x0);
				this.taef2.decodeEarly(enc);
				onPicture?.(enc, s, b);
			},
			done: onBlock ? () => onBlock(s, b) : undefined
		};
	}

	/** The painter's weights (for drawing them): the ternary codes and scales, the dense weights, and where each is. */
	get weights() {
		return {
			codes: this.codesBuf,
			scales: this.scalesBuf,
			dense: this.dense,
			ternary: (name: string) => this.t(name),
			dense_at: (name: string) => this.w(name),
			names: [...this.tern.keys()]
		};
	}

	/** One of the painter's ternary weights exactly: its code (-1, 0, +1) and its group's scale. */
	async weight(name: string, row: number, col: number): Promise<{ code: number; scale: number }> {
		const t = this.t(name);
		const dev = this.device;
		const buf = dev.createBuffer({
			size: 8,
			usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
		});
		const enc = dev.createCommandEncoder();
		enc.copyBufferToBuffer(
			this.codesBuf,
			(t.codes + row * (t.cols / 16) + Math.floor(col / 16)) * 4,
			buf,
			0,
			4
		);
		enc.copyBufferToBuffer(
			this.scalesBuf,
			(t.scales + row * (t.cols / 128) + Math.floor(col / 128)) * 4,
			buf,
			4,
			4
		);
		dev.queue.submit([enc.finish()]);
		await buf.mapAsync(GPUMapMode.READ);
		const u = new Uint32Array(buf.getMappedRange().slice(0));
		buf.destroy();
		return { code: ((u[0] >>> ((col % 16) * 2)) & 3) - 1, scale: new Float32Array(u.buffer)[1] };
	}

	/** Floats read back from the painter's activations (its arena), its dense weights or its ternary scales. */
	async readFloats(
		from: 'arena' | 'dense' | 'scales',
		offset: number,
		count: number
	): Promise<Float32Array> {
		const dev = this.device;
		const buf = dev.createBuffer({
			size: count * 4,
			usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
		});
		const enc = dev.createCommandEncoder();
		const src = from === 'arena' ? this.arena : from === 'dense' ? this.dense : this.scalesBuf;
		enc.copyBufferToBuffer(src, offset * 4, buf, 0, count * 4);
		dev.queue.submit([enc.finish()]);
		await buf.mapAsync(GPUMapMode.READ);
		const v = new Float32Array(buf.getMappedRange().slice(0));
		buf.destroy();
		return v;
	}

	/** A dense weight exactly. */
	async denseWeight(name: string, index: number): Promise<number> {
		const dev = this.device;
		const buf = dev.createBuffer({
			size: 4,
			usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
		});
		const enc = dev.createCommandEncoder();
		enc.copyBufferToBuffer(this.dense, (this.w(name) + index) * 4, buf, 0, 4);
		dev.queue.submit([enc.finish()]);
		await buf.mapAsync(GPUMapMode.READ);
		const v = new Float32Array(buf.getMappedRange().slice(0))[0];
		buf.destroy();
		return v;
	}

	/** Shapes and positions (for the labs). */
	static readonly dims = { D, H, HD, MLP, NT, NI, NJ, CIN, CTX, TAP, POOL };

	/** The finished picture as tasks: the final latent through TAEF2 into taef2.texture, then onDone. */
	finalTasks(onDone: () => void, budget = 9): GpuTask[] {
		const at = this.at;
		return [
			{
				cost: 1,
				record: (enc, sub) => {
					// unpatchify the latent (the lens kernel with sigma 0 is exactly that)
					this.record(
						[
							{
								k: 'kernel',
								name: 'lens_x0',
								p: [NI * CIN, at.vl, 0, at.lat, at.x0],
								f: [0],
								wg: [(NI * CIN) / 256, 1, 1]
							}
						],
						enc,
						sub
					);
					this.taef2.copyLatent(enc, this.arena, at.x0);
				}
			},
			...this.taef2.decodeTasks(budget),
			{ cost: 0, record: () => {}, done: onDone }
		];
	}

	private run(ops: Op[]) {
		const enc = this.device.createCommandEncoder();
		this.record(ops, enc);
		this.device.queue.submit([enc.finish()]);
	}

	private slotSub = -1;
	private slotAt = 0;
	private gemmAt = 0;
	private denseAt16 = 0;

	/**
	 * GPU time of every op of one step (each op in a pass of its own, with timestamps), labelled by what it is. For
	 * finding where a step's time goes; the step itself is run as usual.
	 */
	async profile(s: number): Promise<{ label: string; ms: number }[]> {
		const dev = this.device;
		if (!dev.features.has('timestamp-query'))
			throw new Error('This device has no timestamp queries.');
		const ops = this.stepOps(s).ops.filter((o) => o.k !== 'copy');
		const qs = dev.createQuerySet({ type: 'timestamp', count: ops.length * 2 });
		const res = dev.createBuffer({
			size: ops.length * 16,
			usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC
		});
		const rb = dev.createBuffer({
			size: ops.length * 16,
			usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
		});
		const enc = dev.createCommandEncoder();
		this.record(ops, enc, undefined, qs);
		enc.resolveQuerySet(qs, 0, ops.length * 2, res, 0);
		enc.copyBufferToBuffer(res, 0, rb, 0, ops.length * 16);
		dev.queue.submit([enc.finish()]);
		await rb.mapAsync(GPUMapMode.READ);
		const t = new BigInt64Array(rb.getMappedRange().slice(0));
		rb.destroy();
		res.destroy();
		qs.destroy();
		return ops.map((o, i) => ({
			label:
				o.k === 'gemm'
					? `gemm ${o.job.M}x${o.job.N}x${o.job.K}`
					: o.k === 'dense'
						? `dense ${o.job.M}x${o.job.N}x${o.job.K} x${o.job.batch}`
						: o.k === 'kernel' && o.name === 'dgemm'
							? `dgemm ${o.p[0]}x${o.p[1]}x${o.p[2]}${o.wg[2] > 1 ? ` x${o.wg[2]}` : ''}`
							: o.k === 'kernel'
								? o.name
								: 'copy',
			ms: Number(t[2 * i + 1] - t[2 * i]) / 1e6
		}));
	}

	/**
	 * Record ops; within one scheduler submission (`sub`) each call gets uniform slots of its own. With `timed`, every
	 * op gets a pass of its own that writes its start and end to the query set (2 per op, in order).
	 */
	private record(ops: Op[], enc: GPUCommandEncoder, sub?: number, timed?: GPUQuerySet) {
		const dev = this.device;
		if (sub === undefined || sub !== this.slotSub) {
			this.slotSub = sub ?? -1;
			this.slotAt = 0;
			this.gemmAt = 0;
			this.denseAt16 = 0;
		}
		const k0 = this.slotAt,
			g0 = this.gemmAt;
		const kernels = ops.filter((o): o is Extract<Op, { k: 'kernel' }> => o.k === 'kernel');
		const data = new ArrayBuffer(Math.max(1, kernels.length) * 256);
		kernels.forEach((o, i) => {
			new Uint32Array(data, i * 256, 16).set(o.p.map((x) => x >>> 0));
			new Float32Array(data, i * 256 + 64, 4).set(o.f ?? []);
		});
		const jobs = ops.flatMap((o) => (o.k === 'gemm' ? [o.job] : []));
		const djobs = ops.flatMap((o) => (o.k === 'dense' ? [o.job] : []));
		if (k0 + kernels.length > 1024 || g0 + jobs.length > 1024)
			throw new Error('Too much painter work in one submission.');
		if (kernels.length) dev.queue.writeBuffer(this.params, k0 * 256, data);
		if (jobs.length) this.gemm.prepare(jobs, g0);
		const d0 = this.denseAt16;
		if (djobs.length) this.dense16!.prepare(djobs, d0);
		this.slotAt += kernels.length;
		this.gemmAt += jobs.length;
		this.denseAt16 += djobs.length;
		let pass: GPUComputePassEncoder | null = null;
		let ki = k0,
			gi = g0,
			di = d0,
			qi = 0;
		for (const o of ops) {
			if (o.k === 'copy') {
				pass?.end();
				pass = null;
				enc.copyBufferToBuffer(this.arena, o.from * 4, o.to, o.at * 4, o.count * 4);
				continue;
			}
			if (timed) {
				pass?.end();
				pass = enc.beginComputePass({
					timestampWrites: {
						querySet: timed,
						beginningOfPassWriteIndex: qi++,
						endOfPassWriteIndex: qi++
					}
				});
			}
			pass ??= enc.beginComputePass();
			if (o.k === 'kernel') {
				pass.setPipeline(this.pipes[o.name]);
				pass.setBindGroup(0, this.bind, [ki++ * 256]);
				pass.dispatchWorkgroups(...o.wg);
			} else if (o.k === 'dense') this.dense16!.dispatch(pass, di++, o.job);
			else this.gemm.dispatch(pass, gi++, o.job);
		}
		pass?.end();
	}

	/** Step counts this painter can run (4 always; others when schedules.json is present). */
	get stepChoices(): number[] {
		return this.schedules
			? Object.keys(this.schedules.meta)
					.map(Number)
					.filter((n) => n > 0)
					.sort((a, b) => a - b)
			: [4];
	}

	/**
	 * Use a schedule (sigmas and the per-step modulation): n steps, with the few-step LoRA at 1 or 2 when the file has
	 * it, or '1r': 1 step with EPFL's 1-step fine-tune.
	 */
	setSteps(n: Schedule) {
		const sc = this.schedules?.meta[String(n)];
		if (n === '1r' && !this.hasOneStep) throw new Error('The painter has no 1-step fine-tune.');
		this.branch =
			n === '1r'
				? 'rdm'
				: n === 4 && this.lora?.ranks.best
					? 'best'
					: this.hasLora && n <= 2
						? 'lora'
						: undefined;
		if (!sc || !this.schedules) {
			if (n !== 4) throw new Error(`The painter has no ${n}-step schedule.`);
			return;
		}
		const steps = sc.sigmas.length - 1;
		this.steps = steps;
		this.sigmas = sc.sigmas;
		this.device.queue.writeBuffer(
			this.arena,
			this.at.mod * 4,
			this.schedules.data.subarray(sc.offset, sc.offset + steps * 17 * D)
		);
	}

	get hasLens() {
		return this.denseAt.has('viz.lens');
	}

	/**
	 * What the visuals show after block b (0..24) of step s, read back from the GPU:
	 *   rgb   [64 x 64 x 4]  the painting this block has in mind (tuned lens, x0 = x_t - sigma v, colour probe)
	 *   attn  [1024 x count] how much each image patch attends to each prompt word (mean over heads)
	 *   text  [count x 3072] the prompt words' states in the painter
	 * words: joint index of the first prompt word and how many there are. Block -1 gives only text.
	 */
	async readouts(s: number, b: number, first: number, count: number, full = false) {
		const at = this.at,
			dev = this.device;
		const ops: Op[] = [];
		const block = b >= 0 && b < 25;
		const lens = block && this.hasLens && this.taef2.hasEarly;
		if (lens) {
			const L = this.w('viz.lens') + b * 3073 * CIN;
			ops.push({
				k: 'kernel',
				name: 'dgemm',
				p: [NI, CIN, D, at.h + this.nt * D, D, 0, L, CIN, 0, at.vl, CIN, 0, 3],
				f: [1],
				wg: [CIN / 64, NI / 64, 1]
			});
			ops.push({
				k: 'kernel',
				name: 'lens_x0',
				p: [NI * CIN, at.vl, L + 3072 * CIN, at.lat, at.x0],
				f: [this.sigmas[s]],
				wg: [(NI * CIN) / 256, 1, 1]
			});
		}
		if (block)
			ops.push({
				k: 'kernel',
				name: 'attn_words',
				p: [NI, count, at.s, first, at.aw, this.nt, this.nj, H],
				wg: [NI / 64, count, 1]
			});
		const enc = dev.createCommandEncoder();
		if (ops.length) this.record(ops, enc);
		const textN = count * D,
			attnN = block ? NI * count : 0,
			rgbN = lens ? 4096 * 4 : 0;
		const out = dev.createBuffer({
			size: (textN + attnN + rgbN) * 4,
			usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
		});
		enc.copyBufferToBuffer(this.arena, (at.h + first * D) * 4, out, 0, textN * 4);
		if (attnN) enc.copyBufferToBuffer(this.arena, at.aw * 4, out, textN * 4, attnN * 4);
		if (lens) {
			this.taef2.copyLatent(enc, this.arena, at.x0);
			this.taef2.decodeEarly(enc);
			this.taef2.copyEarly(enc, out, (textN + attnN) * 4);
			if (full) this.taef2.decode(enc);
		}
		dev.queue.submit([enc.finish()]);
		await out.mapAsync(GPUMapMode.READ);
		const all = new Float32Array(out.getMappedRange().slice(0));
		out.destroy();
		return {
			text: all.subarray(0, textN),
			attn: attnN ? all.subarray(textN, textN + attnN) : null,
			rgb: rgbN ? all.subarray(textN + attnN) : null
		};
	}

	/**
	 * Decode with TAEF2 into this.taef2.texture: the current latent, or with sigma > 0 the picture the painter has in
	 * mind after a step (x0 = latent - sigma x velocity, sigma the step's new noise level).
	 */
	async decode(sigma = 0, clean?: Float32Array) {
		// unpatchify: token (h, w), channel 4c + 2dy + dx -> latent pixel (2h + dy, 2w + dx) of channel c
		const lat = clean ?? (await this.read(this.at.lat, NI * CIN));
		if (!clean && sigma > 0) {
			const vel = await this.read(this.at.vel, NI * CIN);
			for (let i = 0; i < lat.length; i++) lat[i] -= sigma * vel[i];
		}
		const z = new Float32Array(32 * 64 * 64);
		for (let h = 0; h < 32; h++)
			for (let w = 0; w < 32; w++)
				for (let c = 0; c < 32; c++)
					for (let dy = 0; dy < 2; dy++)
						for (let dx = 0; dx < 2; dx++)
							z[(c * 64 + 2 * h + dy) * 64 + 2 * w + dx] =
								lat[(h * 32 + w) * CIN + 4 * c + 2 * dy + dx];
		this.taef2.writeLatent(z);
		const enc = this.device.createCommandEncoder();
		this.taef2.decode(enc);
		this.device.queue.submit([enc.finish()]);
		await this.device.queue.onSubmittedWorkDone();
	}

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
