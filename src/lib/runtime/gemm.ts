// Tiled ternary GEMM for many tokens at once (the painter runs 1536 tokens through 3.7B ternary weights per step).
//   Y[m][n] = sum_k X[m][k] * W[n][k]  (+ bias[n])
// W uses the same packing as the language model: 16 two-bit codes per u32 (trit = code - 1), one f32 scale per 128
// inputs, row-major by output. X and Y live in one f32 arena. A workgroup computes a BM x BN tile of Y; each K-chunk of
// X is staged in workgroup memory and the matching W chunk is decoded (trit * scale) once and shared by BM tokens.

export const GEMM_TILE = { BM: 64, BN: 64, BK: 32 } as const;

export const GEMM_WGSL = /* wgsl */ `
struct G { M: u32, N: u32, K: u32, codes: u32, scales: u32, x: u32, y: u32, bias: u32, flags: u32, _a: u32, _b: u32, _c: u32 };
@group(0) @binding(0) var<uniform> U: G;
@group(0) @binding(1) var<storage, read> CODES: array<u32>;
@group(0) @binding(2) var<storage, read> SCALES: array<f32>;
@group(0) @binding(3) var<storage, read_write> A: array<f32>;

const BM = 64u;
const BN = 64u;
const BK = 32u;
var<workgroup> As: array<f32, 2048>; // [BK][BM]
var<workgroup> Bs: array<f32, 2048>; // [BK][BN]

@compute @workgroup_size(16, 16)
fn gemm(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_id) li: vec3u, @builtin(local_invocation_index) lid: u32) {
  let m0 = wg.y * BM;
  let n0 = wg.x * BN;
  let K = U.K;
  var acc0 = vec4f(0.0); var acc1 = vec4f(0.0); var acc2 = vec4f(0.0); var acc3 = vec4f(0.0);
  for (var k0 = 0u; k0 < K; k0 += BK) {
    // stage X: 64 tokens x 32 inputs, 8 per thread, stored k-major
    for (var i = 0u; i < 8u; i++) {
      let idx = lid + i * 256u;
      let r = idx / BK;
      let kk = idx % BK;
      let m = m0 + r;
      var v = 0.0;
      if (m < U.M) { v = A[U.x + m * K + k0 + kk]; }
      As[kk * BM + r] = v;
    }
    // decode W: 64 outputs x 32 inputs = 2 code words per output; each thread decodes 8 trits
    {
      let r = lid / 4u;
      let part = lid % 4u;
      let n = n0 + r;
      var w = 0x55555555u; // all zeros (code 1)
      var s = 0.0;
      if (n < U.N) {
        w = CODES[U.codes + n * (K / 16u) + k0 / 16u + part / 2u];
        s = SCALES[U.scales + n * (K / 128u) + k0 / 128u];
      }
      let half = (part % 2u) * 8u;
      let kb = (part / 2u) * 16u + half;
      for (var j = 0u; j < 8u; j++) {
        Bs[(kb + j) * BN + r] = (f32((w >> ((half + j) * 2u)) & 3u) - 1.0) * s;
      }
    }
    workgroupBarrier();
    for (var kk = 0u; kk < BK; kk++) {
      let ab = kk * BM + li.y * 4u;
      let a = vec4f(As[ab], As[ab + 1u], As[ab + 2u], As[ab + 3u]);
      let bb = kk * BN + li.x * 4u;
      let b = vec4f(Bs[bb], Bs[bb + 1u], Bs[bb + 2u], Bs[bb + 3u]);
      acc0 += a.x * b; acc1 += a.y * b; acc2 += a.z * b; acc3 += a.w * b;
    }
    workgroupBarrier();
  }
  let accs = array<vec4f, 4>(acc0, acc1, acc2, acc3);
  for (var i = 0u; i < 4u; i++) {
    let m = m0 + li.y * 4u + i;
    if (m >= U.M) { continue; }
    for (var j = 0u; j < 4u; j++) {
      let n = n0 + li.x * 4u + j;
      if (n >= U.N) { continue; }
      var v = accs[i][j];
      if ((U.flags & 1u) == 1u) { v += A[U.bias + n]; }
      if ((U.flags & 2u) == 2u) { v += A[U.y + m * U.N + n]; }
      if ((U.flags & 4u) == 4u) { v = A[U.y + m * U.N + n] + A[U.bias + n] * v; }
      A[U.y + m * U.N + n] = v;
    }
  }
}`;

// The fast path: 128 x 128 tiles, 8 x 8 outputs per thread from vector reads of workgroup memory (16 multiply-adds per
// read instead of 2). Half the workgroup decodes W (4 outputs x 4 inputs each), the other half stages X (4 rows x 4
// inputs each, transposed); the arena is read and written as vec4, so every offset and N must be multiples of 4.
const FAST = { BM: 128, BN: 128 } as const;
const acc = (i: number, c: number) => `c${i}${c}`;
const FAST_ACC = Array.from({ length: 8 }, (_, i) => [0, 1].map((c) => acc(i, c)))
	.flat()
	.map((v) => `var ${v} = vec4f(0.0);`)
	.join(' ');
const FAST_FMA = Array.from(
	{ length: 8 },
	(_, i) =>
		`${acc(i, 0)} += a${i >> 2}.${'xyzw'[i & 3]} * b0; ${acc(i, 1)} += a${i >> 2}.${'xyzw'[i & 3]} * b1;`
).join('\n      ');
const FAST_OUT = Array.from(
	{ length: 8 },
	(_, i) => `put(${i}u, ${acc(i, 0)}, ${acc(i, 1)});`
).join(' ');

/** The fast kernel for a K-chunk of BK inputs (16 or 32: workgroup memory holds 2 x BK x 128 floats). */
export const gemmFastWGSL = (BK: 16 | 32) => /* wgsl */ `
struct G { M: u32, N: u32, K: u32, codes: u32, scales: u32, x: u32, y: u32, bias: u32, flags: u32, lt: u32, lb: u32, lr: u32 };
@group(0) @binding(0) var<uniform> U: G;
@group(0) @binding(1) var<storage, read> CODES: array<u32>;
@group(0) @binding(2) var<storage, read> SCALES: array<f32>;
@group(0) @binding(3) var<storage, read_write> A: array<vec4f>;
@group(0) @binding(4) var<storage, read> LB: array<vec4f>; // a low-rank side branch's B, transposed: [rank][N]

const BM = ${FAST.BM}u;
const BN = ${FAST.BN}u;
const BK = ${BK}u;
var<workgroup> As: array<vec4f, ${(BK * FAST.BM) / 4}>; // [BK][BM / 4]
var<workgroup> Bs: array<vec4f, ${(BK * FAST.BN) / 4}>; // [BK][BN / 4]
var<private> m0: u32;
var<private> n0: u32;
var<private> tx: u32;
var<private> ty: u32;

fn put(i: u32, lo: vec4f, hi: vec4f) {
  let m = m0 + ty * 8u + i;
  if (m >= U.M) { return; }
  for (var c = 0u; c < 2u; c++) {
    let n = n0 + tx * 8u + c * 4u;
    if (n >= U.N) { continue; }
    var v = select(lo, hi, c == 1u);
    let at = (U.y + m * U.N + n) / 4u;
    if ((U.flags & 1u) == 1u) { v += A[(U.bias + n) / 4u]; }
    if ((U.flags & 2u) == 2u) { v += A[at]; }
    if ((U.flags & 4u) == 4u) { v = A[at] + A[(U.bias + n) / 4u] * v; }
    A[at] = v;
  }
}

@compute @workgroup_size(256)
fn gemm(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) lid: u32) {
  m0 = wg.y * BM;
  n0 = wg.x * BN;
  tx = lid % 16u;
  ty = lid / 16u;
  let K = U.K;
  ${FAST_ACC}
  for (var k0 = 0u; k0 < K; k0 += BK) {
    // each thread writes whole vec4s (a component write to shared memory is a read-modify-write of the vector)
    if (lid < 128u) {
      // decode W for outputs n0 + 4g .. + 3 and inputs k0 + 4kq .. + 3 into Bs[k][n]
      let g = lid / 4u;
      for (var kq = lid % 4u; kq < BK / 4u; kq += 4u) {
        var t: array<vec4f, 4>; // [output][input]
        for (var o = 0u; o < 4u; o++) {
          let n = n0 + 4u * g + o;
          var w = 0x55555555u; // all zeros (code 1)
          var s = 0.0;
          if (n < U.N) {
            w = CODES[U.codes + n * (K / 16u) + k0 / 16u + kq / 4u];
            s = SCALES[U.scales + n * (K / 128u) + k0 / 128u];
          }
          let b = w >> ((kq % 4u) * 8u);
          t[o] = (vec4f(f32(b & 3u), f32((b >> 2u) & 3u), f32((b >> 4u) & 3u), f32((b >> 6u) & 3u)) - 1.0) * s;
        }
        for (var e = 0u; e < 4u; e++) {
          Bs[(4u * kq + e) * 32u + g] = vec4f(t[0][e], t[1][e], t[2][e], t[3][e]);
        }
      }
    } else {
      // stage X for rows m0 + 4rb .. + 3 and inputs k0 + 4kq .. + 3, transposed into As[k][m]
      let t = lid - 128u;
      let rb = t / 4u;
      for (var kq = t % 4u; kq < BK / 4u; kq += 4u) {
        var v: array<vec4f, 4>; // [row][input]
        for (var o = 0u; o < 4u; o++) {
          let m = m0 + 4u * rb + o;
          v[o] = vec4f(0.0);
          if (m < U.M) { v[o] = A[(U.x + m * K + k0) / 4u + kq]; }
        }
        for (var e = 0u; e < 4u; e++) {
          As[(4u * kq + e) * 32u + rb] = vec4f(v[0][e], v[1][e], v[2][e], v[3][e]);
        }
      }
    }
    workgroupBarrier();
    for (var kk = 0u; kk < BK; kk++) {
      let a0 = As[kk * 32u + ty * 2u];
      let a1 = As[kk * 32u + ty * 2u + 1u];
      let b0 = Bs[kk * 32u + tx * 2u];
      let b1 = Bs[kk * 32u + tx * 2u + 1u];
      ${FAST_FMA}
    }
    workgroupBarrier();
  }
  // a low-rank side branch (a LoRA), as more inputs: T = X A^T [M][rank] (computed before) times B^T [rank][N]
  for (var r0 = 0u; r0 < U.lr; r0 += BK) {
    if (lid < 128u) {
      let g = lid / 4u;
      for (var kq = lid % 4u; kq < BK / 4u; kq += 4u) {
        for (var e = 0u; e < 4u; e++) {
          let k = 4u * kq + e;
          var b = vec4f(0.0);
          if (n0 + 4u * g < U.N) { b = LB[(U.lb + (r0 + k) * U.N + n0) / 4u + g]; }
          Bs[k * 32u + g] = b;
        }
      }
    } else {
      let t = lid - 128u;
      let rb = t / 4u;
      for (var kq = t % 4u; kq < BK / 4u; kq += 4u) {
        var v: array<vec4f, 4>; // [row][input]
        for (var o = 0u; o < 4u; o++) {
          let m = m0 + 4u * rb + o;
          v[o] = vec4f(0.0);
          if (m < U.M) { v[o] = A[(U.lt + m * U.lr + r0) / 4u + kq]; }
        }
        for (var e = 0u; e < 4u; e++) {
          As[(4u * kq + e) * 32u + rb] = vec4f(v[0][e], v[1][e], v[2][e], v[3][e]);
        }
      }
    }
    workgroupBarrier();
    for (var kk = 0u; kk < BK; kk++) {
      let a0 = As[kk * 32u + ty * 2u];
      let a1 = As[kk * 32u + ty * 2u + 1u];
      let b0 = Bs[kk * 32u + tx * 2u];
      let b1 = Bs[kk * 32u + tx * 2u + 1u];
      ${FAST_FMA}
    }
    workgroupBarrier();
  }
  ${FAST_OUT}
}`;

/** Whether a job can take the fast path (vec4 access: offsets and N multiples of 4; K in whole code words). */
export function gemmFastOk(j: GemmJob) {
	const l = j.lora;
	return (
		j.N % 4 === 0 &&
		j.K % 128 === 0 &&
		j.x % 4 === 0 &&
		j.y % 4 === 0 &&
		(j.bias ?? 0) % 4 === 0 &&
		(!l || (l.rank % 32 === 0 && l.t % 4 === 0 && l.b % 4 === 0))
	);
}

export interface GemmJob {
	M: number; // tokens
	N: number; // outputs
	K: number; // inputs (multiple of 128)
	codes: number; // u32 offset of the weight codes
	scales: number; // f32 offset of the weight scales
	x: number; // arena offset of X [M][K]
	y: number; // arena offset of Y [M][N]
	bias?: number; // arena offset of a bias [N]
	accumulate?: boolean; // add into Y instead of overwriting it (a residual update)
	gated?: boolean; // Y += gate[n] * result, with the gate vector at `bias` (a gated residual update)
	/** A low-rank side branch added to the product (before the bias and the gate): T [M][rank] at arena offset t
	 * (T = X A^T, computed before), times B^T [rank][N] at float offset b of the side-branch buffer. Fast path only. */
	lora?: { t: number; b: number; rank: number };
}

/** Records ternary GEMMs into a compute pass; parameters go through one dynamic-offset uniform buffer. */
export class TernaryGemm {
	private pipe: GPUComputePipeline;
	private fast: GPUComputePipeline | null;
	private lora: GPUBuffer;
	private layout: GPUBindGroupLayout;
	private params: GPUBuffer;
	private bind: GPUBindGroup;
	private data: Uint32Array;

	constructor(
		private device: GPUDevice,
		codes: GPUBuffer,
		scales: GPUBuffer,
		arena: GPUBuffer,
		private capacity = 1024,
		fastBK: 0 | 16 | 32 = 16, // 0: the first kernel only
		lora?: GPUBuffer // side branches' B matrices (see GemmJob.lora)
	) {
		this.layout = device.createBindGroupLayout({
			entries: [
				{
					binding: 0,
					visibility: GPUShaderStage.COMPUTE,
					buffer: { type: 'uniform', hasDynamicOffset: true, minBindingSize: 48 }
				},
				{ binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
				{ binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
				{ binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
				{ binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }
			]
		});
		this.lora =
			lora ??
			device.createBuffer({ label: 'no side branch', size: 16, usage: GPUBufferUsage.STORAGE });
		this.pipe = device.createComputePipeline({
			layout: device.createPipelineLayout({ bindGroupLayouts: [this.layout] }),
			compute: {
				module: device.createShaderModule({ label: 'ternary gemm', code: GEMM_WGSL }),
				entryPoint: 'gemm'
			}
		});
		this.fast = fastBK
			? device.createComputePipeline({
					layout: device.createPipelineLayout({ bindGroupLayouts: [this.layout] }),
					compute: {
						module: device.createShaderModule({
							label: 'ternary gemm (fast)',
							code: gemmFastWGSL(fastBK)
						}),
						entryPoint: 'gemm'
					}
				})
			: null;
		this.params = device.createBuffer({
			size: capacity * 256,
			usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
		});
		this.data = new Uint32Array(capacity * 64);
		this.bind = device.createBindGroup({
			layout: this.layout,
			entries: [
				{ binding: 0, resource: { buffer: this.params, size: 48 } },
				{ binding: 1, resource: { buffer: codes } },
				{ binding: 2, resource: { buffer: scales } },
				{ binding: 3, resource: { buffer: arena } },
				{ binding: 4, resource: { buffer: this.lora } }
			]
		});
	}

	/** Write the parameters of jobs[i] into slot (first + i); call before submitting the encoder that runs them. */
	prepare(jobs: GemmJob[], first = 0) {
		jobs.forEach((j, i) => {
			const o = (first + i) * 64;
			if (first + i >= this.capacity) throw new Error('Too many GEMM jobs for one submission.');
			this.data.set(
				[
					j.M,
					j.N,
					j.K,
					j.codes,
					j.scales,
					j.x,
					j.y,
					j.bias ?? 0,
					(j.bias === undefined || j.gated ? 0 : 1) | (j.accumulate ? 2 : 0) | (j.gated ? 4 : 0),
					j.lora?.t ?? 0,
					j.lora?.b ?? 0,
					j.lora?.rank ?? 0
				],
				o
			);
		});
		this.device.queue.writeBuffer(
			this.params,
			first * 256,
			this.data,
			first * 64,
			jobs.length * 64
		);
	}

	/** Dispatch the job prepared in `slot`. */
	dispatch(pass: GPUComputePassEncoder, slot: number, job: GemmJob) {
		pass.setBindGroup(0, this.bind, [slot * 256]);
		if (this.fast && gemmFastOk(job)) {
			pass.setPipeline(this.fast);
			pass.dispatchWorkgroups(Math.ceil(job.N / FAST.BN), Math.ceil(job.M / FAST.BM));
		} else {
			if (job.lora)
				throw new Error(
					'A side branch needs the fast GEMM (aligned offsets, rank a multiple of 32).'
				);
			pass.setPipeline(this.pipe);
			pass.dispatchWorkgroups(Math.ceil(job.N / GEMM_TILE.BN), Math.ceil(job.M / GEMM_TILE.BM));
		}
	}
}
