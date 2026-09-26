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
}

/** Records ternary GEMMs into a compute pass; parameters go through one dynamic-offset uniform buffer. */
export class TernaryGemm {
	private pipe: GPUComputePipeline;
	private layout: GPUBindGroupLayout;
	private params: GPUBuffer;
	private bind: GPUBindGroup;
	private data: Uint32Array;

	constructor(
		private device: GPUDevice,
		codes: GPUBuffer,
		scales: GPUBuffer,
		arena: GPUBuffer,
		private capacity = 1024
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
				{ binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }
			]
		});
		this.pipe = device.createComputePipeline({
			layout: device.createPipelineLayout({ bindGroupLayouts: [this.layout] }),
			compute: {
				module: device.createShaderModule({ label: 'ternary gemm', code: GEMM_WGSL }),
				entryPoint: 'gemm'
			}
		});
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
				{ binding: 3, resource: { buffer: arena } }
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
					(j.bias === undefined || j.gated ? 0 : 1) | (j.accumulate ? 2 : 0) | (j.gated ? 4 : 0)
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
		pass.setPipeline(this.pipe);
		pass.setBindGroup(0, this.bind, [slot * 256]);
		pass.dispatchWorkgroups(Math.ceil(job.N / GEMM_TILE.BN), Math.ceil(job.M / GEMM_TILE.BM));
	}
}
