// Dense GEMMs over the painter's arena, staged and multiplied in f16 with f32 accumulation (each 32-input chunk summed
// in f16, then added into f32), for attention: Q K^T and P V for all heads at once (a batch per head).
//   C[z][m][n] = scale * sum_k A[z][m][k] B[z](k, n)   (+ C[z][m][n] when accumulating)
// B(k, n) is B[n][k] (nt: rows by output, as keys are) or B[k][n] (nn: as values are). Tiles of 128 x 128, 8 x 8 outputs
// per thread; every offset, stride and K must be a multiple of 4 (the arena is read as vec4). Needs shader-f16.

export interface DenseJob {
	M: number;
	N: number;
	K: number;
	a: number; // arena offset of A (floats), row stride lda, batch stride sa
	lda: number;
	sa: number;
	b: number;
	ldb: number;
	sb: number;
	c: number;
	ldc: number;
	sc: number;
	batch: number;
	nn: boolean; // B is [K][N] (else [N][K])
	scale: number;
	accumulate?: boolean;
}

const acc = (i: number, c: number) => `c${i}${c}`;
const part = (i: number, c: number) => `p${i}${c}`;
const ACC = Array.from({ length: 16 }, (_, q) => `var ${acc(q >> 1, q & 1)} = vec4f(0.0);`).join(
	' '
);
const PART = Array.from(
	{ length: 16 },
	(_, q) => `var ${part(q >> 1, q & 1)} = vec4<f16>(0.0h);`
).join(' ');
const FMA = Array.from(
	{ length: 8 },
	(_, i) =>
		`${part(i, 0)} = fma(vec4<f16>(a${i >> 2}.${'xyzw'[i & 3]}), b0, ${part(i, 0)}); ` +
		`${part(i, 1)} = fma(vec4<f16>(a${i >> 2}.${'xyzw'[i & 3]}), b1, ${part(i, 1)});`
).join('\n      ');
const FLUSH = Array.from(
	{ length: 16 },
	(_, q) =>
		`${acc(q >> 1, q & 1)} += vec4f(${part(q >> 1, q & 1)}); ${part(q >> 1, q & 1)} = vec4<f16>(0.0h);`
).join('\n    ');
const OUT = Array.from({ length: 8 }, (_, i) => `put(${i}u, ${acc(i, 0)}, ${acc(i, 1)});`).join(
	' '
);

export const DENSE_WGSL = /* wgsl */ `
enable f16;
struct D { M: u32, N: u32, K: u32, a: u32, lda: u32, sa: u32, b: u32, ldb: u32, sb: u32, c: u32, ldc: u32, sc: u32, flags: u32, _0: u32, _1: u32, scale: f32 };
@group(0) @binding(0) var<uniform> U: D;
@group(0) @binding(1) var<storage, read_write> A: array<vec4f>;

const BM = 128u;
const BN = 128u;
const BK = 32u;
var<workgroup> As: array<vec4<f16>, 1024>; // [BK][BM / 4]
var<workgroup> Bs: array<vec4<f16>, 1024>; // [BK][BN / 4]
var<private> m0: u32;
var<private> n0: u32;
var<private> tx: u32;
var<private> ty: u32;
var<private> cb: u32;

fn put(i: u32, lo: vec4f, hi: vec4f) {
  let m = m0 + ty * 8u + i;
  if (m >= U.M) { return; }
  for (var c = 0u; c < 2u; c++) {
    let n = n0 + tx * 8u + c * 4u;
    if (n >= U.N) { continue; }
    let at = (cb + m * U.ldc + n) / 4u;
    var v = U.scale * select(lo, hi, c == 1u);
    if ((U.flags & 4u) != 0u) { v += A[at]; }
    A[at] = v;
  }
}

// rows r0 + 4 rb .. + 3 of a row-major matrix (stride ld) at inputs k0 + 4 kq .. + 3, transposed into dst[k][r]
fn rows_t(base: u32, ld: u32, r0: u32, rmax: u32, k0: u32, t: u32, dst: u32) {
  let rb = t / 8u;
  let kq = t % 8u;
  var v: array<vec4f, 4>;
  for (var o = 0u; o < 4u; o++) {
    let r = r0 + 4u * rb + o;
    v[o] = vec4f(0.0);
    if (r < rmax && k0 + 4u * kq < U.K) { v[o] = A[(base + r * ld + k0) / 4u + kq]; }
  }
  for (var e = 0u; e < 4u; e++) {
    let h = vec4<f16>(vec4f(v[0][e], v[1][e], v[2][e], v[3][e]));
    if (dst == 0u) { As[(4u * kq + e) * 32u + rb] = h; } else { Bs[(4u * kq + e) * 32u + rb] = h; }
  }
}

@compute @workgroup_size(256)
fn dense(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) lid: u32) {
  m0 = wg.y * BM;
  n0 = wg.x * BN;
  tx = lid % 16u;
  ty = lid / 16u;
  let z = wg.z;
  let ab = U.a + z * U.sa;
  let bb = U.b + z * U.sb;
  cb = U.c + z * U.sc;
  let nn = (U.flags & 2u) != 0u;
  ${ACC}
  ${PART}
  for (var k0 = 0u; k0 < U.K; k0 += BK) {
    rows_t(ab, U.lda, m0, U.M, k0, lid, 0u);
    if (nn) {
      // B[k][n]: 32 inputs x 32 vec4 of outputs, straight into Bs[k][n]
      for (var i = 0u; i < 4u; i++) {
        let idx = lid + 256u * i;
        let k = idx / 32u;
        let q = idx % 32u;
        var v = vec4f(0.0);
        if (k0 + k < U.K && n0 + 4u * q < U.N) { v = A[(bb + (k0 + k) * U.ldb + n0) / 4u + q]; }
        Bs[k * 32u + q] = vec4<f16>(v);
      }
    } else {
      rows_t(bb, U.ldb, n0, U.N, k0, lid, 1u);
    }
    workgroupBarrier();
    for (var kk = 0u; kk < BK; kk++) {
      let a0 = As[kk * 32u + ty * 2u];
      let a1 = As[kk * 32u + ty * 2u + 1u];
      let b0 = Bs[kk * 32u + tx * 2u];
      let b1 = Bs[kk * 32u + tx * 2u + 1u];
      ${FMA}
    }
    workgroupBarrier();
    ${FLUSH}
  }
  ${OUT}
}`;

/** Whether a job fits the kernel (vec4 access). */
export function denseOk(j: DenseJob) {
	return [j.K, j.a, j.lda, j.sa, j.b, j.ldb, j.sb, j.c, j.ldc, j.sc, j.N].every((v) => v % 4 === 0);
}

/** Records dense f16 GEMMs into a compute pass; parameters go through one dynamic-offset uniform buffer. */
export class DenseGemm {
	private pipe: GPUComputePipeline;
	private params: GPUBuffer;
	private bind: GPUBindGroup;
	private data: ArrayBuffer;

	constructor(
		private device: GPUDevice,
		arena: GPUBuffer,
		private capacity = 256
	) {
		const layout = device.createBindGroupLayout({
			entries: [
				{
					binding: 0,
					visibility: GPUShaderStage.COMPUTE,
					buffer: { type: 'uniform', hasDynamicOffset: true, minBindingSize: 64 }
				},
				{ binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }
			]
		});
		this.pipe = device.createComputePipeline({
			layout: device.createPipelineLayout({ bindGroupLayouts: [layout] }),
			compute: {
				module: device.createShaderModule({ label: 'dense gemm (f16)', code: DENSE_WGSL }),
				entryPoint: 'dense'
			}
		});
		this.params = device.createBuffer({
			size: capacity * 256,
			usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
		});
		this.data = new ArrayBuffer(capacity * 256);
		this.bind = device.createBindGroup({
			layout,
			entries: [
				{ binding: 0, resource: { buffer: this.params, size: 64 } },
				{ binding: 1, resource: { buffer: arena } }
			]
		});
	}

	/** Write the parameters of jobs[i] into slot (first + i); call before submitting the encoder that runs them. */
	prepare(jobs: DenseJob[], first = 0) {
		if (first + jobs.length > this.capacity)
			throw new Error('Too many dense GEMMs for one submission.');
		jobs.forEach((j, i) => {
			const o = (first + i) * 256;
			new Uint32Array(this.data, o, 15).set([
				j.M,
				j.N,
				j.K,
				j.a,
				j.lda,
				j.sa,
				j.b,
				j.ldb,
				j.sb,
				j.c,
				j.ldc,
				j.sc,
				(j.nn ? 2 : 0) | (j.accumulate ? 4 : 0),
				0,
				0
			]);
			new Float32Array(this.data, o + 60, 1)[0] = j.scale;
		});
		this.device.queue.writeBuffer(
			this.params,
			first * 256,
			this.data,
			first * 256,
			jobs.length * 256
		);
	}

	/** Dispatch the job prepared in `slot`. */
	dispatch(pass: GPUComputePassEncoder, slot: number, job: DenseJob) {
		pass.setPipeline(this.pipe);
		pass.setBindGroup(0, this.bind, [slot * 256]);
		pass.dispatchWorkgroups(Math.ceil(job.N / 128), Math.ceil(job.M / 128), job.batch);
	}
}
