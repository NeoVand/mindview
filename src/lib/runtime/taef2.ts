import type { GpuTask } from './scheduler';

// TAEF2 (madebyollin/taef2): the tiny decoder that turns the painter's 32-channel latent (64 x 64 for a 512^2 image)
// into pixels. Mirrors taesd.Decoder(32, use_midblock_gn=True):
//   Clamp, conv 32->64, ReLU, 3 x Block(gn), Up, conv, 3 x Block, Up, conv, 3 x Block, Up, conv, Block, conv 64->3
//   Block(x) = ReLU(conv(ReLU(conv(ReLU(conv(x'))))) + x'),  x' = x + pool(x) for the gn blocks,
//   pool = conv1x1 64->256, GroupNorm(4, 256), ReLU, conv1x1 256->64.
// Activations are f32, channel-major [C][H][W], in one arena. Weights are f32 in PyTorch layout, one buffer.

const WGSL = /* wgsl */ `
struct C { H: u32, W: u32, Cin: u32, Cout: u32, x: u32, y: u32, w: u32, b: u32, flags: u32, skip: u32, g: u32, gb: u32 };
@group(0) @binding(0) var<uniform> U: C;
@group(0) @binding(1) var<storage, read> WT: array<f32>;
@group(0) @binding(2) var<storage, read_write> A: array<f32>;
@group(0) @binding(3) var<storage, read_write> ST: array<f32>;
@group(0) @binding(4) var OUT: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(5) var EARLY: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(6) var S128: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(7) var S256: texture_storage_2d<rgba8unorm, write>;

// flags: 1 bias, 2 relu, 4 input is half size (nearest upsample on read), 8 clamp input (tanh(x/3)*3),
//        16 add skip tensor before relu
fn inp(c: u32, y: i32, x: i32) -> f32 {
  var h = i32(U.H); var w = i32(U.W); var yy = y; var xx = x;
  if (yy < 0 || xx < 0 || yy >= h || xx >= w) { return 0.0; }
  if ((U.flags & 4u) != 0u) { yy = yy / 2; xx = xx / 2; h = h / 2; w = w / 2; }
  var v = A[U.x + (c * u32(h) + u32(yy)) * u32(w) + u32(xx)];
  if ((U.flags & 8u) != 0u) { v = tanh(v / 3.0) * 3.0; }
  return v;
}

var<workgroup> tile: array<f32, 1600>; // 16 channels x 10 x 10

// 3x3 convolution, padding 1. A workgroup makes an 8x8 pixel tile for 64 output channels (4 groups of 16).
@compute @workgroup_size(8, 8, 4)
fn conv3(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_id) li: vec3u, @builtin(local_invocation_index) lid: u32) {
  let ty = wg.y + U.g; // tile row (g offsets it, so a big convolution can run in slices)
  let px = wg.x * 8u + li.x;
  let py = ty * 8u + li.y;
  let cg = li.z;
  var acc: array<f32, 16>;
  for (var ci0 = 0u; ci0 < U.Cin; ci0 += 16u) {
    for (var i = lid; i < 1600u; i += 256u) {
      let c = i / 100u; let r = i % 100u;
      let gy = i32(ty * 8u + r / 10u) - 1;
      let gx = i32(wg.x * 8u + r % 10u) - 1;
      var v = 0.0;
      if (ci0 + c < U.Cin) { v = inp(ci0 + c, gy, gx); }
      tile[i] = v;
    }
    workgroupBarrier();
    let nc = min(16u, U.Cin - ci0);
    for (var c = 0u; c < nc; c++) {
      for (var t = 0u; t < 9u; t++) {
        let v = tile[c * 100u + (li.y + t / 3u) * 10u + li.x + t % 3u];
        let wb = U.w + (ci0 + c) * 9u + t;
        for (var o = 0u; o < 16u; o++) {
          let co = cg * 16u + o;
          if (co < U.Cout) { acc[o] += v * WT[wb + co * U.Cin * 9u]; }
        }
      }
    }
    workgroupBarrier();
  }
  if (px >= U.W || py >= U.H) { return; }
  for (var o = 0u; o < 16u; o++) {
    let co = cg * 16u + o;
    if (co >= U.Cout) { break; }
    var v = acc[o];
    if ((U.flags & 1u) != 0u) { v += WT[U.b + co]; }
    let at = (co * U.H + py) * U.W + px;
    if ((U.flags & 16u) != 0u) { v += A[U.skip + at]; }
    if ((U.flags & 2u) != 0u) { v = max(v, 0.0); }
    A[U.y + at] = v;
  }
}

// 3x3 convolution, padding 1, for 64 output channels (the decoder's body): a workgroup makes a 16 x 16 pixel tile, each
// thread 2 x 2 pixels for 16 channels. Per 4 input channels the input tile (18 x 18) and the weights (as vec4 over output
// channels) are staged in workgroup memory, so each read serves 4 to 16 multiply-adds (conv3: one weight read each).
var<workgroup> tb: array<f32, 1296>; // 4 channels x 18 x 18
var<workgroup> wb: array<vec4f, 576>; // [4 channels][9 taps][16 vec4 = 64 output channels]
@compute @workgroup_size(8, 8, 4)
fn conv3b(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_id) li: vec3u, @builtin(local_invocation_index) lid: u32) {
  let ty = wg.y + U.g; // tile row (g offsets it, so a big convolution can run in slices)
  let x0 = wg.x * 16u + 2u * li.x;
  let y0 = ty * 16u + 2u * li.y;
  let cg = li.z; // output channels 16 cg .. + 15
  var a00 = vec4f(0.0); var a01 = vec4f(0.0); var a02 = vec4f(0.0); var a03 = vec4f(0.0); // pixel (x0, y0)
  var a10 = vec4f(0.0); var a11 = vec4f(0.0); var a12 = vec4f(0.0); var a13 = vec4f(0.0); // (x0 + 1, y0)
  var a20 = vec4f(0.0); var a21 = vec4f(0.0); var a22 = vec4f(0.0); var a23 = vec4f(0.0); // (x0, y0 + 1)
  var a30 = vec4f(0.0); var a31 = vec4f(0.0); var a32 = vec4f(0.0); var a33 = vec4f(0.0); // (x0 + 1, y0 + 1)
  for (var ci0 = 0u; ci0 < U.Cin; ci0 += 4u) {
    for (var i = lid; i < 1296u; i += 256u) {
      let c = i / 324u; let r = i % 324u;
      let gy = i32(ty * 16u + r / 18u) - 1;
      let gx = i32(wg.x * 16u + r % 18u) - 1;
      tb[i] = inp(ci0 + c, gy, gx);
    }
    for (var i = lid; i < 576u; i += 256u) {
      let q = i % 16u; let ct = i / 16u; let c = ct / 9u; let t = ct % 9u;
      let base = U.w + (ci0 + c) * 9u + t;
      let stride = U.Cin * 9u;
      wb[i] = vec4f(WT[base + (4u * q) * stride], WT[base + (4u * q + 1u) * stride],
                    WT[base + (4u * q + 2u) * stride], WT[base + (4u * q + 3u) * stride]);
    }
    workgroupBarrier();
    for (var c = 0u; c < 4u; c++) {
      for (var t = 0u; t < 9u; t++) {
        let o = c * 324u + (2u * li.y + t / 3u) * 18u + 2u * li.x + t % 3u;
        let p0 = tb[o]; let p1 = tb[o + 1u]; let p2 = tb[o + 18u]; let p3 = tb[o + 19u];
        let wi = (c * 9u + t) * 16u + cg * 4u;
        let w0 = wb[wi]; let w1 = wb[wi + 1u]; let w2 = wb[wi + 2u]; let w3 = wb[wi + 3u];
        a00 += p0 * w0; a01 += p0 * w1; a02 += p0 * w2; a03 += p0 * w3;
        a10 += p1 * w0; a11 += p1 * w1; a12 += p1 * w2; a13 += p1 * w3;
        a20 += p2 * w0; a21 += p2 * w1; a22 += p2 * w2; a23 += p2 * w3;
        a30 += p3 * w0; a31 += p3 * w1; a32 += p3 * w2; a33 += p3 * w3;
      }
    }
    workgroupBarrier();
  }
  let px = array<u32, 4>(x0, x0 + 1u, x0, x0 + 1u);
  let py = array<u32, 4>(y0, y0, y0 + 1u, y0 + 1u);
  let acc = array<vec4f, 16>(a00, a01, a02, a03, a10, a11, a12, a13, a20, a21, a22, a23, a30, a31, a32, a33);
  for (var k = 0u; k < 4u; k++) {
    if (px[k] >= U.W || py[k] >= U.H) { continue; }
    for (var q = 0u; q < 4u; q++) {
      let v4 = acc[k * 4u + q];
      for (var j = 0u; j < 4u; j++) {
        let co = cg * 16u + q * 4u + j;
        var v = v4[j];
        if ((U.flags & 1u) != 0u) { v += WT[U.b + co]; }
        let at = (co * U.H + py[k]) * U.W + px[k];
        if ((U.flags & 16u) != 0u) { v += A[U.skip + at]; }
        if ((U.flags & 2u) != 0u) { v = max(v, 0.0); }
        A[U.y + at] = v;
      }
    }
  }
}

// 1x1 convolution (no bias): y[co][p] = sum_ci w[co][ci] x[ci][p] (+ skip)
@compute @workgroup_size(64)
fn conv1(@builtin(global_invocation_id) g: vec3u) {
  let p = g.x; let co = g.y;
  let n = U.H * U.W;
  if (p >= n || co >= U.Cout) { return; }
  var v = 0.0;
  for (var ci = 0u; ci < U.Cin; ci++) { v += WT[U.w + co * U.Cin + ci] * A[U.x + ci * n + p]; }
  if ((U.flags & 16u) != 0u) { v += A[U.skip + co * n + p]; }
  A[U.y + co * n + p] = v;
}

// GroupNorm statistics: one workgroup per group; writes mean and 1/std to ST
var<workgroup> red: array<f32, 512>;
@compute @workgroup_size(256)
fn gn_stats(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_id) li: vec3u) {
  let per = (U.Cin / 4u) * U.H * U.W;
  let base = U.x + wg.x * per;
  var s = 0.0; var ss = 0.0;
  for (var i = li.x; i < per; i += 256u) { let v = A[base + i]; s += v; ss += v * v; }
  red[li.x] = s; red[256u + li.x] = ss;
  workgroupBarrier();
  for (var st = 128u; st > 0u; st >>= 1u) {
    if (li.x < st) { red[li.x] += red[li.x + st]; red[256u + li.x] += red[256u + li.x + st]; }
    workgroupBarrier();
  }
  if (li.x == 0u) {
    let mean = red[0] / f32(per);
    ST[wg.x * 2u] = mean;
    ST[wg.x * 2u + 1u] = inverseSqrt(max(red[256] / f32(per) - mean * mean, 0.0) + 1e-5);
  }
}

// GroupNorm apply + ReLU, in place (g = gamma offset, gb = beta offset in WT)
@compute @workgroup_size(256)
fn gn_apply(@builtin(global_invocation_id) g: vec3u) {
  let n = U.H * U.W;
  let i = g.x;
  if (i >= U.Cin * n) { return; }
  let c = i / n;
  let grp = c / (U.Cin / 4u);
  let v = (A[U.x + i] - ST[grp * 2u]) * ST[grp * 2u + 1u] * WT[U.g + c] + WT[U.gb + c];
  A[U.x + i] = max(v, 0.0);
}

// the early picture: a linear colour probe on the 64 channels after the first stage (64 x 64)
// H, W, x = features, y = rgba floats out, w = probe [65][3]
@compute @workgroup_size(8, 8)
fn probe64(@builtin(global_invocation_id) g: vec3u) {
  if (g.x >= U.W || g.y >= U.H) { return; }
  let n = U.H * U.W;
  let p = g.y * U.W + g.x;
  var rgb = vec3f(WT[U.w + 192u], WT[U.w + 193u], WT[U.w + 194u]);
  for (var c = 0u; c < 64u; c++) {
    rgb += A[U.x + c * n + p] * vec3f(WT[U.w + c * 3u], WT[U.w + c * 3u + 1u], WT[U.w + c * 3u + 2u]);
  }
  rgb = clamp(rgb, vec3f(0.0), vec3f(1.0));
  A[U.y + p * 4u] = rgb.x; A[U.y + p * 4u + 1u] = rgb.y; A[U.y + p * 4u + 2u] = rgb.z; A[U.y + p * 4u + 3u] = 1.0;
  textureStore(EARLY, vec2i(g.xy), vec4f(rgb, 1.0));
}

// the picture partway through the decoder: the same kind of probe on the 64 channels after the 128 x 128 or the
// 256 x 256 stage (x = features, w = probe [65][3])
@compute @workgroup_size(8, 8)
fn probe_stage(@builtin(global_invocation_id) g: vec3u) {
  if (g.x >= U.W || g.y >= U.H) { return; }
  let n = U.H * U.W;
  let p = g.y * U.W + g.x;
  var rgb = vec3f(WT[U.w + 192u], WT[U.w + 193u], WT[U.w + 194u]);
  for (var c = 0u; c < 64u; c++) {
    rgb += A[U.x + c * n + p] * vec3f(WT[U.w + c * 3u], WT[U.w + c * 3u + 1u], WT[U.w + c * 3u + 2u]);
  }
  rgb = clamp(rgb, vec3f(0.0), vec3f(1.0));
  if (U.H == 128u) { textureStore(S128, vec2i(g.xy), vec4f(rgb, 1.0)); }
  else { textureStore(S256, vec2i(g.xy), vec4f(rgb, 1.0)); }
}

// the picture: channels 0..2 at x, clamped to 0..1
@compute @workgroup_size(8, 8)
fn present(@builtin(global_invocation_id) g: vec3u) {
  if (g.x >= U.W || g.y >= U.H) { return; }
  let n = U.H * U.W;
  let p = g.y * U.W + g.x;
  let rgb = clamp(vec3f(A[U.x + p], A[U.x + n + p], A[U.x + 2u * n + p]), vec3f(0.0), vec3f(1.0));
  textureStore(OUT, vec2i(g.xy), vec4f(rgb, 1.0));
}`;

/** Names follow the decoder's layer indices in madebyollin/taef2 (0 = first conv ... 18 = last), e.g. '2.pool.1.weight'. */
export type Taef2Weights = Map<string, Float32Array>;

type Step =
	| { k: 'conv3' | 'conv3b' | 'conv1'; p: number[]; wg: [number, number, number] }
	| {
			k: 'gn_stats' | 'gn_apply' | 'present' | 'probe64' | 'probe_stage';
			p: number[];
			wg: [number, number, number];
	  };

export class Taef2 {
	readonly texture: GPUTexture;
	/** The quick picture after the first stage (64 x 64), when a probe64 weight is given. */
	readonly earlyTexture: GPUTexture;
	/** The picture partway through the full decode, after the 128 and 256 stages (with probe128 / probe256). */
	readonly stageTextures: Map<number, GPUTexture> = new Map();
	private earlySteps: Step[] = [];
	private sliced: Step[] = []; // the full recipe with big convolutions cut into row slices
	private rgbAt = 0;
	private pipes: Record<string, GPUComputePipeline>;
	private bind: GPUBindGroup;
	private params: GPUBuffer;
	private arena: GPUBuffer;
	private steps: Step[] = [];
	private latentAt = 0;
	readonly size: number;

	/** latent: 32 x (size/8) x (size/8), channel-major; call decode() after writing it with writeLatent(). */
	constructor(
		private device: GPUDevice,
		weights: Taef2Weights,
		size = 512
	) {
		this.size = size;
		// pack the weights into one buffer and remember offsets
		const offsets = new Map<string, number>();
		let total = 0;
		for (const [k, v] of weights) {
			offsets.set(k, total);
			total += v.length;
		}
		const wbuf = device.createBuffer({
			label: 'taef2 weights',
			size: total * 4,
			usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
		});
		for (const [k, v] of weights) device.queue.writeBuffer(wbuf, offsets.get(k)! * 4, v);
		const W = (name: string) => {
			const o = offsets.get(name);
			if (o === undefined) throw new Error(`TAEF2 weight ${name} is missing.`);
			return o;
		};

		// activation arena: latent, then three 64 x size x size buffers (x, t1, t2) and a pool buffer
		const s0 = size / 8;
		const big = 64 * size * size;
		let o = 0;
		const take = (n: number) => ((o += n), o - n);
		this.latentAt = take(32 * s0 * s0);
		const X = take(big),
			T1 = take(big),
			T2 = take(big),
			P = take(256 * s0 * s0);
		this.rgbAt = take(s0 * s0 * 4);
		this.arena = device.createBuffer({
			label: 'taef2 activations',
			size: o * 4,
			usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
		});
		const stats = device.createBuffer({ size: 64, usage: GPUBufferUsage.STORAGE });
		this.earlyTexture = device.createTexture({
			size: [s0, s0],
			format: 'rgba8unorm',
			usage:
				GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC
		});
		this.texture = device.createTexture({
			size: [size, size],
			format: 'rgba8unorm',
			usage:
				GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC
		});
		for (const r of [128, 256])
			this.stageTextures.set(
				r,
				device.createTexture({
					size: [r, r],
					format: 'rgba8unorm',
					usage:
						GPUTextureUsage.STORAGE_BINDING |
						GPUTextureUsage.TEXTURE_BINDING |
						GPUTextureUsage.COPY_SRC
				})
			);

		// the recipe
		const conv3 = (
			res: number,
			cin: number,
			cout: number,
			x: number,
			y: number,
			layer: string,
			flags: number,
			skip = 0
		) =>
			this.steps.push({
				k: cout === 64 && cin % 4 === 0 ? 'conv3b' : 'conv3',
				p: [
					res,
					res,
					cin,
					cout,
					x,
					y,
					W(`${layer}.weight`),
					flags & 1 ? W(`${layer}.bias`) : 0,
					flags,
					skip,
					0,
					0
				],
				wg:
					cout === 64 && cin % 4 === 0
						? [Math.ceil(res / 16), Math.ceil(res / 16), 1]
						: [Math.ceil(res / 8), Math.ceil(res / 8), 1]
			});
		const conv1 = (
			res: number,
			cin: number,
			cout: number,
			x: number,
			y: number,
			layer: string,
			flags = 0,
			skip = 0
		) =>
			this.steps.push({
				k: 'conv1',
				p: [res, res, cin, cout, x, y, W(`${layer}.weight`), 0, flags, skip, 0, 0],
				wg: [Math.ceil((res * res) / 64), cout, 1]
			});
		const BIAS = 1,
			RELU = 2,
			UP = 4,
			CLAMP = 8,
			SKIP = 16;
		// Buffers rotate, so the recipe tracks where the current activation is.
		let cur = X;
		const bufs = [X, T1, T2];
		const other = (...not: number[]) => bufs.find((b) => !not.includes(b))!;
		const blockAt = (res: number, i: number, gn: boolean) => {
			let x = cur;
			if (gn) {
				const xp = other(x);
				conv1(res, 64, 256, x, P, `${i}.pool.0`);
				this.steps.push({
					k: 'gn_stats',
					p: [res, res, 256, 0, P, 0, 0, 0, 0, 0, 0, 0],
					wg: [4, 1, 1]
				});
				this.steps.push({
					k: 'gn_apply',
					p: [res, res, 256, 0, P, 0, 0, 0, 0, 0, W(`${i}.pool.1.weight`), W(`${i}.pool.1.bias`)],
					wg: [Math.ceil((256 * res * res) / 256), 1, 1]
				});
				conv1(res, 256, 64, P, xp, `${i}.pool.3`, SKIP, x);
				x = xp;
			}
			const a = other(x),
				b = other(x, a);
			conv3(res, 64, 64, x, a, `${i}.conv.0`, BIAS | RELU);
			conv3(res, 64, 64, a, b, `${i}.conv.2`, BIAS | RELU);
			conv3(res, 64, 64, b, a, `${i}.conv.4`, BIAS | RELU | SKIP, x);
			cur = a;
		};
		let res = s0;
		conv3(res, 32, 64, this.latentAt, X, '0', BIAS | RELU | CLAMP);
		cur = X;
		for (const i of [2, 3, 4]) blockAt(res, i, true);
		if (weights.has('probe64'))
			this.earlySteps = [
				...this.steps,
				{
					k: 'probe64',
					p: [res, res, 0, 0, cur, this.rgbAt, W('probe64'), 0, 0, 0, 0, 0],
					wg: [res / 8, res / 8, 1]
				}
			];
		for (const [up, blocks] of [
			[6, [7, 8, 9]],
			[11, [12, 13, 14]],
			[16, [17]]
		] as [number, number[]][]) {
			res *= 2;
			const nxt = other(cur);
			conv3(res, 64, 64, cur, nxt, `${up}`, UP);
			cur = nxt;
			for (const i of blocks) blockAt(res, i, false);
			if (weights.has(`probe${res}`))
				this.steps.push({
					k: 'probe_stage',
					p: [res, res, 0, 0, cur, 0, W(`probe${res}`), 0, 0, 0, 0, 0],
					wg: [res / 8, res / 8, 1]
				});
		}
		const fin = other(cur);
		conv3(res, 64, 3, cur, fin, '18', BIAS);
		this.steps.push({
			k: 'present',
			p: [res, res, 0, 0, fin, 0, 0, 0, 0, 0, 0, 0],
			wg: [res / 8, res / 8, 1]
		});

		const module = device.createShaderModule({ label: 'taef2', code: WGSL });
		const layout = device.createBindGroupLayout({
			entries: [
				{
					binding: 0,
					visibility: GPUShaderStage.COMPUTE,
					buffer: { type: 'uniform', hasDynamicOffset: true, minBindingSize: 48 }
				},
				{ binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
				{ binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
				{ binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
				{
					binding: 4,
					visibility: GPUShaderStage.COMPUTE,
					storageTexture: { access: 'write-only', format: 'rgba8unorm' }
				},
				...[5, 6, 7].map((binding): GPUBindGroupLayoutEntry => ({
					binding,
					visibility: GPUShaderStage.COMPUTE,
					storageTexture: { access: 'write-only', format: 'rgba8unorm' }
				}))
			]
		});
		const pl = device.createPipelineLayout({ bindGroupLayouts: [layout] });
		this.pipes = Object.fromEntries(
			['conv3', 'conv3b', 'conv1', 'gn_stats', 'gn_apply', 'present', 'probe64', 'probe_stage'].map(
				(e) => [e, device.createComputePipeline({ layout: pl, compute: { module, entryPoint: e } })]
			)
		);
		// slices of at most 64 pixel rows for convolutions at 128 x 128 and above
		for (const st of this.steps) {
			if ((st.k !== 'conv3' && st.k !== 'conv3b') || st.p[0] < 128) {
				this.sliced.push(st);
				continue;
			}
			const px = st.k === 'conv3b' ? 16 : 8; // the kernel's tile
			const tiles = st.p[0] / px,
				per = 64 / px;
			for (let t0 = 0; t0 < tiles; t0 += per)
				this.sliced.push({
					k: st.k,
					p: st.p.map((v, i) => (i === 10 ? t0 : v)),
					wg: [tiles, Math.min(per, tiles - t0), 1]
				});
		}
		const all = [...this.steps, ...this.earlySteps, ...this.sliced];
		this.params = device.createBuffer({
			size: Math.max(1, all.length) * 256,
			usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
		});
		const data = new Uint32Array(Math.max(1, all.length) * 64);
		all.forEach((st, i) => data.set(st.p, i * 64));
		device.queue.writeBuffer(this.params, 0, data);
		this.bind = device.createBindGroup({
			layout,
			entries: [
				{ binding: 0, resource: { buffer: this.params, size: 48 } },
				{ binding: 1, resource: { buffer: wbuf } },
				{ binding: 2, resource: { buffer: this.arena } },
				{ binding: 3, resource: { buffer: stats } },
				{ binding: 4, resource: this.texture.createView() },
				{ binding: 5, resource: this.earlyTexture.createView() },
				{ binding: 6, resource: this.stageTextures.get(128)!.createView() },
				{ binding: 7, resource: this.stageTextures.get(256)!.createView() }
			]
		});
	}

	/** The latent to decode: 32 x (size/8) x (size/8) floats, channel-major (the bn-normalised latent). */
	writeLatent(latent: Float32Array) {
		this.device.queue.writeBuffer(this.arena, this.latentAt * 4, latent);
	}

	/** Copy the latent from another GPU buffer (offset in floats). */
	copyLatent(enc: GPUCommandEncoder, src: GPUBuffer, offset: number) {
		const n = 32 * (this.size / 8) ** 2;
		enc.copyBufferToBuffer(src, offset * 4, this.arena, this.latentAt * 4, n * 4);
	}

	/** For checking: read part of the activation arena (offset in floats). */
	async debugRead(offset: number, count: number) {
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

	/** The recipe, for checking. */
	get recipe() {
		return this.steps.map((s) => ({ k: s.k, p: s.p, wg: s.wg }));
	}

	get hasEarly() {
		return this.earlySteps.length > 0;
	}

	/** Record the quick first-stage decode; the picture lands in earlyTexture (and as floats, see copyEarly). */
	decodeEarly(enc: GPUCommandEncoder) {
		const base = this.steps.length;
		const pass = enc.beginComputePass();
		this.earlySteps.forEach((st, i) => {
			pass.setPipeline(this.pipes[st.k]);
			pass.setBindGroup(0, this.bind, [(base + i) * 256]);
			pass.dispatchWorkgroups(...st.wg);
		});
		pass.end();
	}

	/**
	 * The full decode as scheduler tasks of a few milliseconds each (row slices of the big convolutions). stage, if
	 * given, is told when each stage's picture is ready: 128 and 256 (in stageTextures, with probes) and 512 (the
	 * decoded picture, in texture); record runs while the commands are recorded, done once they have run.
	 */
	decodeTasks(
		budget = 9,
		stage?: { record?: (enc: GPUCommandEncoder, res: number) => void; done?: (res: number) => void }
	): GpuTask[] {
		const base = this.steps.length + this.earlySteps.length;
		const cost = (st: Step) =>
			st.k === 'conv3'
				? (st.wg[0] * st.wg[1] * 64 * st.p[2] * st.p[3] * 18) / 2.5e8
				: st.k === 'conv3b' // 256 pixels a workgroup, about 4x the rate
					? (st.wg[0] * st.wg[1] * 256 * st.p[2] * st.p[3] * 18) / 1e9
					: st.k === 'conv1'
						? 0.5
						: 0.3;
		const tasks: GpuTask[] = [];
		let group: number[] = [],
			acc = 0;
		const flush = () => {
			if (!group.length) return;
			const idx = group;
			tasks.push({
				cost: acc,
				record: (enc) => {
					const pass = enc.beginComputePass();
					for (const i of idx) {
						const st = this.sliced[i];
						pass.setPipeline(this.pipes[st.k]);
						pass.setBindGroup(0, this.bind, [(base + i) * 256]);
						pass.dispatchWorkgroups(...st.wg);
					}
					pass.end();
				}
			});
			group = [];
			acc = 0;
		};
		this.sliced.forEach((st, i) => {
			const c = cost(st);
			if (acc + c > budget) flush();
			group.push(i);
			acc += c;
			if (stage && (st.k === 'probe_stage' || st.k === 'present')) {
				flush();
				const res = st.p[0];
				tasks.push({
					cost: 0.1,
					record: (enc) => stage.record?.(enc, res),
					done: () => stage.done?.(res)
				});
			}
		});
		flush();
		return tasks;
	}

	/** Copy the early picture's rgba floats (size/8 squared x 4) into another buffer. */
	copyEarly(enc: GPUCommandEncoder, dst: GPUBuffer, dstOffsetBytes: number) {
		enc.copyBufferToBuffer(
			this.arena,
			this.rgbAt * 4,
			dst,
			dstOffsetBytes,
			(this.size / 8) ** 2 * 16
		);
	}

	/** Record the decode into an encoder; the picture lands in this.texture. */
	decode(enc: GPUCommandEncoder) {
		const pass = enc.beginComputePass();
		this.steps.forEach((st, i) => {
			pass.setPipeline(this.pipes[st.k]);
			pass.setBindGroup(0, this.bind, [i * 256]);
			pass.dispatchWorkgroups(...st.wg);
		});
		pass.end();
	}
}
