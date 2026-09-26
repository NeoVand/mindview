// Act I, Reading: the language model's residual stream as a loom. Every token is a column; 256 of its
// hidden dimensions rise through all layers as threads, displaced by their value and coloured by sign.
import { FRAME_WGSL, storageBuffer } from '../gpu';
import type { Trace } from '../trace';
import { clean } from '../text';
import type { WordInstance } from '../words';

export const LOOM = { sx: 0.62, sy: 0.36, depth: 1.8 };

const WGSL = /* wgsl */ `${FRAME_WGSL}
struct Loom { n: u32, layers: u32, dims: u32, _a: u32, tokenReveal: f32, layerFront: f32, tapGlow: f32, fade: f32, taps: vec4f, sx: f32, sy: f32, depth: f32, _b: f32 };
@group(0) @binding(0) var<uniform> F: Frame;
@group(0) @binding(1) var<uniform> U: Loom;
@group(0) @binding(2) var<storage, read> TH: array<f32>; // [layer][token][dim]
struct VO { @builtin(position) pos: vec4f, @location(0) col: vec3f, @location(1) edge: f32 };

fn value(l: u32, t: u32, d: u32) -> f32 { return TH[(l * U.n + t) * U.dims + d]; }
fn point(l: u32, t: u32, d: u32) -> vec3f {
  let v = clamp(value(l, t, d), -4.0, 4.0);
  let x = (f32(t) - f32(U.n - 1u) * 0.5) * U.sx + v * 0.05;
  return vec3f(x, f32(l) * U.sy, (f32(d) / f32(U.dims - 1u) - 0.5) * U.depth);
}

@vertex fn vs(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VO {
  let segs = U.layers - 1u;
  let l = ii % segs;
  let d = (ii / segs) % U.dims;
  let t = ii / (segs * U.dims);
  let a = point(l, t, d);
  let b = point(l + 1u, t, d);
  let side = array<f32, 6>(-1, 1, -1, -1, 1, 1)[vi];
  let along = array<f32, 6>(0, 0, 1, 1, 0, 1)[vi];
  let p = mix(a, b, along);
  let dir = normalize(b - a);
  let view = normalize(F.camPos.xyz - p);
  let perp = normalize(cross(dir, view));
  let dist = length(F.camPos.xyz - p);
  let width = 1.1 * F.pixel * dist;
  var o: VO;
  o.pos = F.viewProj * vec4f(p + perp * side * width, 1.0);
  let v = mix(value(l, t, d), value(l + 1u, t, d), along);
  // reveal: tokens enter left to right, then the front climbs through the layers
  let tokenOn = clamp(U.tokenReveal - f32(t), 0.0, 1.0);
  let y = f32(l) + along;
  let front = clamp((U.layerFront - y) * 1.5, 0.0, 1.0);
  let tipGlow = exp(-abs(U.layerFront - y) * 2.0) * 1.5;
  var tap = 0.0;
  for (var k = 0u; k < 3u; k++) { tap += exp(-abs(y - U.taps[k]) * 6.0); }
  let mag = min(abs(v), 3.0) / 3.0;
  // most dimensions whisper; the few large ones carry the light
  let strong = mag * mag;
  let intensity = (0.006 + strong * 0.3) * front * tokenOn + tipGlow * strong * tokenOn * step(y, U.layerFront + 0.3) + tap * U.tapGlow * (0.04 + strong);
  o.col = signColor(v) * intensity * U.fade;
  o.edge = side;
  return o;
}
@fragment fn fs(i: VO) -> @location(0) vec4f {
  let soft = 1.0 - i.edge * i.edge;
  return vec4f(i.col * soft * 0.3, 0.0);
}`;

export interface ReadingState {
	tokenReveal: number; // how many tokens have entered (float)
	layerFront: number; // highest layer reached (float)
	tapGlow: number;
	fade: number;
}

export class Reading {
	private pipe: GPURenderPipeline;
	private bind: GPUBindGroup;
	private uniform: GPUBuffer;
	private instances: number;
	readonly n: number;
	readonly layers: number;
	readonly taps: number[];
	private tokens: string[];
	private lens: Trace['manifest']['act1']['lens'];

	constructor(
		private device: GPUDevice,
		frame: GPUBuffer,
		trace: Trace,
		format: GPUTextureFormat
	) {
		const th = trace.arrays.enc_threads;
		const [layers, n, dims] = th.shape;
		this.n = n;
		this.layers = layers;
		this.taps = trace.manifest.act1.taps;
		this.tokens = trace.manifest.act1.tokens;
		this.lens = trace.manifest.act1.lens;
		this.instances = n * dims * (layers - 1);
		const module = device.createShaderModule({ code: WGSL });
		this.pipe = device.createRenderPipeline({
			layout: 'auto',
			vertex: { module, entryPoint: 'vs' },
			fragment: {
				module,
				entryPoint: 'fs',
				targets: [
					{
						format,
						blend: {
							color: { srcFactor: 'one', dstFactor: 'one' },
							alpha: { srcFactor: 'one', dstFactor: 'one' }
						}
					}
				]
			}
		});
		this.uniform = device.createBuffer({
			size: 64,
			usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
		});
		const u = new ArrayBuffer(64);
		new Uint32Array(u, 0, 4).set([n, layers, dims, 0]);
		device.queue.writeBuffer(this.uniform, 0, u);
		this.bind = device.createBindGroup({
			layout: this.pipe.getBindGroupLayout(0),
			entries: [
				{ binding: 0, resource: { buffer: frame } },
				{ binding: 1, resource: { buffer: this.uniform } },
				{ binding: 2, resource: { buffer: storageBuffer(device, th.data, 'threads') } }
			]
		});
	}

	/** World position of a token column at a given layer (used by the handoff and the camera). */
	anchor(token: number, layer: number): [number, number, number] {
		return [(token - (this.n - 1) / 2) * LOOM.sx, layer * LOOM.sy, 0];
	}

	update(s: ReadingState) {
		const f = new Float32Array([
			s.tokenReveal,
			s.layerFront,
			s.tapGlow,
			s.fade,
			...this.taps,
			0,
			LOOM.sx,
			LOOM.sy,
			LOOM.depth,
			0
		]);
		this.device.queue.writeBuffer(this.uniform, 16, f);
	}

	/** The words: each input token under its column, and the logit-lens guess at a few depths. */
	words(s: ReadingState): WordInstance[] {
		const out: WordInstance[] = [];
		const bone: [number, number, number] = [0.807, 0.761, 0.672];
		const dim: [number, number, number] = [0.35, 0.37, 0.4];
		for (let t = 0; t < this.n; t++) {
			const on = Math.min(1, Math.max(0, s.tokenReveal - t)) * s.fade;
			const text = clean(this.tokens[t]);
			const frame = /^<\|.*\|>$|^<\/?think>$|^(user|assistant)$/.test(text);
			const [x] = this.anchor(t, 0);
			out.push({
				word: text,
				pos: [x, -0.5, LOOM.depth / 2],
				height: 0.3,
				color: frame ? dim : bone,
				alpha: on * (frame ? 0.5 : 1.6)
			});
			if (frame) continue;
			for (let l = 4; l < this.layers; l += 4) {
				const [w, p] = this.lens[l][t][0];
				const reached = Math.min(1, Math.max(0, (s.layerFront - l) * 0.8));
				const last = l >= this.layers - 1;
				out.push({
					word: w,
					pos: [x, l * LOOM.sy, LOOM.depth / 2 + 0.05],
					height: last ? 0.26 : 0.19,
					color: last ? bone : [0.62, 0.64, 0.68],
					alpha: on * reached * (0.45 + Math.sqrt(p) * (last ? 1.6 : 1.1))
				});
			}
		}
		return out;
	}

	/** Every string the words() pass may ask for, for the atlas. */
	allWords(): string[] {
		const ws = this.tokens.map(clean);
		for (let l = 4; l < this.layers; l += 4)
			for (let t = 0; t < this.n; t++) ws.push(this.lens[l][t][0][0]);
		return ws;
	}

	draw(pass: GPURenderPassEncoder) {
		pass.setPipeline(this.pipe);
		pass.setBindGroup(0, this.bind);
		pass.draw(6, this.instances);
	}
}
