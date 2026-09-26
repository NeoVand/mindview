// Glowing polylines for millions of tiny segments: each vertex carries a position, the moment it is reached (w), and a
// colour index. Segment i joins vertex i to i+1 unless the two belong to different strands. Drawn as screen-space
// quads with additive blending, so overlapping light adds up and a line's brightness per pixel stays constant.
import { DEPTH_FORMAT, FRAME_WGSL, HDR_FORMAT } from '$lib/engine/gpu';

const WGSL = /* wgsl */ `${FRAME_WGSL}
struct Params { now: f32, width: f32, gain: f32, fresh: f32, vw: f32, vh: f32, fade: f32, _p: f32 };
@group(0) @binding(0) var<uniform> F: Frame;
@group(0) @binding(1) var<uniform> U: Params;
@group(0) @binding(2) var<storage, read> P: array<vec4f>;
@group(0) @binding(3) var<storage, read> ATTR: array<u32>; // strand (16 bits) | colour (8 bits) << 16 | level (8 bits) << 24
@group(0) @binding(4) var<storage, read> PAL: array<vec4f>;
@group(0) @binding(5) var<storage, read> COL: array<vec4f>; // per-vertex colours, used where the colour slot is 254
struct VO { @builtin(position) pos: vec4f, @location(0) col: vec3f, @location(1) side: f32, @location(2) w: f32 };

@vertex fn vs(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VO {
  var o: VO;
  o.pos = vec4f(2.0, 2.0, 2.0, 1.0);
  o.col = vec3f(0.0);
  o.side = 0.0;
  let a = P[ii];
  let b = P[ii + 1u];
  let ta = ATTR[ii];
  let tb = ATTR[ii + 1u];
  if ((ta & 0xffffu) != (tb & 0xffffu) || b.w > U.now) { return o; }
  let ca = F.viewProj * vec4f(a.xyz, 1.0);
  let cb = F.viewProj * vec4f(b.xyz, 1.0);
  if (ca.w <= 0.0 || cb.w <= 0.0) { return o; }
  let sa = ca.xy / ca.w;
  let sb = cb.xy / cb.w;
  let px = vec2f(U.vw, U.vh) * 0.5;
  var d = (sb - sa) * px;
  let len = length(d);
  d = select(vec2f(1.0, 0.0), d / len, len > 1e-6);
  // a box of half-width w pixels, filtered over one pixel, so a line keeps the same brightness wherever it falls
  // between pixel rows (thinner lines are drawn at w = 0.6 and dimmed instead, which reads as thinner without beading)
  let w = max(U.width, 0.6);
  let hw = w + 1.0;
  let n = vec2f(-d.y, d.x) * hw / px;
  let corner = array<vec2f, 6>(vec2f(0,-1), vec2f(1,-1), vec2f(0,1), vec2f(0,1), vec2f(1,-1), vec2f(1,1))[vi];
  let c = mix(ca, cb, corner.x);
  o.pos = vec4f(c.xy + n * corner.y * c.w, c.z, c.w);
  o.side = corner.y * hw;
  o.w = w;
  let age = U.now - b.w;
  let fresh = 1.0 + U.fresh * exp(-age * 12.0);
  let level = f32(tb >> 24u) / 255.0;
  let slot = (tb >> 16u) & 0xffu;
  var base = PAL[slot].rgb;
  if (slot == 254u) { base = COL[ii + 1u].rgb; }
  o.col = base * U.gain * fresh * (0.25 + 0.75 * level) * mix(1.0, exp(-age * 0.35), U.fade)
    * min(1.0, U.width / 0.6) * 0.45;
  return o;
}

@fragment fn fs(i: VO) -> @location(0) vec4f {
  let cover = clamp(i.w + 0.5 - abs(i.side), 0.0, 1.0);
  return vec4f(i.col * cover, 0.0);
}`;

export class LineLayer {
	private pipe: GPURenderPipeline;
	private params: GPUBuffer;
	private bind?: GPUBindGroup;
	private layout: GPUBindGroupLayout;
	private count = 0;
	private buffers: GPUBuffer[] = [];
	now = 1e9;
	width = 1.0; // pixels (half width)
	gain = 1;
	fresh = 0; // extra brightness for segments just drawn
	fade = 0; // 0..1: how much older segments dim

	/** depth: the pass has a depth attachment; lines are hidden behind what wrote depth (they write none). */
	constructor(
		private device: GPUDevice,
		private frame: GPUBuffer,
		depth = false
	) {
		const module = device.createShaderModule({ label: 'lines', code: WGSL });
		const ro = (binding: number): GPUBindGroupLayoutEntry => ({
			binding,
			visibility: GPUShaderStage.VERTEX,
			buffer: { type: 'read-only-storage' }
		});
		this.layout = device.createBindGroupLayout({
			entries: [
				{
					binding: 0,
					visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
					buffer: { type: 'uniform' }
				},
				{ binding: 1, visibility: GPUShaderStage.VERTEX, buffer: { type: 'uniform' } },
				ro(2),
				ro(3),
				ro(4),
				ro(5)
			]
		});
		this.pipe = device.createRenderPipeline({
			layout: device.createPipelineLayout({ bindGroupLayouts: [this.layout] }),
			vertex: { module, entryPoint: 'vs' },
			fragment: {
				module,
				entryPoint: 'fs',
				targets: [
					{
						format: HDR_FORMAT,
						blend: {
							color: { srcFactor: 'one', dstFactor: 'one' },
							alpha: { srcFactor: 'one', dstFactor: 'one' }
						}
					}
				]
			},
			primitive: { topology: 'triangle-list' },
			depthStencil: depth
				? { format: DEPTH_FORMAT, depthWriteEnabled: false, depthCompare: 'less' }
				: undefined
		});
		this.params = device.createBuffer({
			size: 32,
			usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
		});
	}

	/**
	 * points: xyzw per vertex (w = when it is reached); attr: strand | colour << 16 | level << 24; palette: rgba.
	 * colours (optional): rgba per vertex, used by vertices whose colour slot is 254.
	 */
	set(points: Float32Array, attr: Uint32Array, palette: Float32Array, colours?: Float32Array) {
		for (const b of this.buffers) b.destroy();
		const make = (data: Float32Array | Uint32Array) => {
			const buf = this.device.createBuffer({
				size: Math.max(16, data.byteLength),
				usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
			});
			this.device.queue.writeBuffer(buf, 0, data.buffer, data.byteOffset, data.byteLength);
			return buf;
		};
		const p = make(points),
			a = make(attr),
			c = make(palette),
			v = make(colours ?? new Float32Array(4));
		this.buffers = [p, a, c, v];
		this.count = Math.max(0, points.length / 4 - 1);
		this.bind = this.device.createBindGroup({
			layout: this.layout,
			entries: [
				{ binding: 0, resource: { buffer: this.frame } },
				{ binding: 1, resource: { buffer: this.params } },
				{ binding: 2, resource: { buffer: p } },
				{ binding: 3, resource: { buffer: a } },
				{ binding: 4, resource: { buffer: c } },
				{ binding: 5, resource: { buffer: v } }
			]
		});
	}

	/** Replace the points given to set() (same length); cheap, no new buffers. */
	setPoints(points: Float32Array) {
		if (this.buffers[0]) this.device.queue.writeBuffer(this.buffers[0], 0, points);
	}

	/** Replace the per-vertex colours given to set() (same length); cheap, no new buffers. */
	setColours(colours: Float32Array) {
		if (this.buffers[3]) this.device.queue.writeBuffer(this.buffers[3], 0, colours);
	}

	draw(pass: GPURenderPassEncoder, viewW: number, viewH: number) {
		if (!this.bind || !this.count) return;
		this.device.queue.writeBuffer(
			this.params,
			0,
			new Float32Array([this.now, this.width, this.gain, this.fresh, viewW, viewH, this.fade, 0])
		);
		pass.setPipeline(this.pipe);
		pass.setBindGroup(0, this.bind);
		pass.draw(6, this.count);
	}

	destroy() {
		for (const b of this.buffers) b.destroy();
		this.params.destroy();
	}
}
