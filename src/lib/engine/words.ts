// Billboarded, glowing words drawn from a TextAtlas. Instances are rewritten from the CPU each frame (a few hundred).
import { DEPTH_FORMAT, FRAME_WGSL } from './gpu';
import type { TextAtlas } from './text';

export interface WordInstance {
	word: string;
	pos: [number, number, number];
	height: number; // world units
	color: [number, number, number]; // linear rgb
	alpha: number;
	anchor?: number; // 0 = left edge at pos, 0.5 = centred (default)
}

const FLOATS = 16;

const WGSL = /* wgsl */ `${FRAME_WGSL}
struct Word { pos: vec3f, height: f32, uv: vec4f, aspect: f32, alpha: f32, anchor: f32, _p: f32, color: vec3f, _q: f32 };
@group(0) @binding(0) var<uniform> F: Frame;
@group(0) @binding(1) var<storage, read> W: array<Word>;
@group(0) @binding(2) var atlas: texture_2d<f32>;
@group(0) @binding(3) var samp: sampler;
struct VO { @builtin(position) pos: vec4f, @location(0) uv: vec2f, @location(1) col: vec4f };
@vertex fn vs(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VO {
  let w = W[ii];
  let corner = array<vec2f, 6>(vec2f(0,0), vec2f(1,0), vec2f(0,1), vec2f(0,1), vec2f(1,0), vec2f(1,1))[vi];
  let size = vec2f(w.height * w.aspect, w.height);
  let off = (corner - vec2f(w.anchor, 0.5)) * size;
  let p = w.pos + F.camRight.xyz * off.x + F.camUp.xyz * off.y;
  var o: VO;
  o.pos = F.viewProj * vec4f(p, 1.0);
  o.uv = vec2f(mix(w.uv.x, w.uv.z, corner.x), mix(w.uv.w, w.uv.y, corner.y));
  o.col = vec4f(w.color, w.alpha);
  return o;
}
@fragment fn fs(i: VO) -> @location(0) vec4f {
  let a = textureSample(atlas, samp, i.uv).r;
  return vec4f(i.col.rgb * a * i.col.a, 0.0);
}`;

export class WordLayer {
	private pipe: GPURenderPipeline;
	private buf: GPUBuffer;
	private bind: GPUBindGroup;
	private data: Float32Array;
	private count = 0;

	constructor(
		private device: GPUDevice,
		frame: GPUBuffer,
		private atlas: TextAtlas,
		private capacity: number,
		format: GPUTextureFormat,
		depth = false // the pass has a depth attachment: words hide behind what wrote depth
	) {
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
			},
			depthStencil: depth
				? { format: DEPTH_FORMAT, depthWriteEnabled: false, depthCompare: 'less' }
				: undefined
		});
		this.data = new Float32Array(capacity * FLOATS);
		this.buf = device.createBuffer({
			size: this.data.byteLength,
			usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
		});
		this.bind = device.createBindGroup({
			layout: this.pipe.getBindGroupLayout(0),
			entries: [
				{ binding: 0, resource: { buffer: frame } },
				{ binding: 1, resource: { buffer: this.buf } },
				{ binding: 2, resource: atlas.texture.createView() },
				{ binding: 3, resource: device.createSampler({ magFilter: 'linear', minFilter: 'linear' }) }
			]
		});
	}

	set(words: WordInstance[]) {
		let n = 0;
		for (const w of words) {
			if (n >= this.capacity || w.alpha <= 0.002) continue;
			const g = this.atlas.get(w.word);
			if (!g) continue;
			const o = n * FLOATS;
			this.data.set(
				[
					w.pos[0],
					w.pos[1],
					w.pos[2],
					w.height,
					g.u0,
					g.v0,
					g.u1,
					g.v1,
					g.aspect,
					w.alpha,
					w.anchor ?? 0.5,
					0,
					...w.color,
					0
				],
				o
			);
			n++;
		}
		this.count = n;
		if (n) this.device.queue.writeBuffer(this.buf, 0, this.data.buffer, 0, n * FLOATS * 4);
	}

	draw(pass: GPURenderPassEncoder) {
		if (!this.count) return;
		pass.setPipeline(this.pipe);
		pass.setBindGroup(0, this.bind);
		pass.draw(6, this.count);
	}
}
