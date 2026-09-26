// Act II, Handoff: the three tapped layers leave the loom as particles and fly to the painter.
// The 29 real prompt positions carry bright light; the ~480 padding slots follow as faint dust
// (they are never masked, and the image still attends to them).
import { FRAME_WGSL, storageBuffer } from '../gpu';

const WGSL = /* wgsl */ `${FRAME_WGSL}
struct P { a: vec4f, b: vec4f, c: vec4f, d: vec4f }; // start.xyz + delay, ctrl1.xyz + brightness, ctrl2.xyz + tap, end.xyz + size
@group(0) @binding(0) var<uniform> F: Frame;
@group(0) @binding(1) var<uniform> U: vec4f; // x = progress 0..1, y = fade
@group(0) @binding(2) var<storage, read> PS: array<P>;
struct VO { @builtin(position) pos: vec4f, @location(0) col: vec3f, @location(1) uv: vec2f };
fn bez(p0: vec3f, p1: vec3f, p2: vec3f, p3: vec3f, t: f32) -> vec3f {
  let u = 1.0 - t;
  return u*u*u*p0 + 3.0*u*u*t*p1 + 3.0*u*t*t*p2 + t*t*t*p3;
}
@vertex fn vs(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VO {
  let q = PS[ii];
  let t = clamp((U.x - q.a.w) / 0.55, 0.0, 1.0);
  let e = t * t * (3.0 - 2.0 * t);
  let p = bez(q.a.xyz, q.b.xyz, q.c.xyz, q.d.xyz, e);
  let corner = array<vec2f, 6>(vec2f(-1,-1), vec2f(1,-1), vec2f(-1,1), vec2f(-1,1), vec2f(1,-1), vec2f(1,1))[vi];
  let size = q.d.w * (1.0 + sin(e * 3.14159) * 1.5);
  var o: VO;
  o.pos = F.viewProj * vec4f(p + (F.camRight.xyz * corner.x + F.camUp.xyz * corner.y) * size, 1.0);
  let tapCol = array<vec3f, 3>(MINUS, BONE, PLUS)[u32(q.c.w)];
  let alive = smoothstep(0.0, 0.05, t) * (1.0 - smoothstep(0.92, 1.0, t));
  o.col = tapCol * q.b.w * alive * U.y;
  o.uv = corner;
  return o;
}
@fragment fn fs(i: VO) -> @location(0) vec4f {
  let r = dot(i.uv, i.uv);
  return vec4f(i.col * exp(-r * 4.0), 0.0);
}`;

export class Handoff {
	private pipe: GPURenderPipeline;
	private bind: GPUBindGroup;
	private uniform: GPUBuffer;
	private count: number;

	constructor(
		private device: GPUDevice,
		frame: GPUBuffer,
		format: GPUTextureFormat,
		starts: { pos: [number, number, number]; tap: number; real: boolean; strength: number }[],
		end: (i: number, real: boolean) => [number, number, number]
	) {
		const data = new Float32Array(starts.length * 16);
		starts.forEach((s, i) => {
			const e = end(i, s.real);
			const mid = [
				(s.pos[0] + e[0]) / 2,
				Math.max(s.pos[1], e[1]) + 3 + Math.random() * 2,
				(s.pos[2] + e[2]) / 2
			];
			const delay = s.real ? 0.05 + (i % 29) * 0.008 + s.tap * 0.04 : 0.15 + Math.random() * 0.3;
			data.set(
				[
					...s.pos,
					delay,
					mid[0] + (Math.random() - 0.5) * 4,
					mid[1],
					s.pos[2] + (e[2] - s.pos[2]) * 0.25,
					s.real ? 1.4 * s.strength : 0.05 * s.strength,
					mid[0] + (Math.random() - 0.5) * 2,
					mid[1] * 0.7 + e[1] * 0.3,
					s.pos[2] + (e[2] - s.pos[2]) * 0.75,
					s.tap,
					...e,
					s.real ? 0.07 : 0.025
				],
				i * 16
			);
		});
		this.count = starts.length;
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
			size: 16,
			usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
		});
		this.bind = device.createBindGroup({
			layout: this.pipe.getBindGroupLayout(0),
			entries: [
				{ binding: 0, resource: { buffer: frame } },
				{ binding: 1, resource: { buffer: this.uniform } },
				{ binding: 2, resource: { buffer: storageBuffer(device, data, 'handoff particles') } }
			]
		});
	}

	update(progress: number, fade: number) {
		this.device.queue.writeBuffer(this.uniform, 0, new Float32Array([progress, fade, 0, 0]));
	}

	draw(pass: GPURenderPassEncoder) {
		pass.setPipeline(this.pipe);
		pass.setBindGroup(0, this.bind);
		pass.draw(6, this.count);
	}
}
