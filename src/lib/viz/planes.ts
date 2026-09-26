// Pictures standing in the scene: textured quads. A print is opaque (with depth it hides whatever lies behind it); a
// glow is light (added to the scene, see-through where dark). Colours in the textures are display sRGB; they are
// linearised here.
import { DEPTH_FORMAT, FRAME_WGSL, HDR_FORMAT } from '$lib/engine/gpu';

const WGSL = /* wgsl */ `${FRAME_WGSL}
struct Plane { origin: vec3f, alpha: f32, ax: vec3f, gain: f32, ay: vec3f, shape: f32, glow: f32, _a: f32, _b: f32, _c: f32 };
@group(0) @binding(0) var<uniform> F: Frame;
@group(0) @binding(1) var<uniform> Q: Plane;
@group(0) @binding(2) var tex: texture_2d<f32>;
@group(0) @binding(3) var samp: sampler;
struct VO { @builtin(position) pos: vec4f, @location(0) uv: vec2f, @location(1) c: vec2f };
@vertex fn vs(@builtin(vertex_index) vi: u32) -> VO {
  let c = array<vec2f, 6>(vec2f(0,0), vec2f(1,0), vec2f(0,1), vec2f(0,1), vec2f(1,0), vec2f(1,1))[vi];
  var o: VO;
  o.pos = F.viewProj * vec4f(Q.origin + Q.ax * c.x + Q.ay * c.y, 1.0);
  // seen from behind, mirror it so the picture (and any writing in it) still reads the right way round
  let n = cross(Q.ax, Q.ay);
  let behind = dot(n, F.camPos.xyz - Q.origin) > 0.0;
  o.uv = vec2f(select(c.x, 1.0 - c.x, behind), c.y);
  o.c = c;
  return o;
}
// shape 0: a sharp square; 1: a disc with a soft rim; 2: a square whose edges fade out
@fragment fn fs(i: VO) -> @location(0) vec4f {
  let s = textureSample(tex, samp, i.uv).rgb;
  let d = abs(i.c - 0.5);
  var mask = 1.0;
  if (Q.shape > 1.5) { mask = smoothstep(0.5, 0.42, max(d.x, d.y)); }
  else if (Q.shape > 0.5) { mask = smoothstep(0.5, 0.45, length(i.c - 0.5)); }
  let a = mask * Q.alpha;
  if (Q.glow > 0.5) { return vec4f(pow(s, vec3f(2.2)) * Q.gain * a, 0.0); }
  if (a < 0.5) { discard; }
  return vec4f(pow(s, vec3f(2.2)) * Q.gain * a, a);
}`;

interface PlaneState {
	bind: GPUBindGroup;
	origin: [number, number, number];
	ax: [number, number, number];
	ay: [number, number, number];
	alpha: number;
	gain: number;
	shape: number;
	glow: boolean;
}

const STRIDE = 256;
const MAX = 256;

export class ImagePlanes {
	private pipe: GPURenderPipeline;
	private glowPipe: GPURenderPipeline;
	private layout: GPUBindGroupLayout;
	private uniforms: GPUBuffer;
	private sampler: GPUSampler;
	private planes: PlaneState[] = [];

	constructor(
		private device: GPUDevice,
		private frame: GPUBuffer,
		depth = false
	) {
		const module = device.createShaderModule({ label: 'image planes', code: WGSL });
		this.layout = device.createBindGroupLayout({
			entries: [
				{
					binding: 0,
					visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
					buffer: { type: 'uniform' }
				},
				{
					binding: 1,
					visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
					buffer: { type: 'uniform', hasDynamicOffset: true, minBindingSize: 64 }
				},
				{ binding: 2, visibility: GPUShaderStage.FRAGMENT, texture: {} },
				{ binding: 3, visibility: GPUShaderStage.FRAGMENT, sampler: {} }
			]
		});
		const make = (glow: boolean) =>
			device.createRenderPipeline({
				layout: device.createPipelineLayout({ bindGroupLayouts: [this.layout] }),
				vertex: { module, entryPoint: 'vs' },
				fragment: {
					module,
					entryPoint: 'fs',
					targets: [
						{
							format: HDR_FORMAT,
							blend: glow
								? {
										color: { srcFactor: 'one', dstFactor: 'one' },
										alpha: { srcFactor: 'one', dstFactor: 'one' }
									}
								: {
										color: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha' },
										alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha' }
									}
						}
					]
				},
				depthStencil: depth
					? { format: DEPTH_FORMAT, depthWriteEnabled: !glow, depthCompare: 'less' }
					: undefined
			});
		this.pipe = make(false);
		this.glowPipe = make(true);
		this.uniforms = device.createBuffer({
			size: MAX * STRIDE,
			usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
		});
		this.sampler = device.createSampler({ magFilter: 'linear', minFilter: 'linear' });
	}

	/** Add a picture; returns its index. */
	add(texture: GPUTexture): number {
		if (this.planes.length >= MAX) throw new Error('Too many pictures.');
		const i = this.planes.length;
		this.planes.push({
			bind: this.device.createBindGroup({
				layout: this.layout,
				entries: [
					{ binding: 0, resource: { buffer: this.frame } },
					{ binding: 1, resource: { buffer: this.uniforms, size: 64 } },
					{ binding: 2, resource: texture.createView() },
					{ binding: 3, resource: this.sampler }
				]
			}),
			origin: [0, 0, 0],
			ax: [1, 0, 0],
			ay: [0, 1, 0],
			alpha: 0,
			gain: 1,
			shape: 0,
			glow: false
		});
		return i;
	}

	/**
	 * Place picture i: origin is the corner where uv = (0, 0) (the picture's top left), ax / ay span it. shape: 0 a
	 * sharp square, 1 a disc, 2 a square with soft edges. glow: light rather than a print.
	 */
	place(
		i: number,
		origin: [number, number, number],
		ax: [number, number, number],
		ay: [number, number, number],
		alpha: number,
		gain = 1,
		shape = 0,
		glow = false
	) {
		Object.assign(this.planes[i], { origin, ax, ay, alpha, gain, shape, glow });
	}

	clear() {
		this.planes = [];
	}

	/** Draw the placed pictures; eye (the camera) orders them nearest first. */
	draw(pass: GPURenderPassEncoder, eye?: [number, number, number]) {
		if (!this.planes.length) return;
		const data = new Float32Array((this.planes.length * STRIDE) / 4);
		this.planes.forEach((p, i) =>
			data.set(
				[...p.origin, p.alpha, ...p.ax, p.gain, ...p.ay, p.shape, p.glow ? 1 : 0],
				(i * STRIDE) / 4
			)
		);
		this.device.queue.writeBuffer(this.uniforms, 0, data);
		const centre = (p: PlaneState) =>
			[0, 1, 2].map((k) => p.origin[k] + (p.ax[k] + p.ay[k]) / 2) as [number, number, number];
		const order = this.planes.map((_, i) => i).filter((i) => this.planes[i].alpha > 0);
		if (eye) {
			const d = order.map((i) => {
				const c = centre(this.planes[i]);
				return (c[0] - eye[0]) ** 2 + (c[1] - eye[1]) ** 2 + (c[2] - eye[2]) ** 2;
			});
			const rank = new Map(order.map((i, k) => [i, d[k]]));
			order.sort((a, b) => rank.get(a)! - rank.get(b)!);
		}
		// prints first (nearest first, so nearer ones hide farther ones), then the glows over them
		for (const glow of [false, true]) {
			pass.setPipeline(glow ? this.glowPipe : this.pipe);
			for (const i of order) {
				if (this.planes[i].glow !== glow) continue;
				pass.setBindGroup(0, this.planes[i].bind, [i * STRIDE]);
				pass.draw(6);
			}
		}
	}
}
