// The painter's work as slices along the journey, in the order it happens. Every block leaves two: first its reading
// (where the picture's patches read each of your words, in the words' colours; attention happens inside the block),
// then its picture (what the block's output has in mind: tuned lens -> TAEF2's quick 64 x 64 decode). Slices are
// opaque, like pages: a later one covers the earlier ones behind it. They are shown up to a cut (the front of the
// computation, or wherever the visitor has ridden back to: nothing after it is drawn, so riding back uncovers the
// past) and dim a little with age. One instanced draw from two atlases.
import { DEPTH_FORMAT, FRAME_WGSL, HDR_FORMAT } from '$lib/engine/gpu';

export const PICTURE = 64;
export const READING = 32;
const ATLAS = 2048; // 1,024 pictures, 4,096 readings
const FLOATS = 16;

const WGSL = /* wgsl */ `${FRAME_WGSL}
struct Item { origin: vec3f, tick: f32, ax: vec3f, kind: f32, ay: vec3f, slot: f32, gain: f32, _a: f32, _b: f32, _c: f32 };
struct P { cut: f32, tau: f32, floor: f32, _p: f32 };
@group(0) @binding(0) var<uniform> F: Frame;
@group(0) @binding(1) var<uniform> U: P;
@group(0) @binding(2) var<storage, read> I: array<Item>;
@group(0) @binding(3) var pictures: texture_2d<f32>;
@group(0) @binding(4) var readings: texture_2d<f32>;
@group(0) @binding(5) var samp: sampler;
struct VO {
  @builtin(position) pos: vec4f, @location(0) c: vec2f, @location(1) uv: vec2f,
  @location(2) @interpolate(flat) ii: u32, @location(3) light: f32
};

@vertex fn vs(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VO {
  var o: VO;
  o.pos = vec4f(2.0, 2.0, 2.0, 1.0);
  o.ii = ii;
  let it = I[ii];
  let age = U.cut - it.tick;
  o.light = it.gain * max(U.floor, exp(-age / U.tau));
  if (age < -0.001 || o.light < 0.004) { return o; }
  let c = array<vec2f, 6>(vec2f(0,0), vec2f(1,0), vec2f(0,1), vec2f(0,1), vec2f(1,0), vec2f(1,1))[vi];
  o.pos = F.viewProj * vec4f(it.origin + it.ax * c.x + it.ay * c.y, 1.0);
  o.c = c;
  // seen from behind, mirror it so it still reads the right way round
  let behind = dot(cross(it.ax, it.ay), F.camPos.xyz - it.origin) > 0.0;
  o.uv = vec2f(select(c.x, 1.0 - c.x, behind), c.y);
  return o;
}

fn atlas(slot: u32, uv: vec2f, size: f32) -> vec2f {
  let per = u32(${ATLAS}.0 / size);
  let inner = clamp(uv, vec2f(0.5 / size), vec2f(1.0 - 0.5 / size));
  return (vec2f(f32(slot % per), f32(slot / per)) + inner) * size / ${ATLAS}.0;
}

@fragment fn fs(i: VO) -> @location(0) vec4f {
  let it = I[i.ii];
  let slot = u32(it.slot);
  var s: vec3f;
  if (it.kind < 0.5) {
    s = textureSampleLevel(pictures, samp, atlas(slot, i.uv, ${PICTURE}.0), 0.0).rgb;
  } else {
    s = textureSampleLevel(readings, samp, atlas(slot, i.uv, ${READING}.0), 0.0).rgb;
  }
  // rounded corners
  let q = max(abs(i.c - 0.5) - vec2f(0.44), vec2f(0.0));
  if (length(q) > 0.06) { discard; }
  return vec4f(pow(s, vec3f(2.2)) * i.light, 1.0);
}`;

export class Gallery {
	private pipe: GPURenderPipeline;
	private pictures: GPUTexture;
	private readings: GPUTexture;
	private params: GPUBuffer;
	private items: GPUBuffer;
	private bind: GPUBindGroup;
	private data: Float32Array;
	private count = 0;
	private dirty = false;
	private next = [0, 0];
	/** Ticks over which the past dims (to `floor`). */
	tau = 30;
	floor = 0.4;

	constructor(
		private device: GPUDevice,
		frame: GPUBuffer,
		private capacity: number
	) {
		const module = device.createShaderModule({ label: 'gallery', code: WGSL });
		this.pipe = device.createRenderPipeline({
			layout: 'auto',
			vertex: { module, entryPoint: 'vs' },
			fragment: {
				module,
				entryPoint: 'fs',
				targets: [{ format: HDR_FORMAT }]
			},
			depthStencil: { format: DEPTH_FORMAT, depthWriteEnabled: true, depthCompare: 'less' }
		});
		const usage = GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST;
		this.pictures = device.createTexture({ size: [ATLAS, ATLAS], format: 'rgba8unorm', usage });
		this.readings = device.createTexture({ size: [ATLAS, ATLAS], format: 'rgba8unorm', usage });
		this.params = device.createBuffer({
			size: 16,
			usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
		});
		this.data = new Float32Array(capacity * FLOATS);
		this.items = device.createBuffer({
			size: this.data.byteLength,
			usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
		});
		this.bind = device.createBindGroup({
			layout: this.pipe.getBindGroupLayout(0),
			entries: [
				{ binding: 0, resource: { buffer: frame } },
				{ binding: 1, resource: { buffer: this.params } },
				{ binding: 2, resource: { buffer: this.items } },
				{ binding: 3, resource: this.pictures.createView() },
				{ binding: 4, resource: this.readings.createView() },
				{
					binding: 5,
					resource: device.createSampler({
						magFilter: 'linear',
						minFilter: 'linear',
						addressModeU: 'clamp-to-edge',
						addressModeV: 'clamp-to-edge'
					})
				}
			]
		});
	}

	/** Reserve an atlas slot of the given kind (0 picture, 1 reading); -1 when full. */
	private slot(kind: number) {
		const per = ATLAS / (kind ? READING : PICTURE);
		if (this.next[kind] >= per * per || this.count >= this.capacity) return -1;
		const slot = this.next[kind]++;
		return slot;
	}

	private origin(kind: number, slot: number): [number, number] {
		const size = kind ? READING : PICTURE,
			per = ATLAS / size;
		return [(slot % per) * size, Math.floor(slot / per) * size];
	}

	private push(
		tick: number,
		kind: number,
		slot: number,
		corner: number[],
		ax: number[],
		ay: number[],
		gain: number
	) {
		this.data.set([...corner, tick, ...ax, kind, ...ay, slot, gain, 0, 0, 0], this.count * FLOATS);
		this.count++;
		this.dirty = true;
	}

	/**
	 * Record a copy of a 64 x 64 picture (rgba8, display sRGB) into the atlas and stand it at `corner` (its top left,
	 * spanned by ax / ay) from `tick` on. Call while recording the commands that make the picture.
	 */
	addPicture(
		enc: GPUCommandEncoder,
		src: GPUTexture,
		tick: number,
		corner: number[],
		ax: number[],
		ay: number[],
		gain = 1
	) {
		const slot = this.slot(0);
		if (slot < 0) return;
		enc.copyTextureToTexture(
			{ texture: src },
			{ texture: this.pictures, origin: this.origin(0, slot) },
			[PICTURE, PICTURE]
		);
		this.push(tick, 0, slot, corner, ax, ay, gain);
	}

	/** A 32 x 32 reading (rgba8, display sRGB) standing from `tick` on. */
	addReading(
		pixels: Uint8Array,
		tick: number,
		corner: number[],
		ax: number[],
		ay: number[],
		gain = 1
	) {
		const slot = this.slot(1);
		if (slot < 0) return;
		this.device.queue.writeTexture(
			{ texture: this.readings, origin: this.origin(1, slot) },
			pixels,
			{ bytesPerRow: READING * 4 },
			[READING, READING]
		);
		this.push(tick, 1, slot, corner, ax, ay, gain);
	}

	/** Draw every slice up to `cut` (in ticks), the older ones fainter. */
	draw(pass: GPURenderPassEncoder, cut: number) {
		if (!this.count) return;
		if (this.dirty) {
			this.device.queue.writeBuffer(this.items, 0, this.data, 0, this.count * FLOATS);
			this.dirty = false;
		}
		this.device.queue.writeBuffer(this.params, 0, new Float32Array([cut, this.tau, this.floor, 0]));
		pass.setPipeline(this.pipe);
		pass.setBindGroup(0, this.bind);
		pass.draw(6, this.count);
	}

	destroy() {
		this.pictures.destroy();
		this.readings.destroy();
		this.items.destroy();
		this.params.destroy();
	}
}
