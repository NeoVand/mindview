// HDR scene target, physically-inspired bloom (13-tap down / tent up), filmic tonemap to the canvas.
import { DEPTH_FORMAT, HDR_FORMAT, type GPU } from './gpu';

const FULLSCREEN = /* wgsl */ `
struct VO { @builtin(position) pos: vec4f, @location(0) uv: vec2f };
@vertex fn vs(@builtin(vertex_index) i: u32) -> VO {
  let p = vec2f(f32((i << 1u) & 2u), f32(i & 2u));
  var o: VO;
  o.pos = vec4f(p * 2.0 - 1.0, 0.0, 1.0);
  o.uv = vec2f(p.x, 1.0 - p.y);
  return o;
}
@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var samp: sampler;
@group(0) @binding(2) var<uniform> P: vec4f; // xy = source texel size, z = threshold (first pass) / radius, w = flag
`;

const DOWN = /* wgsl */ `${FULLSCREEN}
fn s(uv: vec2f) -> vec3f { return textureSampleLevel(src, samp, uv, 0.0).rgb; }
@fragment fn fs(i: VO) -> @location(0) vec4f {
  let t = P.xy;
  let a = s(i.uv + t * vec2f(-2, -2)); let b = s(i.uv + t * vec2f(0, -2)); let c = s(i.uv + t * vec2f(2, -2));
  let d = s(i.uv + t * vec2f(-2, 0));  let e = s(i.uv);                     let f = s(i.uv + t * vec2f(2, 0));
  let g = s(i.uv + t * vec2f(-2, 2));  let h = s(i.uv + t * vec2f(0, 2));  let k = s(i.uv + t * vec2f(2, 2));
  let j = s(i.uv + t * vec2f(-1, -1)); let l = s(i.uv + t * vec2f(1, -1));
  let m = s(i.uv + t * vec2f(-1, 1));  let n = s(i.uv + t * vec2f(1, 1));
  var col = e * 0.125 + (a + c + g + k) * 0.03125 + (b + d + f + h) * 0.0625 + (j + l + m + n) * 0.125;
  if (P.w > 0.5) { // first pass: soft threshold so only light that is really bright blooms
    let br = max(col.r, max(col.g, col.b));
    let knee = P.z * 0.5;
    var soft = clamp(br - P.z + knee, 0.0, 2.0 * knee);
    soft = soft * soft / (4.0 * knee + 1e-4);
    col *= max(soft, br - P.z) / max(br, 1e-4);
  }
  return vec4f(col, 1.0);
}`;

const UP = /* wgsl */ `${FULLSCREEN}
fn s(uv: vec2f) -> vec3f { return textureSampleLevel(src, samp, uv, 0.0).rgb; }
@fragment fn fs(i: VO) -> @location(0) vec4f {
  let t = P.xy * P.z;
  var col = s(i.uv) * 4.0;
  col += (s(i.uv + vec2f(-t.x, 0)) + s(i.uv + vec2f(t.x, 0)) + s(i.uv + vec2f(0, -t.y)) + s(i.uv + vec2f(0, t.y))) * 2.0;
  col += s(i.uv + vec2f(-t.x, -t.y)) + s(i.uv + vec2f(t.x, -t.y)) + s(i.uv + vec2f(-t.x, t.y)) + s(i.uv + vec2f(t.x, t.y));
  return vec4f(col / 16.0, 1.0);
}`;

const COMPOSITE = /* wgsl */ `
struct VO { @builtin(position) pos: vec4f, @location(0) uv: vec2f };
@vertex fn vs(@builtin(vertex_index) i: u32) -> VO {
  let p = vec2f(f32((i << 1u) & 2u), f32(i & 2u));
  var o: VO; o.pos = vec4f(p * 2.0 - 1.0, 0.0, 1.0); o.uv = vec2f(p.x, 1.0 - p.y); return o;
}
@group(0) @binding(0) var scene: texture_2d<f32>;
@group(0) @binding(1) var bloom: texture_2d<f32>;
@group(0) @binding(2) var samp: sampler;
@group(0) @binding(3) var<uniform> P: vec4f; // x = bloom strength, y = exposure, z = time, w = fade
fn aces(x: vec3f) -> vec3f {
  let a = 2.51; let b = 0.03; let c = 2.43; let d = 0.59; let e = 0.14;
  return clamp((x * (a * x + b)) / (x * (c * x + d) + e), vec3f(0.0), vec3f(1.0));
}
fn toSrgb(c: vec3f) -> vec3f {
  return select(1.055 * pow(c, vec3f(1.0 / 2.4)) - 0.055, c * 12.92, c <= vec3f(0.0031308));
}
fn hash(p: vec2f) -> f32 { return fract(sin(dot(p, vec2f(12.9898, 78.233))) * 43758.5453); }
@fragment fn fs(i: VO) -> @location(0) vec4f {
  let hdr = textureSampleLevel(scene, samp, i.uv, 0.0).rgb;
  let bl = textureSampleLevel(bloom, samp, i.uv, 0.0).rgb;
  var col = (hdr + bl * P.x) * P.y;
  let v = i.uv - 0.5;
  col *= 1.0 - dot(v, v) * 0.55;               // gentle vignette
  col = toSrgb(aces(col)) * P.w;
  col += (hash(i.pos.xy + P.z) - 0.5) / 255.0;  // dither the dark gradients
  return vec4f(col, 1.0);
}`;

export class Post {
	sceneView!: GPUTextureView;
	/** Depth for the scene pass (scenes that want it attach it themselves). */
	depthView!: GPUTextureView;
	private scene!: GPUTexture;
	private depth?: GPUTexture;
	private bloom!: GPUTexture;
	private mips: GPUTextureView[] = [];
	private downPipe: GPURenderPipeline;
	private upPipe: GPURenderPipeline;
	private compPipe: GPURenderPipeline;
	private sampler: GPUSampler;
	private passes: {
		pipe: GPURenderPipeline;
		bind: GPUBindGroup;
		target: GPUTextureView;
		load: GPULoadOp;
		uniform: GPUBuffer;
	}[] = [];
	private compBind!: GPUBindGroup;
	private compUniform: GPUBuffer;
	bloomStrength = 0.9;
	exposure = 1.0;
	fade = 1.0;

	constructor(private gpu: GPU) {
		const { device } = gpu;
		this.sampler = device.createSampler({
			magFilter: 'linear',
			minFilter: 'linear',
			addressModeU: 'clamp-to-edge',
			addressModeV: 'clamp-to-edge'
		});
		const mk = (code: string, format: GPUTextureFormat, blend?: GPUBlendState) =>
			device.createRenderPipeline({
				layout: 'auto',
				vertex: { module: device.createShaderModule({ code }), entryPoint: 'vs' },
				fragment: {
					module: device.createShaderModule({ code }),
					entryPoint: 'fs',
					targets: [{ format, blend }]
				},
				primitive: { topology: 'triangle-list' }
			});
		const add: GPUBlendState = {
			color: { srcFactor: 'one', dstFactor: 'one' },
			alpha: { srcFactor: 'one', dstFactor: 'one' }
		};
		this.downPipe = mk(DOWN, HDR_FORMAT);
		this.upPipe = mk(UP, HDR_FORMAT, add);
		this.compPipe = mk(COMPOSITE, gpu.format);
		this.compUniform = device.createBuffer({
			size: 16,
			usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
		});
	}

	resize(width: number, height: number) {
		const { device } = this.gpu;
		this.scene?.destroy();
		this.bloom?.destroy();
		this.passes.forEach((p) => p.uniform.destroy());
		const usage = GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING;
		this.scene = device.createTexture({ size: [width, height], format: HDR_FORMAT, usage });
		this.sceneView = this.scene.createView();
		this.depth?.destroy();
		this.depth = device.createTexture({
			size: [width, height],
			format: DEPTH_FORMAT,
			usage: GPUTextureUsage.RENDER_ATTACHMENT
		});
		this.depthView = this.depth.createView();
		const bw = Math.max(1, width >> 1),
			bh = Math.max(1, height >> 1);
		const levels = Math.max(1, Math.min(7, Math.floor(Math.log2(Math.min(bw, bh))) - 2));
		this.bloom = device.createTexture({
			size: [bw, bh],
			format: HDR_FORMAT,
			usage,
			mipLevelCount: levels
		});
		this.mips = Array.from({ length: levels }, (_, i) =>
			this.bloom.createView({ baseMipLevel: i, mipLevelCount: 1 })
		);
		this.passes = [];
		const pass = (
			pipe: GPURenderPipeline,
			src: GPUTextureView,
			target: GPUTextureView,
			params: number[],
			load: GPULoadOp
		) => {
			const uniform = device.createBuffer({
				size: 16,
				usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
			});
			device.queue.writeBuffer(uniform, 0, new Float32Array(params));
			const bind = device.createBindGroup({
				layout: pipe.getBindGroupLayout(0),
				entries: [
					{ binding: 0, resource: src },
					{ binding: 1, resource: this.sampler },
					{ binding: 2, resource: { buffer: uniform } }
				]
			});
			this.passes.push({ pipe, bind, target, load, uniform });
		};
		pass(this.downPipe, this.sceneView, this.mips[0], [1 / width, 1 / height, 1.0, 1], 'clear');
		for (let i = 1; i < levels; i++)
			pass(
				this.downPipe,
				this.mips[i - 1],
				this.mips[i],
				[1 / (bw >> (i - 1)), 1 / (bh >> (i - 1)), 0, 0],
				'clear'
			);
		for (let i = levels - 2; i >= 0; i--)
			pass(
				this.upPipe,
				this.mips[i + 1],
				this.mips[i],
				[1 / (bw >> (i + 1)), 1 / (bh >> (i + 1)), 1.0, 0],
				'load'
			);
		this.compBind = device.createBindGroup({
			layout: this.compPipe.getBindGroupLayout(0),
			entries: [
				{ binding: 0, resource: this.sceneView },
				{ binding: 1, resource: this.mips[0] },
				{ binding: 2, resource: this.sampler },
				{ binding: 3, resource: { buffer: this.compUniform } }
			]
		});
	}

	/** Run bloom + tonemap after the scene has been drawn into sceneView (timestamps: for the last pass). */
	finish(enc: GPUCommandEncoder, time: number, timestamps?: GPURenderPassTimestampWrites) {
		for (const p of this.passes) {
			const rp = enc.beginRenderPass({
				colorAttachments: [
					{ view: p.target, loadOp: p.load, storeOp: 'store', clearValue: [0, 0, 0, 1] }
				]
			});
			rp.setPipeline(p.pipe);
			rp.setBindGroup(0, p.bind);
			rp.draw(3);
			rp.end();
		}
		this.gpu.device.queue.writeBuffer(
			this.compUniform,
			0,
			new Float32Array([this.bloomStrength, this.exposure, time % 97, this.fade])
		);
		const rp = enc.beginRenderPass({
			colorAttachments: [
				{
					view: this.gpu.context.getCurrentTexture().createView(),
					loadOp: 'clear',
					storeOp: 'store',
					clearValue: [0, 0, 0, 1]
				}
			],
			timestampWrites: timestamps
		});
		rp.setPipeline(this.compPipe);
		rp.setBindGroup(0, this.compBind);
		rp.draw(3);
		rp.end();
	}
}
