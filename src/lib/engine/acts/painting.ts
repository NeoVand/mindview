// Act III, Painting: the diffusion transformer as a tower of 25 translucent planes (one per block).
// Each plane is the 32x32 lattice of image patches, coloured by that block's own principal components.
// A scan runs through the blocks at every denoising step; the focused prompt word casts beams onto the
// patches it is attending to; the canvas at the end shows what the image would be if the network stopped here.
import { FRAME_WGSL, storageBuffer, textureArrayFromBitmaps } from '../gpu';
import type { Trace } from '../trace';
import { clean } from '../text';
import type { WordInstance } from '../words';

export const TOWER = {
	x: 0,
	y: 4.2,
	z0: -26,
	dz: 1.05,
	size: 3.4,
	canvasGap: 4,
	canvasSize: 7.5
};

const COMMON = /* wgsl */ `${FRAME_WGSL}
struct Tower { steps: u32, blocks: u32, grid: u32, nText: u32,
  step: f32, scan: f32, fade: f32, palette: f32,
  origin: vec4f,        // x, y, z0, dz
  size: f32, focus: f32, beam: f32, _a: f32 };
@group(0) @binding(0) var<uniform> F: Frame;
@group(0) @binding(1) var<uniform> U: Tower;
fn planeZ(b: f32) -> f32 { return U.origin.z - b * U.origin.w; }
fn patchPos(b: u32, p: u32) -> vec3f {
  let g = f32(U.grid);
  let ij = vec2f(f32(p % U.grid), f32(p / U.grid));
  let xy = (ij + 0.5) / g - 0.5;
  return vec3f(U.origin.x + xy.x * U.size, U.origin.y - xy.y * U.size, planeZ(f32(b)));
}
fn planeLight(b: f32) -> f32 {
  let d = U.scan - b;
  // blocks already computed at this step stay lit; the block being computed flares; the rest wait in the dark
  return select(0.025, 0.06 + 0.3 * exp(-d * 0.45), d >= 0.0) + 1.3 * exp(-d * d * 5.0);
}
`;

const PATCHES = /* wgsl */ `${COMMON}
@group(0) @binding(2) var<storage, read> C: array<vec4f>; // [step][block][patch] rgb (pca, normalised) + activity
struct VO { @builtin(position) pos: vec4f, @location(0) col: vec3f, @location(1) uv: vec2f };
@vertex fn vs(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VO {
  let n = U.grid * U.grid;
  let b = ii / n;
  let p = ii % n;
  let s = u32(clamp(U.step, 0.0, f32(U.steps - 1u)));
  let c = C[(s * U.blocks + b) * n + p];
  let corner = array<vec2f, 6>(vec2f(-1,-1), vec2f(1,-1), vec2f(-1,1), vec2f(-1,1), vec2f(1,-1), vec2f(1,1))[vi];
  let hs = U.size / f32(U.grid) * 0.46;
  let pos = patchPos(b, p) + vec3f(corner * hs, 0.0);
  var o: VO;
  o.pos = F.viewProj * vec4f(pos, 1.0);
  var rgb = c.rgb;
  if (U.palette < 0.5) {
    // ternary duotone: first component runs glacier (-) to ember (+), second sets brightness
    let t = c.r * 2.0 - 1.0;
    rgb = mix(MINUS, PLUS, c.r) * (0.25 + c.g) * (0.35 + abs(t));
  }
  o.col = rgb * planeLight(f32(b)) * (0.35 + c.a) * U.fade;
  o.uv = corner;
  return o;
}
@fragment fn fs(i: VO) -> @location(0) vec4f {
  let e = max(abs(i.uv.x), abs(i.uv.y));
  let soft = 1.0 - smoothstep(0.7, 1.0, e);
  return vec4f(i.col * soft * 0.55, 0.0);
}`;

const BEAMS = /* wgsl */ `${COMMON}
@group(0) @binding(2) var<storage, read> A: array<f32>; // [step][block][patch][word], attention normalised per (step, block, word)
@group(0) @binding(3) var<uniform> Src: vec4f;         // world position of the focused word
struct VO { @builtin(position) pos: vec4f, @location(0) col: vec3f, @location(1) edge: f32 };
@vertex fn vs(@builtin(vertex_index) vi: u32, @builtin(instance_index) p: u32) -> VO {
  let n = U.grid * U.grid;
  let s = u32(clamp(U.step, 0.0, f32(U.steps - 1u)));
  let b = u32(clamp(floor(U.scan), 0.0, f32(U.blocks - 1u)));
  let w = u32(U.focus);
  let a = A[((s * U.blocks + b) * n + p) * U.nText + w];
  let dst = patchPos(b, p);
  let side = array<f32, 6>(-1, 1, -1, -1, 1, 1)[vi];
  let along = array<f32, 6>(0, 0, 1, 1, 0, 1)[vi];
  let pos = mix(Src.xyz, dst, along);
  let dir = normalize(dst - Src.xyz);
  let perp = normalize(cross(dir, normalize(F.camPos.xyz - pos)));
  let width = 0.9 * F.pixel * length(F.camPos.xyz - pos);
  var o: VO;
  o.pos = F.viewProj * vec4f(pos + perp * side * width, 1.0);
  let strength = pow(a, 2.2) * U.beam * U.fade;
  o.col = mix(BONE, PLUS, along) * strength * (0.25 + along * 0.9);
  o.edge = side;
  return o;
}
@fragment fn fs(i: VO) -> @location(0) vec4f { return vec4f(i.col * (1.0 - i.edge * i.edge), 0.0); }`;

const CANVAS = /* wgsl */ `${COMMON}
@group(0) @binding(2) var lens: texture_2d_array<f32>;  // tuned-lens readout of every (step, block), TAEF2-decoded
@group(0) @binding(3) var dec: texture_2d_array<f32>;   // the VAE decoder's stages, coarse to fine, ending in the image
@group(0) @binding(4) var samp: sampler;
struct Canvas { lensLayer: f32, decAmount: f32, decA: f32, decB: f32, decT: f32, size: f32, opacity: f32, _p: f32 };
@group(0) @binding(5) var<uniform> V: Canvas;
struct VO { @builtin(position) pos: vec4f, @location(0) uv: vec2f };
@vertex fn vs(@builtin(vertex_index) vi: u32) -> VO {
  let c = array<vec2f, 6>(vec2f(0,0), vec2f(1,0), vec2f(0,1), vec2f(0,1), vec2f(1,0), vec2f(1,1))[vi];
  let z = planeZ(f32(U.blocks)) - ${TOWER.canvasGap.toFixed(2)};
  let p = vec3f(U.origin.x + (c.x - 0.5) * V.size, U.origin.y + (c.y - 0.5) * V.size, z);
  var o: VO;
  o.pos = F.viewProj * vec4f(p, 1.0);
  o.uv = vec2f(c.x, 1.0 - c.y);
  return o;
}
@fragment fn fs(i: VO) -> @location(0) vec4f {
  // every pixel shown here was computed by the model: no blending with earlier results, no second decoder mid-way
  let l0 = floor(V.lensLayer);
  let a = textureSample(lens, samp, i.uv, i32(l0)).rgb;
  let b = textureSample(lens, samp, i.uv, i32(min(l0 + 1.0, f32(textureNumLayers(lens) - 1u)))).rgb;
  let lensCol = mix(a, b, smoothstep(0.0, 1.0, fract(V.lensLayer)));
  let d = mix(textureSample(dec, samp, i.uv, i32(V.decA)).rgb, textureSample(dec, samp, i.uv, i32(V.decB)).rgb, V.decT);
  let col = mix(lensCol, d, V.decAmount);
  let edge = min(min(i.uv.x, 1.0 - i.uv.x), min(i.uv.y, 1.0 - i.uv.y));
  return vec4f(col * smoothstep(0.0, 0.004, edge) * V.opacity, 0.0);
}`;

const smooth = (x: number) => {
	const t = Math.min(1, Math.max(0, x));
	return t * t * (3 - 2 * t);
};

export interface PaintingState {
	step: number; // current denoising step (0..steps-1)
	scan: number; // block being computed (float, 0..blocks)
	fade: number;
	focus: number; // index into the prompt's real tokens
	beam: number;
	palette: number; // 0 = ternary duotone, 1 = false colour
	canvasLens: number; // float layer into the lens array (step*blocks + block)
	decoder: number; // position along the decoder stages (float), -1 = not decoding yet
	canvasSize: number;
	canvasOpacity: number;
	towerFade: number;
}

export class Painting {
	readonly steps: number;
	readonly blocks: number;
	readonly grid: number;
	readonly nText: number;
	readonly words: { token: number; text: string }[];
	private uniform: GPUBuffer;
	private srcUniform: GPUBuffer;
	private canvasUniform: GPUBuffer;
	decoderStages = 1;
	private patchPipe: GPURenderPipeline;
	private beamPipe: GPURenderPipeline;
	private canvasPipe: GPURenderPipeline;
	private patchBind: GPUBindGroup;
	private beamBind: GPUBindGroup;
	private canvasBind: GPUBindGroup;

	constructor(
		private device: GPUDevice,
		frame: GPUBuffer,
		trace: Trace,
		format: GPUTextureFormat
	) {
		const m = trace.manifest;
		this.steps = m.steps;
		this.blocks = m.act3.blocks;
		this.grid = m.act3.grid;
		this.nText = m.act3.n_text_tokens;
		const N = this.grid * this.grid;
		// the prompt's own words (not the chat template around them)
		this.words = m.act1.tokens
			.map((t, token) => ({ token, text: clean(t) }))
			.filter((w) => w.text && !/^<\|.*\|>$|^<\/?think>$|^(user|assistant)$/.test(w.text));

		// per (step, block): normalise PCA to 1st..99th percentile, activity = token norm relative to the block
		const pca = trace.arrays.dit_pca.data,
			norm = trace.arrays.dit_norm_img.data;
		const colors = new Float32Array(this.steps * this.blocks * N * 4);
		for (let sb = 0; sb < this.steps * this.blocks; sb++) {
			for (let c = 0; c < 3; c++) {
				const vals = Array.from({ length: N }, (_, p) => pca[(sb * N + p) * 3 + c]).sort(
					(a, b) => a - b
				);
				const lo = vals[Math.floor(N * 0.01)],
					hi = vals[Math.floor(N * 0.99)];
				for (let p = 0; p < N; p++)
					colors[(sb * N + p) * 4 + c] = Math.min(
						1,
						Math.max(0, (pca[(sb * N + p) * 3 + c] - lo) / (hi - lo + 1e-6))
					);
			}
			let mean = 0;
			for (let p = 0; p < N; p++) mean += norm[sb * N + p] / N;
			for (let p = 0; p < N; p++)
				colors[(sb * N + p) * 4 + 3] = Math.min(2, norm[sb * N + p] / (mean + 1e-6)) - 0.5;
		}
		// attention image->word, normalised per (step, block, word) so every word has a visible footprint
		const att = trace.arrays.dit_attn.data;
		const attn = new Float32Array(att.length);
		const nt = this.nText;
		for (let sb = 0; sb < this.steps * this.blocks; sb++)
			for (let w = 0; w < nt; w++) {
				let mx = 1e-9;
				for (let p = 0; p < N; p++) mx = Math.max(mx, att[(sb * N + p) * nt + w]);
				for (let p = 0; p < N; p++) attn[(sb * N + p) * nt + w] = att[(sb * N + p) * nt + w] / mx;
			}

		this.uniform = device.createBuffer({
			size: 64,
			usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
		});
		const u = new ArrayBuffer(64);
		new Uint32Array(u, 0, 4).set([this.steps, this.blocks, this.grid, nt]);
		device.queue.writeBuffer(this.uniform, 0, u);
		this.srcUniform = device.createBuffer({
			size: 16,
			usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
		});
		this.canvasUniform = device.createBuffer({
			size: 32,
			usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
		});

		const additive: GPUColorTargetState = {
			format,
			blend: {
				color: { srcFactor: 'one', dstFactor: 'one' },
				alpha: { srcFactor: 'one', dstFactor: 'one' }
			}
		};
		const pipe = (code: string) => {
			const module = device.createShaderModule({ code });
			return device.createRenderPipeline({
				layout: 'auto',
				vertex: { module, entryPoint: 'vs' },
				fragment: { module, entryPoint: 'fs', targets: [additive] }
			});
		};
		this.patchPipe = pipe(PATCHES);
		this.beamPipe = pipe(BEAMS);
		this.canvasPipe = pipe(CANVAS);
		const base = [
			{ binding: 0, resource: { buffer: frame } },
			{ binding: 1, resource: { buffer: this.uniform } }
		];
		this.patchBind = device.createBindGroup({
			layout: this.patchPipe.getBindGroupLayout(0),
			entries: [
				...base,
				{ binding: 2, resource: { buffer: storageBuffer(device, colors, 'patch colours') } }
			]
		});
		this.beamBind = device.createBindGroup({
			layout: this.beamPipe.getBindGroupLayout(0),
			entries: [
				...base,
				{ binding: 2, resource: { buffer: storageBuffer(device, attn, 'attention') } },
				{ binding: 3, resource: { buffer: this.srcUniform } }
			]
		});
		const lensTex = textureArrayFromBitmaps(device, trace.lens, trace.lens[0].width, 'lens');
		const stages = trace.decoder.length ? trace.decoder.map((d) => d.bitmap) : [trace.final];
		this.decoderStages = stages.length;
		const bigTex = textureArrayFromBitmaps(device, stages, stages[0].width, 'decoder stages');
		this.canvasBind = device.createBindGroup({
			layout: this.canvasPipe.getBindGroupLayout(0),
			entries: [
				...base,
				{ binding: 2, resource: lensTex.createView({ dimension: '2d-array' }) },
				{ binding: 3, resource: bigTex.createView({ dimension: '2d-array' }) },
				{
					binding: 4,
					resource: device.createSampler({ magFilter: 'linear', minFilter: 'linear' })
				},
				{ binding: 5, resource: { buffer: this.canvasUniform } }
			]
		});
	}

	planeZ(block: number) {
		return TOWER.z0 - block * TOWER.dz;
	}

	canvasZ() {
		return this.planeZ(this.blocks) - TOWER.canvasGap;
	}

	/** Where the prompt words hang, in a column beside the block being computed. */
	wordPos(i: number, scan: number): [number, number, number] {
		const k = this.words.length;
		return [
			TOWER.x - TOWER.size * 0.5 - 1.6,
			TOWER.y + ((k - 1) / 2 - i) * 0.3,
			this.planeZ(Math.min(scan, this.blocks - 1)) + 0.6
		];
	}

	update(s: PaintingState) {
		const d = this.device;
		d.queue.writeBuffer(
			this.uniform,
			16,
			new Float32Array([
				s.step,
				s.scan,
				s.fade * s.towerFade,
				s.palette,
				TOWER.x,
				TOWER.y,
				TOWER.z0,
				TOWER.dz,
				TOWER.size,
				this.words[s.focus]?.token ?? 0,
				s.beam,
				0
			])
		);
		d.queue.writeBuffer(
			this.srcUniform,
			0,
			new Float32Array([...this.wordPos(s.focus, s.scan), 1])
		);
		const dPos = Math.max(0, Math.min(this.decoderStages - 1, s.decoder));
		const decA = Math.floor(dPos);
		d.queue.writeBuffer(
			this.canvasUniform,
			0,
			new Float32Array([
				s.canvasLens,
				s.decoder < 0 ? 0 : smooth(s.decoder + 1),
				decA,
				Math.min(decA + 1, this.decoderStages - 1),
				smooth((dPos - decA - 0.55) / 0.45),
				s.canvasSize,
				s.canvasOpacity * s.fade,
				0
			])
		);
	}

	wordInstances(s: PaintingState): WordInstance[] {
		return this.words.map((w, i) => ({
			word: w.text,
			pos: this.wordPos(i, s.scan),
			height: 0.2,
			color: i === s.focus ? [1.0, 0.434, 0.069] : [0.807, 0.761, 0.672],
			alpha: s.fade * s.towerFade * (i === s.focus ? 1.6 : 0.45),
			anchor: 1
		}));
	}

	draw(pass: GPURenderPassEncoder, s: PaintingState) {
		pass.setPipeline(this.patchPipe);
		pass.setBindGroup(0, this.patchBind);
		pass.draw(6, this.blocks * this.grid * this.grid);
		if (s.beam > 0.001) {
			pass.setPipeline(this.beamPipe);
			pass.setBindGroup(0, this.beamBind);
			pass.draw(6, this.grid * this.grid);
		}
		pass.setPipeline(this.canvasPipe);
		pass.setBindGroup(0, this.canvasBind);
		pass.draw(6);
	}
}
