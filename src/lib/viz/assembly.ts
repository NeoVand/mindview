// Live study: "meaning assembling". A prompt is read by the ternary LLM on the GPU; then, layer by layer, each
// token is placed by what its hidden state now says (3D MDS of directions, aligned layer to layer) and linked to
// the tokens it actually read (attention, strongest head, attention sinks set aside). Particles carry information
// from source to reader. Nothing here is recorded: every number comes from this tab's forward pass.
import { mat4, vec3 } from 'wgpu-matrix';
import { FRAME_BYTES, FRAME_WGSL, HDR_FORMAT, type GPU } from '$lib/engine/gpu';
import { Post } from '$lib/engine/post';
import { TextAtlas, clean } from '$lib/engine/text';
import { WordLayer, type WordInstance } from '$lib/engine/words';
import { BonsaiLLM, MAX_TOKENS } from '$lib/runtime/bonsai-llm';
import { chatPrompt } from '$lib/runtime/tokenizer';
import { align3, mds } from './linalg';

const BODY_WGSL = /* wgsl */ `${FRAME_WGSL}
struct Body { pos: vec3f, size: f32, color: vec3f, alpha: f32 };
@group(0) @binding(0) var<uniform> F: Frame;
@group(0) @binding(1) var<storage, read> B: array<Body>;
struct VO { @builtin(position) pos: vec4f, @location(0) uv: vec2f, @location(1) col: vec4f };
@vertex fn vs(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VO {
  let b = B[ii];
  let c = array<vec2f, 6>(vec2f(-1,-1), vec2f(1,-1), vec2f(-1,1), vec2f(-1,1), vec2f(1,-1), vec2f(1,1))[vi];
  var o: VO;
  o.pos = F.viewProj * vec4f(b.pos + (F.camRight.xyz * c.x + F.camUp.xyz * c.y) * b.size, 1.0);
  o.uv = c;
  o.col = vec4f(b.color, b.alpha);
  return o;
}
@fragment fn fs(i: VO) -> @location(0) vec4f {
  let r = length(i.uv);
  let core = exp(-r * r * 9.0) * 2.2 + exp(-r * r * 2.2) * 0.35;
  return vec4f(i.col.rgb * core * i.col.a, 0.0);
}`;

// one instance per particle: a bond (reader <- source) carries a stream of particles along a gentle arc
const FLOW_WGSL = /* wgsl */ `${FRAME_WGSL}
struct Bond { src: vec3f, strength: f32, dst: vec3f, phase: f32, color: vec3f, _p: f32 };
@group(0) @binding(0) var<uniform> F: Frame;
@group(0) @binding(1) var<storage, read> BD: array<Bond>;
@group(0) @binding(2) var<uniform> PER: vec4f; // x = particles per bond
struct VO { @builtin(position) pos: vec4f, @location(0) uv: vec2f, @location(1) col: vec3f };
@vertex fn vs(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VO {
  let per = u32(PER.x);
  let bd = BD[ii / per];
  let k = f32(ii % per) / f32(per);
  let t = fract(k + F.time * 0.35 + bd.phase);
  let mid = (bd.src + bd.dst) * 0.5 + vec3f(0.0, length(bd.dst - bd.src) * 0.28, 0.0);
  let p = mix(mix(bd.src, mid, t), mix(mid, bd.dst, t), t);
  let c = array<vec2f, 6>(vec2f(-1,-1), vec2f(1,-1), vec2f(-1,1), vec2f(-1,1), vec2f(1,-1), vec2f(1,1))[vi];
  let size = 0.018 + 0.03 * bd.strength;
  var o: VO;
  o.pos = F.viewProj * vec4f(p + (F.camRight.xyz * c.x + F.camUp.xyz * c.y) * size, 1.0);
  o.uv = c;
  o.col = bd.color * bd.strength * sin(t * 3.14159);
  return o;
}
@fragment fn fs(i: VO) -> @location(0) vec4f { return vec4f(i.col * exp(-dot(i.uv, i.uv) * 3.0) * 1.6, 0.0); }`;

const PARTICLES_PER_BOND = 28;
const MAX_BONDS = 600;
const SINKS = 3; // '<|im_start|>', 'user', '\n' where attention rests when a head has nothing to read

export interface AssemblyStatus {
	caption: string;
	layer: number;
	layers: number;
	busy: boolean;
	ms?: number;
}

export class Assembly {
	private post: Post;
	private frame: GPUBuffer;
	private bodyPipe: GPURenderPipeline;
	private flowPipe: GPURenderPipeline;
	private bodyBuf: GPUBuffer;
	private bondBuf: GPUBuffer;
	private bodyBind: GPUBindGroup;
	private flowBind: GPUBindGroup;
	private words?: WordLayer;
	private raf = 0;
	private last = 0;
	private observer: ResizeObserver;
	// per prompt
	private tokens: string[] = [];
	private n = 0;
	private pos: Float64Array[] = []; // per layer, n x 3
	private colors: Float32Array[] = []; // per layer, n x 3
	private links: { i: number; j: number; s: number }[][] = []; // per layer: reader i read source j with strength s
	private started = 0;
	time = 0;
	paused = false;
	layerTime = 1.1; // seconds per layer

	constructor(
		private gpu: GPU,
		private llm: BonsaiLLM,
		private onStatus: (s: AssemblyStatus) => void
	) {
		const { device } = gpu;
		this.post = new Post(gpu);
		this.frame = device.createBuffer({
			size: FRAME_BYTES,
			usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
		});
		const add: GPUColorTargetState = {
			format: HDR_FORMAT,
			blend: {
				color: { srcFactor: 'one', dstFactor: 'one' },
				alpha: { srcFactor: 'one', dstFactor: 'one' }
			}
		};
		const mk = (code: string) => {
			const module = device.createShaderModule({ code });
			return device.createRenderPipeline({
				layout: 'auto',
				vertex: { module, entryPoint: 'vs' },
				fragment: { module, entryPoint: 'fs', targets: [add] }
			});
		};
		this.bodyPipe = mk(BODY_WGSL);
		this.flowPipe = mk(FLOW_WGSL);
		this.bodyBuf = device.createBuffer({
			size: MAX_TOKENS * 32,
			usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
		});
		this.bondBuf = device.createBuffer({
			size: MAX_BONDS * 48,
			usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
		});
		const per = device.createBuffer({
			size: 16,
			usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
		});
		device.queue.writeBuffer(per, 0, new Float32Array([PARTICLES_PER_BOND, 0, 0, 0]));
		this.bodyBind = device.createBindGroup({
			layout: this.bodyPipe.getBindGroupLayout(0),
			entries: [
				{ binding: 0, resource: { buffer: this.frame } },
				{ binding: 1, resource: { buffer: this.bodyBuf } }
			]
		});
		this.flowBind = device.createBindGroup({
			layout: this.flowPipe.getBindGroupLayout(0),
			entries: [
				{ binding: 0, resource: { buffer: this.frame } },
				{ binding: 1, resource: { buffer: this.bondBuf } },
				{ binding: 2, resource: { buffer: per } }
			]
		});
		this.observer = new ResizeObserver(() => this.resize());
		this.observer.observe(gpu.canvas);
		this.resize();
		this.last = performance.now();
		this.raf = requestAnimationFrame(this.loop);
	}

	destroy() {
		cancelAnimationFrame(this.raf);
		this.observer.disconnect();
	}

	get layers() {
		return this.llm.config.layers;
	}

	/** Read a prompt with the live model, then lay out every layer from its real hidden states and attention. */
	async read(prompt: string) {
		const llm = this.llm,
			c = llm.config,
			L = llm.layout,
			D = c.dim,
			H = c.heads,
			N = MAX_TOKENS;
		this.onStatus({ caption: 'Reading', layer: 0, layers: c.layers, busy: true });
		const ids = llm.tokenizer.encode(chatPrompt(prompt));
		const r = await llm.prefill(ids.slice(0, MAX_TOKENS));
		const n = r.ids.length;
		const resid = await llm.read(L.resid, (c.layers + 1) * N * D);
		const probs = await llm.read(L.probs, c.layers * H * N * N);

		// positions: each layer's token directions (centred) embedded in 3D, each aligned to the layer before
		const pos: Float64Array[] = [];
		for (let l = 0; l <= c.layers; l++) {
			const X = resid.subarray(l * N * D, l * N * D + n * D);
			let P = mds(X, n, D, 3);
			if (l > 0) P = align3(P, pos[l - 1], n);
			let rad = 1e-9;
			for (let i = 0; i < n * 3; i++) rad = Math.max(rad, Math.abs(P[i]));
			for (let i = 0; i < n * 3; i++) P[i] = (P[i] / rad) * 4.2;
			pos.push(P);
		}
		// colour: each layer's first three principal directions over the same tokens, into the ternary palette
		const colors = pos.map((P) => {
			const col = new Float32Array(n * 3);
			for (let i = 0; i < n; i++) {
				const a = Math.tanh(P[i * 3] / 2.5),
					b = Math.tanh(P[i * 3 + 1] / 2.5);
				const glacier = [0.093, 0.578, 1.0],
					ember = [1.0, 0.434, 0.069],
					bone = [0.807, 0.761, 0.672];
				const t = (a + 1) / 2;
				for (let k = 0; k < 3; k++)
					col[i * 3 + k] =
						(glacier[k] * (1 - t) + ember[k] * t) * (0.55 + 0.45 * Math.abs(b)) + bone[k] * 0.12;
			}
			return col;
		});
		// links: for the attention that produced layer l+1, the strongest head's weight from reader i to source j
		const links: { i: number; j: number; s: number }[][] = [];
		for (let l = 0; l < c.layers; l++) {
			const out: { i: number; j: number; s: number }[] = [];
			for (let i = SINKS; i < n; i++) {
				const row: { j: number; s: number }[] = [];
				for (let j = SINKS; j < i; j++) {
					let s = 0;
					for (let h = 0; h < H; h++) s = Math.max(s, probs[((l * H + h) * N + i) * N + j]);
					row.push({ j, s });
				}
				row.sort((a, b) => b.s - a.s);
				for (const x of row.slice(0, 3)) if (x.s > 0.12) out.push({ i, j: x.j, s: x.s });
			}
			links.push(out);
		}
		// swap everything in at once so the render loop never sees a half-built prompt
		this.pos = pos;
		this.colors = colors;
		this.links = links;
		this.tokens = r.tokens;
		this.n = n;
		const words = [...r.tokens.map(clean)];
		this.words = new WordLayer(
			this.gpu.device,
			this.frame,
			new TextAtlas(this.gpu.device, words, { px: 64 }),
			MAX_TOKENS,
			HDR_FORMAT
		);
		this.time = 0;
		this.started = performance.now();
		this.onStatus({
			caption: 'The words as the model first sees them',
			layer: 0,
			layers: c.layers,
			busy: false,
			ms: r.ms
		});
	}

	seekLayer(delta: number) {
		const l = Math.round(this.time / this.layerTime) + delta;
		this.time = Math.max(0, Math.min(this.layers, l)) * this.layerTime;
	}

	private resize() {
		const cv = this.gpu.canvas;
		const dpr = Math.min(window.devicePixelRatio || 1, 2);
		const w = Math.max(1, Math.floor(cv.clientWidth * dpr)),
			h = Math.max(1, Math.floor(cv.clientHeight * dpr));
		if (w === cv.width && h === cv.height && this.post.sceneView) return;
		cv.width = w;
		cv.height = h;
		this.post.resize(w, h);
	}

	private loop = (now: number) => {
		const dt = Math.min(0.1, (now - this.last) / 1000);
		this.last = now;
		if (!this.paused && this.n)
			this.time = Math.min(this.time + dt, this.layers * this.layerTime + 1e-3);
		this.raf = requestAnimationFrame(this.loop);
		try {
			this.render(now / 1000);
		} catch (e) {
			console.error('assembly frame failed', e);
		}
	};

	private render(clock: number) {
		const { device, canvas } = this.gpu;
		const n = this.n;
		const lf = Math.min(this.layers, this.time / this.layerTime); // continuous layer position
		const l0 = Math.floor(lf),
			l1 = Math.min(this.layers, l0 + 1);
		const k = smooth(lf - l0);

		// camera: slow orbit around the cloud
		const ang = clock * 0.06;
		const eye: [number, number, number] = [
			Math.sin(ang) * 12.5,
			2.2 + Math.sin(clock * 0.05) * 0.8,
			Math.cos(ang) * 12.5
		];
		const target: [number, number, number] = [0, 0, 0];
		const aspect = canvas.width / canvas.height,
			fov = (38 * Math.PI) / 180;
		const vp = mat4.multiply(
			mat4.perspective(fov, aspect, 0.05, 200),
			mat4.lookAt(eye, target, [0, 1, 0])
		);
		const fwd = vec3.normalize(vec3.subtract(target, eye));
		const right = vec3.normalize(vec3.cross(fwd, [0, 1, 0]));
		const up = vec3.cross(right, fwd);
		const f = new Float32Array(FRAME_BYTES / 4);
		f.set(vp, 0);
		f.set([...right, 0], 16);
		f.set([...up, 0], 20);
		f.set([...eye, 1], 24);
		f.set([clock, aspect, (2 * Math.tan(fov / 2)) / canvas.height, 0], 28);
		device.queue.writeBuffer(this.frame, 0, f);

		let bondCount = 0;
		const words: WordInstance[] = [];
		if (n) {
			const A = this.pos[l0],
				B = this.pos[l1],
				CA = this.colors[l0],
				CB = this.colors[l1];
			const at = (i: number): [number, number, number] => [
				A[i * 3] + (B[i * 3] - A[i * 3]) * k,
				A[i * 3 + 1] + (B[i * 3 + 1] - A[i * 3 + 1]) * k,
				A[i * 3 + 2] + (B[i * 3 + 2] - A[i * 3 + 2]) * k
			];
			const bodies = new Float32Array(MAX_TOKENS * 8);
			for (let i = 0; i < n; i++) {
				const p = at(i);
				const frame = i < SINKS || /^<|^(user|assistant)$/.test(clean(this.tokens[i]));
				bodies.set(
					[
						...p,
						frame ? 0.1 : 0.2,
						CA[i * 3] + (CB[i * 3] - CA[i * 3]) * k,
						CA[i * 3 + 1] + (CB[i * 3 + 1] - CA[i * 3 + 1]) * k,
						CA[i * 3 + 2] + (CB[i * 3 + 2] - CA[i * 3 + 2]) * k,
						frame ? 0.25 : 1
					],
					i * 8
				);
				const text = clean(this.tokens[i]);
				if (text)
					words.push({
						word: text,
						pos: [p[0], p[1] - 0.32, p[2]],
						height: frame ? 0.16 : 0.24,
						color: [0.807, 0.761, 0.672],
						alpha: frame ? 0.35 : 1.1
					});
			}
			device.queue.writeBuffer(this.bodyBuf, 0, bodies);
			// the attention that turns layer l0 into l1 is what we show while moving between them
			const links = lf < this.layers ? this.links[Math.min(this.layers - 1, l0)] : [];
			const bonds = new Float32Array(MAX_BONDS * 12);
			const glow = Math.sin(Math.min(1, k * 1.15) * Math.PI) * 0.8 + 0.2;
			for (const lk of links) {
				if (bondCount >= MAX_BONDS) break;
				const src = at(lk.j),
					dst = at(lk.i);
				bonds.set(
					[...src, lk.s * glow, ...dst, (lk.i * 0.37 + lk.j * 0.11) % 1, 1.0, 0.55, 0.12, 0],
					bondCount * 12
				);
				bondCount++;
			}
			device.queue.writeBuffer(this.bondBuf, 0, bonds, 0, Math.max(12, bondCount * 12));
		}
		this.words?.set(words);

		const enc = device.createCommandEncoder();
		const pass = enc.beginRenderPass({
			colorAttachments: [
				{ view: this.post.sceneView, loadOp: 'clear', storeOp: 'store', clearValue: [0, 0, 0, 1] }
			]
		});
		if (n) {
			if (bondCount) {
				pass.setPipeline(this.flowPipe);
				pass.setBindGroup(0, this.flowBind);
				pass.draw(6, bondCount * PARTICLES_PER_BOND);
			}
			pass.setPipeline(this.bodyPipe);
			pass.setBindGroup(0, this.bodyBind);
			pass.draw(6, n);
			this.words?.draw(pass);
		}
		pass.end();
		this.post.finish(enc, clock);
		device.queue.submit([enc.finish()]);

		if (n) {
			const layer = Math.min(this.layers, Math.round(lf));
			const caption =
				lf < 0.5
					? 'The words as the model first sees them'
					: lf >= this.layers
						? `After all ${this.layers} layers: what the model made of your words`
						: `Layer ${l0 + 1} of ${this.layers}: each word reads the others (${this.links[Math.min(this.layers - 1, l0)].length} strong links)`;
			this.onStatus({ caption, layer, layers: this.layers, busy: false });
		}
	}
}

const smooth = (x: number) => {
	const t = Math.min(1, Math.max(0, x));
	return t * t * (3 - 2 * t);
};
