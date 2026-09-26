// Live study: threads. Every word of the prompt is a thread falling through the 28 layers of Ternary Bonsai 1.7B.
// A layer moves a word by the sum of (a) attention: one contribution per head and earlier word, P[h,t,s] * Wo_h v_s,
// and (b) the MLP: one contribution per neuron, act_j * Wdown[:, j]. The thread is drawn through that sum term by term,
// projected onto two fixed directions of meaning-space: height is how far through the sum we are, the sideways
// position is the running total, so each step lands exactly where the word is at the next layer. Nothing is
// illustrative: every vertex is a partial sum of the model's real products, computed from this tab's forward pass.
import { mat4, vec3 } from 'wgpu-matrix';
import { FRAME_BYTES, HDR_FORMAT, type GPU } from '$lib/engine/gpu';
import { Post } from '$lib/engine/post';
import { TextAtlas, clean } from '$lib/engine/text';
import { WordLayer, type WordInstance } from '$lib/engine/words';
import { BonsaiLLM, MAX_TOKENS } from '$lib/runtime/bonsai-llm';
import { chatPrompt } from '$lib/runtime/tokenizer';
import type { Painter } from '$lib/runtime/painter';
import { GpuScheduler } from '$lib/runtime/scheduler';
import { LineLayer } from './lines';
import { Painting, TICKS_PER_STEP } from './painting';
import { ImagePlanes } from './planes';

type V3 = [number, number, number];

export interface ThreadsStatus {
	caption: string;
	busy: boolean;
	front?: number; // how far the computation has got, in journey units
	ride?: number; // where the camera is, in journey units
	total?: number; // length of the journey in units (28 layers, then the painter)
	marks?: number[]; // where layers and painting steps begin, in units
	mode?: string;
}

const WORLD = 5; // radius of the bundle in world units
const DX = 1.3; // length of one layer along the journey
const ATTN_SHARE = 0.3; // fraction of a layer's length given to the attention sum
const GAP = 5; // world length of the handoff from reader to painter
const NEURON_STRIDE = 4; // draw every 4th neuron's running total
const TICK_LEN = 3.2; // world length of one painter tick before a painting has laid its columns out
const HANDOFF = 2; // journey units taken by the handoff
const OTHER = 250,
	PLATE = 251,
	TAP_PLATE = 252; // palette slots
const TAP_LAYERS = [7, 14, 21]; // where the painter listens: the words as they stand after these layers

// Project every column of a ternary matrix onto two directions of the residual stream: out[j] = E^T W[:, j]
const PROJECT_WGSL = /* wgsl */ `
struct P { rows: u32, K: u32, codes: u32, scales: u32, out: u32, _a: u32, _b: u32, _c: u32 };
@group(0) @binding(0) var<uniform> U: P;
@group(0) @binding(1) var<storage, read> CODES: array<u32>;
@group(0) @binding(2) var<storage, read> SCALES: array<f32>;
@group(0) @binding(3) var<storage, read> E: array<vec2f>;
@group(0) @binding(4) var<storage, read_write> OUT: array<vec2f>;
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) gid: vec3u) {
  let j = gid.x;
  if (j >= U.K) { return; }
  var acc = vec2f(0.0);
  for (var i = 0u; i < U.rows; i++) {
    let w = CODES[U.codes + i * (U.K / 16u) + j / 16u];
    let t = f32((w >> ((j % 16u) * 2u)) & 3u) - 1.0;
    if (t != 0.0) { acc += E[i] * (t * SCALES[U.scales + i * (U.K / 128u) + j / 128u]); }
  }
  OUT[U.out + j] = acc;
}`;

// timeline: how long each layer takes to grow (seconds)
const layerDur = (l: number) => (l === 0 ? 7 : l === 1 ? 4 : 2.2);

export class Threads {
	private post: Post;
	private frame: GPUBuffer;
	private threads: LineLayer;
	private links: LineLayer;
	private words?: WordLayer;
	private labels: WordInstance[] = [];
	// ghost words: what the model would say each word has become, at every layer (the logit lens)
	private ghostLayer?: WordLayer;
	private ghosts: { text: string; layer: number; word: number; prob: number; pos: V3 }[] = [];
	private raf = 0;
	private last = 0;
	private observer: ResizeObserver;
	private project: GPUComputePipeline;
	private ready = false;
	private layers = 28;
	private wordCount = 0;
	private tips: Float32Array[] = []; // each word's thread (xyzw per vertex), for placing names on the growing tips
	private tipWord = '';
	private rest: V3[] = [];
	private reach = new Float32Array(1); // per layer: how far the threads reach from the axis
	private starts: number[] = []; // time at which each layer starts growing
	private total = 0;
	time = 0;
	paused = false;
	// the camera rides alongside the front ('live'), then shows the whole ('overview'), then travels back through
	// the finished past ('rewind'); the visitor can take the ride position ('manual') and turn / zoom at any time
	mode: 'live' | 'finale' | 'overview' | 'rewind' | 'manual' = 'live';
	ride = 0; // where along the journey the camera is, in layers
	private modeTime = 0;
	private rewound = false;
	private yawOff = 0;
	private pitchOff = 0;
	private distMul = 1;
	private cam = { c: [0, 0, 0] as V3, dist: 12, yaw: 0.5, pitch: 0.2 };
	private idle = 0; // seconds since the visitor last touched anything
	// the painter, once it has downloaded: it paints when the reader is done
	private painter?: Painter;
	private painting?: Painting;
	private planes: ImagePlanes;
	/** All compute for the painter goes through this queue, a slice per frame, in order. */
	readonly scheduler: GpuScheduler;
	/** Painting options, used by the next painting. */
	options = { steps: 4, seed: 7 };
	private prompt = '';
	private palette = new Float32Array(256 * 4);
	private threadPos: V3[][] = [];
	private paintFront = 0; // ticks shown so far (smoothed)
	private paintError = '';
	// a new prompt: the old scene fades out while the new one is read (and nothing is painted from the old one)
	private reading = false;
	private readId = 0;
	private quiet = 0; // 0..1: the finished picture on its own (lines dimmed, since they run straight at the camera)

	constructor(
		private gpu: GPU,
		private llm: BonsaiLLM,
		private onStatus: (s: ThreadsStatus) => void
	) {
		const { device } = gpu;
		this.post = new Post(gpu);
		this.post.bloomStrength = 0.6;
		this.frame = device.createBuffer({
			size: FRAME_BYTES,
			usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
		});
		this.threads = new LineLayer(device, this.frame, true);
		this.links = new LineLayer(device, this.frame, true);
		this.scheduler = new GpuScheduler(device);
		this.planes = new ImagePlanes(device, this.frame, true);
		this.project = device.createComputePipeline({
			layout: 'auto',
			compute: {
				module: device.createShaderModule({ label: 'project columns', code: PROJECT_WGSL }),
				entryPoint: 'main'
			}
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
		this.threads.destroy();
		this.links.destroy();
		this.scheduler.clear();
		this.painting?.destroy();
	}

	/** The painter is ready; the journey continues into it after the reader's last layer. */
	attachPainter(painter: Painter) {
		this.painter = painter;
	}

	private startPainting() {
		if (!this.painter) return;
		const x0 = (this.layers * DX) / 2;
		const painting = new Painting(
			this.gpu.device,
			this.frame,
			this.planes,
			{ x0: x0 + GAP, readerEnd: x0, layers: this.layers },
			this.palette,
			this.wordCount,
			this.options.steps
		);
		this.painting = painting;
		try {
			const taps = this.threadPos.map((ps) => [ps[7], ps[14], ps[21]]);
			painting.start(this.painter, this.llm, this.scheduler, this.prompt, taps, this.options.seed);
		} catch (err) {
			this.paintError = err instanceof Error ? err.message : String(err);
		}
	}

	private get paintTicks() {
		return this.painting?.ticksTotal ?? TICKS_PER_STEP * this.options.steps;
	}

	/** Journey units: 0..28 the reader's layers, then the handoff, then one unit per painter readout. */
	private get journeyLength() {
		return this.painter ? this.layers + HANDOFF + this.paintTicks : this.layers;
	}

	private xAt(u: number) {
		const x0 = (this.layers * DX) / 2;
		if (u <= this.layers) return u * DX - x0;
		if (u <= this.layers + HANDOFF) return x0 + ((u - this.layers) / HANDOFF) * GAP;
		const t = u - this.layers - HANDOFF;
		return this.painting ? this.painting.xOf(t) : x0 + GAP + t * TICK_LEN;
	}

	get duration() {
		return this.total;
	}

	get layerCount() {
		return this.layers;
	}

	/** How far the computation has got, in layers. */
	get front() {
		if (!this.ready) return 0;
		const p = this.progress(this.time);
		if (p < this.layers || !this.painter) return p;
		if (!this.painting || this.painting.encoded < 1 || this.painting.ticks < 1)
			return this.layers + HANDOFF * (this.painting ? this.painting.encoded : 0);
		return this.layers + HANDOFF + this.paintFront;
	}

	/** The visitor turned the camera. */
	orbit(dx: number, dy: number) {
		this.yawOff -= dx * 0.005;
		this.pitchOff = Math.max(-1.2, Math.min(1.2, this.pitchOff + dy * 0.005));
		this.idle = 0;
	}

	zoom(f: number) {
		this.distMul = Math.max(0.15, Math.min(6, this.distMul * f));
		this.idle = 0;
	}

	/** The visitor moved along the journey (in layers); the computation already done stays drawn. */
	rideTo(layer: number) {
		this.ride = Math.max(0, Math.min(this.front, layer));
		this.cam.c = [this.xAt(this.ride), 0, 0];
		this.mode = 'manual';
		this.idle = 0;
	}

	/** Paint again with the current options (steps, seed), keeping what the reader did. */
	repaint() {
		if (!this.painter || !this.ready) return;
		this.scheduler.clear();
		this.painting?.destroy();
		this.painting = undefined;
		this.planes.clear();
		this.paintFront = 0;
		this.paintError = '';
		if (this.progress(this.time) >= this.layers) {
			this.startPainting();
			this.mode = 'live';
			this.ride = this.layers;
		}
	}

	/** Travel back through everything computed so far, from the front to the start. */
	rideBack() {
		this.ride = this.front;
		this.mode = 'rewind';
		this.modeTime = 0;
		this.idle = 0;
	}

	/** Back to riding alongside the computation. */
	follow() {
		this.mode = this.front >= this.journeyLength - 0.01 ? 'overview' : 'live';
		this.modeTime = 0;
		this.idle = 0;
	}

	seek(dt: number) {
		if (this.mode === 'manual' || this.mode === 'overview' || this.mode === 'rewind') {
			this.rideTo(this.ride + dt / 2);
			return;
		}
		this.time = Math.max(0, Math.min(this.total, this.time + dt));
	}

	async read(prompt: string) {
		const id = ++this.readId;
		this.reading = true;
		this.prompt = prompt;
		this.painting?.destroy();
		this.painting = undefined;
		this.planes.clear();
		this.scheduler.clear();
		this.paintFront = 0;
		this.paintError = '';
		const llm = this.llm,
			c = llm.config,
			L = llm.layout,
			dev = this.gpu.device;
		const N = MAX_TOKENS,
			D = c.dim,
			F = c.ffn,
			H = c.heads,
			HD = c.headDim,
			KV = c.kvHeads * HD,
			Q = H * HD;
		// the painter uses the words only as they stand after layer 21, so the reader stops there
		const NL = TAP_LAYERS[2],
			G = H / c.kvHeads;
		// the reading is long JavaScript work: give frames a turn every few milliseconds, and stop if a newer
		// prompt has come in meanwhile (breathe() then returns true)
		let since = performance.now();
		const breathe = async () => {
			if (performance.now() - since > 6) {
				await new Promise((r) => setTimeout(r, 0));
				since = performance.now();
			}
			return id !== this.readId;
		};
		this.onStatus({ caption: `Reading your words through ${NL} layers`, busy: true });
		const r = await llm.prefill(
			llm.tokenizer.encode(chatPrompt(prompt)).slice(0, MAX_TOKENS),
			this.scheduler,
			NL
		);
		if (await breathe()) return;
		const n = r.ids.length;
		const end = r.tokens.findIndex((t, i) => i > 3 && t.startsWith('<|im_end'));
		const W = Array.from({ length: Math.max(1, (end > 0 ? end : n) - 3) }, (_, i) => 3 + i);
		const nw = W.length;

		// ---- read back what the pass left in the arena
		this.onStatus({ caption: 'Collecting every partial sum', busy: true });
		const regions: { offset: number; count: number }[] = [];
		for (let l = 0; l <= NL; l++)
			for (const t of W) regions.push({ offset: L.resid + (l * N + t) * D, count: D });
		for (let l = 0; l < NL; l++)
			for (const t of W) regions.push({ offset: L.act + (l * N + t) * F, count: F });
		for (let l = 0; l < NL; l++) regions.push({ offset: L.v + l * N * KV, count: n * KV });
		for (let l = 0; l < NL; l++)
			for (let h = 0; h < H; h++)
				regions.push({ offset: L.probs + ((l * H + h) * N + W[0]) * N, count: nw * N });
		const got = await llm.readMany(regions);
		if (await breathe()) return;
		let k = 0;
		const resid = Array.from({ length: NL + 1 }, () => W.map(() => got[k++]));
		const act = Array.from({ length: NL }, () => W.map(() => got[k++]));
		const vals = Array.from({ length: NL }, () => got[k++]);
		const probs = Array.from({ length: NL }, () => Array.from({ length: H }, () => got[k++])); // [l][h][(t - W0) * N + s]

		// ---- two fixed directions: the main axes along which the words move, each layer rescaled to a common size
		const norm = (v: Float32Array) => Math.sqrt(v.reduce((a, x) => a + x * x, 0));
		const scale = resid.map((row) => median(row.map(norm)));
		const mu = resid.map((row, l) => {
			const m = new Float64Array(D);
			for (const v of row) for (let i = 0; i < D; i++) m[i] += v[i] / scale[l] / nw;
			return m;
		});
		const X: Float64Array[] = [];
		for (let l = 1; l <= NL; l++)
			for (const v of resid[l]) {
				const x = new Float64Array(D);
				for (let i = 0; i < D; i++) x[i] = v[i] / scale[l] - mu[l][i];
				X.push(x);
			}
		const two = await topTwo(X, D, breathe);
		if (!two) return;
		const [e0, e1] = two;
		const dot2 = (v: ArrayLike<number>): [number, number] => {
			let a = 0,
				b = 0;
			for (let i = 0; i < D; i++) {
				a += v[i] * e0[i];
				b += v[i] * e1[i];
			}
			return [a, b];
		};
		const hE = resid.map((row) => row.map(dot2));
		const muE = mu.map(dot2);

		// ---- every column of Wo and Wdown, projected onto those directions (on the GPU)
		const Ebuf = dev.createBuffer({
			size: D * 8,
			usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
		});
		const Einter = new Float32Array(D * 2);
		for (let i = 0; i < D; i++) {
			Einter[2 * i] = e0[i];
			Einter[2 * i + 1] = e1[i];
		}
		dev.queue.writeBuffer(Ebuf, 0, Einter);
		const per = F + Q;
		const out = dev.createBuffer({
			size: NL * per * 8,
			usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
		});
		const params = dev.createBuffer({
			size: NL * 2 * 256,
			usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
		});
		const { codes, scales } = llm.weightBuffers;
		const enc = dev.createCommandEncoder();
		const pass = enc.beginComputePass();
		pass.setPipeline(this.project);
		const pdata = new Uint32Array(NL * 2 * 64);
		const binds: GPUBindGroup[] = [];
		for (let l = 0; l < NL; l++) {
			const jobs = [
				llm.tensor(`blk.${l}.ffn_down.weight`),
				llm.tensor(`blk.${l}.attn_output.weight`)
			];
			jobs.forEach((w, i) => {
				const slot = l * 2 + i;
				pdata.set([w.rows, w.cols, w.codes, w.scales, l * per + (i ? F : 0)], slot * 64);
				binds[slot] = dev.createBindGroup({
					layout: this.project.getBindGroupLayout(0),
					entries: [
						{ binding: 0, resource: { buffer: params, offset: slot * 256, size: 32 } },
						{ binding: 1, resource: { buffer: codes } },
						{ binding: 2, resource: { buffer: scales } },
						{ binding: 3, resource: { buffer: Ebuf } },
						{ binding: 4, resource: { buffer: out } }
					]
				});
			});
		}
		dev.queue.writeBuffer(params, 0, pdata);
		for (let l = 0; l < NL; l++)
			[F, Q].forEach((K, i) => {
				pass.setBindGroup(0, binds[l * 2 + i]);
				pass.dispatchWorkgroups(Math.ceil(K / 64));
			});
		pass.end();
		const read = dev.createBuffer({
			size: NL * per * 8,
			usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
		});
		enc.copyBufferToBuffer(out, 0, read, 0, NL * per * 8);
		dev.queue.submit([enc.finish()]);
		await read.mapAsync(GPUMapMode.READ);
		if (id !== this.readId) {
			read.destroy();
			return;
		}
		const proj = new Float32Array(read.getMappedRange()).slice();
		read.destroy();
		out.destroy();
		params.destroy();
		Ebuf.destroy();

		// ---- the threads: partial sums, term by term
		this.onStatus({ caption: 'Laying out the threads', busy: true });
		let lo0 = Infinity,
			lo1 = Infinity,
			hi0 = -Infinity,
			hi1 = -Infinity;
		for (let l = 0; l <= NL; l++)
			for (let i = 0; i < nw; i++) {
				const x = hE[l][i][0] / scale[l] - muE[l][0],
					y = hE[l][i][1] / scale[l] - muE[l][1];
				lo0 = Math.min(lo0, x);
				hi0 = Math.max(hi0, x);
				lo1 = Math.min(lo1, y);
				hi1 = Math.max(hi1, y);
			}
		const cx = (lo0 + hi0) / 2,
			cz = (lo1 + hi1) / 2,
			ws = WORLD / (Math.max(hi0 - lo0, hi1 - lo1) / 2 || 1);
		const x0 = (NL * DX) / 2;
		const cap = nw * (2 + NL * (H * n + F + 1) + 17) + (NL + 1) * 129;
		const pts = new Float32Array(cap * 4),
			attr = new Uint32Array(cap);
		let nv = 0;
		const push = (
			x: number,
			y: number,
			z: number,
			t: number,
			strand: number,
			colour: number,
			level: number
		) => {
			pts.set([x, y, z, t], nv * 4);
			attr[nv++] =
				(strand & 0xffff) |
				((colour & 0xff) << 16) |
				((Math.max(0, Math.min(255, level * 255)) | 0) << 24);
		};
		const at = (l: number, frac: number, sum0: number, sum1: number, i: number) => {
			const n1 = Math.min(l + 1, NL);
			const s = scale[l] + (scale[n1] - scale[l]) * frac;
			const m0 = muE[l][0] + (muE[n1][0] - muE[l][0]) * frac,
				m1 = muE[l][1] + (muE[n1][1] - muE[l][1]) * frac;
			return [
				(l + frac) * DX - x0,
				((hE[l][i][0] + sum0) / s - m0 - cx) * ws,
				((hE[l][i][1] + sum1) / s - m1 - cz) * ws
			];
		};
		const linkPts: number[] = [],
			linkAttr: number[] = [];
		const threadPos: V3[][] = []; // [word][layer]: position at the start of each layer (for links and labels)
		const landings: Map<number, { pos: V3; w: number }>[][] = []; // [word][layer]: source token -> landing
		let strand = 0;
		const ranges: [number, number][] = [];
		for (let i = 0; i < nw; i++) {
			const t = W[i];
			const from = nv;
			const first = at(0, 0, 0, 0, i);
			push(first[0], first[1], first[2], 0, strand, i, 1);
			threadPos.push([]);
			landings.push([]);
			for (let l = 0; l < NL; l++) {
				if (await breathe()) return;
				threadPos[i].push(at(l, 0, 0, 0, i) as V3);
				const Pl = proj.subarray(l * per * 2, (l + 1) * per * 2); // [F down columns, then Q output columns] x 2
				const terms: { d0: number; d1: number; colour: number; level: number; src: number }[] = [];
				// attention: one term per (earlier word, head)
				const v = vals[l];
				for (let s = 0; s <= t; s++)
					for (let h = 0; h < H; h++) {
						const p = probs[l][h][i * N + s];
						if (p < 2e-3) continue;
						let d0 = 0,
							d1 = 0;
						const vo = s * KV + Math.floor(h / G) * HD,
							po = (F + h * HD) * 2;
						for (let d = 0; d < HD; d++) {
							d0 += Pl[po + 2 * d] * v[vo + d];
							d1 += Pl[po + 2 * d + 1] * v[vo + d];
						}
						const wi = s - W[0];
						terms.push({
							d0: d0 * p,
							d1: d1 * p,
							colour: wi >= 0 && wi < nw ? wi : OTHER,
							level: Math.min(1, 0.3 + p),
							src: s
						});
					}
				const na = terms.length;
				// MLP: one term per neuron
				const a = act[l][i];
				for (let j = 0; j < F; j++) {
					const x = a[j];
					if (x === 0) continue;
					terms.push({ d0: Pl[2 * j] * x, d1: Pl[2 * j + 1] * x, colour: i, level: 0.8, src: -1 });
				}
				let s0 = 0,
					s1 = 0;
				const landed = new Map<number, { pos: V3; w: number }>(); // where each source's terms end on the thread
				landings[i].push(landed);
				terms.forEach((term, q) => {
					s0 += term.d0;
					s1 += term.d1;
					const frac =
						q < na
							? (ATTN_SHARE * (q + 1)) / na
							: ATTN_SHARE + ((1 - ATTN_SHARE) * (q - na + 1)) / (terms.length - na);
					// every attention term, and every 4th neuron's running total (still exact partial sums, a quarter
					// of the vertices, which leaves the GPU free for the painter)
					if (q >= na && (q - na) % NEURON_STRIDE !== NEURON_STRIDE - 1 && q !== terms.length - 1)
						return;
					const p3 = at(l, frac, s0, s1, i);
					push(p3[0], p3[1], p3[2], l + frac, strand, term.colour, term.level);
					if (q < na && (q === na - 1 || terms[q + 1].src !== term.src))
						landed.set(term.src, { pos: p3 as V3, w: l + frac });
				});
			}
			threadPos[i].push(at(NL, 0, 0, 0, i) as V3);
			ranges.push([from, nv]);
			strand++;
		}
		// attention links: light lent by the words a word reads most, drawn from the source thread to the reader
		for (let l = 0; l < NL; l++)
			for (let i = 0; i < nw; i++) {
				const share: [number, number][] = [];
				for (let j = 0; j < i; j++) {
					let p = 0;
					for (let h = 0; h < H; h++) p += probs[l][h][i * N + W[j]] / H;
					share.push([j, p]);
				}
				share.sort((a, b) => b[1] - a[1]);
				for (const [j, p] of share.slice(0, 3)) {
					if (p < 0.04) break;
					// from the source word where it enters the layer (its keys and values come from there) to the
					// point on the reader's thread where that word's attention terms have just been added
					const end = landings[i][l].get(W[j]);
					if (!end) continue;
					const a = threadPos[j][l],
						b = end.pos;
					const my = (a[1] + b[1]) / 2,
						mz = (a[2] + b[2]) / 2,
						mr = Math.hypot(my, mz) || 1;
					const lift = Math.hypot(b[1] - a[1], b[2] - a[2]) * 0.15;
					const K = 32;
					for (let q = 0; q <= K; q++) {
						const u = q / K,
							bulge = Math.sin(Math.PI * u) * lift;
						linkPts.push(
							a[0] + (b[0] - a[0]) * u,
							a[1] + (b[1] - a[1]) * u + (my / mr) * bulge,
							a[2] + (b[2] - a[2]) * u + (mz / mr) * bulge,
							l + (end.w - l) * u
						);
						linkAttr.push(
							(strand & 0xffff) |
								(j << 16) |
								((Math.min(255, 255 * Math.min(1, p * 2.5)) | 0) << 24)
						);
					}
					strand++;
				}
			}
		// the layers: faint rings the bundle passes through
		for (let l = 0; l <= NL; l++) {
			const x = l * DX - x0,
				R = WORLD * 1.15;
			for (let q = 0; q <= 128; q++)
				push(
					x,
					R * Math.cos((q / 128) * 2 * Math.PI),
					R * Math.sin((q / 128) * 2 * Math.PI),
					l,
					strand,
					TAP_LAYERS.includes(l) ? TAP_PLATE : PLATE,
					TAP_LAYERS.includes(l) ? 0.8 : 0.5
				);
			strand++;
		}
		// beads where the painter will take each word: its place after layers 7, 14 and 21
		for (const l of TAP_LAYERS)
			for (let i = 0; i < nw; i++) {
				const p = threadPos[i][l];
				for (let q = 0; q <= 24; q++) {
					const a = (q / 24) * 2 * Math.PI;
					push(p[0], p[1] + 0.12 * Math.cos(a), p[2] + 0.12 * Math.sin(a), l, strand, i, 1);
				}
				strand++;
			}
		const palette = new Float32Array(256 * 4);
		for (let i = 0; i < nw; i++)
			palette.set([...wordColour(nw > 1 ? i / (nw - 1) : 0.5), 1], i * 4);
		palette.set([0.25, 0.24, 0.22, 1], OTHER * 4);
		palette.set([0.07, 0.066, 0.06, 1], PLATE * 4);
		palette.set([0.13, 0.12, 0.1, 1], TAP_PLATE * 4);
		this.threads.set(pts.subarray(0, nv * 4), attr.subarray(0, nv), palette);
		this.palette = palette;
		this.threadPos = threadPos;
		this.links.set(new Float32Array(linkPts), new Uint32Array(linkAttr), palette);

		// the model's own reading of every word at every layer
		this.onStatus({
			caption: 'Asking the model what each word has become at every layer',
			busy: true
		});
		const vecs = new Float32Array((NL + 1) * nw * D);
		for (let l = 0; l <= NL; l++)
			for (let i = 0; i < nw; i++) vecs.set(resid[l][i], (l * nw + i) * D);
		const lens = await llm.readout(vecs, (NL + 1) * nw, this.scheduler);
		if (await breathe()) return;
		this.ghosts = [];
		for (let l = 0; l <= NL; l++)
			for (let i = 0; i < nw; i++) {
				const k = l * nw + i;
				const text = clean(llm.tokenizer.decode([lens.ids[k]]));
				if (text)
					this.ghosts.push({ text, layer: l, word: i, prob: lens.probs[k], pos: threadPos[i][l] });
			}
		this.ghostLayer = new WordLayer(
			dev,
			this.frame,
			new TextAtlas(
				dev,
				this.ghosts.map((g) => g.text),
				{ px: 56, italic: true }
			),
			1024,
			HDR_FORMAT,
			true
		);

		// labels at the top of each thread
		const texts = W.map((t) => clean(r.tokens[t]));
		this.words = new WordLayer(
			dev,
			this.frame,
			new TextAtlas(dev, texts, { px: 64 }),
			256,
			HDR_FORMAT,
			true
		);
		this.labels = W.map((_, i) => ({
			word: texts[i],
			pos: [0, 0, 0] as V3,
			height: 0.3,
			color: wordColour(nw > 1 ? i / (nw - 1) : 0.5),
			alpha: 1,
			anchor: 0
		}));

		this.layers = NL;
		this.wordCount = nw;
		this.tips = ranges.map(([a, b]) => pts.slice(a * 4, b * 4));
		// how far from the axis the threads reach within each layer (the running sums swing wide in early layers)
		this.reach = new Float32Array(NL + 1).fill(WORLD * 1.15);
		for (const tip of this.tips)
			for (let v = 0; v < tip.length; v += 4) {
				const l = Math.min(NL, Math.floor(tip[v + 3]));
				this.reach[l] = Math.max(this.reach[l], Math.hypot(tip[v + 1], tip[v + 2]));
			}
		this.tipWord = texts[nw - 1];
		// where each name rests at the end: the layer where its thread stands furthest out
		this.rest = threadPos.map((ps) =>
			ps
				.slice(2, NL)
				.reduce((best, p) => (Math.hypot(p[1], p[2]) > Math.hypot(best[1], best[2]) ? p : best))
		);
		this.mode = 'live';
		this.ride = 0;
		this.modeTime = 0;
		this.rewound = false;
		this.yawOff = this.pitchOff = 0;
		this.distMul = 1;
		this.starts = [];
		let t = 1.5;
		for (let l = 0; l < NL; l++) {
			this.starts.push(t);
			t += layerDur(l);
		}
		this.starts.push(t);
		this.total = t + 4;
		this.time = 0;
		this.ready = true;
		this.reading = false;
		console.info(`threads: ${nw} words, ${(nv / 1e6).toFixed(2)}M vertices`);
		this.onStatus({ caption: '', busy: false });
	}

	/** Layer-time (0..28) reached at clock time t. */
	private progress(t: number) {
		if (t <= this.starts[0]) return 0;
		for (let l = 0; l < this.layers; l++)
			if (t < this.starts[l + 1])
				return l + (t - this.starts[l]) / (this.starts[l + 1] - this.starts[l]);
		return this.layers;
	}

	/** Where the followed word's thread has reached at layer-time p. */
	private tipAt(p: number, i: number): V3 {
		const P = this.tips[i];
		let lo = 0,
			hi = P.length / 4 - 1;
		while (lo < hi) {
			const m = (lo + hi + 1) >> 1;
			if (P[m * 4 + 3] <= p) lo = m;
			else hi = m - 1;
		}
		return [P[lo * 4], P[lo * 4 + 1], P[lo * 4 + 2]];
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
		if (!this.paused && this.ready) this.time = Math.min(this.time + dt, this.total);
		this.idle += dt;
		this.scheduler.pump();
		this.raf = requestAnimationFrame(this.loop);
		try {
			this.render(dt);
		} catch (e) {
			console.error('threads frame failed', e);
		}
	};

	private render(dt: number) {
		const { device, canvas } = this.gpu;
		const prog = this.ready ? this.progress(this.time) : 0;
		const NL = this.layers;
		const total = this.journeyLength;
		// the painter starts when the reader is done
		if (this.ready && !this.reading && this.painter && prog >= NL && !this.painting)
			this.startPainting();
		// fade out while a new prompt is being read, back in when its journey begins
		this.post.fade += ((this.reading ? 0 : 1) - this.post.fade) * (1 - Math.exp(-dt * 5));
		if (this.painting?.ticks && !this.paused) {
			const target = this.painting.done
				? this.painting.ticksTotal
				: Math.max(0, this.painting.ticks - 1);
			this.paintFront = Math.min(
				target,
				this.paintFront + dt * Math.max(1.5, (target - this.paintFront) * 2)
			);
		}
		const front = this.front;
		// the ride
		if (this.ready && !this.paused) {
			this.modeTime += dt;
			if (this.mode === 'live') {
				this.ride = front;
				const finished = this.painter ? !!this.painting?.done && front >= total - 0.01 : prog >= NL;
				if (finished) {
					this.mode = this.painting?.done ? 'finale' : 'overview';
					this.modeTime = 0;
				}
			} else if (this.mode === 'finale' && this.modeTime > 9) {
				this.mode = 'overview';
				this.modeTime = 0;
			} else if (this.mode === 'rewind') {
				this.ride = Math.max(0, this.ride - (dt * total) / 70);
				if (this.ride <= 0) {
					this.mode = 'overview';
					this.modeTime = 0;
					this.rewound = true;
				}
			}
		}
		// where the camera wants to be: beside the ride point, a little ahead in the direction of travel
		const rx = this.xAt(this.ride);
		const inPainter = this.ride > NL + HANDOFF * 0.5;
		const lift = inPainter ? 0.4 : 0;
		// in the painter, whether riding along, back, or placed by hand: the picture and its ring of words a little
		// right of centre, nearly face on, with the trail behind it to the left
		// far enough back that the threads near the ride fit on screen (with room for their names)
		const near = Math.max(
			...Array.from(
				{ length: 4 },
				(_, i) => this.reach[Math.max(0, Math.min(NL, Math.floor(this.ride) - 1 + i))] ?? WORLD
			)
		);
		const fit = Math.max(12, 3.3 * (near + 0.8));
		// in the painter: from the side, like the reader, a little ahead of the front; its threads along the axis and
		// its slices standing above them
		let want = inPainter
			? { c: [rx - 5, 0.3, 0] as V3, dist: 31, yaw: 0.34, pitch: 0.2 }
			: { c: [rx - DX * 0.8, 0, 0] as V3, dist: fit, yaw: 0.55, pitch: 0.18 };
		if (this.mode === 'rewind' && !inPainter)
			want = { c: [rx + DX * 0.8, lift, 0], dist: fit, yaw: -0.55, pitch: 0.15 };
		if (this.mode === 'manual' && !inPainter)
			want = { c: [rx, lift, 0], dist: fit, yaw: 0.35, pitch: 0.2 };
		if (this.mode === 'finale' && this.painting) {
			// face the finished picture
			want = { c: this.painting.finalAt, dist: 33, yaw: 0, pitch: 0.02 };
		}
		if (this.mode === 'overview') {
			const a = this.xAt(0),
				z = this.painting ? this.painting.finalAt[0] + 14 : this.xAt(total);
			// nearly side on, so the whole journey runs across the screen, turning very slowly
			want = {
				c: [(a + z) / 2, 1, 0],
				dist: (z - a) * 1.0,
				yaw: 0.3 + 0.12 * Math.sin(this.modeTime * 0.05),
				pitch: 0.22
			};
		}
		if (prog < 1 && this.mode === 'live') want.dist = Math.max(10, fit * 0.85);
		const k = 1 - Math.exp(-dt * 2.5);
		const cam = this.cam;
		cam.c = lerp3(cam.c, want.c, k);
		cam.dist += (want.dist - cam.dist) * k;
		cam.yaw += (want.yaw - cam.yaw) * k;
		cam.pitch += (want.pitch - cam.pitch) * k;
		const target = cam.c,
			yaw = cam.yaw + this.yawOff,
			pitch = Math.max(-1.4, Math.min(1.4, cam.pitch + this.pitchOff)),
			dist = cam.dist * this.distMul;
		const eye: V3 = [
			target[0] + dist * Math.cos(pitch) * Math.sin(yaw),
			target[1] + dist * Math.sin(pitch),
			target[2] + dist * Math.cos(pitch) * Math.cos(yaw)
		];
		const aspect = canvas.width / canvas.height,
			fov = (38 * Math.PI) / 180;
		const vp = mat4.multiply(
			mat4.perspective(fov, aspect, 0.05, 500),
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
		f.set([this.time, aspect, (2 * Math.tan(fov / 2)) / canvas.height, 0], 28);
		device.queue.writeBuffer(this.frame, 0, f);

		// what is shown: everything computed so far, or, when the visitor has ridden back, only up to where they are
		const cut =
			this.mode === 'manual' || this.mode === 'rewind' ? Math.min(front, this.ride) : front;
		const shownLayers = Math.min(prog, cut);
		const paintCut = cut - NL - HANDOFF;
		this.threads.now = shownLayers;
		this.threads.width = 0.9;
		this.quiet += ((this.mode === 'finale' ? 1 : 0) - this.quiet) * (1 - Math.exp(-dt * 2));
		const lineGain = 1 - 0.92 * this.quiet;
		this.threads.gain = 0.9 * lineGain;
		this.threads.fresh = 3;
		this.links.now = shownLayers;
		this.links.width = 0.8;
		this.links.gain = 0.8 * lineGain;
		this.links.fresh = 2;
		this.links.fade = 1;
		if (this.ready) {
			const names = this.painting?.names(paintCut, 0.013 * dist * 1.1) ?? null;
			this.labels.forEach((lb, i) => {
				// names ride the tips while the reader runs; in the painter they stand where the latest reading slice
				// reads each word most; they stay the same size on screen
				lb.height = 0.016 * dist;
				lb.alpha = 1;
				if (this.mode === 'overview') {
					lb.pos = [this.rest[i][0], this.rest[i][1] + 0.14, this.rest[i][2]];
					return;
				}
				if (inPainter && names) {
					lb.pos = names[i].pos;
					lb.alpha = 0.3 + 0.7 * names[i].light;
					lb.height = 0.013 * dist;
					return;
				}
				const p = this.tipAt(Math.min(shownLayers, this.ride + 0.5), i);
				lb.pos = [p[0], p[1] + 0.14, p[2]];
			});
			this.words?.set(this.labels);
			// ghost words near the camera, on the layers already computed
			const near: WordInstance[] = [];
			for (const g of this.ghosts) {
				const d = Math.abs(g.layer - this.ride);
				if (g.layer > shownLayers + 1e-3 || (this.mode !== 'overview' && d > 3.5)) continue;
				const a =
					(this.mode === 'overview' ? 0.25 : 1 - d / 3.5) * (0.35 + 0.65 * Math.min(1, g.prob * 3));
				near.push({
					word: g.text,
					pos: [g.pos[0], g.pos[1] - 0.1, g.pos[2]],
					height: 0.012 * dist,
					color: wordColour(this.wordCount > 1 ? g.word / (this.wordCount - 1) : 0.5),
					alpha: a * 0.8,
					anchor: 0.5
				});
			}
			this.ghostLayer?.set(near);
		}

		const enc = device.createCommandEncoder();
		const timing = this.scheduler.timer.span('render');
		const pass = enc.beginRenderPass({
			colorAttachments: [
				{ view: this.post.sceneView, loadOp: 'clear', storeOp: 'store', clearValue: [0, 0, 0, 1] }
			],
			depthStencilAttachment: {
				view: this.post.depthView,
				depthClearValue: 1,
				depthLoadOp: 'clear',
				depthStoreOp: 'discard'
			},
			timestampWrites: timing.first
		});
		if (this.ready) {
			// pictures first: where they are bright they hide the lines and words behind them
			if (this.painting) {
				this.painting.place(paintCut);
				this.planes.draw(pass, eye);
				this.painting.drawGallery(pass, paintCut);
			}
			this.threads.draw(pass, canvas.width, canvas.height);
			this.links.draw(pass, canvas.width, canvas.height);
			this.painting?.drawLines(
				pass,
				canvas.width,
				canvas.height,
				Math.max(0, Math.min(1, (cut - NL) / HANDOFF)),
				lineGain,
				paintCut
			);
			this.ghostLayer?.draw(pass);
			this.words?.draw(pass);
		}
		pass.end();
		this.post.finish(enc, this.time, timing.last);
		const collect = timing.finish(enc);
		device.queue.submit([enc.finish()]);
		collect();

		if (this.ready) {
			const l = Math.min(this.layers, Math.floor(prog) + 1);
			const frac = prog - Math.floor(prog);
			const word = this.tipWord;
			let caption: string;
			const at = Math.min(NL, Math.floor(this.ride) + 1);
			const tick = Math.floor(this.ride - NL - HANDOFF);
			const passes = this.painting?.steps ?? this.options.steps;
			if (this.paintError) caption = `The painter stopped: ${this.paintError}`;
			else if (this.mode === 'finale')
				caption = `Your words, painted: 1.4 billion ternary weights read them, 3.7 billion turned noise into this in ${passes} ${passes === 1 ? 'pass' : 'passes'}, all in this browser tab.`;
			else if (this.mode === 'overview')
				caption = this.painting?.done
					? `The whole journey: 1.4 billion weights read your ${this.wordCount} words, 3.7 billion painted them. Drag to turn, scroll to come closer, or drag along the timeline to go back.`
					: `All ${NL} layers: 1.4 billion weights moved your ${this.wordCount} words, left to right. Drag to turn, scroll to come closer, or drag along the timeline to go back.`;
			else if ((this.mode === 'rewind' || this.mode === 'manual') && !inPainter)
				caption = `Looking back at layer ${at} of ${NL}. Each jagged step is the sum of everything that layer added to the word.`;
			else if (this.ride >= NL && this.ride - NL - HANDOFF < 0.5)
				caption = this.painter
					? `The reading stops after layer 21: the painter needs nothing later. It takes each word as it stood after layers 7, 14 and 21 (the beads); the 1.7B reads the prompt once more, padded to the 512 rows the painter expects, and here the adapter merges each word's three states into one (${Math.round((this.painting?.encoded ?? 0) * 100)}%).`
					: 'The painter is still downloading.';
			else if (inPainter && this.painting && tick >= this.painting.ticksTotal)
				caption = `The finished picture, decoded from the painter's last latent: ${passes} ${passes === 1 ? 'pass' : 'passes'} of 3.7 billion ternary weights, starting from noise.`;
			else if (inPainter) {
				// what the ride point has reached: a block (its reading, then its picture) or a stage of the decoder
				const tf = Math.max(0, this.ride - NL - HANDOFF);
				const p = Math.min(passes - 1, Math.floor(tf / TICKS_PER_STEP)),
					r = tf - p * TICKS_PER_STEP;
				const head = `The painter, pass ${p + 1} of ${passes}`;
				caption =
					r < 25.5
						? `${head}, block ${Math.min(25, Math.max(1, Math.round(r)))} of 25. Inside each block the picture's 1,024 patches read your words; the lower row shows where, in the words' colours. Your words run beneath as a cable: a word the picture reads hard climbs out to the spot it is read most, and its name lights up. The arcs are the words reading each other. Then the block changes the picture (the upper row).`
						: r < 27.5
							? `${head}, decoding. The small pictures were only the decoder's first stage (64 × 64); now the last block's guess of the finished picture goes through the rest of it, doubling each time: ${r < 26.5 ? '128 × 128' : '256 × 256'}. The decoder never reads your words: it only sharpens what the blocks painted, so the threads run on under it, untouched.`
							: `${head}: its guess of the finished picture, decoded in full at 512 × 512. The next pass starts from where this one moved the noise, and reads your words again.`;
			} else if (prog < 1)
				caption =
					frac < ATTN_SHARE
						? `Layer 1 of ${NL}. "${word}" first takes in light from the words before it: each coloured stroke is one attention head reading one word.`
						: `Layer 1 of ${NL}. Then its 6,144 neurons each push it a little: ${Math.round(((frac - ATTN_SHARE) / (1 - ATTN_SHARE)) * 6144).toLocaleString()} of 6,144 added so far. Each push is one column of 2,048 weights, each −1, 0 or +1.`;
			else if (this.painter && TAP_LAYERS.some((t) => prog >= t && prog < t + 1.5))
				caption = `Layer ${l} of ${NL}. The beads on the brighter ring mark where the painter will take each word: as it stands after layers 7, 14 and 21.`;
			else
				caption = `Layer ${l} of ${NL}. Each thread is one of your words; each jagged step is the sum of everything that layer adds to it. The faint italic words are what the model would say each word has become.`;
			this.onStatus({
				caption,
				busy: false,
				front,
				ride: this.ride,
				total,
				marks: [
					...Array.from({ length: NL + 1 }, (_, l) => l),
					...(this.painter
						? Array.from(
								{ length: (this.painting?.steps ?? this.options.steps) + 1 },
								(_, s) => NL + HANDOFF + s * TICKS_PER_STEP
							)
						: [])
				],
				mode: this.mode
			});
		}
	}
}

// ---- helpers

function median(xs: number[]) {
	const s = [...xs].sort((a, b) => a - b);
	return s[Math.floor(s.length / 2)] || 1;
}

/**
 * Top two principal directions of the rows of X (already centred), by subspace iteration. breathe() is awaited between
 * iterations (it gives frames a turn); if it returns true the work is abandoned (null).
 */
async function topTwo(
	X: Float64Array[],
	D: number,
	breathe: () => Promise<boolean>
): Promise<[Float64Array, Float64Array] | null> {
	let a = new Float64Array(D).map((_, i) => Math.sin(i * 12.9898) * 0.5 + 0.5);
	let b = new Float64Array(D).map((_, i) => Math.cos(i * 78.233));
	const apply = (v: Float64Array) => {
		const out = new Float64Array(D);
		for (const x of X) {
			let d = 0;
			for (let i = 0; i < D; i++) d += x[i] * v[i];
			for (let i = 0; i < D; i++) out[i] += x[i] * d;
		}
		return out;
	};
	const unit = (v: Float64Array) => {
		const n = Math.sqrt(v.reduce((s, x) => s + x * x, 0)) || 1;
		return v.map((x) => x / n);
	};
	for (let it = 0; it < 40; it++) {
		if (await breathe()) return null;
		a = unit(apply(a));
		b = apply(b);
		const d = b.reduce((s, x, i) => s + x * a[i], 0);
		b = unit(b.map((x, i) => x - d * a[i]));
	}
	return [a, b];
}

/** A word's colour by its place in the prompt: glacier, through bone, to ember (linear rgb). */
function wordColour(u: number): [number, number, number] {
	const A: V3 = [0.093, 0.578, 1.0],
		B: V3 = [0.807, 0.761, 0.672],
		C: V3 = [1.0, 0.434, 0.069];
	const mix = (p: V3, q: V3, k: number): V3 => [
		p[0] + (q[0] - p[0]) * k,
		p[1] + (q[1] - p[1]) * k,
		p[2] + (q[2] - p[2]) * k
	];
	return u < 0.5 ? mix(A, B, u * 2) : mix(B, C, (u - 0.5) * 2);
}

const lerp3 = (a: V3, b: V3, k: number): V3 => [
	a[0] + (b[0] - a[0]) * k,
	a[1] + (b[1] - a[1]) * k,
	a[2] + (b[2] - a[2]) * k
];
