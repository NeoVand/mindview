// Live study: threads. Every word of the prompt is a thread falling through the layers of Ternary Bonsai 1.7B.
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
import type { ReaderTrace, Recording } from './recording';

type V3 = [number, number, number];

/** What the model is: the layers the painter listens to (the reader runs up to the last), and sizes for the captions. */
export interface ThreadsModel {
	taps: number[];
	readerWeights: string; // e.g. '1.4 billion'
	painterWeights: string;
	textRows: number; // rows the painter's text side is padded to
	/** Fast: 2 passes, the first a sketch at 256 x 256 (the one-file model with its few-step LoRA). */
	fast: boolean;
	name?: string; // the model file, for recordings
}

/** The lab's models: the 1.7B read to layer 21 and the painter bundle. */
export const LAB_MODEL: ThreadsModel = {
	taps: [7, 14, 21],
	readerWeights: '1.4 billion',
	painterWeights: '3.7 billion',
	textRows: 512,
	fast: false
};

export interface ThreadsStatus {
	caption: string;
	busy: boolean;
	front?: number; // how far the computation has got, in journey units
	ride?: number; // where the camera is, in journey units
	total?: number; // length of the journey in units (28 layers, then the painter)
	marks?: number[]; // where layers and painting steps begin, in units
	mode?: string;
	done?: boolean; // the picture is finished
}

const WORLD = 5; // radius of the bundle in world units
const DX = 1.3; // length of one layer along the journey
const ATTN_SHARE = 0.3; // fraction of a layer's length given to the attention sum
const GAP = 5; // world length of the handoff from reader to painter
const NEURON_STRIDE = 4; // draw every 4th neuron's running total
const TICK_LEN = 3.2; // world length of one painter tick before a painting has laid its columns out
const HANDOFF = 2; // journey units taken by the handoff
const PAINT_RATE = 1.7; // painter ticks shown per second (a tick is a block, or a stage of the decoder)
const OTHER = 250,
	PLATE = 251,
	TAP_PLATE = 252; // palette slots

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
const layerDur = (l: number) => (l === 0 ? 4.5 : l === 1 ? 2.5 : 1.4);
const HANDOFF_SECS = 2; // the ride's time through the handoff, once the reader's animation is done

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
	// the drawing's resolution, as a share of the screen's: lowered while the painter works and frames run slow (the
	// GPU draws the scene and paints in the same frames), raised again when they run fast
	private scale = 1;
	private slowFor = 0;
	private fastFor = 0;
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
	// before a run, the landing shows a recorded reading, finished and turning slowly ('idle', see preview)
	mode: 'live' | 'finale' | 'overview' | 'rewind' | 'manual' | 'idle' = 'live';
	ride = 0; // where along the journey the camera is, in layers
	private modeTime = 0;
	private rewound = false;
	private yawOff = 0;
	private pitchOff = 0;
	private distMul = 1;
	private panOff: V3 = [0, 0, 0]; // the visitor moved the view (world units)
	// the last frame's camera, for turning a drag on the screen into a move in the world
	private view = {
		right: [1, 0, 0] as V3,
		up: [0, 1, 0] as V3,
		dist: 12,
		fov: (38 * Math.PI) / 180
	};
	private idling = false; // a preview (see preview)
	/** While previewing: where the picture sits on the screen (clip space, -1..1), beside the page's own text. */
	idleShift: [number, number] = [0, 0];
	private shift: [number, number] = [0, 0];
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
	private handoffAt = -1; // time at which the reader's animation ended and the ride entered the handoff
	private paintError = '';
	// a new prompt: the old scene fades out while the new one is read (and nothing is painted from the old one)
	private reading = false;
	private readId = 0;
	private readingCaption = ''; // what the reading is doing, while it runs
	private quiet = 0; // 0..1: the finished picture on its own (lines dimmed, since they run straight at the camera)
	private far = 0; // 0..1: the whole journey seen from far away (lines dimmed)
	// a recorded run being played (nothing is computed), or one being recorded from a live run (record = true)
	private playback?: Recording;
	private recorder?: Recording;
	record = false;

	constructor(
		private gpu: GPU,
		private llm: BonsaiLLM | undefined,
		private onStatus: (s: ThreadsStatus) => void,
		readonly model: ThreadsModel = LAB_MODEL
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

	/** A painter is downloading (attachPainter will follow): the ride waits for it after the reader. */
	painterComing = false;

	/** The reader is ready (a piece made to play a recording can then read live). */
	attachReader(llm: BonsaiLLM) {
		this.llm = llm;
	}

	/** The run just recorded (record = true), once its painting is done. */
	get recording(): Recording | undefined {
		return this.painting?.recorded ? this.recorder : undefined;
	}

	/** Whether there is a painting to follow the reader: a painter to compute it, or a recording to play. */
	private get canPaint() {
		return !!this.painter || !!this.playback;
	}

	private startPainting() {
		if (!this.canPaint) return;
		const x0 = (this.layers * DX) / 2;
		const painting = new Painting(
			this.gpu.device,
			this.frame,
			this.planes,
			{ x0: x0 + GAP, readerEnd: x0, layers: this.layers, taps: this.model.taps },
			this.palette,
			this.wordCount,
			this.model.fast ? 2 : this.options.steps,
			this.model.fast
		);
		this.painting = painting;
		try {
			const taps = this.threadPos.map((ps) => this.model.taps.map((l) => ps[l]));
			if (this.playback) painting.play(this.playback.paint, taps);
			else if (this.painter && this.llm)
				painting.start(
					this.painter,
					this.llm,
					this.scheduler,
					this.prompt,
					taps,
					this.options.seed,
					this.recorder?.paint
				);
		} catch (err) {
			this.paintError = err instanceof Error ? err.message : String(err);
		}
	}

	private get paintTicks() {
		return this.painting?.ticksTotal ?? TICKS_PER_STEP * this.options.steps;
	}

	/** Journey units: 0..28 the reader's layers, then the handoff, then one unit per painter readout. */
	private get journeyLength() {
		return this.canPaint ? this.layers + HANDOFF + this.paintTicks : this.layers;
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
		if (p < this.layers || !this.canPaint) return p;
		const h = this.handoffProgress;
		if (h < 1) return this.layers + HANDOFF * h;
		return this.layers + HANDOFF + this.paintFront;
	}

	/**
	 * How far the ride is through the handoff (0..1): the painter's text side as it is computed, but never faster
	 * than HANDOFF_SECS once the reader's animation is done (the computation usually runs ahead of the show).
	 */
	private get handoffProgress() {
		if (!this.painting) return 0;
		const clock = this.handoffAt < 0 ? 0 : (this.time - this.handoffAt) / HANDOFF_SECS;
		const p = Math.min(this.painting.encoded, clock);
		return this.painting.ticks >= 1 && p >= 1 ? 1 : Math.min(p, 0.999);
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

	/** The visitor moved the view (dx, dy in CSS pixels): the world under the pointer follows it. */
	pan(dx: number, dy: number) {
		const { right, up, dist, fov } = this.view;
		const k = (dist * 2 * Math.tan(fov / 2)) / Math.max(1, this.gpu.canvas.clientHeight);
		for (let i = 0; i < 3; i++) this.panOff[i] += (-right[i] * dx + up[i] * dy) * k;
		this.idle = 0;
	}

	/** Forget where the visitor moved the view (each of the piece's own views starts centred). */
	private recentre() {
		this.panOff = [0, 0, 0];
	}

	/** The visitor moved along the journey (in layers); the computation already done stays drawn. */
	rideTo(layer: number) {
		this.ride = Math.max(0, Math.min(this.front, layer));
		this.cam.c = [this.xAt(this.ride), 0, 0];
		this.mode = 'manual';
		this.recentre();
		this.idle = 0;
	}

	/** Paint again with the current options (steps, seed), keeping what the reader did. */
	repaint() {
		if (!this.painter || !this.ready || this.playback) return;
		this.scheduler.clear();
		this.painting?.destroy();
		this.painting = undefined;
		this.planes.clear();
		this.paintFront = 0;
		this.handoffAt = -1;
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
		this.recentre();
		this.idle = 0;
	}

	/** Back to riding alongside the computation. */
	follow() {
		this.mode = this.front >= this.journeyLength - 0.01 ? 'overview' : 'live';
		this.modeTime = 0;
		this.recentre();
		this.distMul = 1;
		this.idle = 0;
	}

	/** The finished picture, face on (or back to riding along while it is still being made). */
	showPicture() {
		if (!this.painting?.done) return this.follow();
		this.ride = this.front;
		this.mode = 'finale';
		this.modeTime = 0;
		this.recentre();
		this.distMul = 1;
		this.idle = 0;
	}

	seek(dt: number) {
		if (this.mode === 'manual' || this.mode === 'overview' || this.mode === 'rewind') {
			this.rideTo(this.ride + dt / 2);
			return;
		}
		this.time = Math.max(0, Math.min(this.total, this.time + dt));
	}

	/** What the reading is doing (shown until the new journey begins). */
	private say(caption: string) {
		this.readingCaption = caption;
		this.onStatus({ caption, busy: true });
	}

	/** Read a prompt live: the reader runs here, then the painter when it is attached. */
	read(prompt: string) {
		return this.run(prompt);
	}

	/** Play a recorded run (see recording.ts): drawn as a live run is, from the numbers in the recording. */
	play(rec: Recording) {
		return this.run(rec.prompt, rec);
	}

	/**
	 * Before a run: a recorded run's reading, finished (every thread drawn as the run left it), turning slowly, with
	 * nothing painted. A run (play, read) starts from the beginning.
	 */
	preview(rec: Recording) {
		return this.run(rec.prompt, rec, true);
	}

	private async run(prompt: string, rec?: Recording, idle = false) {
		const id = ++this.readId;
		this.idling = idle;
		this.reading = true;
		this.prompt = prompt;
		this.playback = rec;
		this.recorder = undefined;
		this.readingCaption = rec ? 'Loading the recording' : 'Reading your words';
		this.painting?.destroy();
		this.painting = undefined;
		this.planes.clear();
		this.scheduler.clear();
		this.paintFront = 0;
		this.handoffAt = -1;
		this.paintError = '';
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
		const trace = rec ? rec.reader : await this.gather(prompt, breathe);
		if (!trace || (await breathe())) return;
		if (this.record && !rec)
			this.recorder = {
				prompt,
				seed: this.options.seed,
				steps: this.model.fast ? 2 : this.options.steps,
				fast: this.model.fast,
				model: this.model.name ?? '',
				reader: trace,
				paint: []
			};
		const dev = this.gpu.device,
			{ tokens, W, N, NL, resid, act, vals, probs, proj } = trace,
			{ D, F, H, HD, KV } = trace.dims,
			G = H / (KV / HD),
			nw = W.length,
			n = vals[0].length / KV,
			per = F + H * HD,
			TAP_LAYERS = this.model.taps;
		const [e0, e1] = trace.e;
		const { scale, mu } = centre(resid, D);
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

		// ---- the threads: partial sums, term by term
		this.say('Laying out the threads');
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
		// beads where the painter will take each word: its place after the tapped layers
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

		// the model's own reading of every word at every layer, beside its thread
		this.ghosts = [];
		for (let l = 0; l <= NL; l++)
			for (let i = 0; i < nw; i++) {
				const g = trace.ghosts[l * nw + i];
				if (g.text)
					this.ghosts.push({ text: g.text, layer: l, word: i, prob: g.prob, pos: threadPos[i][l] });
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
		const texts = W.map((t) => clean(tokens[t]));
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
		this.recentre();
		this.starts = [];
		let t = 1.5;
		for (let l = 0; l < NL; l++) {
			this.starts.push(t);
			t += layerDur(l);
		}
		this.starts.push(t);
		this.total = t + 4;
		this.time = 0;
		if (idle) {
			// the reading as the run left it
			this.time = this.starts[NL] + 1;
			this.mode = 'idle';
		}
		this.ready = true;
		this.reading = false;
		console.info(`threads: ${nw} words, ${(nv / 1e6).toFixed(2)}M vertices`);
		this.onStatus({ caption: '', busy: false });
	}

	/**
	 * The reader's pass over the prompt, and what the piece needs from it: the states, activations, attention and
	 * values at every layer, the two directions the threads are drawn along, every column of Wo and Wdown projected
	 * on them (on the GPU), and the logit lens's reading of each word at each layer. Null if a newer prompt came in.
	 */
	private async gather(
		prompt: string,
		breathe: () => Promise<boolean>
	): Promise<ReaderTrace | null> {
		const llm = this.llm;
		if (!llm) throw new Error('The reader is not loaded.');
		const c = llm.config,
			L = llm.layout,
			dev = this.gpu.device;
		const N = MAX_TOKENS,
			D = c.dim,
			F = c.ffn,
			H = c.heads,
			HD = c.headDim,
			KV = c.kvHeads * HD,
			Q = H * HD;
		// the painter uses the words only as they stand after its last tap, so the reader stops there
		const TAP_LAYERS = this.model.taps;
		const NL = TAP_LAYERS[TAP_LAYERS.length - 1];
		const t0 = performance.now(),
			laps: string[] = [];
		const lap = (what: string) => laps.push(`${what} ${(performance.now() - t0).toFixed(0)}`);
		this.say(`Reading your words through ${NL} layers`);
		const r = await llm.prefill(
			llm.tokenizer.encode(chatPrompt(prompt)).slice(0, MAX_TOKENS),
			this.scheduler,
			NL
		);
		if (await breathe()) return null;
		const n = r.ids.length;
		const end = r.tokens.findIndex((t, i) => i > 3 && t.startsWith('<|im_end'));
		const W = Array.from({ length: Math.max(1, (end > 0 ? end : n) - 3) }, (_, i) => 3 + i);
		const nw = W.length;

		// ---- read back what the pass left in the arena
		this.say('Collecting every partial sum');
		const regions: { offset: number; count: number }[] = [];
		for (let l = 0; l <= NL; l++)
			for (const t of W) regions.push({ offset: L.resid + (l * N + t) * D, count: D });
		for (let l = 0; l < NL; l++)
			for (const t of W) regions.push({ offset: L.act + (l * N + t) * F, count: F });
		for (let l = 0; l < NL; l++) regions.push({ offset: L.v + l * N * KV, count: n * KV });
		for (let l = 0; l < NL; l++)
			for (let h = 0; h < H; h++)
				regions.push({ offset: L.probs + ((l * H + h) * N + W[0]) * N, count: nw * N });
		lap('prefill');
		const got = await llm.readMany(regions);
		lap('readback');
		if (await breathe()) return null;
		let k = 0;
		const resid = Array.from({ length: NL + 1 }, () => W.map(() => got[k++]));
		const act = Array.from({ length: NL }, () => W.map(() => got[k++]));
		const vals = Array.from({ length: NL }, () => got[k++]);
		const probs = Array.from({ length: NL }, () => Array.from({ length: H }, () => got[k++])); // [l][h][(t - W0) * N + s]

		// ---- two fixed directions: the main axes along which the words move, each layer rescaled to a common size
		const { scale, mu } = centre(resid, D);
		const X: Float64Array[] = [];
		for (let l = 1; l <= NL; l++)
			for (const v of resid[l]) {
				const x = new Float64Array(D);
				for (let i = 0; i < D; i++) x[i] = v[i] / scale[l] - mu[l][i];
				X.push(x);
			}
		const two = await topTwo(X, D, breathe);
		lap('axes');
		if (!two) return null;
		const [e0, e1] = two;

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
		const proj = new Float32Array(read.getMappedRange()).slice();
		lap('projection');
		read.destroy();
		out.destroy();
		params.destroy();
		Ebuf.destroy();
		if (await breathe()) return null;

		// ---- the model's own reading of every word at every layer
		this.say('Asking the model what each word has become at every layer');
		const vecs = new Float32Array((NL + 1) * nw * D);
		for (let l = 0; l <= NL; l++)
			for (let i = 0; i < nw; i++) vecs.set(resid[l][i], (l * nw + i) * D);
		const lens = await llm.readout(vecs, (NL + 1) * nw, this.scheduler);
		const ghosts = Array.from({ length: (NL + 1) * nw }, (_, k) => ({
			text: clean(llm.tokenizer.decode([lens.ids[k]])),
			prob: lens.probs[k]
		}));
		lap('lens');
		console.info(`threads reading, ms from the start: ${laps.join(', ')}`);
		return {
			tokens: r.tokens,
			W,
			NL,
			N,
			dims: { D, F, H, HD, KV },
			resid,
			act,
			vals,
			probs,
			e: [Float32Array.from(e0), Float32Array.from(e1)],
			proj,
			ghosts
		};
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
		const screen = Math.min(window.devicePixelRatio || 1, 2);
		const dpr = Math.max(Math.min(screen, 1), screen * this.scale);
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
		if (!this.paused && this.ready && !this.idling)
			this.time = Math.min(this.time + dt, this.total);
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
		// the painter starts computing as soon as the words are read (the reader's animation is a replay); the show
		// reaches its work only after the reader's last layer, and never runs ahead of what has been computed
		if (this.ready && !this.reading && !this.idling && this.canPaint && !this.painting) {
			this.startPainting();
			// the painter came after the reader had finished and the whole was on show: ride on into it
			if (this.mode === 'overview') {
				this.mode = 'live';
				this.modeTime = 0;
			}
		}
		this.painting?.update();
		if (this.ready && prog >= NL && this.handoffAt < 0) this.handoffAt = this.time;
		// the GPU queue: bigger slices while nothing on screen needs to stay smooth (the reading, the handoff); while
		// the painting is on show, never less than a third of each frame, however heavy the scene
		const frameMs = this.scheduler.frameMs;
		this.scheduler.floor =
			!this.ready || this.reading
				? 30
				: this.painting && this.painting.ticks >= 1 && prog >= NL
					? Math.max(4, frameMs / 3)
					: 10;
		// while the painter works on a slow frame rate, draw at a lower resolution (a step at a time, at most half the
		// screen's), and back up once frames come fast again
		const working = !!this.painting && !this.painting.done && this.scheduler.pending > 0;
		this.slowFor = working && frameMs > 21 ? this.slowFor + dt : 0;
		this.fastFor = frameMs < 18 || !working ? this.fastFor + dt : 0;
		if (this.slowFor > 1.5 && this.scale > 0.5) {
			this.scale = Math.max(0.5, this.scale * 0.8);
			this.slowFor = 0;
			this.resize();
		} else if (this.fastFor > 3 && this.scale < 1) {
			this.scale = Math.min(1, this.scale / 0.8);
			this.fastFor = 0;
			this.resize();
		}
		// fade out while a new prompt is being read, back in when its journey begins
		this.post.fade += ((this.reading ? 0 : 1) - this.post.fade) * (1 - Math.exp(-dt * 5));
		if (this.painting?.ticks && !this.paused && prog >= NL && this.handoffProgress >= 1) {
			const target = this.painting.done
				? this.painting.ticksTotal
				: Math.max(0, this.painting.ticks - 1);
			// a steady, readable pace (one block every ~0.6 s), never ahead of what has been computed; when the
			// computation is far ahead (a fast painter), a little quicker, so the picture is not kept waiting long
			const lag = target - this.paintFront;
			const rate = PAINT_RATE * (1 + Math.max(0, lag - TICKS_PER_STEP) / TICKS_PER_STEP);
			this.paintFront = Math.min(target, this.paintFront + dt * rate);
		}
		const front = this.front;
		// the ride
		if (this.ready && !this.paused) {
			this.modeTime += dt;
			if (this.mode === 'live') {
				this.ride = front;
				// with a painter on its way, the ride waits at the handoff for it
				const finished = this.canPaint
					? !!this.painting?.done && front >= total - 0.01
					: prog >= NL && !this.painterComing;
				if (finished) {
					this.mode = this.painting?.done ? 'finale' : 'overview';
					this.modeTime = 0;
				}
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
		const aspect = canvas.width / canvas.height;
		if (this.mode === 'overview') {
			const a = this.xAt(0),
				z = this.painting ? this.painting.finalAt[0] + 14 : this.xAt(total);
			// nearly side on and a little from above, so the whole journey runs across the screen, turning very slowly; on
			// a screen held upright, far enough back for its whole length to fit the width, slanting down the screen
			const upright = aspect < 0.9;
			want = {
				c: [(a + z) / 2, 2, 0],
				dist: (z - a) * 0.92 * (upright ? (1.6 / aspect) * 0.9 : 1),
				yaw: (upright ? 0.42 : 0.18) + 0.08 * Math.sin(this.modeTime * 0.05),
				pitch: upright ? 0.42 : 0.3
			};
		}
		if (this.mode === 'idle') {
			// before a run: the finished reading, turning slowly all the way round and gently up and down
			const a = this.xAt(0),
				z = this.xAt(NL);
			want = {
				c: [(a + z) / 2, 0, 0],
				dist: Math.max(16, (z - a) * 1.25 + 10),
				yaw: 0.6 + this.modeTime * 0.07,
				pitch: 0.22 + 0.1 * Math.sin(this.modeTime * 0.11)
			};
		}
		if (prog < 1 && this.mode === 'live') want.dist = Math.max(10, fit * 0.85);
		const k = 1 - Math.exp(-dt * 2.5);
		const cam = this.cam;
		cam.c = lerp3(cam.c, want.c, k);
		cam.dist += (want.dist - cam.dist) * k;
		cam.yaw += (want.yaw - cam.yaw) * k;
		cam.pitch += (want.pitch - cam.pitch) * k;
		// the views are framed for a landscape screen; on a narrower one (a phone held upright) step back until the
		// same width fits (the overview on such a screen is framed for it already)
		const narrow =
			this.mode === 'overview' && aspect < 0.9 ? 1 : Math.max(1, Math.pow(1.45 / aspect, 0.85));
		const target: V3 = [
				cam.c[0] + this.panOff[0],
				cam.c[1] + this.panOff[1],
				cam.c[2] + this.panOff[2]
			],
			yaw = cam.yaw + this.yawOff,
			pitch = Math.max(-1.4, Math.min(1.4, cam.pitch + this.pitchOff)),
			dist = cam.dist * this.distMul * narrow;
		const eye: V3 = [
			target[0] + dist * Math.cos(pitch) * Math.sin(yaw),
			target[1] + dist * Math.sin(pitch),
			target[2] + dist * Math.cos(pitch) * Math.cos(yaw)
		];
		const fov = (38 * Math.PI) / 180;
		// the picture moved on the screen: while previewing, beside the landing's text; the whole journey on a screen
		// held upright, a little up, clear of the caption below it
		const sh = this.idling
			? this.idleShift
			: this.mode === 'overview' && aspect < 0.9
				? [0, 0.2]
				: [0, 0];
		const ke = 1 - Math.exp(-dt * 2);
		this.shift = [
			this.shift[0] + (sh[0] - this.shift[0]) * ke,
			this.shift[1] + (sh[1] - this.shift[1]) * ke
		];
		const vp = mat4.multiply(
			mat4.multiply(
				mat4.translation([this.shift[0], this.shift[1], 0]),
				// far enough for the whole journey seen from a long way off
				mat4.perspective(fov, aspect, 0.05, Math.max(500, dist * 3))
			),
			mat4.lookAt(eye, target, [0, 1, 0])
		);
		const fwd = vec3.normalize(vec3.subtract(target, eye));
		const right = vec3.normalize(vec3.cross(fwd, [0, 1, 0]));
		const up = vec3.cross(right, fwd);
		this.view = { right: [...right] as V3, up: [...up] as V3, dist, fov };
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
		// seen whole from far away, the lines crowd into few pixels and add up: dim them
		const far = this.mode === 'overview' && this.painting ? 1 : 0;
		this.far += (far - this.far) * (1 - Math.exp(-dt * 2));
		const lineGain = (1 - 0.92 * this.quiet) * (1 - 0.7 * this.far);
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
			if (this.reading) caption = this.readingCaption;
			else if (this.paintError) caption = `The painter stopped: ${this.paintError}`;
			else if (this.mode === 'finale')
				caption = this.playback
					? `A recording of a real run: ${this.model.readerWeights} ternary weights read these words, ${this.model.painterWeights} turned noise into this in ${passes} ${passes === 1 ? 'pass' : 'passes'}, in a browser tab. Change the words to run it here, live.`
					: `Your words, painted: ${this.model.readerWeights} ternary weights read them, ${this.model.painterWeights} turned noise into this in ${passes} ${passes === 1 ? 'pass' : 'passes'}, all in this browser tab.`;
			else if (this.mode === 'overview')
				caption = this.painting?.done
					? `The whole journey: ${this.model.readerWeights} weights read your ${this.wordCount} words, ${this.model.painterWeights} painted them. Drag to turn, pinch or scroll to come closer, move it with two fingers (or a right-drag), or drag along the timeline to go back.`
					: `All ${NL} layers: ${this.model.readerWeights} weights moved your ${this.wordCount} words, left to right. Drag to turn, pinch or scroll to come closer, move it with two fingers (or a right-drag), or drag along the timeline to go back.`;
			else if ((this.mode === 'rewind' || this.mode === 'manual') && !inPainter)
				caption = `Looking back at layer ${at} of ${NL}. Each jagged step is the sum of everything that layer added to the word.`;
			else if (this.ride >= NL && this.ride - NL - HANDOFF < 0.5)
				caption = this.canPaint
					? `The reading stops after layer ${NL}: the painter needs nothing later. It takes each word as it stood after layers ${listed(this.model.taps)} (the beads); the reader reads the prompt once more, padded to the ${this.model.textRows} rows the painter expects, and here a linear map merges each word's three states into one (${Math.round(this.handoffProgress * 100)}%).`
					: 'The painter is still downloading.';
			else if (inPainter && this.painting && tick >= this.painting.ticksTotal)
				caption = `The finished picture, decoded from the painter's last latent: ${passes} ${passes === 1 ? 'pass' : 'passes'} of ${this.model.painterWeights} ternary weights, starting from noise.`;
			else if (inPainter) {
				// what the ride point has reached: a block (its reading, then its picture) or a stage of the decoder
				const tf = Math.max(0, this.ride - NL - HANDOFF);
				const p = Math.min(passes - 1, Math.floor(tf / TICKS_PER_STEP)),
					r = tf - p * TICKS_PER_STEP;
				const sketch = this.model.fast && p === 0;
				const head = sketch
					? `The painter, pass 1 of ${passes}: a sketch at a quarter of the size`
					: `The painter, pass ${p + 1} of ${passes}`;
				const patches = sketch ? '256' : '1,024';
				// a picture per block: live where the painter has the tuned lens, in a recording where it kept them
				const lens = this.playback
					? this.playback.paint.some((e) => e.kind === 'picture')
					: !!this.painter?.hasLens;
				caption =
					r < 25.5
						? `${head}, block ${Math.min(25, Math.max(1, Math.round(r)))} of 25. Inside each block the picture's ${patches} patches read your words; the lower row shows where, in the words' colours. Your words run beneath as a cable: a word the picture reads hard climbs out to the spot it is read most, and its name lights up. The arcs are the words reading each other.${lens ? ' Then the block changes the picture (the upper row).' : ''}`
						: r < 27.5
							? `${head}, decoding. ${sketch ? "The sketch's guess of the finished picture is enlarged to 512 × 512 and" : lens ? "The small pictures were only the decoder's first stage (64 × 64); now the last block's guess of the finished picture" : "The pass's guess of the finished picture"} goes through the decoder, doubling each time: ${r < 26.5 ? '128 × 128' : '256 × 256'}. The decoder never reads your words: it only sharpens what the blocks painted, so the threads run on under it, untouched.`
							: sketch
								? `${head}: the sketch, enlarged to 512 × 512. Mixed with fresh noise, it is where the second pass starts: at full size, reading your words again.`
								: p < passes - 1
									? `${head}: its guess of the finished picture, decoded in full at 512 × 512. The next pass starts from where this one moved the noise, and reads your words again.`
									: `${head}: the finished picture, decoded in full at 512 × 512.`;
			} else if (prog < 1)
				caption =
					frac < ATTN_SHARE
						? `Layer 1 of ${NL}. "${word}" first takes in light from the words before it: each coloured stroke is one attention head reading one word.`
						: `Layer 1 of ${NL}. Then its 6,144 neurons each push it a little: ${Math.round(((frac - ATTN_SHARE) / (1 - ATTN_SHARE)) * 6144).toLocaleString()} of 6,144 added so far. Each push is one column of 2,048 weights, each −1, 0 or +1.`;
			else if (this.canPaint && this.model.taps.some((t) => prog >= t && prog < t + 1.5))
				caption = `Layer ${l} of ${NL}. The beads on the brighter ring mark where the painter will take each word: as it stands after layers ${listed(this.model.taps)}.`;
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
					...(this.canPaint
						? Array.from(
								{ length: (this.painting?.steps ?? this.options.steps) + 1 },
								(_, s) => NL + HANDOFF + s * TICKS_PER_STEP
							)
						: [])
				],
				mode: this.mode,
				done: !!this.painting?.done
			});
		}
	}
}

// ---- helpers

/** 7, 14 and 21 */
function listed(xs: number[]) {
	return xs.length < 2
		? String(xs[0] ?? '')
		: `${xs.slice(0, -1).join(', ')} and ${xs[xs.length - 1]}`;
}

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

/** Each layer's typical size (the median norm of its words) and its mean word, at that size. */
function centre(resid: Float32Array[][], D: number) {
	const norm = (v: Float32Array) => Math.sqrt(v.reduce((a, x) => a + x * x, 0));
	const scale = resid.map((row) => median(row.map(norm)));
	const mu = resid.map((row, l) => {
		const m = new Float64Array(D);
		for (const v of row) for (let i = 0; i < D; i++) m[i] += v[i] / scale[l] / row.length;
		return m;
	});
	return { scale, mu };
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
