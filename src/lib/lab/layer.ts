// Lab: one layer, every number. One word goes through one layer of Ternary Bonsai 1.7B, operation by operation, laid
// out left to right as stations the camera visits in turn. Every number drawn is the model's own, from this tab's
// forward pass (or computed here from those numbers, exactly as the model computes it, where the model does not keep
// it: the normed vectors before rotation, the attention scores, the running sums).
//   the word arrives, is normed, and is read by three matrices (q, k, v): the input lies along the top of each matrix,
//   and a sweep runs across it while each row's running sum grows in the waveform on its right (a ternary matrix never
//   multiplies: each weight adds its input, subtracts it, or skips it; one scale per 128 weights)
//   q and k turn: 64 pairs per head, each a hand on a dial turned by the word's position
//   each head scores every earlier word (q . k), softmax makes the scores shares, and the shares mix the words' values
//   the output projection adds the result to the word; the MLP (gate, up, 6,144 neurons, down) adds its part
//   the word leaves, and the 28 layers' words are stacked to show where this one sits
import type { BonsaiLLM, Ternary } from '$lib/runtime/bonsai-llm';
import { MAX_TOKENS } from '$lib/runtime/bonsai-llm';
import { chatPrompt } from '$lib/runtime/tokenizer';
import { clean } from '$lib/engine/text';
import { Panels, type PanelDesc } from './panels';
import type { Scene, Stage, V3 } from './stage';
import { fmt } from './machine';

const U = 256; // cells per world unit (a 2,048 x 2,048 matrix is 8 x 8)
const STATION_GAP = 26;

export interface LayerLabel {
	text: string;
	x: number;
	y: number;
	size: number;
	alpha: number;
	align?: 'left' | 'center' | 'right';
	italic?: boolean;
}

interface Station {
	name: string;
	first: number; // its panels: [first, last)
	last: number;
	x: number; // left edge in the world
	w: number;
	h: number; // extent below y = top
	top: number;
	caption: () => string;
	sweep?: { mats: number[]; outs: { panel: number; K: number }[]; K: number };
	labels: {
		text: () => string;
		at: V3;
		size: number;
		align?: 'left' | 'center' | 'right';
		italic?: boolean;
	}[];
}

// running sums of a ternary matrix times a vector: out[m][j] = sum over the first 16 (j + 1) inputs; and, if fin is
// not 0, the finished sums side by side at fin
export const PREFIX_WGSL = /* wgsl */ `
struct P { codes: u32, scales: u32, K: u32, M: u32, x: u32, out: u32, nck: u32, fin: u32 };
@group(0) @binding(0) var<uniform> U: P;
@group(0) @binding(1) var<storage, read> CODES: array<u32>;
@group(0) @binding(2) var<storage, read> SCALES: array<f32>;
@group(0) @binding(3) var<storage, read> ARENA: array<f32>;
@group(0) @binding(4) var<storage, read_write> AUX: array<f32>;
@compute @workgroup_size(64) fn prefix(@builtin(global_invocation_id) g: vec3u) {
  let m = g.x;
  if (m >= U.M) { return; }
  let wpr = U.K / 16u;
  var acc = 0.0;
  for (var w = 0u; w < wpr; w++) {
    let bits = CODES[U.codes + m * wpr + w];
    let s = SCALES[U.scales + m * (U.K / 128u) + w / 8u];
    var part = 0.0;
    for (var q = 0u; q < 16u; q++) {
      let t = f32((bits >> (q * 2u)) & 3u) - 1.0;
      part += t * ARENA[U.x + w * 16u + q];
    }
    acc += s * part;
    AUX[U.out + m * U.nck + w] = acc;
  }
  if (U.fin != 0u) { AUX[U.fin + m] = acc; } // the finished sums together (0: not wanted)
}`;

// scratch layout (floats)
const A = {
	xr: 0, // x / rms
	g1: 2048, // the attention norm's gains
	qn: 4096, // q after its per-head norm, before rotation [16][128]
	kn: 6144, // k likewise [8][128]
	scores: 7168, // [16][T]
	dA: 9216, // what attention adds
	mr: 11264, // mid / rms
	g2: 13312, // the MLP norm's gains
	silu: 15360, // SiLU(gate) [6144]
	dM: 21504, // what the MLP adds
	hist: 23552, // SiLU scatter as a 2D histogram [96][96]
	sumd: 32768, // the chosen dot product's running sum [128]
	prefix: 36864 // running sums of the seven matrices
};
const HB = 96;

export class Layer implements Scene {
	panels: Panels;
	private prefixPipe: GPUComputePipeline;
	private prefixBind?: GPUBindGroup;
	private prefixUniforms: GPUBuffer;
	private stations: Station[] = [];
	private tokens: string[] = [];
	private ids: number[] = [];
	private words: number[] = [];
	token = 0;
	layer = 4;
	station = 0;
	/** Seconds into the current station (sweeps run in the first part of it). */
	t = 0;
	playing = false;
	busy = '';
	private readId = 0;
	private cpu?: Record<string, Float32Array> & { stack?: Float32Array[] };
	private nums: Record<string, number> = {};
	private head = 0; // the head shown up close (scores, mixing)
	private src = 0; // the earlier word whose dot product is shown
	private prefixAt = new Map<string, { off: number; nck: number; M: number }>();
	private rowPick = new Map<string, number>(); // matrix -> row shown in the inspector
	private p: Record<string, number> = {};
	private mats = new Map<
		number,
		{ key: string; tensor: Ternary; xOff: () => number; out: string }
	>();

	constructor(
		private stage: Stage,
		private llm: BonsaiLLM
	) {
		const c = llm.config;
		let prefixFloats = 0;
		for (const k of ['q', 'k', 'v', 'o', 'gate', 'up', 'down']) {
			const t = this.tensorFor(k, 0);
			this.prefixAt.set(k, { off: A.prefix + prefixFloats, nck: t.cols / 16, M: t.rows });
			prefixFloats += (t.rows * t.cols) / 16;
		}
		this.panels = new Panels(stage, llm, { capacity: 256, aux: A.prefix + prefixFloats, idx: 4 });
		this.prefixUniforms = stage.device.createBuffer({
			size: 8 * 256,
			usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
		});
		this.prefixPipe = stage.device.createComputePipeline({
			layout: 'auto',
			compute: {
				module: stage.device.createShaderModule({ label: 'running sums', code: PREFIX_WGSL }),
				entryPoint: 'prefix'
			}
		});
		void c;
		this.build();
		stage.focusPlane = { n: [0, 0, 1], d: 0 };
	}

	private tensorFor(k: string, l: number): Ternary {
		const n = {
			q: 'attn_q',
			k: 'attn_k',
			v: 'attn_v',
			o: 'attn_output',
			gate: 'ffn_gate',
			up: 'ffn_up',
			down: 'ffn_down'
		}[k];
		return this.llm.tensor(`blk.${l}.${n}.weight`);
	}

	get tokenTexts() {
		return this.words.map((t) => clean(this.tokens[t]) || '·');
	}
	get wordIndex() {
		return this.words.indexOf(this.token);
	}
	get stationNames() {
		return this.stations.map((s) => s.name);
	}
	get caption() {
		return this.cpu ? this.stations[this.station].caption() : '';
	}

	// ---- layout

	private build() {
		const c = this.llm.config,
			D = c.dim,
			F = c.ffn,
			Q = c.heads * c.headDim,
			KV = c.kvHeads * c.headDim,
			N = MAX_TOKENS,
			H = c.heads;
		const L = this.llm.layout;
		const P = this.panels;
		const tok = () => this.token;
		const lay = () => this.layer;
		const wave = (
			key: string,
			origin: V3,
			len: number,
			thick: number,
			n: number,
			a: () => { offset: number; aux?: boolean },
			vertical = false,
			extra: Partial<PanelDesc> = {}
		) => {
			const src = a();
			this.p[key] = P.add({
				origin,
				ax: vertical ? [0, -len, 0] : [len, 0, 0],
				ay: vertical ? [thick, 0, 0] : [0, -thick, 0],
				rows: 1,
				cols: n,
				kind: 'bars',
				a: { offset: src.offset, rs: 0, cs: 1, aux: src.aux },
				gain: 1,
				digits: true,
				lift: 1.2,
				...extra
			});
			this.srcs.set(this.p[key], a);
			return this.p[key];
		};
		const matrix = (key: string, k: string, origin: V3, xOff: () => number, out: string) => {
			const t = this.tensorFor(k, 0);
			const w = t.cols / U,
				h = t.rows / U;
			this.p[key] = P.add({
				origin,
				ax: [w, 0, 0],
				ay: [0, -h, 0],
				rows: t.rows,
				cols: t.cols,
				kind: 'ternary',
				tensor: t,
				x: { offset: 0 },
				gain: 1,
				pyramid: true,
				digits: true,
				lift: 1.5
			});
			this.mats.set(this.p[key], { key: k, tensor: t, xOff, out });
			return { w, h, panel: this.p[key] };
		};
		// a running sum along the right of a matrix (it shows the finished sums once the sweep is done)
		const sums = (key: string, k: string, origin: V3, h: number) => {
			const pa = this.prefixAt.get(k)!;
			const panel = P.add({
				origin,
				ax: [0, -h, 0],
				ay: [1.4, 0, 0],
				rows: 1,
				cols: pa.M,
				kind: 'bars',
				a: { offset: pa.off, rs: 0, cs: pa.nck, aux: true },
				sweepStride: 1,
				sweep: pa.nck * 16 - 1,
				gain: 1,
				digits: true,
				lift: 1.2
			});
			this.p[key] = panel;
			return panel;
		};
		// the chosen row of a matrix, spelled out: its weights, their products with the input, the running sum
		const inspector = (key: string, k: string, origin: V3) => {
			const t = this.tensorFor(k, 0);
			const w = t.cols / U;
			const wts = P.add({
				origin,
				ax: [w, 0, 0],
				ay: [0, -0.5, 0],
				rows: 1,
				cols: t.cols,
				kind: 'ternary',
				tensor: t,
				gain: 1,
				digits: true,
				lift: 1.5
			});
			const prod = P.add({
				origin: [origin[0], origin[1] - 0.8, 0],
				ax: [w, 0, 0],
				ay: [0, -0.5, 0],
				rows: 1,
				cols: t.cols,
				kind: 'ternary',
				tensor: t,
				x: { offset: 0 },
				gain: 1,
				digits: true,
				lift: 1.5
			});
			const pa = this.prefixAt.get(k)!;
			const run = P.add({
				origin: [origin[0], origin[1] - 1.6, 0],
				ax: [w, 0, 0],
				ay: [0, -1.6, 0],
				rows: 1,
				cols: pa.nck,
				kind: 'bars',
				a: { offset: pa.off, rs: 0, cs: 1, aux: true },
				gain: 1,
				digits: true,
				lift: 1.2
			});
			this.inspectors.set(key, { k, wts, prod, run });
		};

		let x = 0;
		const top = 0;
		let first = 0;
		const add = (s: Omit<Station, 'x' | 'first' | 'last'>, width: number) => {
			this.stations.push({ ...s, x, first, last: P.count });
			first = P.count;
			x += width + STATION_GAP;
		};
		const n = this.nums;
		const f = (v: number | undefined) => (v === undefined ? '' : fmt(v));

		// 1. the word arrives
		{
			const x0 = x;
			wave('x', [x0, top, 0], 8, 2.6, D, () => ({ offset: L.resid + (lay() * N + tok()) * D }));
			add(
				{
					name: 'The word arrives',
					w: 8,
					h: 2.6,
					top,
					caption: () =>
						`“${this.word}” as it enters layer ${lay() + 1}: 2,048 numbers, the running total of everything the layers before added to it. Its size (root mean square) is ${f(n.rms1)}.`,
					labels: [
						{
							text: () => `“${this.word}”, entering layer ${lay() + 1}`,
							at: [x0, top + 0.7, 0],
							size: 0.45,
							align: 'left',
							italic: true
						}
					]
				},
				8
			);
		}
		// 2. normed
		{
			const x0 = x;
			const g = 2.3;
			wave('n_x', [x0, top, 0], 8, 1.6, D, () => ({ offset: L.resid + (lay() * N + tok()) * D }));
			wave('n_xr', [x0, top - g, 0], 8, 1.6, D, () => ({ offset: A.xr, aux: true }));
			wave('n_g', [x0, top - 2 * g, 0], 8, 1.6, D, () => ({ offset: A.g1, aux: true }));
			wave('n_xn', [x0, top - 3 * g, 0], 8, 1.6, D, () => ({
				offset: L.xn + (lay() * N + tok()) * D
			}));
			add(
				{
					name: 'Normed',
					w: 8,
					h: 3 * g + 1.6,
					top,
					caption: () =>
						`First the word is scaled to a standard size: every number divided by ${f(n.rms1)}, then multiplied by its own gain (the norm's 2,048 learned weights, the only ones in the layer that are not ternary).`,
					labels: [
						{ text: () => 'the word', at: [x0 - 0.4, top - 0.8, 0], size: 0.36, align: 'right' },
						{
							text: () => `÷ ${f(n.rms1)}`,
							at: [x0 - 0.4, top - g - 0.8, 0],
							size: 0.36,
							align: 'right'
						},
						{
							text: () => '× gains',
							at: [x0 - 0.4, top - 2 * g - 0.8, 0],
							size: 0.36,
							align: 'right'
						},
						{
							text: () => '= normed',
							at: [x0 - 0.4, top - 3 * g - 0.8, 0],
							size: 0.36,
							align: 'right'
						}
					]
				},
				8
			);
		}
		// 3. q, k, v
		{
			const x0 = x;
			wave('qkv_in', [x0, top + 1.9, 0], 8, 1.5, D, () => ({
				offset: L.xn + (lay() * N + tok()) * D
			}));
			const xn = () => L.xn + (lay() * N + tok()) * D;
			const q = matrix('q', 'q', [x0, top, 0], xn, 'qraw');
			sums('q_sum', 'q', [x0 + 8.3, top, 0], q.h);
			const ky = top - q.h - 0.5;
			const k = matrix('k', 'k', [x0, ky, 0], xn, 'kraw');
			sums('k_sum', 'k', [x0 + 8.3, ky, 0], k.h);
			const vy = ky - k.h - 0.5;
			const v = matrix('v', 'v', [x0, vy, 0], xn, 'v');
			sums('v_sum', 'v', [x0 + 8.3, vy, 0], v.h);
			const iy = vy - v.h - 1.4;
			inspector('qkv', 'q', [x0, iy, 0]);
			add(
				{
					name: 'Three projections',
					w: 9.7,
					h: -(iy - 3.4 - top),
					top: top + 3.4,
					sweep: {
						mats: [q.panel, k.panel, v.panel],
						outs: [
							{ panel: this.p.q_sum, K: D },
							{ panel: this.p.k_sum, K: D },
							{ panel: this.p.v_sum, K: D }
						],
						K: D
					},
					caption: () =>
						`Three matrices read the normed word: q (2,048 rows), k and v (1,024 each). Each cell is a weight times the input above it; each row adds its cells up into the waveform on the right. No multiplications: each weight adds its input (+1, amber), subtracts it (−1, blue) or skips it (0), and one scale per 128 weights sets the size. Below, row ${this.rowPick.get('qkv') ?? 0} of q spelled out: its weights, what they make of the input, and its running sum, ${f(n.qkvRow)} at the end. Click any row to spell it out.`,
					labels: [
						{ text: () => 'the normed word', at: [x0, top + 3.7, 0], size: 0.36, align: 'left' },
						{
							text: () => 'q',
							at: [x0 - 0.35, top - 4, 0],
							size: 0.6,
							align: 'right',
							italic: true
						},
						{
							text: () => 'k',
							at: [x0 - 0.35, ky - 2, 0],
							size: 0.6,
							align: 'right',
							italic: true
						},
						{
							text: () => 'v',
							at: [x0 - 0.35, vy - 2, 0],
							size: 0.6,
							align: 'right',
							italic: true
						},
						{
							text: () => `row ${this.rowPick.get('qkv') ?? 0} of q: weights`,
							at: [x0 - 0.3, iy - 0.25, 0],
							size: 0.28,
							align: 'right'
						},
						{ text: () => '× the input', at: [x0 - 0.3, iy - 1.05, 0], size: 0.28, align: 'right' },
						{ text: () => 'running sum', at: [x0 - 0.3, iy - 2.4, 0], size: 0.28, align: 'right' }
					]
				},
				9.7
			);
		}
		// 4. rotary dials
		{
			const x0 = x;
			const dq = P.add({
				origin: [x0, top, 0],
				ax: [16, 0, 0],
				ay: [0, -4, 0],
				rows: H,
				cols: 64,
				kind: 'dial',
				a: { offset: A.qn, rs: 128, cs: 1, aux: true },
				b: { offset: 0, rs: 128, cs: 1 },
				gain: 1
			});
			this.p.dial_q = dq;
			const ky = top - 5.2;
			const dk = P.add({
				origin: [x0, ky, 0],
				ax: [16, 0, 0],
				ay: [0, -2, 0],
				rows: c.kvHeads,
				cols: 64,
				kind: 'dial',
				a: { offset: A.kn, rs: 128, cs: 1, aux: true },
				b: { offset: 0, rs: 128, cs: 1 },
				gain: 1
			});
			this.p.dial_k = dk;
			add(
				{
					name: 'Turned by position',
					w: 16,
					h: 7.2,
					top,
					caption: () =>
						`q splits into 16 heads, k into 8, each 128 numbers, normed on its own. Then each is turned: its 64 pairs of numbers are hands on dials, each turned by an angle that grows with the word's position (${tok()}) — the first pairs spin fast, the last barely move. Faint: before; bright: after. The turning is how attention knows how far apart two words are.`,
					labels: [
						{
							text: () => 'q: 16 heads × 64 pairs',
							at: [x0, top + 0.5, 0],
							size: 0.36,
							align: 'left'
						},
						{
							text: () => 'k: 8 heads × 64 pairs',
							at: [x0, ky + 0.5, 0],
							size: 0.36,
							align: 'left'
						},
						{ text: () => 'fast', at: [x0, ky - 2.5, 0], size: 0.3, align: 'left', italic: true },
						{
							text: () => 'slow',
							at: [x0 + 16, ky - 2.5, 0],
							size: 0.3,
							align: 'right',
							italic: true
						}
					]
				},
				16
			);
		}
		// 5. scores and shares
		{
			const x0 = x;
			const cell = 0.5;
			const sc = P.add({
				origin: [x0, top, 0],
				ax: [cell, 0, 0], // resized to the prompt length later
				ay: [0, -H * cell, 0],
				rows: H,
				cols: 1,
				kind: 'dense',
				a: { offset: A.scores, rs: MAX_TOKENS, cs: 1, aux: true },
				gain: 1,
				digits: true
			});
			this.p.scores = sc;
			const pr = P.add({
				origin: [x0, top - H * cell - 1.5, 0],
				ax: [cell, 0, 0],
				ay: [0, -H * cell, 0],
				rows: H,
				cols: 1,
				kind: 'dense',
				a: { offset: 0, rs: MAX_TOKENS * MAX_TOKENS, cs: 1 },
				gain: 1,
				digits: true
			});
			this.p.probs = pr;
			// the chosen head and earlier word's dot product: q_d x k_d for the 128 d, and its running sum
			const dy = top - 2 * H * cell - 3.6;
			const dp = P.add({
				origin: [x0, dy, 0],
				ax: [8, 0, 0],
				ay: [0, -1.6, 0],
				rows: 1,
				cols: 128,
				kind: 'bars',
				a: { offset: 0, rs: 0, cs: 1 },
				b: { offset: 0, rs: 0, cs: 1 },
				gain: 1,
				digits: true,
				lift: 1.2
			});
			this.p.dot = dp;
			const ds = P.add({
				origin: [x0, dy - 2.3, 0],
				ax: [8, 0, 0],
				ay: [0, -1.6, 0],
				rows: 1,
				cols: 128,
				kind: 'bars',
				a: { offset: A.sumd, rs: 0, cs: 1, aux: true },
				gain: 1,
				digits: true,
				lift: 1.2
			});
			this.p.dotsum = ds;
			add(
				{
					name: 'Scores and shares',
					w: 12,
					h: 2 * H * cell + 8,
					top,
					caption: () =>
						`Each of the 16 heads compares this word's q with the k of every word so far (one row per head, one column per word): a dot product of 128 pairs, divided by √128. Softmax turns each row of scores into shares that add up to 1. Below: head ${this.head + 1} and “${this.srcWord}”, the 128 products and their running sum, ${f(n.dot)} (÷ √128 = ${f(n.dotScaled)}). Click a score to spell it out.`,
					labels: [
						{
							text: () => 'scores (q · k ÷ √128)',
							at: [x0, top + 0.5, 0],
							size: 0.36,
							align: 'left'
						},
						{
							text: () => 'shares (softmax)',
							at: [x0, top - H * cell - 1.0, 0],
							size: 0.36,
							align: 'left'
						},
						{
							text: () => `head ${this.head + 1} × “${this.srcWord}”: q × k, pair by pair`,
							at: [x0, dy + 0.5, 0],
							size: 0.32,
							align: 'left'
						},
						{ text: () => 'running sum', at: [x0, dy - 1.8, 0], size: 0.32, align: 'left' }
					]
				},
				12
			);
			this.scoreX = x0;
		}
		// 6. mixing the values
		{
			const x0 = x;
			const mix = P.add({
				origin: [x0, top, 0],
				ax: [8, 0, 0],
				ay: [0, -6, 0],
				rows: 1,
				cols: 128,
				kind: 'dense',
				a: { offset: 0, rs: 1, cs: 0 }, // share of word s (row)
				b: { offset: 0, rs: KV, cs: 1 }, // its value
				gain: 1,
				digits: true
			});
			this.p.mix = mix;
			const oh = wave('mix_out', [x0, top - 7.4, 0], 8, 1.6, 128, () => ({ offset: 0 }));
			this.srcs.delete(oh); // set with the chosen head
			wave('attn', [x0, top - 10.6, 0], 8, 1.6, Q, () => ({
				offset: L.attn + (lay() * N + tok()) * Q
			}));
			add(
				{
					name: 'Mixing the values',
					w: 8,
					h: 12.2,
					top,
					caption: () =>
						`Each head then takes its shares of the earlier words' values (v): above, head ${this.head + 1}'s share of each word (a row) times that word's 128 values; summed down the rows, they make the head's output. The 16 heads' outputs side by side are attention's result, 2,048 numbers.`,
					labels: [
						{
							text: () => `head ${this.head + 1}: share × value, one row per word`,
							at: [x0, top + 0.5, 0],
							size: 0.34,
							align: 'left'
						},
						{
							text: () => `summed down the rows: head ${this.head + 1}'s output`,
							at: [x0, top - 6.9, 0],
							size: 0.34,
							align: 'left'
						},
						{
							text: () => 'all 16 heads side by side',
							at: [x0, top - 10.1, 0],
							size: 0.34,
							align: 'left'
						}
					]
				},
				8
			);
			this.mixX = x0;
		}
		// 7. output projection, added to the word
		{
			const x0 = x;
			const attn = () => L.attn + (lay() * N + tok()) * Q;
			wave('o_in', [x0, top + 1.9, 0], 8, 1.5, Q, () => ({ offset: attn() }));
			const o = matrix('o', 'o', [x0, top, 0], attn, 'dA');
			sums('o_sum', 'o', [x0 + 8.3, top, 0], o.h);
			const ay = top - o.h - 1.2;
			wave('add1_x', [x0, ay, 0], 8, 1.2, D, () => ({ offset: L.resid + (lay() * N + tok()) * D }));
			wave('add1_d', [x0, ay - 1.8, 0], 8, 1.2, D, () => ({ offset: A.dA, aux: true }));
			wave('add1_m', [x0, ay - 3.6, 0], 8, 1.2, D, () => ({
				offset: L.mid + (lay() * N + tok()) * D
			}));
			inspector('o', 'o', [x0, ay - 6, 0]);
			add(
				{
					name: 'Back into the word',
					w: 9.7,
					h: -(ay - 9 - top - 3.4),
					top: top + 3.4,
					sweep: { mats: [o.panel], outs: [{ panel: this.p.o_sum, K: Q }], K: Q },
					caption: () =>
						`The output projection turns attention's result into 2,048 changes to the word, and they are added to it: this is the residual stream, where every part of the model writes by adding. Below, row ${this.rowPick.get('o') ?? 0} spelled out.`,
					labels: [
						{ text: () => "attention's result", at: [x0, top + 3.7, 0], size: 0.36, align: 'left' },
						{ text: () => 'the word', at: [x0 - 0.3, ay - 0.6, 0], size: 0.32, align: 'right' },
						{ text: () => '+ the change', at: [x0 - 0.3, ay - 2.4, 0], size: 0.32, align: 'right' },
						{ text: () => '=', at: [x0 - 0.3, ay - 4.2, 0], size: 0.32, align: 'right' }
					]
				},
				9.7
			);
		}
		// 8. normed again
		{
			const x0 = x;
			const g = 2.3;
			wave('n2_x', [x0, top, 0], 8, 1.6, D, () => ({ offset: L.mid + (lay() * N + tok()) * D }));
			wave('n2_xr', [x0, top - g, 0], 8, 1.6, D, () => ({ offset: A.mr, aux: true }));
			wave('n2_g', [x0, top - 2 * g, 0], 8, 1.6, D, () => ({ offset: A.g2, aux: true }));
			wave('n2_xn', [x0, top - 3 * g, 0], 8, 1.6, D, () => ({
				offset: L.xn2 + (lay() * N + tok()) * D
			}));
			add(
				{
					name: 'Normed again',
					w: 8,
					h: 3 * g + 1.6,
					top,
					caption: () =>
						`Scaled again before the neurons: divided by ${f(n.rms2)}, times the second norm's gains.`,
					labels: [
						{ text: () => 'the word', at: [x0 - 0.4, top - 0.8, 0], size: 0.36, align: 'right' },
						{
							text: () => `÷ ${f(n.rms2)}`,
							at: [x0 - 0.4, top - g - 0.8, 0],
							size: 0.36,
							align: 'right'
						},
						{
							text: () => '× gains',
							at: [x0 - 0.4, top - 2 * g - 0.8, 0],
							size: 0.36,
							align: 'right'
						},
						{
							text: () => '= normed',
							at: [x0 - 0.4, top - 3 * g - 0.8, 0],
							size: 0.36,
							align: 'right'
						}
					]
				},
				8
			);
		}
		// 9. gate and up
		{
			const x0 = x;
			const xn2 = () => L.xn2 + (lay() * N + tok()) * D;
			wave('g_in', [x0, top + 1.9, 0], 8, 1.5, D, () => ({ offset: xn2() }));
			const g = matrix('gate', 'gate', [x0, top, 0], xn2, 'gate');
			sums('gate_sum', 'gate', [x0 + 8.3, top, 0], g.h);
			const ux = x0 + 11;
			wave('u_in', [ux, top + 1.9, 0], 8, 1.5, D, () => ({ offset: xn2() }));
			const u = matrix('up', 'up', [ux, top, 0], xn2, 'up');
			sums('up_sum', 'up', [ux + 8.3, top, 0], u.h);
			add(
				{
					name: 'Gate and up',
					w: 20.7,
					h: g.h + 3.4,
					top: top + 3.4,
					sweep: {
						mats: [g.panel, u.panel],
						outs: [
							{ panel: this.p.gate_sum, K: D },
							{ panel: this.p.up_sum, K: D }
						],
						K: D
					},
					caption: () =>
						`Two matrices of 6,144 rows read the normed word, in the same way: every row a neuron. Gate decides how much each neuron opens; up is what it lets through.`,
					labels: [
						{ text: () => 'gate', at: [x0, top + 3.7, 0], size: 0.5, align: 'left', italic: true },
						{ text: () => 'up', at: [ux, top + 3.7, 0], size: 0.5, align: 'left', italic: true }
					]
				},
				20.7
			);
		}
		// 10. the neurons
		{
			const x0 = x;
			const g = 2.4;
			const wlen = 24;
			wave('ne_g', [x0, top, 0], wlen, 1.6, F, () => ({
				offset: L.gate + (lay() * N + tok()) * F
			}));
			wave('ne_s', [x0, top - g, 0], wlen, 1.6, F, () => ({ offset: A.silu, aux: true }));
			wave('ne_u', [x0, top - 2 * g, 0], wlen, 1.6, F, () => ({
				offset: L.up + (lay() * N + tok()) * F
			}));
			wave('ne_a', [x0, top - 3 * g, 0], wlen, 1.6, F, () => ({
				offset: L.act + (lay() * N + tok()) * F
			}));
			const hx = x0 + wlen + 2;
			this.p.hist = P.add({
				origin: [hx, top, 0],
				ax: [7.5, 0, 0],
				ay: [0, -7.5, 0],
				rows: HB,
				cols: HB,
				kind: 'dense',
				a: { offset: A.hist, rs: HB, cs: 1, aux: true },
				gain: 1,
				grid: false,
				lift: 2
			});
			add(
				{
					name: 'The neurons',
					w: wlen + 9.5,
					h: 3 * g + 1.6,
					top,
					caption: () =>
						`The 6,144 neurons: each gate value goes through SiLU (right: every neuron placed by its gate value and what SiLU makes of it, so they trace the curve), then is multiplied by up. ${n.fired ?? 0} of 6,144 come out larger than a tenth of the largest.`,
					labels: [
						{ text: () => 'gate', at: [x0 - 0.4, top - 0.8, 0], size: 0.36, align: 'right' },
						{
							text: () => 'SiLU(gate)',
							at: [x0 - 0.4, top - g - 0.8, 0],
							size: 0.36,
							align: 'right'
						},
						{
							text: () => '× up',
							at: [x0 - 0.4, top - 2 * g - 0.8, 0],
							size: 0.36,
							align: 'right'
						},
						{
							text: () => '= the neurons',
							at: [x0 - 0.4, top - 3 * g - 0.8, 0],
							size: 0.36,
							align: 'right'
						},
						{
							text: () => 'SiLU(gate) against gate',
							at: [hx, top + 0.5, 0],
							size: 0.34,
							align: 'left'
						},
						{ text: () => `${f(-n.hr)}`, at: [hx, top - 7.9, 0], size: 0.28, align: 'left' },
						{ text: () => `${f(n.hr)}`, at: [hx + 7.5, top - 7.9, 0], size: 0.28, align: 'right' }
					]
				},
				wlen + 9.5
			);
		}
		// 11. down, added to the word
		{
			const x0 = x;
			const act = () => L.act + (lay() * N + tok()) * F;
			wave('d_in', [x0, top + 1.9, 0], 24, 1.5, F, () => ({ offset: act() }));
			const d = matrix('down', 'down', [x0, top, 0], act, 'dM');
			sums('down_sum', 'down', [x0 + 24.3, top, 0], d.h);
			const ay = top - d.h - 1.2;
			wave('add2_m', [x0, ay, 0], 8, 1.2, D, () => ({ offset: L.mid + (lay() * N + tok()) * D }));
			wave('add2_d', [x0, ay - 1.8, 0], 8, 1.2, D, () => ({ offset: A.dM, aux: true }));
			wave('add2_o', [x0, ay - 3.6, 0], 8, 1.2, D, () => ({
				offset: L.resid + ((lay() + 1) * N + tok()) * D
			}));
			inspector('down', 'down', [x0, ay - 6, 0]);
			add(
				{
					name: 'Down, added to the word',
					w: 25.7,
					h: -(ay - 9 - top - 3.4),
					top: top + 3.4,
					sweep: { mats: [d.panel], outs: [{ panel: this.p.down_sum, K: F }], K: F },
					caption: () =>
						`Down reads the 6,144 neurons and writes 2,048 changes, added to the word like attention's. The word leaves layer ${lay() + 1}. Below, row ${this.rowPick.get('down') ?? 0} spelled out: 6,144 weights.`,
					labels: [
						{ text: () => 'the neurons', at: [x0, top + 3.7, 0], size: 0.36, align: 'left' },
						{ text: () => 'the word', at: [x0 - 0.3, ay - 0.6, 0], size: 0.32, align: 'right' },
						{ text: () => '+ the change', at: [x0 - 0.3, ay - 2.4, 0], size: 0.32, align: 'right' },
						{ text: () => '=', at: [x0 - 0.3, ay - 4.2, 0], size: 0.32, align: 'right' }
					]
				},
				25.7
			);
		}
		// 12. the whole stack
		{
			const x0 = x;
			const step = 0.62;
			this.p.stack = P.add({
				origin: [x0, top, 0],
				ax: [8, 0, 0],
				ay: [0, -(c.layers + 1) * step, 0],
				rows: c.layers + 1,
				cols: D,
				kind: 'dense',
				a: { offset: 0, rs: N * D, cs: 1 },
				gain: 1,
				grid: false,
				lift: 1.2
			});
			this.stackTop = top;
			this.stackStep = step;
			this.stackX = x0;
			add(
				{
					name: 'All the layers',
					w: 8,
					h: (c.layers + 1) * step,
					top,
					caption: () =>
						`The word after each of the 28 layers, top to bottom (each row the same 2,048 numbers, brighter where larger). This was layer ${lay() + 1}: every layer does exactly what you just watched, with its own weights.`,
					labels: [
						{
							text: () => 'embedding',
							at: [x0 - 0.3, top - step * 0.5, 0],
							size: 0.28,
							align: 'right'
						},
						{
							text: () => `after layer ${lay() + 1} →`,
							at: [x0 - 0.3, top - step * (lay() + 1.5), 0],
							size: 0.3,
							align: 'right'
						},
						{
							text: () => 'after 28',
							at: [x0 - 0.3, top - step * (c.layers + 0.5), 0],
							size: 0.28,
							align: 'right'
						}
					]
				},
				8
			);
		}
		this.panels.allocate();
	}

	private srcs = new Map<number, () => { offset: number; aux?: boolean }>();
	private inspectors = new Map<string, { k: string; wts: number; prod: number; run: number }>();
	private scoreX = 0;
	private mixX = 0;
	private stackTop = 0;
	private stackStep = 0.6;
	private stackX = 0;

	private get word() {
		return clean(this.tokens[this.token] ?? '') || '·';
	}
	private get srcWord() {
		return clean(this.tokens[this.src] ?? '') || '·';
	}

	// ---- reading

	async read(prompt: string) {
		const id = ++this.readId;
		this.busy = 'Reading your words through all 28 layers';
		const ids = this.llm.tokenizer.encode(chatPrompt(prompt)).slice(0, MAX_TOKENS);
		const r = await this.llm.prefill(ids);
		if (id !== this.readId) return;
		this.ids = r.ids;
		this.tokens = r.tokens;
		const end = r.tokens.findIndex((t, i) => i > 3 && t.startsWith('<|im_end'));
		this.words = Array.from(
			{ length: Math.max(1, (end > 0 ? end : r.ids.length) - 3) },
			(_, i) => 3 + i
		);
		this.token = this.words[this.words.length - 1];
		await this.load();
	}

	async choose(token: number, layer: number) {
		this.token = token;
		this.layer = layer;
		await this.load();
	}

	/** Everything for the chosen word and layer: read back, derive what the model does not keep, rebuild. */
	private async load() {
		const id = ++this.readId;
		this.busy = 'Collecting every number of this layer';
		const llm = this.llm,
			c = llm.config,
			L = llm.layout,
			N = MAX_TOKENS;
		const t = this.token,
			l = this.layer;
		const D = c.dim,
			F = c.ffn,
			H = c.heads,
			KVH = c.kvHeads,
			HD = c.headDim,
			Q = H * HD,
			KV = KVH * HD,
			T = t + 1;
		const at = (base: number, width: number, layer = l, tk = t) => ({
			offset: base + (layer * N + tk) * width,
			count: width
		});
		const regs = [
			at(L.resid, D),
			at(L.qraw, Q),
			at(L.q, Q),
			at(L.kraw, KV),
			at(L.mid, D),
			at(L.gate, F),
			at(L.up, F),
			at(L.act, F),
			at(L.resid, D, l + 1),
			{ offset: L.k + l * N * KV, count: T * KV },
			{ offset: L.probs + l * H * N * N, count: H * N * N }
		];
		const [x, qraw, q, kraw, mid, gate, , act, out, kAll, probs] = await llm.readMany(regs);
		if (id !== this.readId) return;
		const b = `blk.${l}.`;
		const g1 = llm.normWeight(b + 'attn_norm.weight'),
			g2 = llm.normWeight(b + 'ffn_norm.weight'),
			gq = llm.normWeight(b + 'attn_q_norm.weight'),
			gk = llm.normWeight(b + 'attn_k_norm.weight');
		const rms = (v: Float32Array, eps = c.eps) =>
			Math.sqrt(v.reduce((s, a) => s + a * a, 0) / v.length + eps);
		const rms1 = rms(x),
			rms2 = rms(mid);
		const aux = new Float32Array(A.sumd + 128);
		aux.set(
			x.map((v) => v / rms1),
			A.xr
		);
		aux.set(g1, A.g1);
		for (let h = 0; h < H; h++) {
			const seg = qraw.subarray(h * HD, (h + 1) * HD);
			const r = rms(seg);
			for (let i = 0; i < HD; i++) aux[A.qn + h * HD + i] = (seg[i] / r) * gq[i];
		}
		for (let h = 0; h < KVH; h++) {
			const seg = kraw.subarray(h * HD, (h + 1) * HD);
			const r = rms(seg);
			for (let i = 0; i < HD; i++) aux[A.kn + h * HD + i] = (seg[i] / r) * gk[i];
		}
		// scores: q (after turning) . k of each earlier word, / sqrt(128), as the model computes them
		const scale = 1 / Math.sqrt(HD);
		let bestHead = 0,
			bestSrc = 0,
			bestP = -1;
		for (let h = 0; h < H; h++) {
			const kvh = Math.floor(h / (H / KVH));
			for (let s = 0; s < T; s++) {
				let d = 0;
				for (let i = 0; i < HD; i++) d += q[h * HD + i] * kAll[s * KV + kvh * HD + i];
				aux[A.scores + h * MAX_TOKENS + s] = d * scale;
				const p = probs[(h * N + t) * N + s];
				// the strongest share that is not on the first words (where heads rest when they have nothing to read)
				if (s >= 3 && p > bestP) {
					bestP = p;
					bestHead = h;
					bestSrc = s;
				}
			}
		}
		this.head = bestHead;
		this.src = bestSrc;
		const dA = mid.map((v, i) => v - x[i]);
		aux.set(dA, A.dA);
		aux.set(
			mid.map((v) => v / rms2),
			A.mr
		);
		aux.set(g2, A.g2);
		const silu = gate.map((g) => g / (1 + Math.exp(-g)));
		aux.set(silu, A.silu);
		const dM = out.map((v, i) => v - mid[i]);
		aux.set(dM, A.dM);
		// the SiLU scatter as a histogram: x = gate, y = SiLU(gate)
		let hr = 0;
		for (const g of gate) hr = Math.max(hr, Math.abs(g));
		hr = Math.max(hr, 1e-6);
		for (let i = 0; i < F; i++) {
			const bx = Math.min(HB - 1, Math.floor(((gate[i] / hr) * 0.5 + 0.5) * HB));
			const by = Math.min(HB - 1, Math.max(0, Math.floor((0.5 - (silu[i] / hr) * 0.5) * HB)));
			aux[A.hist + by * HB + bx] += 1;
		}
		for (let i = 0; i < HB * HB; i++) aux[A.hist + i] = Math.log1p(aux[A.hist + i]);
		let mact = 0;
		for (const a of act) mact = Math.max(mact, Math.abs(a));
		Object.assign(this.nums, {
			rms1,
			rms2,
			hr,
			fired: act.reduce((n, a) => n + (Math.abs(a) > mact * 0.1 ? 1 : 0), 0)
		});
		this.cpu = { x, qraw, q, kraw, mid, gate, act, out, kAll, probs };
		this.stage.device.queue.writeBuffer(this.panels.aux, 0, aux);

		// point every panel at this word and layer
		const P = this.panels;
		for (const [panel, src] of this.srcs) {
			const d = P.desc(panel);
			const s = src();
			P.set(panel, { a: { ...d.a!, offset: s.offset, aux: s.aux } });
		}
		P.set(this.p.dial_q, { b: { offset: L.q + (l * N + t) * Q, rs: HD, cs: 1 } });
		P.set(this.p.dial_k, { b: { offset: L.k + (l * N + t) * KV, rs: HD, cs: 1 } });
		this.setDot(this.head, this.src);
		for (const [panel, m] of this.mats) {
			const tensor = this.tensorFor(m.key, l);
			P.set(panel, { tensor, x: { offset: m.xOff() } });
		}
		// scores and shares: one column per word so far
		const cell = 0.5;
		P.set(this.p.scores, { cols: T, ax: [T * cell, 0, 0] });
		P.set(this.p.probs, {
			cols: T,
			ax: [T * cell, 0, 0],
			a: { offset: L.probs + (l * H * N + t) * N, rs: N * N, cs: 1 }
		});
		// the stack of the word after every layer
		P.set(this.p.stack, { a: { offset: L.resid + t * D, rs: N * D, cs: 1 } });
		// running sums, then the pyramids, then the brightnesses
		const enc = this.stage.device.createCommandEncoder();
		this.runPrefix(enc);
		P.buildPyramids(enc);
		this.stage.device.queue.submit([enc.finish()]);
		const matPanels = [...this.mats.keys()];
		const means = await P.meanAbs(matPanels);
		if (id !== this.readId) return;
		const gains = new Map<number, number>();
		matPanels.forEach((p, j) => gains.set(p, means[j] * 18));
		this.gainsFor(gains, aux, { x, qraw, kraw, mid, gate, act, out, q });
		for (const [p, g] of gains) P.set(p, { gain: g });
		// the inspectors: default to the row with the largest output
		for (const key of this.inspectors.keys()) if (!this.rowPick.has(key)) this.rowPick.set(key, -1);
		this.pickRows();
		this.busy = '';
	}

	/** Brightness for each waveform: its largest value (so every waveform fills its height). */
	private gainsFor(gains: Map<number, number>, aux: Float32Array, v: Record<string, Float32Array>) {
		const c = this.llm.config;
		const mx = (a: ArrayLike<number>) => {
			let m = 0;
			for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs(a[i]));
			return m || 1;
		};
		const auxv = (off: number, n: number) => aux.subarray(off, off + n);
		const p = this.p,
			D = c.dim,
			F = c.ffn;
		const set = (key: string, g: number) => p[key] !== undefined && gains.set(p[key], g);
		set('x', mx(v.x));
		set('n_x', mx(v.x));
		set('n_xr', mx(auxv(A.xr, D)));
		set('n_g', mx(auxv(A.g1, D)));
		const xn = new Float32Array(D).map((_, i) => aux[A.xr + i] * aux[A.g1 + i]);
		set('n_xn', mx(xn));
		set('qkv_in', mx(xn));
		set('q_sum', mx(v.qraw));
		set('k_sum', mx(v.kraw));
		set('v_sum', mx(this.cpuV ?? v.kraw));
		set('dial_q', mx(v.q) * 1.1);
		set('dial_k', mx(auxv(A.kn, 1024)) * 1.1);
		set('scores', mx(auxv(A.scores, c.heads * 128)));
		set('probs', 1);
		set('o_in', mx(this.cpuAttn ?? v.q));
		set('o_sum', mx(auxv(A.dA, D)));
		set('add1_x', mx(v.x));
		set('add1_d', mx(v.x));
		set('add1_m', mx(v.x));
		set('attn', mx(this.cpuAttn ?? v.q));
		set('n2_x', mx(v.mid));
		set('n2_xr', mx(auxv(A.mr, D)));
		set('n2_g', mx(auxv(A.g2, D)));
		const xn2 = new Float32Array(D).map((_, i) => aux[A.mr + i] * aux[A.g2 + i]);
		set('n2_xn', mx(xn2));
		set('g_in', mx(xn2));
		set('u_in', mx(xn2));
		set('gate_sum', mx(v.gate));
		set('up_sum', mx(this.cpuUp ?? v.gate));
		set('ne_g', mx(v.gate));
		set('ne_s', mx(v.gate));
		set('ne_u', mx(this.cpuUp ?? v.gate));
		set('ne_a', mx(v.act));
		set('d_in', mx(v.act));
		set('down_sum', mx(auxv(A.dM, D)));
		set('add2_m', mx(v.out));
		set('add2_d', mx(v.out));
		set('add2_o', mx(v.out));
		set('hist', Math.log1p(40));
		set('stack', mx(v.out) * 0.6);
		void F;
	}
	private cpuV?: Float32Array;
	private cpuAttn?: Float32Array;
	private cpuUp?: Float32Array;

	/** The dot product of head h's q with the k of word s, pair by pair, and its running sum. */
	private setDot(h: number, s: number) {
		const cpu = this.cpu;
		if (!cpu) return;
		const c = this.llm.config,
			L = this.llm.layout,
			N = MAX_TOKENS,
			HD = c.headDim,
			KV = c.kvHeads * HD,
			Q = c.heads * HD;
		const kvh = Math.floor(h / (c.heads / c.kvHeads));
		const l = this.layer,
			t = this.token;
		this.head = h;
		this.src = s;
		const qo = L.q + (l * N + t) * Q + h * HD,
			ko = L.k + (l * N + s) * KV + kvh * HD;
		const sum = new Float32Array(HD);
		let acc = 0,
			mx = 0,
			mp = 0;
		for (let i = 0; i < HD; i++) {
			const p = cpu.q[h * HD + i] * cpu.kAll[s * KV + kvh * HD + i];
			acc += p;
			sum[i] = acc;
			mx = Math.max(mx, Math.abs(acc));
			mp = Math.max(mp, Math.abs(p));
		}
		this.nums.dot = acc;
		this.nums.dotScaled = acc / Math.sqrt(HD);
		this.stage.device.queue.writeBuffer(this.panels.aux, A.sumd * 4, sum);
		const P = this.panels;
		P.set(this.p.dot, {
			a: { offset: qo, rs: 0, cs: 1 },
			b: { offset: ko, rs: 0, cs: 1 },
			gain: mp || 1
		});
		P.set(this.p.dotsum, { gain: mx || 1 });
		// mixing: this head's shares of every word times their values
		const T = t + 1;
		P.set(this.p.mix, {
			rows: T,
			a: { offset: L.probs + ((l * c.heads + h) * N + t) * N, rs: 1, cs: 0 },
			b: { offset: L.v + l * N * KV + kvh * HD, rs: KV, cs: 1 }
		});
		P.set(this.p.mix_out, { a: { offset: L.attn + (l * N + t) * Q + h * HD, rs: 0, cs: 1 } });
		// brightness of the mixing grid and the head's output: from the values
		void this.llm
			.readMany([
				{ offset: L.v + l * N * KV, count: T * KV },
				{ offset: L.attn + (l * N + t) * Q, count: Q },
				{ offset: L.up + (l * N + t) * c.ffn, count: c.ffn }
			])
			.then(([v, attn, up]) => {
				this.cpuV = v;
				this.cpuAttn = attn;
				this.cpuUp = up;
				let mv = 0,
					mo = 0;
				for (let s2 = 0; s2 < T; s2++)
					for (let i = 0; i < HD; i++) mv = Math.max(mv, Math.abs(v[s2 * KV + kvh * HD + i]));
				for (let i = 0; i < HD; i++) mo = Math.max(mo, Math.abs(attn[h * HD + i]));
				let ma = 0;
				for (const a of attn) ma = Math.max(ma, Math.abs(a));
				let mu = 0;
				for (const a of up) mu = Math.max(mu, Math.abs(a));
				P.set(this.p.mix, { gain: mv * 0.5 || 1 });
				P.set(this.p.mix_out, { gain: mo || 1 });
				P.set(this.p.attn, { gain: ma || 1 });
				P.set(this.p.o_in, { gain: ma || 1 });
				P.set(this.p.up_sum, { gain: mu || 1 });
				P.set(this.p.ne_u, { gain: mu || 1 });
				let mvv = 0;
				for (let i = 0; i < KV; i++) mvv = Math.max(mvv, Math.abs(v[t * KV + i]));
				P.set(this.p.v_sum, { gain: mvv || 1 });
			});
	}

	/** Record the running sums of the seven matrices for this word and layer. */
	private runPrefix(enc: GPUCommandEncoder) {
		const dev = this.stage.device;
		const { codes, scales } = this.llm.weightBuffers;
		if (!this.prefixBind)
			this.prefixBind = dev.createBindGroup({
				layout: this.prefixPipe.getBindGroupLayout(0),
				entries: [
					{ binding: 0, resource: { buffer: this.prefixUniforms, size: 32 } },
					{ binding: 1, resource: { buffer: codes } },
					{ binding: 2, resource: { buffer: scales } },
					{ binding: 3, resource: { buffer: this.llm.arena } },
					{ binding: 4, resource: { buffer: this.panels.aux } }
				]
			});
		const L = this.llm.layout,
			c = this.llm.config,
			N = MAX_TOKENS,
			l = this.layer,
			t = this.token;
		const D = c.dim,
			F = c.ffn,
			Q = c.heads * c.headDim;
		const xOf: Record<string, number> = {
			q: L.xn + (l * N + t) * D,
			k: L.xn + (l * N + t) * D,
			v: L.xn + (l * N + t) * D,
			o: L.attn + (l * N + t) * Q,
			gate: L.xn2 + (l * N + t) * D,
			up: L.xn2 + (l * N + t) * D,
			down: L.act + (l * N + t) * F
		};
		const keys = ['q', 'k', 'v', 'o', 'gate', 'up', 'down'];
		const data = new Uint32Array((keys.length * 256) / 4);
		keys.forEach((k, i) => {
			const tt = this.tensorFor(k, l);
			const pa = this.prefixAt.get(k)!;
			data.set([tt.codes, tt.scales, tt.cols, tt.rows, xOf[k], pa.off, pa.nck, 0], i * 64);
		});
		dev.queue.writeBuffer(this.prefixUniforms, 0, data);
		keys.forEach((k, i) => {
			const pa = this.prefixAt.get(k)!;
			const bg = dev.createBindGroup({
				layout: this.prefixPipe.getBindGroupLayout(0),
				entries: [
					{ binding: 0, resource: { buffer: this.prefixUniforms, offset: i * 256, size: 32 } },
					{ binding: 1, resource: { buffer: codes } },
					{ binding: 2, resource: { buffer: scales } },
					{ binding: 3, resource: { buffer: this.llm.arena } },
					{ binding: 4, resource: { buffer: this.panels.aux } }
				]
			});
			const pass = enc.beginComputePass();
			pass.setPipeline(this.prefixPipe);
			pass.setBindGroup(0, bg);
			pass.dispatchWorkgroups(Math.ceil(pa.M / 64));
			pass.end();
		});
	}

	/** Point the inspectors at their rows (the largest output by default). */
	private pickRows() {
		const cpu = this.cpu;
		if (!cpu) return;
		const outOf: Record<string, Float32Array> = {
			q: cpu.qraw,
			o: cpu.mid.map((v, i) => v - cpu.x[i]),
			down: cpu.out.map((v, i) => v - cpu.mid[i])
		};
		const P = this.panels,
			L = this.llm.layout,
			c = this.llm.config,
			N = MAX_TOKENS,
			l = this.layer,
			t = this.token;
		const xOf: Record<string, number> = {
			q: L.xn + (l * N + t) * c.dim,
			o: L.attn + (l * N + t) * c.heads * c.headDim,
			down: L.act + (l * N + t) * c.ffn
		};
		for (const [key, ins] of this.inspectors) {
			let row = this.rowPick.get(key) ?? -1;
			const outs = outOf[ins.k];
			if (row < 0) {
				row = 0;
				for (let i = 1; i < outs.length; i++) if (Math.abs(outs[i]) > Math.abs(outs[row])) row = i;
			}
			this.rowPick.set(key, row);
			const tensor = this.tensorFor(ins.k, l);
			const pa = this.prefixAt.get(ins.k)!;
			const mean = (P.desc(this.p[ins.k]).gain || 1) / 18;
			P.set(ins.wts, { tensor, m0: row, gain: tensor.scales ? mean * 8 : 1 });
			P.set(ins.prod, { tensor, m0: row, x: { offset: xOf[ins.k] }, gain: mean * 6 });
			P.set(ins.run, {
				a: { offset: pa.off + row * pa.nck, rs: 0, cs: 1, aux: true },
				gain: Math.abs(outs[row]) * 1.3 || 1
			});
			if (key === 'qkv') this.nums.qkvRow = outs[row];
		}
		// the weights row shows weights alone: its brightness from the typical scale
		for (const ins of this.inspectors.values()) {
			const t = this.tensorFor(ins.k, l);
			P.set(ins.wts, { gain: t.meanScale * 1.2 });
		}
	}

	// ---- time and camera

	goTo(i: number) {
		this.station = Math.max(0, Math.min(this.stations.length - 1, i));
		this.t = 0;
		const s = this.stations[this.station];
		const c = this.stage.gpu.canvas;
		const aspect = c.clientWidth / Math.max(1, c.clientHeight);
		const tn = Math.tan((38 * Math.PI) / 360);
		// fit it between the controls above and the caption below (about 60% of the height, 70% of the width)
		const w = s.w + 4,
			h = s.h + 1.5;
		const dist = Math.max(w / (2 * tn * aspect * 0.7), h / (2 * tn * 0.6));
		const shift = dist * tn * 0.08; // a little above the middle
		this.stage.flyTo({
			target: [s.x + s.w / 2 - 1, s.top - s.h / 2 - shift, 0],
			dist,
			yaw: 0.05,
			pitch: 0.02
		});
	}

	next() {
		if (this.station < this.stations.length - 1) this.goTo(this.station + 1);
		else this.playing = false;
	}

	prev() {
		this.goTo(this.station - 1);
	}

	update(dt: number) {
		this.t += dt;
		if (this.playing && this.t > this.duration(this.station)) this.next();
		const P = this.panels;
		// the station being shown in full; the others dim
		this.stations.forEach((s, i) => {
			const a = i === this.station ? 1 : 0.14;
			for (let p = s.first; p < s.last; p++) P.set(p, { alpha: a });
		});
		// sweeps: in the station being shown they run over the first part of its time; elsewhere they are done
		this.stations.forEach((s, i) => {
			if (!s.sweep) return;
			const K = s.sweep.K;
			const here = i === this.station;
			const f = here ? Math.min(1, Math.max(0, (this.t - 1.2) / 5.5)) : 1;
			const sw = f * K;
			for (const m of s.sweep.mats) P.set(m, { sweep: f < 1 ? sw : null });
			for (const o of s.sweep.outs) P.set(o.panel, { sweep: Math.min(o.K - 1, sw) });
		});
	}

	private duration(i: number) {
		return this.stations[i].sweep ? 10 : 7;
	}

	draw(pass: GPURenderPassEncoder) {
		this.panels.draw(pass);
	}

	labels(): LayerLabel[] {
		if (!this.cpu) return [];
		const s = this.stage;
		const out: LayerLabel[] = [];
		const upp = s.unitsPerPixel(s.view.dist);
		for (const st of this.stations) {
			for (const l of st.labels) {
				const q = s.project(l.at);
				if (!q.front) continue;
				const size = (l.size * 1.7) / s.unitsPerPixel(q.depth);
				if (size < 6) continue;
				out.push({
					text: l.text(),
					x: q.x,
					y: q.y,
					size: Math.min(size, 40),
					alpha: 0.8,
					align: l.align,
					italic: l.italic
				});
			}
		}
		// the words under the scores and shares
		const T = this.token + 1;
		const cell = 0.5;
		const H = this.llm.config.heads;
		const near = (i: number) => i === this.station;
		for (let k = 0; k < T; k++) {
			const word = clean(this.tokens[k]) || '·';
			if (near(4)) {
				const q = s.project([this.scoreX + (k + 0.5) * cell, -2 * H * cell - 1.9, 0]);
				const upp = s.unitsPerPixel(q.depth);
				const size = Math.min(0.42 / upp, (cell * 0.95) / upp / Math.max(1, word.length * 0.5));
				if (q.front && size >= 5)
					out.push({
						text: word,
						x: q.x,
						y: q.y,
						size: Math.min(size, 22),
						alpha: k === this.src ? 1 : 0.6,
						align: 'center'
					});
			}
			if (near(5)) {
				const q = s.project([this.mixX - 0.2, -((k + 0.5) / T) * 6, 0]);
				const size = Math.min(0.36, (6 / T) * 0.8) / s.unitsPerPixel(q.depth);
				if (q.front && size >= 5)
					out.push({
						text: word,
						x: q.x,
						y: q.y,
						size: Math.min(size, 20),
						alpha: 0.6,
						align: 'right'
					});
			}
		}
		void upp;
		return out;
	}

	/** Click: pick a row to spell out, or a score to break down. */
	click(clientX: number, clientY: number) {
		const hit = this.panels.pick(this.stage.ray(clientX, clientY));
		if (!hit) return;
		const m = this.mats.get(hit.panel);
		if (m) {
			const key =
				m.key === 'q' || m.key === 'k' || m.key === 'v'
					? 'qkv'
					: m.key === 'o'
						? 'o'
						: m.key === 'down'
							? 'down'
							: '';
			if (key && (key !== 'qkv' || m.key === 'q')) {
				this.rowPick.set(key, hit.row);
				this.pickRows();
			}
			return;
		}
		if (hit.panel === this.p.scores || hit.panel === this.p.probs) this.setDot(hit.row, hit.col);
	}

	/** What is under the pointer. */
	inspect(clientX: number, clientY: number): { title: string; lines: string[] } | null {
		const hit = this.panels.pick(this.stage.ray(clientX, clientY));
		if (!hit || !this.cpu) return null;
		const m = this.mats.get(hit.panel);
		if (m)
			return {
				title: `${m.key}, layer ${this.layer + 1}`,
				lines: [`row ${hit.row}, column ${hit.col}`, 'click to spell this row out']
			};
		if (hit.panel === this.p.scores) {
			const v = this.cpu.q; // not used; the score is in the scratch buffer on the GPU
			void v;
			return {
				title: `head ${hit.row + 1} reading “${clean(this.tokens[hit.col])}”`,
				lines: ['click to see the dot product pair by pair']
			};
		}
		if (hit.panel === this.p.probs) {
			const p = this.cpu.probs[(hit.row * MAX_TOKENS + this.token) * MAX_TOKENS + hit.col];
			return {
				title: `head ${hit.row + 1}'s share for “${clean(this.tokens[hit.col])}”`,
				lines: [fmt(p)]
			};
		}
		return null;
	}

	destroy() {
		this.panels.destroy();
		this.prefixUniforms.destroy();
	}
}
