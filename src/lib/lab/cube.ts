// Lab: the compute cube. Every multiplication of the forward pass, drawn as a voxel. A matrix times the prompt's words
// is a box: across, the matrix's inputs; down, its outputs; in depth, the words, one slice each. Each slice holds that
// word's products, weight x input (one voxel each, glowing by its size). Adding up along the inputs collapses the box
// onto its side face: those sums are the outputs, which become the next box's inputs (its top face).
//   q, k, v       4,096 outputs from 2,048 inputs, for every word
//   scores        every word's q against every earlier word's k (all 16 heads side by side): a box that is as long as
//                 the prompt both ways, and only half full, since a word never looks ahead
//   mixing        every word's shares of every earlier word's values
//   output, gate and up, down: the rest of the layer
// ... 28 times, then the vocabulary scores the next word (for the last word only: 151,936 rows).
// True proportions make the word axis one cell per word, like the others: the boxes are thin plates, and attention
// almost vanishes next to the neurons. The stretch makes each word's slice thick enough to see.
import type { BonsaiLLM, Ternary } from '$lib/runtime/bonsai-llm';
import { MAX_TOKENS } from '$lib/runtime/bonsai-llm';
import { chatPrompt } from '$lib/runtime/tokenizer';
import { clean } from '$lib/engine/text';
import { Panels, type PanelDesc } from './panels';
import type { Scene, Stage, V3 } from './stage';
import type { Painter } from '$lib/runtime/painter';
import type { PaintingRun } from './painting-run';
import { PainterCube } from './cube-painter';

const U = 256; // cells per world unit
const MAX_T = 64; // the cube shows up to this many words
const GAP = 3; // between boxes
const LAYER_GAP = 10;

export interface CubeLabel {
	text: string;
	x: number;
	y: number;
	size: number;
	alpha: number;
}

interface Box {
	name: string;
	layer: number; // -1: the vocabulary
	lo: V3; // corner (min x, min y, min z) for pointing at
	hi: V3;
	flops: number;
	slices: number[]; // panels holding the products
	faceIn: number[];
	faceOut: number[];
	start: number;
	dur: number;
	K: number; // columns (for the sweep)
	painter?: { step: number; block: number }; // the painter's (step -1: before the steps; block -1 / 25: before / after the blocks)
	hull?: number[]; // its walls (painter boxes)
}

export class Cube implements Scene {
	panels: Panels;
	boxes: Box[] = [];
	private tokens: string[] = [];
	private T = 0;
	stretch = 32; // cells per word along the word axis
	busy = '';
	reveal = Infinity;
	revealLength = 0;
	playing = false;
	follow = false;
	private readId = 0;
	private length = 0;
	totalFlops = 0;
	attnFlops = 0;

	constructor(
		private stage: Stage,
		private llm: BonsaiLLM
	) {
		const c = llm.config;
		this.panels = new Panels(stage, llm, {
			capacity: c.layers * 24 + 64 + 2400,
			aux: 4 * c.layers * MAX_T * c.heads * c.headDim,
			idx: 1 << 16
		});
		stage.focusPlane = null;
		stage.maxDist = 8000;
	}

	get wordCount() {
		return this.T;
	}

	pc: PainterCube | null = null;

	/** Add the painter's boxes, lit by a painting's capture. */
	async attachPainter(painter: Painter, run: PaintingRun) {
		const pc = new PainterCube(painter, run, 0);
		await pc.prepare();
		this.panels.setPainter(painter.weights);
		this.panels.setCapture(run.cap.buffer);
		this.pc = pc;
		if (this.T) this.layout();
	}

	/** A new painting (a new prompt). */
	setRun(run: PaintingRun) {
		if (!this.pc) return;
		this.pc.run = run;
		this.panels.setCapture(run.cap.buffer);
		if (this.T) this.layout();
	}

	get painterFlops() {
		return this.pc
			? { all: this.pc.flops, attn: this.pc.attnFlops, reader: this.pc.readerFlops }
			: null;
	}

	async read(prompt: string) {
		const id = ++this.readId;
		this.busy = 'Reading your words through all 28 layers';
		const ids = this.llm.tokenizer.encode(chatPrompt(prompt)).slice(0, MAX_TOKENS);
		const r = await this.llm.prefill(ids);
		if (id !== this.readId) return;
		this.tokens = r.tokens;
		this.T = Math.min(MAX_T, r.ids.length);
		this.busy = 'Collecting the inputs of every box';
		await this.prepare();
		if (id !== this.readId) return;
		this.layout();
		this.busy = '';
		this.play();
	}

	/** The expanded keys and values (one copy per head) and the two added parts, into the scratch buffer. */
	private aux = { kx: 0, vx: 0, dA: 0, dM: 0 };
	private gains = new Map<string, number>();
	private async prepare() {
		const llm = this.llm,
			c = llm.config,
			L = llm.layout,
			N = MAX_TOKENS,
			T = this.T;
		const D = c.dim,
			F = c.ffn,
			H = c.heads,
			HD = c.headDim,
			Q = H * HD,
			KV = c.kvHeads * HD;
		const regions: { offset: number; count: number }[] = [];
		const per = (base: number, width: number, layers = c.layers) =>
			Array.from(
				{ length: layers },
				(_, l) => regions.push({ offset: base + l * N * width, count: T * width }) - 1
			);
		const ix = {
			k: per(L.k, KV),
			v: per(L.v, KV),
			q: per(L.q, Q),
			xn: per(L.xn, D),
			attn: per(L.attn, Q),
			xn2: per(L.xn2, D),
			act: per(L.act, F),
			resid: per(L.resid, D, c.layers + 1),
			mid: per(L.mid, D)
		};
		const fin = regions.push({ offset: L.final + (T - 1) * D, count: D }) - 1;
		const got = await llm.readMany(regions);
		const per4 = c.layers * T * Q;
		this.aux = { kx: 0, vx: per4, dA: 2 * per4, dM: 3 * per4 };
		const aux = new Float32Array(4 * per4);
		const meanAbs = (a: Float32Array) => {
			let s = 0;
			for (const v of a) s += Math.abs(v);
			return s / Math.max(1, a.length);
		};
		let mk = 0,
			mv = 0,
			mq = 0,
			mdA = 0,
			mdM = 0;
		for (let l = 0; l < c.layers; l++) {
			const k = got[ix.k[l]],
				v = got[ix.v[l]],
				resid = got[ix.resid[l]],
				mid = got[ix.mid[l]],
				next = got[ix.resid[l + 1]];
			mq += meanAbs(got[ix.q[l]]);
			for (let t = 0; t < T; t++)
				for (let cc = 0; cc < Q; cc++) {
					const src = t * KV + Math.floor(cc / 256) * HD + (cc % HD);
					aux[this.aux.kx + (l * T + t) * Q + cc] = k[src];
					aux[this.aux.vx + (l * T + t) * Q + cc] = v[src];
				}
			for (let i = 0; i < T * D; i++) {
				const a = mid[i] - resid[i],
					b = next[i] - mid[i];
				aux[this.aux.dA + l * T * Q + i] = a;
				aux[this.aux.dM + l * T * Q + i] = b;
				mdA += Math.abs(a);
				mdM += Math.abs(b);
			}
			mk += meanAbs(k);
			mv += meanAbs(v);
		}
		this.stage.device.queue.writeBuffer(this.panels.aux, 0, aux);
		const n = c.layers;
		const g = this.gains;
		g.set('xn', avg(ix.xn.map((i) => meanAbs(got[i]))));
		g.set('attn', avg(ix.attn.map((i) => meanAbs(got[i]))));
		g.set('xn2', avg(ix.xn2.map((i) => meanAbs(got[i]))));
		g.set('act', avg(ix.act.map((i) => meanAbs(got[i]))));
		g.set('final', meanAbs(got[fin]));
		g.set('q', mq / n);
		g.set('k', mk / n);
		g.set('v', mv / n);
		g.set('dA', mdA / (n * T * D));
		g.set('dM', mdM / (n * T * D));
		g.set('p', 1 / Math.max(1, T / 2));
	}

	/** Build every box for the current prompt and stretch. */
	layout() {
		const P = this.panels;
		P.clear();
		this.boxes = [];
		const llm = this.llm,
			c = llm.config,
			L = llm.layout,
			N = MAX_TOKENS,
			T = this.T;
		const D = c.dim,
			F = c.ffn,
			H = c.heads,
			HD = c.headDim,
			Q = H * HD,
			KV = c.kvHeads * HD;
		const dz = this.stretch / U; // depth of one word
		const depth = T * dz;
		const g = this.gains;
		// the slices add up along the view: each gives off a share of the light, so a box glows the same however
		// many words it holds
		const sliceAlpha = Math.min(1, 2.2 / T);
		let time = 0;
		let x = 0;
		this.totalFlops = 0;
		this.attnFlops = 0;
		// products of a ternary matrix with every word: one slice per word
		const slices = (
			t: Ternary,
			origin: V3,
			rows: number,
			m0: number,
			xBase: number,
			xStride: number,
			gain: number
		): number =>
			P.add({
				origin,
				ax: [t.cols / U, 0, 0],
				ay: [0, -rows / U, 0],
				rows,
				cols: t.cols,
				kind: 'ternary',
				tensor: t,
				m0,
				x: { offset: xBase },
				instances: { count: T, step: [0, 0, dz], x: xStride },
				gain: gain * 2.2,
				additive: true,
				alpha: sliceAlpha,
				samples: 2,
				grid: false,
				lift: 2
			});
		// a face: rows along one axis, columns along another, values from the arena or the scratch buffer
		const face = (
			d: Partial<PanelDesc> &
				Pick<PanelDesc, 'origin' | 'ax' | 'ay' | 'rows' | 'cols' | 'a' | 'gain'>
		) => P.add({ kind: 'dense', additive: true, samples: 3, grid: false, lift: 2, alpha: 1, ...d });
		// the words' inputs lying on top of a box: rows are the words (in depth), columns the inputs
		const topFace = (
			x0: number,
			w: number,
			cols: number,
			base: number,
			stride: number,
			gain: number,
			aux = false
		) =>
			face({
				origin: [x0, 0.08, 0],
				ax: [w, 0, 0],
				ay: [0, 0, depth],
				rows: T,
				cols,
				a: { offset: base, rs: stride, cs: 1, aux },
				gain: gain * 3
			});
		// the sums on a box's side: rows are the outputs (down), columns the words (in depth)
		const sideFace = (
			x0: number,
			y0: number,
			h: number,
			rows: number,
			base: number,
			stride: number,
			gain: number,
			aux = false
		) =>
			face({
				origin: [x0, y0, 0],
				ax: [0, 0, depth],
				ay: [0, -h, 0],
				rows,
				cols: T,
				a: { offset: base, rs: 1, cs: stride, aux },
				gain: gain * 3
			});
		const box = (b: Omit<Box, 'start' | 'dur'>, dur: number) => {
			this.boxes.push({ ...b, start: time, dur });
			time += dur;
			this.totalFlops += b.flops;
		};
		const ms = (name: string) => llm.tensor(name).meanScale * 0.61;

		// the layers in rows of seven, read like text
		const PER_ROW = 7,
			ROW_H = (2 * F) / U + 14;
		let layerW = 0;
		for (let l = 0; l < c.layers; l++) {
			const b = `blk.${l}.`;
			const firstPanel = P.count,
				firstBox = this.boxes.length,
				layerX0 = x;
			// q, k, v
			{
				const x0 = x;
				const qT = llm.tensor(b + 'attn_q.weight'),
					kT = llm.tensor(b + 'attn_k.weight'),
					vT = llm.tensor(b + 'attn_v.weight');
				const xb = L.xn + l * N * D;
				const gx = g.get('xn')!;
				const sl = [
					slices(qT, [x0, 0, 0], Q, 0, xb, D, ms(b + 'attn_q.weight') * gx),
					slices(kT, [x0, -Q / U - 0.2, 0], KV, 0, xb, D, ms(b + 'attn_k.weight') * gx),
					slices(vT, [x0, -(Q + KV) / U - 0.4, 0], KV, 0, xb, D, ms(b + 'attn_v.weight') * gx)
				];
				const fi = [topFace(x0, D / U, D, xb, D, gx)];
				const fo = [
					sideFace(x0 + D / U + 0.08, 0, Q / U, Q, L.qraw + l * N * Q, Q, g.get('q')! * 3),
					sideFace(
						x0 + D / U + 0.08,
						-Q / U - 0.2,
						KV / U,
						KV,
						L.kraw + l * N * KV,
						KV,
						g.get('k')! * 3
					),
					sideFace(
						x0 + D / U + 0.08,
						-(Q + KV) / U - 0.4,
						KV / U,
						KV,
						L.v + l * N * KV,
						KV,
						g.get('v')!
					)
				];
				const h = (Q + 2 * KV) / U + 0.4;
				box(
					{
						name: 'q, k and v',
						layer: l,
						lo: [x0, -h, 0],
						hi: [x0 + D / U, 0, depth],
						flops: 2 * (Q + 2 * KV) * D * T,
						slices: sl,
						faceIn: fi,
						faceOut: fo,
						K: D
					},
					1
				);
				x += D / U + GAP;
			}
			// scores: each word's q (all heads side by side) against every earlier word's k
			{
				const x0 = x;
				const sl = [
					P.add({
						origin: [x0, 0, 0],
						ax: [Q / U, 0, 0],
						ay: [0, -depth, 0],
						rows: T,
						cols: Q,
						kind: 'dense',
						a: { offset: L.q + l * N * Q, rs: 0, cs: 1 },
						b: { offset: this.aux.kx + l * T * Q, rs: Q, cs: 1, aux: true },
						instances: { count: T, step: [0, 0, dz], a: Q, causal: true },
						gain: g.get('q')! * g.get('k')! * 10,
						additive: true,
						alpha: sliceAlpha,
						samples: 2,
						grid: false,
						lift: 2
					})
				];
				const fl = 2 * Q * ((T * (T + 1)) / 2);
				this.attnFlops += fl;
				box(
					{
						name: 'scores (q · k)',
						layer: l,
						lo: [x0, -depth, 0],
						hi: [x0 + Q / U, 0, depth],
						flops: fl,
						slices: sl,
						faceIn: [],
						faceOut: [],
						K: Q
					},
					0.5
				);
				x += Q / U + GAP;
			}
			// mixing: each word's shares of every earlier word's values; the sums collapse downwards
			{
				const x0 = x;
				const sl = [
					P.add({
						origin: [x0, 0, 0],
						ax: [Q / U, 0, 0],
						ay: [0, -depth, 0],
						rows: T,
						cols: Q,
						kind: 'dense',
						// share of word s (row) for this head (column / 128) and word t (the slice)
						a: { offset: L.probs + l * H * N * N, rs: 1, cs: 0, div: HD, ds: N * N },
						b: { offset: this.aux.vx + l * T * Q, rs: Q, cs: 1, aux: true },
						instances: { count: T, step: [0, 0, dz], a: N, causal: true },
						gain: g.get('v')! * g.get('p')! * 3,
						additive: true,
						alpha: sliceAlpha,
						samples: 2,
						grid: false,
						lift: 2
					})
				];
				const fo = [
					face({
						origin: [x0, -depth - 0.08, 0],
						ax: [Q / U, 0, 0],
						ay: [0, 0, depth],
						rows: T,
						cols: Q,
						a: { offset: L.attn + l * N * Q, rs: Q, cs: 1 },
						gain: g.get('attn')! * 3
					})
				];
				const fl = 2 * Q * ((T * (T + 1)) / 2);
				this.attnFlops += fl;
				box(
					{
						name: 'mixing (shares × values)',
						layer: l,
						lo: [x0, -depth, 0],
						hi: [x0 + Q / U, 0, depth],
						flops: fl,
						slices: sl,
						faceIn: [],
						faceOut: fo,
						K: Q
					},
					0.5
				);
				x += Q / U + GAP;
			}
			// output projection
			{
				const x0 = x;
				const oT = llm.tensor(b + 'attn_output.weight');
				const xb = L.attn + l * N * Q;
				const sl = [
					slices(oT, [x0, 0, 0], D, 0, xb, Q, ms(b + 'attn_output.weight') * g.get('attn')!)
				];
				const fi = [topFace(x0, Q / U, Q, xb, Q, g.get('attn')!)];
				const fo = [
					sideFace(x0 + Q / U + 0.08, 0, D / U, D, this.aux.dA + l * T * Q, D, g.get('dA')!, true)
				];
				box(
					{
						name: 'output projection',
						layer: l,
						lo: [x0, -D / U, 0],
						hi: [x0 + Q / U, 0, depth],
						flops: 2 * D * Q * T,
						slices: sl,
						faceIn: fi,
						faceOut: fo,
						K: Q
					},
					1
				);
				x += Q / U + GAP;
			}
			// gate and up
			{
				const x0 = x;
				const gT = llm.tensor(b + 'ffn_gate.weight'),
					uT = llm.tensor(b + 'ffn_up.weight');
				const xb = L.xn2 + l * N * D;
				const gx = g.get('xn2')!;
				const sl = [
					slices(gT, [x0, 0, 0], F, 0, xb, D, ms(b + 'ffn_gate.weight') * gx),
					slices(uT, [x0, -F / U - 0.2, 0], F, 0, xb, D, ms(b + 'ffn_up.weight') * gx)
				];
				const fi = [topFace(x0, D / U, D, xb, D, gx)];
				const fo = [
					sideFace(x0 + D / U + 0.08, 0, F / U, F, L.gate + l * N * F, F, g.get('act')! * 4),
					sideFace(
						x0 + D / U + 0.08,
						-F / U - 0.2,
						F / U,
						F,
						L.up + l * N * F,
						F,
						g.get('act')! * 4
					)
				];
				box(
					{
						name: 'gate and up',
						layer: l,
						lo: [x0, -(2 * F) / U - 0.2, 0],
						hi: [x0 + D / U, 0, depth],
						flops: 2 * 2 * F * D * T,
						slices: sl,
						faceIn: fi,
						faceOut: fo,
						K: D
					},
					2
				);
				x += D / U + GAP;
			}
			// down
			{
				const x0 = x;
				const dT = llm.tensor(b + 'ffn_down.weight');
				const xb = L.act + l * N * F;
				const sl = [slices(dT, [x0, 0, 0], D, 0, xb, F, ms(b + 'ffn_down.weight') * g.get('act')!)];
				const fi = [topFace(x0, F / U, F, xb, F, g.get('act')!)];
				const fo = [
					sideFace(x0 + F / U + 0.08, 0, D / U, D, this.aux.dM + l * T * Q, D, g.get('dM')!, true)
				];
				box(
					{
						name: 'down',
						layer: l,
						lo: [x0, -D / U, 0],
						hi: [x0 + F / U, 0, depth],
						flops: 2 * D * F * T,
						slices: sl,
						faceIn: fi,
						faceOut: fo,
						K: F
					},
					1.2
				);
				x += F / U + GAP;
			}
			x += LAYER_GAP;
			// into its place in the grid
			layerW ||= x - layerX0;
			const shift: V3 = [(l % PER_ROW) * layerW - layerX0, -Math.floor(l / PER_ROW) * ROW_H, 0];
			for (let p = firstPanel; p < P.count; p++) {
				const o = P.desc(p).origin;
				P.set(p, { origin: [o[0] + shift[0], o[1] + shift[1], o[2]] });
			}
			for (let k = firstBox; k < this.boxes.length; k++) {
				const bx = this.boxes[k];
				bx.lo = [bx.lo[0] + shift[0], bx.lo[1] + shift[1], bx.lo[2]];
				bx.hi = [bx.hi[0] + shift[0], bx.hi[1] + shift[1], bx.hi[2]];
			}
		}
		x = PER_ROW * layerW + LAYER_GAP;
		// the vocabulary, for the last word only (the only one whose next word is wanted), folded into columns
		{
			const x0 = x;
			const E = llm.tensor('token_embd.weight');
			const folds = 8;
			const rows = Math.ceil(c.vocab / folds);
			const sl: number[] = [];
			for (let f = 0; f < folds; f++) {
				const m0 = f * rows,
					r = Math.min(rows, c.vocab - m0);
				sl.push(
					P.add({
						origin: [x0 + f * (D / U + 1), 0, (T - 1) * dz],
						ax: [D / U, 0, 0],
						ay: [0, -r / U, 0],
						rows: r,
						cols: D,
						kind: 'ternary',
						tensor: E,
						m0,
						x: { offset: L.final + (T - 1) * D },
						gain: E.meanScale * 0.61 * g.get('final')! * 2.2,
						additive: true,
						alpha: 1,
						samples: 3,
						grid: false,
						lift: 2
					})
				);
			}
			box(
				{
					name: 'the vocabulary (last word only)',
					layer: -1,
					lo: [x0, -rows / U, 0],
					hi: [x0 + folds * (D / U + 1), 0, depth],
					flops: 2 * c.vocab * D,
					slices: sl,
					faceIn: [],
					faceOut: [],
					K: D
				},
				2
			);
			x += folds * (D / U + 1);
		}
		this.length = x;
		this.height = Math.ceil(c.layers / PER_ROW) * ROW_H;
		// the painter, below
		if (this.pc) {
			const y0 = -this.height - 60;
			const tables = this.pc.layout(P, {
				x0: 0,
				y0,
				dz,
				box: (b, dur) => {
					this.boxes.push({ ...b, start: time, dur });
					time += dur;
				}
			});
			this.stage.device.queue.writeBuffer(P.idx, 0, tables);
			this.length = Math.max(this.length, this.pc.width);
			this.height += 60 + this.pc.height;
			this.painterKey = '';
		}
		this.revealLength = time + 1;
		this.panels.allocate();
		void KV;
	}

	setStretch(s: number) {
		this.stretch = s;
		if (!this.T) return;
		this.layout();
		this.reveal = Infinity;
	}

	play() {
		this.reveal = 0;
		this.playing = true;
	}

	skip() {
		this.reveal = Infinity;
		this.playing = false;
	}

	private height = 0;
	/** Everything in view: the nearest distance at which every corner of the scene is on screen. */
	overview(jump = false) {
		const cv = this.stage.gpu.canvas;
		const aspect = cv.clientWidth / Math.max(1, cv.clientHeight);
		const tn = Math.tan((38 * Math.PI) / 360);
		const dz = this.stretch / U;
		const depth = (this.pc ? 1536 : this.T) * dz;
		const lo: V3 = [0, -this.height, 0],
			hi: V3 = [this.length, 12, depth];
		const target: V3 = [(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, (lo[2] + hi[2]) / 2];
		const yaw = 0.28,
			pitch = 0.18;
		const back: V3 = [
			Math.cos(pitch) * Math.sin(yaw),
			Math.sin(pitch),
			Math.cos(pitch) * Math.cos(yaw)
		];
		const f: V3 = [-back[0], -back[1], -back[2]];
		const r = norm([f[1] * 0 - f[2] * 1, f[2] * 0 - f[0] * 0, f[0] * 1 - f[1] * 0]);
		const u: V3 = [r[1] * f[2] - r[2] * f[1], r[2] * f[0] - r[0] * f[2], r[0] * f[1] - r[1] * f[0]];
		const fits = (dist: number) => {
			const eye = [0, 1, 2].map((k) => target[k] + back[k] * dist);
			for (let c = 0; c < 8; c++) {
				const p = [c & 1 ? hi[0] : lo[0], c & 2 ? hi[1] : lo[1], c & 4 ? hi[2] : lo[2]];
				const v = p.map((x, k) => x - eye[k]);
				const z = v[0] * f[0] + v[1] * f[1] + v[2] * f[2];
				if (z <= 0) return false;
				const x = (v[0] * r[0] + v[1] * r[1] + v[2] * r[2]) / (z * tn * aspect);
				const y = (v[0] * u[0] + v[1] * u[1] + v[2] * u[2]) / (z * tn);
				if (Math.abs(x) > 0.92 || Math.abs(y) > 0.8) return false;
			}
			return true;
		};
		let a = 1,
			b = 1e5;
		for (let i = 0; i < 50; i++) {
			const m = Math.sqrt(a * b);
			if (fits(m)) b = m;
			else a = m;
		}
		this.stage.maxDist = Math.max(this.stage.maxDist, b * 1.5);
		const v = { target, dist: b, yaw, pitch };
		if (jump) this.stage.jumpTo(v);
		else this.stage.flyTo(v);
	}

	/** Fly to a box. */
	focus(i: number) {
		const b = this.boxes[i];
		if (!b) return;
		const c: V3 = [(b.lo[0] + b.hi[0]) / 2, (b.lo[1] + b.hi[1]) / 2, (b.lo[2] + b.hi[2]) / 2];
		const size = Math.max(b.hi[0] - b.lo[0], b.hi[1] - b.lo[1], (b.hi[2] - b.lo[2]) * 0.8, 4);
		this.stage.flyTo({ target: c, dist: size * 2.2, yaw: 0.65, pitch: 0.35 });
	}

	private lastActive = -2;
	private painterKey = '';
	update(dt: number) {
		if (this.playing) {
			this.reveal += (dt * this.revealLength) / 45;
			if (this.reveal >= this.revealLength) this.skip();
		}
		const r = this.reveal;
		const P = this.panels;
		let active = -1;
		const pc = this.pc;
		if (pc) {
			const run = pc.run;
			const key = `${run.at.step}|${run.at.block}|${run.done}`;
			if (key !== this.painterKey) {
				this.painterKey = key;
				void pc.refresh(P);
			}
		}
		this.boxes.forEach((b, i) => {
			let state = r < b.start ? 0 : r < b.start + b.dur ? 1 : 2;
			// the painter's boxes light once the painting has reached them
			if (b.painter && !(pc && pc.ready(b.painter))) state = Math.min(state, 0) - 1;
			if (state === 1) active = i;
			// only touch what changes: boxes before and after the one at work keep their state
			if (state === 1 || (b as Box & { state?: number }).state !== state) {
				(b as Box & { state?: number }).state = state;
				const sweep = state === 1 ? ((r - b.start) / b.dur) * b.K : null;
				for (const p of b.slices) P.set(p, { visible: state > 0, sweep: state > 0 ? sweep : null });
				for (const p of b.faceIn) P.set(p, { visible: state > 0 });
				for (const p of b.faceOut) P.set(p, { visible: state === 2 });
			}
		});
		if (this.follow && active >= 0 && active !== this.lastActive) this.focus(active);
		this.lastActive = active;
		void dt;
	}

	draw(pass: GPURenderPassEncoder) {
		this.panels.draw(pass);
	}

	/** The box under the pointer. */
	boxAt(clientX: number, clientY: number): Box | null {
		const { o, d } = this.stage.ray(clientX, clientY);
		let best: Box | null = null,
			bt = Infinity;
		for (const b of this.boxes) {
			let t0 = 0,
				t1 = Infinity,
				hit = true;
			for (let k = 0; k < 3 && hit; k++) {
				const inv = 1 / (d[k] || 1e-12);
				let ta = (b.lo[k] - o[k]) * inv,
					tb = (b.hi[k] + (k === 2 ? 0.001 : 0) - o[k]) * inv;
				if (ta > tb) [ta, tb] = [tb, ta];
				t0 = Math.max(t0, ta);
				t1 = Math.min(t1, tb);
				if (t0 > t1) hit = false;
			}
			if (hit && t0 < bt) {
				bt = t0;
				best = b;
			}
		}
		return best;
	}

	labels(): CubeLabel[] {
		const s = this.stage;
		const out: CubeLabel[] = [];
		const seen = new Set<number>();
		for (const b of this.boxes) {
			if (b.painter) continue;
			if (b.layer >= 0 && seen.has(b.layer)) continue;
			const q = s.project([b.lo[0] + 4, b.hi[1] + 3, 0]);
			const size = 2 / s.unitsPerPixel(q.depth);
			if (!q.front || size < 8) continue;
			seen.add(b.layer);
			out.push({
				text: b.layer >= 0 ? `layer ${b.layer + 1}` : b.name,
				x: q.x,
				y: q.y,
				size: Math.min(size, 26),
				alpha: 0.75
			});
		}
		// the painter: its steps, its blocks when close
		const pc = this.pc;
		if (pc) {
			const put = (text: string, at: V3, world: number, max: number, min = 8) => {
				const q = s.project(at);
				const size = world / s.unitsPerPixel(q.depth);
				if (q.front && size >= min)
					out.push({ text, x: q.x, y: q.y, size: Math.min(size, max), alpha: 0.75 });
			};
			pc.stepY.forEach((y, i) =>
				put(`the painter, step ${i + 1} of ${pc.stepY.length}`, [60, y + 14, 0], 9, 30)
			);
			const first = this.boxes.find((b) => b.painter?.step === -1);
			if (first) put('the painting begins: once', [first.lo[0] + 40, first.hi[1] + 14, 0], 9, 30);
			pc.blockX.forEach((bx, b) =>
				pc.stepY.forEach((y) => put(`block ${b + 1}`, [bx + 8, y + 5, 0], 3, 22, 9))
			);
		}
		// box names when close
		for (const b of this.boxes) {
			if (b.painter) {
				const q = s.project([(b.lo[0] + b.hi[0]) / 2, b.hi[1] + 1.6, 0]);
				const size = 0.9 / s.unitsPerPixel(q.depth);
				if (q.front && size >= 10)
					out.push({ text: b.name, x: q.x, y: q.y, size: Math.min(size, 20), alpha: 0.6 });
				continue;
			}
			const q = s.project([(b.lo[0] + b.hi[0]) / 2, b.hi[1] + 1.1, 0]);
			const size = 0.7 / s.unitsPerPixel(q.depth);
			if (!q.front || size < 9 || b.layer < 0) continue;
			out.push({ text: b.name, x: q.x, y: q.y, size: Math.min(size, 22), alpha: 0.6 });
		}
		return out;
	}

	get firstWords() {
		return this.tokens.slice(0, this.T).map((t) => clean(t) || '·');
	}

	destroy() {
		this.panels.destroy();
	}
}

function norm(v: V3): V3 {
	const l = Math.hypot(v[0], v[1], v[2]) || 1;
	return [v[0] / l, v[1] / l, v[2] / l];
}

function avg(a: number[]) {
	return a.reduce((s, x) => s + x, 0) / Math.max(1, a.length);
}

/** A count of multiply-adds, readably. */
export function flopsText(n: number) {
	if (n >= 1e12) return `${(n / 1e12).toFixed(2)} trillion`;
	if (n >= 1e9) return `${(n / 1e9).toFixed(2)} billion`;
	if (n >= 1e6) return `${(n / 1e6).toFixed(1)} million`;
	return n.toLocaleString();
}
